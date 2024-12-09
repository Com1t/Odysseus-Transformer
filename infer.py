import argparse
from utils.globals import _set_global_memory_buffer
import torch
import os
from transformers import set_seed
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaModel
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import ShardingStrategy
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    get_model_parallel_group,
)
import time
import numpy as np

from decoder.ulysses import apply_ulysses_attn_patch_llama
from decoder.tensor_parallel import apply_tpsp_attn_patch_llama
from utils.apply_seq_parallel import prepare_attn_inputs
from torch.profiler import profile, record_function, ProfilerActivity


def init_prof(use_profiler):
    activities = []
    # activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    from contextlib import nullcontext

    ctx = (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/"),
            record_shapes=True,
            with_stack=True,
        )
        if use_profiler
        else nullcontext()
    )
    return ctx


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    assert args.batch_size == 1, "Only support batch size 1"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    tp_rank = get_model_parallel_rank()
    tp_pg = get_model_parallel_group()

    set_seed(args.seed)
    dev = torch.device(f"cuda:{local_rank}")
    print(f"local rank: {local_rank}, dev id: {dev}")
    _set_global_memory_buffer(dev)

    if args.use_cfg_init_model:
        cfg = LlamaConfig()
        cfg.hidden_size = 4096
        cfg.intermediate_size = 11008
        cfg.num_attention_heads = 32
        cfg.num_key_value_heads = 8
        cfg.num_hidden_layers = 16
        cfg._attn_implementation = "sdpa"
        cfg.rope_theta = args.rope_theta
        cfg.torch_dtype = torch.bfloat16
        model = transformers.LlamaForCausalLM(cfg).to(dtype=torch.bfloat16).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            rope_theta=args.rope_theta,
            _attn_implementation="sdpa",
            do_sample=True,  # fix warning
        )

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"total parameters {total_params/1e9:.2f}B")

    if args.parallel_mode == "ulysses":
        apply_ulysses_attn_patch_llama(model)
    elif args.parallel_mode == "tpsp":
        apply_tpsp_attn_patch_llama(model, sequence_parallel=False)

    model = model.to(dev)

    if rank == 0:
        print(
            f"{args.parallel_mode} After init model, CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
        )

    assert isinstance(
        model, (transformers.LlamaForCausalLM)
    ), "Only support llama model"

    layer_num = model.config.num_hidden_layers
    if world_size > 1:
        if (
            args.parallel_mode == "ulysses"
        ):
            model = FSDP(
                model,
                process_group=tp_pg
            )
        elif args.parallel_mode == "tpsp":
            ignored_modules = []
            layer_num = model.config.num_hidden_layers
            for i in range(layer_num):
                ignored_modules.append(model.model.layers[i].self_attn)
                ignored_modules.append(model.model.layers[i].mlp)
            model = FSDP(
                model,
                ignored_modules=ignored_modules,
                process_group=tp_pg
            )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with torch.no_grad():
        warmup_num_iterations = 2

        if rank == 0:
            print(
                f"{args.parallel_mode} During infer, CUDA memory allocated: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
            )

        ctx = init_prof(args.use_profiler)
        for tot_seq_length in 1024, 2048, 4096, 8192, 16384, 32768, 65536:
            with ctx as prof:
                elapse = 0.0
                seq_length = tot_seq_length // world_size
                for step in range(args.max_train_steps):
                    if step > warmup_num_iterations:
                        start_time = time.time()
                    vocab_size = model.config.vocab_size
                    batch = torch.randint(vocab_size, size=(1, seq_length + 1))

                    input_ids = batch[..., :-1]
                    target_ids = batch[..., 1:]
                    position_ids = (
                        torch.arange(seq_length)
                        .unsqueeze(0)
                        .expand(input_ids.shape[0], -1)
                    )
                    prepared = prepare_attn_inputs(
                        args.parallel_mode,
                        input_ids,
                        position_ids,
                        target_ids,
                        rank,
                        world_size,
                        dev,
                    )
                    local_input_ids = prepared["local_input_ids"]
                    local_position_ids = prepared["local_position_ids"]
                    local_target_ids = prepared["local_target_ids"]

                    logits = model(
                        local_input_ids,
                        position_ids=local_position_ids,
                    ).logits
                    if rank == 0:
                        print(
                            f"step {step} CUDA memory allocated/reserved: {torch.cuda.memory_allocated(dev) / 1024 ** 3:.2f}/{torch.cuda.memory_reserved(dev) / 1024 ** 3:.2f} GB"
                        )

                    if step > warmup_num_iterations:
                        end_time = time.time()
                        elapse += end_time - start_time

                    if step >= args.max_train_steps:
                        break

                    if args.use_profiler:
                        prof.step()

            if rank == 0:
                print(
                    f"{args.parallel_mode} {tot_seq_length//1024}K {world_size} Time taken: {elapse:.2f} seconds"
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=10)
    args.add_argument("--rope-theta", type=float, default=100000)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--use_cfg_init_model", action="store_true", default=False)
    args.add_argument("--use_profiler", action="store_true", default=False)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["ulysses", "tpsp"],
    )
    main(args.parse_args())
