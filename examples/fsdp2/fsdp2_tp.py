# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example of training with Context Parallel using FSDP2 via Accelerate.
This example demonstrates how to use Accelerate's context_parallel feature for efficient long sequence training.
"""

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed, TorchTensorParallelPlugin
from utils import PerformanceTracker, create_collate_fn, get_dataset, setup_tokenizer


MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"


def main():
    """
    Main function to train the model.
    """
    set_seed(42)

    device_mesh = init_device_mesh(mesh_dim_names=("dp_shard", "tp"), mesh_shape=(4, 2), device_type="cuda")

    fsdp2_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        cpu_ram_efficient_loading=False,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
        device_mesh=device_mesh,
    )
    fsdp2_plugin.set_mixed_precision("bf16")

    accelerator = Accelerator(
        fsdp_plugin=fsdp2_plugin,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        tp_plan="auto",
        device_mesh=device_mesh,
    )

    tokenizer = setup_tokenizer(MODEL_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model, optimizer = accelerator.prepare(model, optimizer)

    dataset = get_dataset(accelerator, tokenizer, 256)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())
    dataloader = accelerator.prepare(dataloader)

    model.train()

    total_num_steps = min(1000, len(dataloader))
    performance_tracker = PerformanceTracker(warmup_steps=10)

    for step, batch in enumerate(dataloader):
        if step >= total_num_steps:
            break

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        batch_tokens = batch["input_ids"].shape[1]
        metrics = performance_tracker.step(batch_tokens)

        print_msg = f"Step {step}/{total_num_steps}, Loss: {loss.item():.4f}"
        log_metrics = {"loss": loss.item()}

        if "warmup_completed" in metrics:
            accelerator.print("Warm up completed! Starting performance tracking...")
        elif metrics:
            print_msg += f" | Average steps/s: {metrics['steps_per_second']:.2f}"

        if step % 10 == 0 or step == total_num_steps - 1:
            accelerator.print(print_msg)

        accelerator.log(log_metrics)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.print("Training completed!")


if __name__ == "__main__":
    main()
