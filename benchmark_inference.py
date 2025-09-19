import argparse
import time
import torch
import numpy as np
import os
import sys

# Ensure the local pytorch_transformers module is in the path
curPath = os.path.abspath(os.path.dirname(__file__))
# Assuming this script is in the root of K-Adapter project
sys.path.append(curPath)

from pytorch_transformers.my_modeling_roberta import RobertaModelwithAdapter
from pytorch_transformers import RobertaTokenizer

def run_benchmark(model, tokenizer, device, batch_size=8, sequence_length=128):
    """
    Runs a benchmark for a given model.

    Args:
        model: The model to benchmark.
        tokenizer: The tokenizer for the model.
        device: The device to run on ('cuda' or 'cpu').
        batch_size: The batch size for inference.
        sequence_length: The sequence length of the input.
    """
    model.eval()
    model.to(device)

    # --- Create dummy input (compatible with older tokenizer) ---
    dummy_text = "some text to encode"
    # 1. Encode text to token IDs
    encoded_ids = tokenizer.encode(dummy_text, add_special_tokens=True)
    
    # 2. Manually pad the sequence to max_length
    pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.eos_token_id

    if len(encoded_ids) < sequence_length:
        padding = [pad_token_id] * (sequence_length - len(encoded_ids))
        encoded_ids += padding
    elif len(encoded_ids) > sequence_length:
        encoded_ids = encoded_ids[:sequence_length]

    input_batch = torch.tensor([encoded_ids] * batch_size).to(device)
    
    # --- GPU Warm-up ---
    print("Warming up GPU...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(input_batch)

    # --- Timing ---
    print("Starting benchmark...")
    timings = []
    
    # Reset peak memory stats
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for _ in range(100):
        if device == 'cuda':
            starter.record()
            with torch.no_grad():
                _ = model(input_batch)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
        else: # CPU timing
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(input_batch)
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000) # convert to ms

    # --- Results ---
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    throughput = (batch_size * 100) / (sum(timings) / 1000)

    print(f"\n--- Results for Batch Size: {batch_size} ---")
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"Latency StdDev:  {std_latency:.3f} ms")
    print(f"Throughput:      {throughput:.2f} samples/sec")

    if device == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Peak GPU Memory:   {peak_memory_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark for K-Adapter")
    parser.add_argument(
        "--test_name", 
        type=str, 
        required=True, 
        choices=["base", "fac", "fac_lin"],
        help="The name of the test to run."
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,8,16,32",
        help="Comma-separated list of batch sizes to test."
    )
    cli_args = parser.parse_args()

    # --- Common Model Arguments (from run_example.sh) ---
    args = argparse.Namespace()
    args.adapter_size = 768 # Corrected to match pre-trained fac-adapter model
    args.adapter_list = [0, 11, 22] # Using 0,11,22 instead of 0,11,23 from example
    args.adapter_transformer_layers = 2
    args.adapter_skip_layers = 0
    args.fusion_mode = 'concat'
    args.freeze_bert = False
    args.freeze_adapter = True
    args.meta_fac_adaptermodel = ""
    args.meta_lin_adaptermodel = ""
    
    # --- Model Loading Logic ---
    print(f"--- Loading Test Case: {cli_args.test_name} ---")
    
    fac_adapter_path = "./pretrained_models/fac-adapter/pytorch_model.bin"
    lin_adapter_path = "./pretrained_models/lin-adapter/pytorch_model.bin"

    if cli_args.test_name == "base":
        # For base, we don't provide any adapter model paths
        pass
    elif cli_args.test_name == "fac":
        args.meta_fac_adaptermodel = fac_adapter_path
    elif cli_args.test_name == "fac_lin":
        args.meta_fac_adaptermodel = fac_adapter_path
        args.meta_lin_adaptermodel = lin_adapter_path

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device # Ensure device is set before model instantiation
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    # Get pad token id, if it's not set, use eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = RobertaModelwithAdapter(args)
    
    # --- Run Benchmarks ---
    batch_sizes = [int(bs) for bs in cli_args.batch_sizes.split(',')]
    for bs in batch_sizes:
        run_benchmark(model, tokenizer, device, batch_size=bs)


if __name__ == "__main__":
    main()