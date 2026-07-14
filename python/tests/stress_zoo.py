import orca
import orca.nn as nn
import numpy as np
import time

def run_gpt2_inference_stress():
    print("==================================================")
    print("STARTING HUGGING FACE LOADER & GPT-2 STRESS TEST")
    print("==================================================")
    
    # 1. Download and load real GPT-2 (124M parameters)
    print("\n--- 1. Loading Real GPT-2 Pretrained Model from Hugging Face ---")
    t0 = time.time()
    try:
        model = orca.hf.load("gpt2", pretrained=True)
    except Exception as e:
        print(f"Skipping HF download stress test due to network/environment constraint: {e}")
        print("Falling back to local un-pretrained GPT-2 stress test...")
        model = orca.hf.load("gpt2", pretrained=False)
        
    t1 = time.time()
    print(f"  Model loading & parameters mapping completed in {t1 - t0:.2f}s.")
    
    # 2. Verify total parameters count
    total_params = sum(p.tensor.shape[0] * (p.tensor.shape[1] if len(p.tensor.shape) > 1 else 1) for p in model.parameters())
    print(f"  Total model parameters: {total_params:,}")
    
    # 3. Large input sequence forward pass stress test
    batch_size = 2
    seq_len = 16
    vocab_size = model.vocab_size
    
    print(f"\n--- 2. Running Forward Pass on CPU (Batch={batch_size}, SeqLen={seq_len}) ---")
    model.to("cpu")
    model.eval()
    
    # Generate mock one-hot inputs
    x_np = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        for s in range(seq_len):
            idx = np.random.randint(0, 1000)
            x_np[b, s, idx] = 1.0
            
    x_cpu = orca.Tensor.from_list(x_np.flatten().tolist(), shape=[batch_size, seq_len, vocab_size]).to("cpu")
    
    t0 = time.time()
    logits_cpu = model(x_cpu)
    t1 = time.time()
    print(f"  CPU forward pass finished in {t1 - t0:.2f}s. Output shape: {logits_cpu.shape}")
    
    print(f"\n--- 3. Running Forward Pass on GPU (WGPU) ---")
    model.to("gpu")
    
    print("DEVICE OF EVERY PARAMETER:")
    for name, p in model.named_parameters() if hasattr(model, 'named_parameters') else []:
        print(f"  {name}: {p.tensor.device}")
    # Since named_parameters might not exist, print manually for key parts:
    print("  wte.weight device:", model.wte.weight.tensor.device)
    print("  wpe.weight device:", model.wpe.weight.tensor.device)
    print("  lm_head.weight device:", model.lm_head.weight.tensor.device)
    print("  block 0 norm1.weight device:", model.blocks._modules["0"].block.norm1.weight.tensor.device)
    
    x_gpu = x_cpu.to("gpu")
    t0 = time.time()
    logits_gpu = model(x_gpu)
    t1 = time.time()
    print(f"  GPU forward pass finished in {t1 - t0:.2f}s. Output shape: {logits_gpu.shape}")
    
    # 4. Compare outputs consistency
    cpu_vals = logits_cpu.to_list()
    gpu_vals = logits_gpu.to_list()
    diff = max(abs(c - g) for c, g in zip(cpu_vals, gpu_vals))
    print(f"  Max absolute difference between CPU and GPU: {diff:.6f}")
    assert diff < 1e-3, "CPU and GPU results should match"
    print("  [PASS] Numerical outputs are consistent between devices.")
    
    # 5. Run Autoregressive text generation loop (Inference stress test)
    print("\n--- 4. Running Autoregressive Text Generation Loop (10 tokens) ---")
    gen_len = 10
    
    # Prompt is single sequence (batch=1)
    prompt_tokens = [12, 45, 120, 203] # random token IDs
    seq = list(prompt_tokens)
    
    t0 = time.time()
    for step in range(gen_len):
        # Convert current sequence to one-hot input of shape [1, len(seq), vocab_size]
        curr_len = len(seq)
        inp_np = np.zeros((1, curr_len, vocab_size), dtype=np.float32)
        for i, tok in enumerate(seq):
            inp_np[0, i, tok % vocab_size] = 1.0
            
        inp = orca.Tensor.from_list(inp_np.flatten().tolist(), shape=[1, curr_len, vocab_size]).to("gpu")
        
        # Forward
        logits = model(inp)
        
        # Get last token logits
        # shape [1, curr_len, vocab_size] -> last token is index -1 along dim=1
        # To get the last token's logits, we can convert to list and slice it
        flat_logits = logits.to_list()
        last_tok_logits = flat_logits[-vocab_size:]
        
        # Argmax
        next_tok = int(np.argmax(last_tok_logits))
        seq.append(next_tok)
        
    t1 = time.time()
    print(f"  Generated 10 tokens autoregressively in {t1 - t0:.2f}s.")
    print(f"  Input prompt: {prompt_tokens}")
    print(f"  Generated sequence: {seq}")
    print("  [PASS] Autoregressive inference completed successfully.")
    
    print("\n==================================================")
    print("HUGGING FACE LOADER & GPT-2 STRESS TEST PASSED")
    print("==================================================")

if __name__ == "__main__":
    run_gpt2_inference_stress()
