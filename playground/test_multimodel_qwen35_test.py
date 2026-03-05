#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B — Local Deployment & Benchmark
Hardware : NVIDIA TITAN RTX 24 GB VRAM  |  32 GB DDR4 RAM
Engine   : llama-cpp-python (CUDA build)
"""

import os, sys, time, threading, argparse, subprocess
from dataclasses import dataclass, field
from typing import Optional

# ── Dependency bootstrap ───────────────────────────────────────────────────────
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import psutil
except ImportError:
    _pip("psutil"); import psutil

try:
    # Replaced deprecated pynvml with nvidia-ml-py
    import pynvml; pynvml.nvmlInit(); _NVML = True
except Exception:
    try: _pip("nvidia-ml-py"); import pynvml; pynvml.nvmlInit(); _NVML = True
    except Exception: _NVML = False

try:
    from llama_cpp import Llama
except ImportError:
    print(
        "❌  llama-cpp-python not found.\n"
        "   Install with CUDA:\n"
        '   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python '
        "--upgrade --force-reinstall --no-cache-dir"
    ); sys.exit(1)

# ── Hardware constants ─────────────────────────────────────────────────────────
VRAM_TOTAL_GB  = 24.0          
RAM_TOTAL_GB   = 32.0
# Increased safety buffer from 1.5 to 2.5 to prevent CUDA Out Of Memory at load
VRAM_USABLE_GB = VRAM_TOTAL_GB - 2.5   

# ── Qwen3.5-35B-A3B architecture constants ────────────────────────────────────
ARCH = dict(
    n_layers          = 40,
    n_attn_full_layers= 10,   
    n_attn_heads      = 16,
    n_kv_heads        = 2,    
    head_dim          = 128,
    hidden_dim        = 2048,
    n_experts         = 256,
    n_active_experts  = 9,
    expert_inter_dim  = 512,
    native_ctx        = 262_144,
    rope_theta        = 1_000_000,   
)

# ── Quantization catalogue ─────────────────────────────────────────────────────
QUANT_CATALOGUE = {
    "UD-Q4_K_XL"        : ("Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",        22.2, "RECOMMENDED — Unsloth Dynamic, SOTA Pareto"),
    "Q4_K_M"            : ("Qwen3.5-35B-A3B-Q4_K_M.gguf",             22.0, "Standard 4-bit"),
    "UD-Q4_K_L"         : ("Qwen3.5-35B-A3B-UD-Q4_K_L.gguf",         20.2, "Smaller, slightly lower quality"),
    "Q5_K_M"            : ("Qwen3.5-35B-A3B-Q5_K_M.gguf",             24.8, "Best quality; needs --hybrid (expert offload)"),
    "UD-Q5_K_XL"        : ("Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf",        24.9, "Best quality Unsloth; needs --hybrid"),
}

# ── KV cache estimator ────────────────────────────────────────────────────────
def estimate_kv_cache_gb(n_ctx: int) -> float:
    a = ARCH
    kv_bytes = (
        2 * a["n_kv_heads"] * a["head_dim"] * n_ctx * a["n_attn_full_layers"] * 2 
    )
    delta_state_bytes = a["hidden_dim"] * a["hidden_dim"] * (a["n_layers"] - a["n_attn_full_layers"]) * 2
    return (kv_bytes + delta_state_bytes) / (1024**3)

# ── Layer offload calculator ───────────────────────────────────────────────────
def compute_layer_plan(model_size_gb: float, n_ctx: int, manual_n_gpu: int = None) -> dict:
    kv_gb        = estimate_kv_cache_gb(n_ctx)
    workspace_gb = 0.3
    budget_gb    = VRAM_USABLE_GB - kv_gb - workspace_gb
    total_llama_layers = ARCH["n_layers"] + 2
    per_layer_gb       = model_size_gb / total_llama_layers

    if manual_n_gpu is not None:
        mode  = "MANUAL OVERRIDE"
        n_gpu = manual_n_gpu
    else:
        if model_size_gb <= budget_gb:
            mode  = "FULL_VRAM ✅"
            n_gpu = 999
        else:
            mode  = "HYBRID VRAM+RAM ⚠️"
            n_gpu = min(int(budget_gb / per_layer_gb), total_llama_layers)

    vram_est_gb = (min(n_gpu, total_llama_layers) * per_layer_gb) + kv_gb + workspace_gb
    ram_est_gb  = max(0, (total_llama_layers - n_gpu)) * per_layer_gb

    return {
        "mode"            : mode,
        "n_gpu_layers"    : n_gpu,
        "kv_cache_gb"     : round(kv_gb, 3),
        "budget_gb"       : round(budget_gb, 2),
        "per_layer_gb"    : round(per_layer_gb, 3),
        "est_vram_gb"     : round(vram_est_gb, 2),
        "est_ram_gb"      : round(ram_est_gb,  2),
    }

# ── VRAM / RAM monitor (background thread) ────────────────────────────────────
class MemoryMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 0.05):
        self._interval = interval
        self._samples  = []
        self._running  = False
        self._thread   = None
        self._handle   = pynvml.nvmlDeviceGetHandleByIndex(gpu_index) if _NVML else None

    def _sample_loop(self):
        while self._running:
            vram = 0.0
            if _NVML:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                vram = info.used / (1024**3)
            ram = psutil.virtual_memory().used / (1024**3)
            self._samples.append({"vram": vram, "ram": ram})
            time.sleep(self._interval)

    def start(self):
        self._samples = []
        self._running = True
        self._thread  = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        self._thread.join(timeout=0.5)
        if not self._samples:
            return {"peak_vram_gb": 0.0, "peak_ram_gb": 0.0, "avg_vram_gb": 0.0}
        return {
            "peak_vram_gb" : round(max(s["vram"] for s in self._samples), 2),
            "peak_ram_gb"  : round(max(s["ram"]  for s in self._samples), 2),
            "avg_vram_gb"  : round(sum(s["vram"] for s in self._samples) / len(self._samples), 2),
        }

# ── Build model configuration ─────────────────────────────────────────────────
def build_llama_config(model_path: str, n_ctx: int, model_size_gb: float,
                       hybrid: bool = False, manual_n_gpu: int = None) -> dict:
    plan = compute_layer_plan(model_size_gb, n_ctx, manual_n_gpu)

    print("\n" + "═" * 62)
    print("  Layer Allocation Plan")
    print("═" * 62)
    for k, v in plan.items():
        print(f"  {k:<22}: {v}")
    print("═" * 62)

    n_gpu_layers = plan["n_gpu_layers"]

    if hybrid and manual_n_gpu is None:
        n_gpu_layers = ARCH["n_layers"] + 2   
        print(
            "\n  ⚡  Hybrid mode active. For precise expert-only CPU routing,\n"
            "  run llama-server instead of this python script.\n"
        )

    return {
        "model_path"      : model_path,
        "n_ctx"           : n_ctx,
        "n_gpu_layers"    : n_gpu_layers,
        "n_batch"         : 512,
        "rope_freq_base"  : float(ARCH["rope_theta"]),
        "flash_attn"      : True,
        "use_mmap"        : False,
        "use_mlock"       : False,
        "n_threads"       : 8,
        "n_threads_batch" : 8,
        "verbose"         : False,
    }

# ── Benchmark prompts ─────────────────────────────────────────────────────────
PROMPTS = [
    "What is Mixture-of-Experts and why does it allow a 35B model to run with only 3B active parameters?",
    "Write a Python function implementing a thread-safe LRU cache using only the standard library. Include type hints, a docstring, and usage example.",
    "A train travels from A to B at 120 km/h, then returns at 80 km/h. Calculate the harmonic mean speed for the full round trip. Show each step.",
    "Compare transformer self-attention, linear attention (DeltaNet/Mamba), and state-space models on: (a) computational complexity vs. sequence length, (b) memory requirements, (c) suitability for long-context tasks. Use concrete model examples in each category.",
    "Implement a production-ready async REST API client in Python with: retry logic with exponential backoff, connection pooling, structured logging, and a typed response model using dataclasses.",
]

# ── Benchmark runner ───────────────────────────────────────────────────────────
def run_benchmark(llm: "Llama", max_tokens: int = 512) -> list[dict]:
    monitor = MemoryMonitor()
    results = []

    GEN_PARAMS = dict(
        temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0.0,
        stop = ["<|im_end|>", "<|endoftext|>"], echo = False,
    )

    print(f"\n{'─'*62}")
    print(f"  Benchmark  ·  {len(PROMPTS)} prompts  ·  max_tokens={max_tokens}")
    print(f"{'─'*62}")

    for idx, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{idx}/{len(PROMPTS)}] {prompt[:72]}…")

        n_prompt = len(llm.tokenize(prompt.encode()))
        monitor.start()
        t_start          = time.perf_counter()
        first_token_time = None
        chunks: list[str]= []

        stream = llm(prompt, max_tokens=max_tokens, stream=True, **GEN_PARAMS)

        for chunk in stream:
            text = chunk["choices"][0]["text"]
            if first_token_time is None and text:
                first_token_time = time.perf_counter()
            chunks.append(text)

        t_end    = time.perf_counter()
        mem      = monitor.stop()

        ttft     = (first_token_time - t_start) if first_token_time else float("nan")
        response = "".join(chunks)
        n_out    = len(llm.tokenize(response.encode()))
        gen_time = t_end - (first_token_time or t_start)
        tps      = n_out / gen_time if gen_time > 0 else 0.0

        rec = {
            "idx": idx, "prompt_tok": n_prompt, "output_tok": n_out,
            "ttft_s": round(ttft, 3), "tps": round(tps, 1),
            "total_s": round(t_end - t_start, 2),
            "peak_vram_gb": mem["peak_vram_gb"], "peak_ram_gb" : mem["peak_ram_gb"],
        }
        results.append(rec)

        print(f"  ✓  TTFT {rec['ttft_s']:.3f}s  |  TPS {rec['tps']:.1f} tok/s  |  VRAM {rec['peak_vram_gb']:.1f} GB  |  RAM {rec['peak_ram_gb']:.1f} GB")

        llm.reset()   

    return results

def print_summary(results: list[dict], quant: str, n_ctx: int) -> None:
    if not results: return
    avg_ttft  = sum(r["ttft_s"] for r in results) / len(results)
    avg_tps   = sum(r["tps"] for r in results) / len(results)
    peak_vram = max(r["peak_vram_gb"] for r in results)
    peak_ram  = max(r["peak_ram_gb"] for r in results)

    print(f"\n{'═'*62}")
    print(f"  BENCHMARK SUMMARY  ·  {quant}  ·  n_ctx={n_ctx}")
    print(f"{'═'*62}")
    print(f"  Avg TTFT        : {avg_ttft:.3f} s")
    print(f"  Avg TPS         : {avg_tps:.1f} tokens/sec")
    print(f"  Peak VRAM       : {peak_vram:.1f} / {VRAM_TOTAL_GB:.0f} GB")
    print(f"  Peak RAM        : {peak_ram:.1f} / {RAM_TOTAL_GB:.0f} GB")
    print(f"{'═'*62}\n")

# ── Model download helper ─────────────────────────────────────────────────────
def ensure_model(quant: str, local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    
    fname, _, _ = QUANT_CATALOGUE[quant]
    path = os.path.join(local_dir, fname)
    if os.path.exists(path):
        print(f"✓  Model found: {path}")
        return path
        
    print(f"⬇  Downloading {fname}  ({QUANT_CATALOGUE[quant][1]} GB) to {local_dir}…")
    try:
        from huggingface_hub import hf_hub_download
        # Added local_dir_use_symlinks=False to prevent cross-drive linking bugs
        path = hf_hub_download(
            repo_id   = "unsloth/Qwen3.5-35B-A3B-GGUF",
            filename  = fname,
            local_dir = local_dir,
            local_dir_use_symlinks = False, 
        )
        return path
    except ImportError:
        print("Install: pip install huggingface-hub")
    except Exception as e:
        print(f"Download error: {e}")
    sys.exit(1)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-35B-A3B local benchmark — TITAN RTX 24 GB")
    parser.add_argument(
        "--quant", default="UD-Q4_K_XL", choices=list(QUANT_CATALOGUE.keys()),
        help="GGUF quantization variant (default: UD-Q4_K_XL)",
    )
    parser.add_argument(
        "--model-dir", default="/media/gatv-projects/ssd/AI_models",
        help="Local directory to cache / load GGUFs",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-ctx", type=int, default=16_384)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    # Added manual VRAM layer override for debugging OOMs
    parser.add_argument("--n-gpu", type=int, default=None, help="Manually set n_gpu_layers to override auto-calculation")
    
    args = parser.parse_args()

    fname, model_size_gb, notes = QUANT_CATALOGUE[args.quant]
    print(f"\n  Qwen3.5-35B-A3B  ·  {args.quant}  ({model_size_gb} GB)")
    print(f"  {notes}\n")

    if model_size_gb > VRAM_USABLE_GB and not args.hybrid and args.n_gpu is None:
        print(f"⚠  {args.quant} ({model_size_gb} GB) exceeds usable VRAM. Enabling --hybrid mode.\n")
        args.hybrid = True

    model_path = args.model_path or ensure_model(args.quant, args.model_dir)
    config     = build_llama_config(model_path, args.n_ctx, model_size_gb, args.hybrid, args.n_gpu)

    print(f"\n🚀  Loading model (This may take a moment if not cached in RAM)…")
    t0  = time.perf_counter()
    
    try:
        llm = Llama(**config)
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL. Error: {e}")
        print("💡 Possible Fixes:")
        print("1. If the previous download created a broken symlink, delete it:")
        print(f"   rm {model_path}")
        print("   Then run the script again to force a hard copy download.")
        print("2. You might still be running out of VRAM during the initial load.")
        print("   Try running with a lower layer count manually:")
        print(f"   python test_multimodel_qwen35_test.py --quant {args.quant} --n-ctx {args.n_ctx} --max-tokens {args.max_tokens} --n-gpu 35")
        sys.exit(1)
        
    print(f"✓   Loaded in {time.perf_counter() - t0:.1f} s\n")

    results = run_benchmark(llm, max_tokens=args.max_tokens)
    print_summary(results, args.quant, args.n_ctx)

    if args.interactive:
        print("Interactive mode (Ctrl-C to exit)\n")
        try:
            while True:
                user = input("You: ").strip()
                if not user: continue
                t0 = time.perf_counter()
                output = llm(user, max_tokens=1024, temperature=0.6, top_p=0.95, top_k=20, stop=["<|im_end|>", "<|endoftext|>"], echo=False)
                elapsed = time.perf_counter() - t0
                text = output["choices"][0]["text"]
                n_tok = output["usage"]["completion_tokens"]
                print(f"\nAssistant: {text}\n[{n_tok} tokens · {n_tok/elapsed:.1f} TPS]\n")
                llm.reset()
        except KeyboardInterrupt:
            print("\nDone.")

if __name__ == "__main__":
    main()
 