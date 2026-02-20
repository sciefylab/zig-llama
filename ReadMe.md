# zig-llama

**MatMul-Free LLM Inference Engine** — Written in pure Zig, zero dependencies.

Runs GGUF models (Qwen2.5, LLaMA, TinyLLaMA, etc.) with a unique **Shift-Add SIMD engine** that uses **zero multiply instructions** in the hot path.

## Features

- **MatMul-Free Inference** — Shift-Add bit decomposition, zero MUL instructions
- **Ternary CSD Engine** — Canonical Signed Digit decomposition (~2.8 terms/weight)
- **Native SIMD Fast Path** — Optional hardware i16 multiply (vpmullw/AVX2)
- **Q8_0 / Q4_0 Quantization** — Direct GGUF file loading
- **Multi-threaded** — 8-thread worker pool with fused QKV/FFN
- **Zero-copy Loading** — Direct file read to aligned buffers, no double allocation
- **Pure Zig** — No C dependencies, no libc, cross-platform

## Quick Start

```bash
# Build and run (default: matmul-free shift-add)
zig build run -Doptimize=ReleaseFast -Dcpu=native -- models/qwen2.5-coder-1.5b-instruct-q8_0.gguf

# With custom prompt
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf \
  -p "Write a hello world in Python" \
  -n 128 --threads 8
```

Engine Modes
zig-llama supports three computation engines. The default is MatMul-Free Shift-Add.

1. MatMul-Free: Shift-Add SIMD (Default)
Zero multiply instructions. Decomposes 7-bit multiplication into shifts and adds.

```bash

zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --profile --threads 8
```  

Mode: MatMul-Free (Shift-Add SIMD)
Generate: 81 tokens in 5538 ms (14.63 tokens/sec)
MatMul-Free: YES (zero MUL instructions)
2. MatMul-Free: Ternary CSD
Zero multiply instructions. Canonical Signed Digit decomposition with ~2.8 terms per weight.

Bash

# On-the-fly (no extra memory)
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --ternary-otf --threads 8

# Cached (fastest matmul-free, uses ~5.6x weight memory)
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --ternary --threads 8
3. Native SIMD (Fastest)
Uses hardware i16 multiply (vpmullw on AVX2). Not matmul-free but fastest.

To enable, edit src/lut_mul.zig:

zig

pub const USE_NATIVE_DOT: bool = true;  // false = matmul-free (default)
text

Mode: Native SIMD (hardware multiply)
Generate: 81 tokens in 4473 ms (18.11 tokens/sec)
MatMul-Free: NO (using hardware multiply)
Performance
Tested on Qwen2.5-Coder-1.5B Q8_0 (1564 MB weights):

text

┌───────────────────────────┬───────────┬───────────┬──────────────┐
│ Mode                      │ Generate  │ MatMul    │ MatMul-Free? │
├───────────────────────────┼───────────┼───────────┼──────────────┤
│ Shift-Add (default)       │ 14.6 t/s  │ 5091 ms   │ ✅ YES       │
│ Ternary CSD               │ ~14 t/s   │ ~5200 ms  │ ✅ YES       │
│ Native i16 multiply       │ 18.1 t/s  │ 3904 ms   │ ❌ NO        │
└───────────────────────────┴───────────┴───────────┴──────────────┘

Bandwidth utilization: ~29 GB/s (~73-85% of DDR4 theoretical)
Bottleneck: Memory bandwidth (not compute)
The matmul-free shift-add engine is only 24% slower than hardware multiply — because inference is memory-bandwidth bound, not compute-bound.

Supported Models
Any GGUF model with Q8_0 or Q4_0 quantization:

Bash

# Qwen2.5 Coder 1.5B (recommended for testing)
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --profile --quiet --threads 8

# Qwen2.5 Coder 3B
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/Qwen2.5-Coder-3B-Instruct-Q8_0.gguf --profile --quiet --threads 8

# Qwen2.5 Coder 7B
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf --profile --quiet --threads 8

# Qwen3 4B
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/Qwen3-4B-Thinking-2507-Q8_0.gguf --profile --quiet --threads 8

# TinyLLaMA 1.1B (Q4_K_M)
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/tinyllama-1.1b-chat.Q4_K_M.gguf --threads 8
CLI Reference
text

zig-llama [options] <model_path>

Model:
  -m, --model <path>       GGUF model file
  -p, --prompt <text>      Input prompt (default: "Write a hello world in Python")
  -n, --max-tokens <N>     Max tokens to generate (default: 128)

Sampling:
  -t, --temperature <F>    Temperature (default: 0.7)
  --top-p <F>              Top-p nucleus sampling (default: 0.9)
  --top-k <N>              Top-k sampling (default: 40)

Performance:
  -j, --threads <N>        Worker threads (default: 8)
  --no-mt-wo               Disable MT for output projection
  --no-mt-w2               Disable MT for FFN down projection
  --profile                Enable profiling breakdown

MatMul-Free Engine:
  --ternary                Ternary CSD cached (recommended matmul-free)
  --ternary-otf            Ternary CSD on-the-fly (no extra memory)
  --ternary-test           Run ternary validation suite
  --ternary-bench          Run kernel benchmark comparison

Output:
  -q, --quiet              Suppress model loading details
  --show-isa               Show ISA/CPU detection info
  --no-validation          Disable input validation
  -h, --help               Show help
Diagnostics
Bash

# Show CPU/ISA detection and kernel info
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --show-isa --quiet

# Run ternary validation suite
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --ternary-test --quiet

# Benchmark all kernel variants
zig build run -Doptimize=ReleaseFast -Dcpu=native -- \
  models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --ternary-bench --quiet
Architecture
text

┌─────────────────────────────────────────────────────┐
│                    main.zig                          │
│              CLI parsing, generation loop             │
├─────────────────────────────────────────────────────┤
│                 inference.zig                        │
│        Forward pass, attention, RMSNorm, SiLU        │
├──────────────┬──────────────┬───────────────────────┤
│  tensor.zig  │  lut_mul.zig │    ternary.zig        │
│  SIMD ops    │  Shift-Add   │    CSD decomp         │
│  MatVec MT   │  Native dot  │    Grouped kernel     │
│  Thread pool │  Dispatcher  │    Weight cache        │
├──────────────┴──────────────┴───────────────────────┤
│                  model.zig                           │
│          GGUF loading, direct file read              │
├─────────────────────────────────────────────────────┤
│     gguf.zig      │   tokenizer.zig  │  sampler.zig │
│   File parser     │   GPT-2/SP encode│  Top-p/k     │
└───────────────────┴──────────────────┴──────────────┘
Project Structure
text

zig-llama/
├── models/                    # GGUF model files
├── scripts/tools/             # Utility scripts
├── src/
│   ├── main.zig               # Entry point, CLI, generation loop
│   ├── inference.zig          # Forward pass, attention, FFN
│   ├── tensor.zig             # SIMD ops, quantized matmul, thread pool
│   ├── lut_mul.zig            # Shift-Add / Native dot kernels
│   ├── ternary.zig            # CSD decomposition engine
│   ├── model.zig              # GGUF model loading
│   ├── gguf.zig               # GGUF file parser
│   ├── tokenizer.zig          # GPT-2 / SentencePiece tokenizer
│   ├── sampler.zig            # Temperature / Top-p / Top-k sampling
│   ├── kv_cache.zig           # KV cache for autoregressive generation
│   └── validation.zig         # Input validation
├── build.zig                  # Zig build configuration
└── ReadMe.md
How MatMul-Free Works
Traditional Q8_0 inference: dot = Σ w[i] × x[i] (uses MUL instruction)

Shift-Add decomposition (7 iterations, all 32 lanes in parallel):

text

for each bit b in x[i]:
    if bit b is set:
        product += w[i] << b    // shift only, no multiply
Ternary CSD decomposition (~2.8 iterations average):

text

w = 73 → CSD: +64 +8 +1  (3 terms instead of 7 bits)
dot(w, x) = (x << 6) + (x << 3) + (x << 0)
Both methods produce bit-identical results to hardware multiply for all i8×i8 products.

Building
Bash

# Debug build
zig build

# Release build (recommended)
zig build -Doptimize=ReleaseFast

# Release with native CPU optimizations (recommended)
zig build -Doptimize=ReleaseFast -Dcpu=native

# Run directly
zig build run -Doptimize=ReleaseFast -Dcpu=native -- <model.gguf> [options]
Requirements
Zig 0.15.x
x86_64 CPU with AVX2 (recommended) or any 64-bit CPU
RAM: Model size + ~100 MB overhead
Models: Any GGUF file with Q8_0 or Q4_0 quantization
License
MIT






