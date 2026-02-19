# Command

zig build run -- models/tinyllama-1.1b-chat.Q4_K_M.gguf

zig build test-dequant -- models/tinyllama-1.1b-chat.Q4_K_M.gguf

zig build debug-q6k -- models/tinyllama-1.1b-chat.Q4_K_M.gguf

python scripts/tools/project_tree.py

## Build dengan optimisasi RELEASE

zig build -Doptimize=ReleaseFast

## Run

./zig-out/bin/zig-llama models/tinyllama-1.1b-chat.Q4_K_M.gguf "Once upon a time"

## Atau langsung

zig build run -Doptimize=ReleaseFast -- models/tinyllama-1.1b-chat.Q4_K_M.gguf "Hello, how are you?"

zig build run -Doptimize=ReleaseFast -- models/qwen2.5-coder-1.5b-instruct-q8_0.gguf 

zig build run -Doptimize=ReleaseFast -Dcpu=native -- models/qwen2.5-coder-1.5b-instruct-q8_0.gguf --profile --quiet --threads 8

zig build run -Doptimize=ReleaseFast -Dcpu=native -- models/Qwen3VL-2B-Instruct-Q8_0.gguf --profile --quiet --threads 8

zig build run -Doptimize=ReleaseFast -Dcpu=native -- models/Qwen2.5-Coder-3B-Instruct-Q8_0.gguf --profile --quiet --threads 8

zig build run -Doptimize=ReleaseFast -Dcpu=native -- models/Qwen3-4B-Thinking-2507-Q8_0.gguf --profile --quiet --threads 8

zig build run -Doptimize=ReleaseFast -Dcpu=native -- models/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf --profile --quiet --threads 8

saya punya main.zig, gguf.zig, model.zig, kv_chace.zig, sampler.zig, tensor.zig, tokenizer.zig, inference.zig, tolong refaktor dari f32 ke q8 SIMD, tolong step by step, jangan langsung dirubah semua, mulai dari mana? 
