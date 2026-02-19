// src/main.zig
const std = @import("std");
const gguf = @import("gguf.zig");
const model_mod = @import("model.zig");
const tensor = @import("tensor.zig");
const inference = @import("inference.zig");
const tokenizer_mod = @import("tokenizer.zig");
const sampler = @import("sampler.zig");
const validation = @import("validation.zig");
const lut = @import("lut_mul.zig");

fn parseUsize(arg: []const u8) !usize {
    return std.fmt.parseInt(usize, arg, 10);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    tensor.initIsa(allocator);
    tensor.printIsaInfo();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_path: []const u8 = "models/model.gguf";
    var enable_profile: bool = false;
    var quiet: bool = false;
    var threads_override: ?usize = null;
    var no_validate: bool = false;
    var use_lut: ?bool = null;

    // MT flags: null = use default from inference init (true)
    var mt_wo_override: ?bool = null;
    var mt_w2_override: ?bool = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];

        if (std.mem.eql(u8, a, "--profile")) {
            enable_profile = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--quiet")) {
            quiet = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--no-validate")) {
            no_validate = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--lut")) {
            use_lut = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--no-lut")) {
            use_lut = false;
            continue;
        }

        // MT flags only override when explicitly passed
        if (std.mem.eql(u8, a, "--mt-wo")) {
            mt_wo_override = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--no-mt-wo")) {
            mt_wo_override = false;
            continue;
        }
        if (std.mem.eql(u8, a, "--mt-w2")) {
            mt_w2_override = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--no-mt-w2")) {
            mt_w2_override = false;
            continue;
        }

        if (std.mem.eql(u8, a, "--threads")) {
            if (i + 1 < args.len) {
                i += 1;
                threads_override = try parseUsize(args[i]);
            }
            continue;
        }
        if (std.mem.startsWith(u8, a, "--threads=")) {
            threads_override = try parseUsize(a["--threads=".len..]);
            continue;
        }

        if (!std.mem.startsWith(u8, a, "--")) {
            model_path = a;
            continue;
        }
    }

    if (use_lut) |v| lut.g_lut_enabled = v;

    const lut_active = tensor.USE_LUT_MUL and lut.g_lut_enabled;

    const validator = validation.Validator.initDefault();
    validator.validateModelPath(model_path) catch |err| {
        std.debug.print("Error: {s}\n", .{validation.errorMessage(err)});
        std.debug.print("Usage: zig-llama <model.gguf> [--quiet] [--profile] [--threads N] [--mt-wo] [--no-mt-wo] [--mt-w2] [--no-mt-w2] [--no-validate] [--lut] [--no-lut]\n", .{});
        return;
    };

    var gguf_file = try gguf.GGUFFile.load(allocator, model_path);
    defer gguf_file.deinit();

    var llama_model = try model_mod.LlamaModel.fromGGUF(&gguf_file, allocator);
    defer llama_model.deinit();

    var tokenizer_inst = try tokenizer_mod.Tokenizer.init(allocator, &gguf_file);
    defer tokenizer_inst.deinit();

    const sampler_params = validation.Validator.SamplerParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repetition_penalty = 1.05,
        .max_tokens = 256,
    };

    validator.validateSamplerParams(sampler_params) catch |err| {
        std.debug.print("Invalid sampler config: {s}\n", .{validation.errorMessage(err)});
        return;
    };

    const sampler_config = sampler.SamplerConfig{
        .temperature = sampler_params.temperature,
        .top_p = sampler_params.top_p,
        .top_k = sampler_params.top_k,
        .repetition_penalty = sampler_params.repetition_penalty,
        .repetition_window = 64,
        .seed = 42,
    };
    var sampler_inst = try sampler.Sampler.init(sampler_config, llama_model.config.vocab_size, allocator);
    defer sampler_inst.deinit();

    var infer_state = try inference.InferenceState.init(&llama_model, allocator);
    defer infer_state.deinit();

    infer_state.profiling_enabled = enable_profile;
    infer_state.validation_enabled = !no_validate;

    // Only override MT if explicitly passed — otherwise keep init defaults (true)
    if (mt_wo_override) |v| infer_state.mt_wo_enabled = v;
    if (mt_w2_override) |v| infer_state.mt_w2_enabled = v;

    if (threads_override) |t| infer_state.n_threads = @max(@as(usize, 1), t);

    std.debug.print("============================================================\n", .{});
    std.debug.print("Model: {s}\n", .{gguf_file.getMetadataString("general.name") orelse "Unknown"});
    std.debug.print("Tied Embeddings: {}\n", .{llama_model.use_tied_embeddings});
    std.debug.print("Quantization: Q8_0 (all weights)\n", .{});
    std.debug.print("Threads: {}\n", .{infer_state.n_threads});
    std.debug.print("Profiling: {}\n", .{infer_state.profiling_enabled});
    std.debug.print("Quiet: {}\n", .{quiet});
    std.debug.print("MT WO: {}\n", .{infer_state.mt_wo_enabled});
    std.debug.print("MT W2: {}\n", .{infer_state.mt_w2_enabled});
    std.debug.print("Validation(inference): {}\n", .{infer_state.validation_enabled});
    if (lut_active) {
        std.debug.print("MatMul-Free: Shift-Add SIMD (ZERO multiply)\n", .{});
    } else {
        std.debug.print("MatMul: Standard SIMD multiply\n", .{});
    }
    std.debug.print("============================================================\n\n", .{});

    const prompt =
        \\<|im_start|>system
        \\You are a SQL expert. Write only SQL code, no explanations.<|im_end|>
        \\<|im_start|>user
        \\Tables:
        \\- users(id, name, email, created_at)
        \\- orders(id, user_id, total_amount, status, order_date)
        \\
        \\Query: Find top 10 customers by total spending with their order count.<|im_end|>
        \\<|im_start|>assistant
        \\
    ;

    validator.validatePrompt(prompt) catch |err| {
        std.debug.print("Invalid prompt: {s}\n", .{validation.errorMessage(err)});
        return;
    };

    const prompt_tokens = try tokenizer_inst.encode(prompt, false, allocator);
    defer allocator.free(prompt_tokens);

    validator.validatePromptTokens(prompt_tokens.len, llama_model.config.max_seq_len) catch |err| {
        std.debug.print("Prompt too long: {s}\n", .{validation.errorMessage(err)});
        return;
    };

    std.debug.print("Prompt: {} tokens\n", .{prompt_tokens.len});

    var logits: []f32 = undefined;
    const start_time = std.time.milliTimestamp();

    if (prompt_tokens.len == 0) {
        std.debug.print("Empty prompt tokens\n", .{});
        return;
    } else if (prompt_tokens.len == 1) {
        const result = infer_state.forward(prompt_tokens[0], 0) catch |err| {
            std.debug.print("Forward pass error: {}\n", .{err});
            return;
        };
        logits = result.data;
    } else {
        for (prompt_tokens[0 .. prompt_tokens.len - 1], 0..) |tok, pos| {
            infer_state.forwardNoLogits(tok, pos) catch |err| {
                std.debug.print("Forward(pre) error at pos {}: {}\n", .{ pos, err });
                return;
            };
        }
        const last_pos = prompt_tokens.len - 1;
        const result = infer_state.forward(prompt_tokens[last_pos], last_pos) catch |err| {
            std.debug.print("Forward(last) error at pos {}: {}\n", .{ last_pos, err });
            return;
        };
        logits = result.data;
    }

    const prefill_time = std.time.milliTimestamp();

    if (!quiet) {
        std.debug.print("\n=== Logits Debug ===\n", .{});
        var max_logit: f32 = -std.math.inf(f32);
        var max_idx: usize = 0;
        var nan_count: usize = 0;
        for (logits, 0..) |l, idx| {
            if (std.math.isNan(l)) {
                nan_count += 1;
            } else if (l > max_logit) {
                max_logit = l;
                max_idx = idx;
            }
        }
        std.debug.print("NaN count: {}\n", .{nan_count});
        std.debug.print("Max logit: {d:.4} at idx {}\n", .{ max_logit, max_idx });
        std.debug.print("Argmax token: \"{s}\"\n", .{tokenizer_inst.decode(@intCast(max_idx)) orelse "<unk>"});
    }

    const gen_start_time = std.time.milliTimestamp();

    if (!quiet) std.debug.print("\n=== Generated SQL ===\n```sql\n", .{});

    const max_tokens = sampler_params.max_tokens;
    var generated_count: usize = 0;
    const eos_token = tokenizer_inst.eos_token_id;
    const im_end_token: u32 = 151645;
    const im_start_token: u32 = 151644;
    var pos: usize = prompt_tokens.len;

    while (generated_count < max_tokens) : (generated_count += 1) {
        const token_id = sampler_inst.sample(logits);
        if (token_id == eos_token or token_id == im_end_token or token_id == im_start_token) break;

        validator.validateToken(token_id, llama_model.config.vocab_size) catch {
            if (!quiet) std.debug.print("\n[Warning] Invalid token: {}\n", .{token_id});
            break;
        };

        if (!quiet) {
            if (tokenizer_inst.decode(token_id)) |tok_str| tokenizer_mod.printToken(tok_str);
        }

        const result = infer_state.forward(token_id, pos) catch |err| {
            if (!quiet) std.debug.print("\nGen error at pos {}: {}\n", .{ pos, err });
            break;
        };
        logits = result.data;
        pos += 1;
    }

    const end_time = std.time.milliTimestamp();
    if (!quiet) std.debug.print("\n```\n", .{});

    const prefill_dur = prefill_time - start_time;
    const gen_dur = end_time - gen_start_time;
    const total_dur = end_time - start_time;

    const prefill_tps = if (prefill_dur > 0) @as(f64, @floatFromInt(prompt_tokens.len)) / (@as(f64, @floatFromInt(prefill_dur)) / 1000.0) else 0;
    const gen_tps = if (gen_dur > 0) @as(f64, @floatFromInt(generated_count)) / (@as(f64, @floatFromInt(gen_dur)) / 1000.0) else 0;

    std.debug.print("\n============================================================\n", .{});
    std.debug.print("                    PERFORMANCE STATS\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Prefill: {} tokens in {} ms ({d:.2} tokens/sec)\n", .{ prompt_tokens.len, prefill_dur, prefill_tps });
    std.debug.print("Generate: {} tokens in {} ms ({d:.2} tokens/sec)\n", .{ generated_count, gen_dur, gen_tps });
    std.debug.print("Total: {} ms\n", .{total_dur});
    std.debug.print("Mode: {s}\n", .{if (lut_active) "MatMul-Free (Shift-Add SIMD)" else "Standard SIMD"});
    std.debug.print("============================================================\n", .{});

    if (infer_state.profiling_enabled) infer_state.profile_stats.print();

    tensor.shutdownKernels();
    std.debug.print("\n[SUCCESS] Done.\n", .{});
}
