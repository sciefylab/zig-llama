// src/main.zig — Entry point kernel with MatMul-Free Shift-Add SIMD
const std = @import("std");
const gguf = @import("gguf.zig");
const model_mod = @import("model.zig");
const tensor_ops = @import("tensor.zig");
const inference = @import("inference.zig");
const tokenizer_mod = @import("tokenizer.zig");
const sampler_mod = @import("sampler.zig");
const lut = @import("lut_mul.zig");
const ternary = @import("ternary.zig");

const LlamaModel = model_mod.LlamaModel;
const InferenceState = inference.InferenceState;

// ============================================================
// Constants
// ============================================================

const separator = "=" ** 60;
const dash42 = "─" ** 42;
const dash12 = "─" ** 12;
const dash14 = "─" ** 14;
const dash10 = "─" ** 10;

// ============================================================
// CLI Arguments
// ============================================================

const Args = struct {
    model_path: []const u8 = "models/qwen2.5-coder-1.5b-instruct-q8_0.gguf",
    prompt: []const u8 = "Write a hello world in Python",
    max_tokens: usize = 128,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    threads: usize = 8,
    profiling: bool = false,
    quiet: bool = false,
    mt_wo: bool = true,
    mt_w2: bool = true,
    validation: bool = true,
    show_isa: bool = false,
    ternary_mode: enum { off, otf, cached } = .off,
    ternary_test: bool = false,
    ternary_bench: bool = false,
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    var args = Args{};
    var iter = try std.process.argsWithAllocator(allocator);
    defer iter.deinit();

    _ = iter.next();

    while (iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            if (iter.next()) |v| args.model_path = v;
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            if (iter.next()) |v| args.prompt = v;
        } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
            if (iter.next()) |v| args.max_tokens = std.fmt.parseInt(usize, v, 10) catch 128;
        } else if (std.mem.eql(u8, arg, "--temperature") or std.mem.eql(u8, arg, "-t")) {
            if (iter.next()) |v| args.temperature = std.fmt.parseFloat(f32, v) catch 0.7;
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            if (iter.next()) |v| args.top_p = std.fmt.parseFloat(f32, v) catch 0.9;
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            if (iter.next()) |v| args.top_k = std.fmt.parseInt(u32, v, 10) catch 40;
        } else if (std.mem.eql(u8, arg, "--threads") or std.mem.eql(u8, arg, "-j")) {
            if (iter.next()) |v| args.threads = std.fmt.parseInt(usize, v, 10) catch 8;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            args.profiling = true;
        } else if (std.mem.eql(u8, arg, "--quiet") or std.mem.eql(u8, arg, "-q")) {
            args.quiet = true;
        } else if (std.mem.eql(u8, arg, "--no-mt-wo")) {
            args.mt_wo = false;
        } else if (std.mem.eql(u8, arg, "--no-mt-w2")) {
            args.mt_w2 = false;
        } else if (std.mem.eql(u8, arg, "--no-validation")) {
            args.validation = false;
        } else if (std.mem.eql(u8, arg, "--show-isa")) {
            args.show_isa = true;
        } else if (std.mem.eql(u8, arg, "--ternary")) {
            args.ternary_mode = .cached;
        } else if (std.mem.eql(u8, arg, "--ternary-otf")) {
            args.ternary_mode = .otf;
        } else if (std.mem.eql(u8, arg, "--ternary-cached")) {
            args.ternary_mode = .cached;
        } else if (std.mem.eql(u8, arg, "--ternary-test")) {
            args.ternary_test = true;
        } else if (std.mem.eql(u8, arg, "--ternary-bench")) {
            args.ternary_bench = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }
    return args;
}

fn printHelp() void {
    std.debug.print(
        \\
        \\zig-llama: MatMul-Free LLM Inference Engine
        \\
        \\Usage: zig-llama [options]
        \\
        \\Model:
        \\  -m, --model <path>       GGUF model file
        \\  -p, --prompt <text>      Input prompt
        \\  -n, --max-tokens <N>     Max tokens to generate (default: 128)
        \\
        \\Sampling:
        \\  -t, --temperature <F>    Temperature (default: 0.7)
        \\  --top-p <F>              Top-p nucleus sampling (default: 0.9)
        \\  --top-k <N>              Top-k sampling (default: 40)
        \\
        \\Performance:
        \\  -j, --threads <N>        Worker threads (default: 8)
        \\  --no-mt-wo               Disable MT for output projection
        \\  --no-mt-w2               Disable MT for FFN down projection
        \\  --profile                Enable profiling breakdown
        \\
        \\MatMul-Free Engine:
        \\  --ternary                Enable Ternary CSD (cached, recommended)
        \\  --ternary-otf            Enable Ternary CSD (on-the-fly)
        \\  --ternary-cached         Enable Ternary CSD (cached, fastest)
        \\  --ternary-test           Run ternary validation suite
        \\  --ternary-bench          Run ternary benchmark comparison
        \\
        \\Output:
        \\  -q, --quiet              Suppress model loading details
        \\  --show-isa               Show ISA/CPU detection info
        \\  --no-validation          Disable input validation
        \\  -h, --help               Show this help
        \\
    , .{});
}

// ============================================================
// Main
// ============================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try parseArgs(allocator);

    // ── ISA Detection ──
    tensor_ops.initIsa(allocator);
    defer tensor_ops.shutdownKernels();

    if (args.show_isa) {
        tensor_ops.printIsaInfo();
    }

    // ── Ternary Mode Setup ──
    switch (args.ternary_mode) {
        .off => {
            lut.g_ternary_mode = .off;
        },
        .otf => {
            lut.g_ternary_mode = .on_the_fly;
            if (!args.quiet) std.debug.print("Ternary CSD: on-the-fly mode\n", .{});
        },
        .cached => {
            lut.g_ternary_mode = .cached;
            lut.initTernaryCache(allocator);
            if (!args.quiet) std.debug.print("Ternary CSD: cached mode\n", .{});
        },
    }
    defer {
        if (args.ternary_mode == .cached) lut.deinitTernaryCache();
    }

    // ── Ternary Test Suite ──
    if (args.ternary_test) {
        runTernaryTests(allocator);
        if (args.prompt.len == 0) return;
    }

    // ── Ternary Benchmark ──
    if (args.ternary_bench) {
        runTernaryBenchmark();
        if (args.prompt.len == 0) return;
    }

    // ── Load Model ──
    if (!args.quiet) {
        std.debug.print("\n" ++ separator ++ "\n", .{});
    }

    var gguf_file = try gguf.GGUFFile.load(allocator, args.model_path);
    defer gguf_file.deinit();

    var mdl = try LlamaModel.fromGGUF(&gguf_file, allocator);
    defer mdl.deinit();

    // ── Tokenizer ──
    var tok = try tokenizer_mod.Tokenizer.init(allocator, &gguf_file);
    defer tok.deinit();

    // ── Inference State ──
    var state = try InferenceState.init(&mdl, allocator);
    defer state.deinit();

    state.profiling_enabled = args.profiling;
    state.validation_enabled = args.validation;
    state.n_threads = args.threads;
    state.mt_wo_enabled = args.mt_wo;
    state.mt_w2_enabled = args.mt_w2;

    // ── Print Config ──
    if (!args.quiet) {
        printConfig(&mdl, &args);
    }

    // ── Encode Prompt ──
    const prompt_tokens = try tok.encode(args.prompt, true, allocator);
    defer allocator.free(prompt_tokens);

    if (!args.quiet) {
        std.debug.print("\nPrompt: {} tokens\n", .{prompt_tokens.len});
    }

    // ── Prefill (Sequential + MT) ──
    var timer = try std.time.Timer.start();

    for (prompt_tokens[0 .. prompt_tokens.len - 1], 0..) |tok_id, pos| {
        try state.forwardNoLogits(tok_id, pos);
    }

    var logits = try state.forward(prompt_tokens[prompt_tokens.len - 1], prompt_tokens.len - 1);
    const prefill_ns = timer.read();

    // ── Sampler ──
    var sampler = try sampler_mod.Sampler.init(
        .{
            .temperature = args.temperature,
            .top_p = args.top_p,
            .top_k = args.top_k,
        },
        @intCast(mdl.config.vocab_size),
        allocator,
    );
    defer sampler.deinit();

    // ── Generate ──
    var gen_timer = try std.time.Timer.start();
    var generated: usize = 0;
    var pos = prompt_tokens.len;

    if (!args.quiet) {
        std.debug.print("\n", .{});
    }

    while (generated < args.max_tokens) : (generated += 1) {
        const next_token = sampler.sample(logits.data);

        // Check EOS
        if (isEosToken(next_token, &tok)) break;

        // Decode and print
        if (tok.decode(next_token)) |text| {
            tokenizer_mod.printToken(text);
        }

        // Forward
        logits = try state.forward(next_token, pos);
        pos += 1;
    }

    const gen_ns = gen_timer.read();

    // ── Stats ──
    if (!args.quiet) {
        std.debug.print("\n", .{});
    }

    printStats(prompt_tokens.len, prefill_ns, generated, gen_ns, &args);

    if (args.profiling) {
        state.profile_stats.print();
    }

    std.debug.print("\n[SUCCESS] Done.\n", .{});
}

// ============================================================
// Config Display
// ============================================================

fn printConfig(mdl: *const LlamaModel, args: *const Args) void {
    const ternary_str: []const u8 = switch (args.ternary_mode) {
        .off => "off",
        .otf => "on-the-fly",
        .cached => "cached (precomputed)",
    };

    const engine_str: []const u8 = if (lut.USE_NATIVE_DOT)
        "Native SIMD (hardware i16 multiply)"
    else switch (args.ternary_mode) {
        .off => "Shift-Add SIMD (ZERO multiply, 7-bit decomp)",
        .otf => "Ternary CSD SIMD (ZERO multiply, ~2.8 terms avg)",
        .cached => "Ternary CSD Cached (ZERO multiply, ~2.8 terms avg)",
    };

    const matmul_free_str: []const u8 = if (lut.USE_NATIVE_DOT)
        "NO (using hardware multiply)"
    else
        "YES (zero MUL instructions)";

    std.debug.print("\n" ++ separator ++ "\n", .{});
    std.debug.print("Model: Qwen2.5 Coder {d:.0}B\n", .{
        @as(f64, @floatFromInt(mdl.config.vocab_size)) * @as(f64, @floatFromInt(mdl.config.dim)) / 1_000_000_000.0,
    });
    std.debug.print("Tied Embeddings: {}\n", .{mdl.use_tied_embeddings});
    std.debug.print("Quantization: Q8_0 (all weights)\n", .{});
    std.debug.print("Threads: {}\n", .{args.threads});
    std.debug.print("Profiling: {}\n", .{args.profiling});
    std.debug.print("MT WO: {}\n", .{args.mt_wo});
    std.debug.print("MT W2: {}\n", .{args.mt_w2});
    std.debug.print("Validation: {}\n", .{args.validation});
    std.debug.print("Ternary CSD: {s}\n", .{ternary_str});
    std.debug.print("Engine: {s}\n", .{engine_str});
    std.debug.print("MatMul-Free: {s}\n", .{matmul_free_str});
    std.debug.print(separator ++ "\n", .{});
}

// ============================================================
// Stats Display
// ============================================================

fn printStats(
    prompt_len: usize,
    prefill_ns: u64,
    generated: usize,
    gen_ns: u64,
    args: *const Args,
) void {
    const prefill_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0;
    const gen_ms = @as(f64, @floatFromInt(gen_ns)) / 1_000_000.0;
    const total_ms = prefill_ms + gen_ms;
    const prefill_tps = if (prefill_ms > 0) @as(f64, @floatFromInt(prompt_len)) / (prefill_ms / 1000.0) else 0;
    const gen_tps = if (gen_ms > 0) @as(f64, @floatFromInt(generated)) / (gen_ms / 1000.0) else 0;

    const mode_str: []const u8 = if (lut.USE_NATIVE_DOT)
        "Native SIMD (hardware multiply)"
    else switch (args.ternary_mode) {
        .off => "MatMul-Free (Shift-Add SIMD)",
        .otf => "MatMul-Free (Ternary CSD on-the-fly)",
        .cached => "MatMul-Free (Ternary CSD cached)",
    };

    std.debug.print("\n" ++ separator ++ "\n", .{});
    std.debug.print("                    PERFORMANCE STATS\n", .{});
    std.debug.print(separator ++ "\n", .{});
    std.debug.print("Prefill: {} tokens in {d:.0} ms ({d:.2} tokens/sec)\n", .{
        prompt_len, prefill_ms, prefill_tps,
    });
    std.debug.print("Generate: {} tokens in {d:.0} ms ({d:.2} tokens/sec)\n", .{
        generated, gen_ms, gen_tps,
    });
    std.debug.print("Total: {d:.0} ms\n", .{total_ms});
    std.debug.print("Mode: {s}\n", .{mode_str});
    std.debug.print(separator ++ "\n", .{});
}

// ============================================================
// EOS Detection
// ============================================================

fn isEosToken(token: u32, tok: *const tokenizer_mod.Tokenizer) bool {
    if (token == 151643 or token == 151644 or token == 151645) return true;

    if (tok.decode(token)) |text| {
        if (std.mem.eql(u8, text, "</s>")) return true;
        if (std.mem.eql(u8, text, "<|endoftext|>")) return true;
        if (std.mem.eql(u8, text, "<|im_end|>")) return true;
    }

    return false;
}

// ============================================================
// Ternary Test Suite
// ============================================================

fn runTernaryTests(allocator: std.mem.Allocator) void {
    _ = allocator;

    std.debug.print("\n" ++ separator ++ "\n", .{});
    std.debug.print("         TERNARY CSD VALIDATION SUITE\n", .{});
    std.debug.print(separator ++ "\n\n", .{});

    std.debug.print("=== CSD Reconstruction (all 256 i8 values) ===\n", .{});

    var exact_count: usize = 0;
    var total_terms: usize = 0;
    var max_error: i16 = 0;
    var total_abs_error: u64 = 0;
    var term_hist: [5]usize = .{ 0, 0, 0, 0, 0 };

    for (0..256) |ui| {
        const val: i8 = @bitCast(@as(u8, @intCast(ui)));
        const csd = ternary.toCSD(val);
        const reconstructed = ternary.csdToValue(csd);
        const err: i16 = @as(i16, val) - reconstructed;
        const abs_err: u16 = @intCast(@abs(err));

        if (err == 0) exact_count += 1;
        total_terms += csd.n_terms;
        if (@abs(err) > @abs(max_error)) max_error = err;
        total_abs_error += abs_err;
        if (csd.n_terms < 5) term_hist[csd.n_terms] += 1;
    }

    std.debug.print("  Exact reconstructions: {}/256 ({d:.1}%)\n", .{
        exact_count,
        @as(f64, @floatFromInt(exact_count)) / 256.0 * 100.0,
    });
    std.debug.print("  Avg terms/weight: {d:.2}\n", .{
        @as(f64, @floatFromInt(total_terms)) / 256.0,
    });
    std.debug.print("  Max absolute error: {}\n", .{max_error});
    std.debug.print("  Avg absolute error: {d:.3}\n", .{
        @as(f64, @floatFromInt(total_abs_error)) / 256.0,
    });
    std.debug.print("  Term distribution: 0t={} 1t={} 2t={} 3t={}\n", .{
        term_hist[0], term_hist[1], term_hist[2], term_hist[3],
    });

    std.debug.print("\n=== Dot Product Accuracy (10000 random blocks) ===\n", .{});

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();

    var dot_exact: usize = 0;
    var dot_total_err: f64 = 0;
    var dot_max_err: f64 = 0;
    const N_DOT_TESTS = 10000;

    for (0..N_DOT_TESTS) |_| {
        var w_buf: [32]u8 align(32) = undefined;
        var x_buf: [32]i8 align(32) = undefined;

        for (0..32) |i| {
            w_buf[i] = @bitCast(rand.intRangeAtMost(i8, -127, 127));
            x_buf[i] = rand.intRangeAtMost(i8, -127, 127);
        }

        var ref_sum: i32 = 0;
        for (0..32) |i| {
            ref_sum += @as(i32, @as(i8, @bitCast(w_buf[i]))) * @as(i32, x_buf[i]);
        }

        const sa_sum = lut.dotI8x32_shiftadd(&w_buf, &x_buf);

        const tblk = ternary.TernaryBlock.fromRawWeights(&w_buf);
        const tern_sum = ternary.dotTernary32(&tblk, &x_buf);

        const gblk = ternary.TernaryBlockGrouped.fromRawWeights(&w_buf);
        const gtern_sum = ternary.dotTernaryGrouped32(&gblk, &x_buf);

        if (sa_sum != ref_sum) {
            std.debug.print("  [FAIL] Shift-add mismatch: ref={} got={}\n", .{ ref_sum, sa_sum });
        }

        const err_t = @abs(@as(f64, @floatFromInt(ref_sum - tern_sum)));
        const err_g = @abs(@as(f64, @floatFromInt(ref_sum - gtern_sum)));
        const err = @max(err_t, err_g);

        if (err == 0) dot_exact += 1;
        dot_total_err += err;
        if (err > dot_max_err) dot_max_err = err;
    }

    std.debug.print("  Exact matches: {}/{} ({d:.1}%)\n", .{
        dot_exact,
        N_DOT_TESTS,
        @as(f64, @floatFromInt(dot_exact)) / @as(f64, @floatFromInt(N_DOT_TESTS)) * 100.0,
    });
    std.debug.print("  Avg dot error: {d:.2}\n", .{dot_total_err / @as(f64, @floatFromInt(N_DOT_TESTS))});
    std.debug.print("  Max dot error: {d:.0}\n", .{dot_max_err});

    std.debug.print("\n=== Specific Value Tests ===\n", .{});

    const test_cases = [_]struct { w: i8, x: i8, expected: i32 }{
        .{ .w = 7, .x = 8, .expected = 56 },
        .{ .w = -3, .x = 4, .expected = -12 },
        .{ .w = -5, .x = -6, .expected = 30 },
        .{ .w = 127, .x = 127, .expected = 16129 },
        .{ .w = -128, .x = 1, .expected = -128 },
        .{ .w = 0, .x = 100, .expected = 0 },
        .{ .w = 1, .x = 1, .expected = 1 },
        .{ .w = 73, .x = 11, .expected = 803 },
        .{ .w = 85, .x = -7, .expected = -595 },
    };

    for (test_cases) |tc| {
        var w_t: [32]u8 align(32) = undefined;
        var x_t: [32]i8 align(32) = undefined;
        @memset(&w_t, 0);
        @memset(&x_t, 0);
        w_t[0] = @bitCast(tc.w);
        x_t[0] = tc.x;

        const sa = lut.dotI8x32_shiftadd(&w_t, &x_t);
        const tblk = ternary.TernaryBlock.fromRawWeights(&w_t);
        const tr = ternary.dotTernary32(&tblk, &x_t);

        const sa_ok: []const u8 = if (sa == tc.expected) "OK" else "FAIL";
        const tr_ok: []const u8 = if (tr == tc.expected) "OK" else "APPROX";

        std.debug.print("  {}x{}: expected={} shift-add={} [{s}] ternary={} [{s}]\n", .{
            tc.w, tc.x, tc.expected, sa, sa_ok, tr, tr_ok,
        });
    }

    std.debug.print("\n" ++ separator ++ "\n", .{});
    std.debug.print("         VALIDATION COMPLETE\n", .{});
    std.debug.print(separator ++ "\n\n", .{});
}

// ============================================================
// Ternary Benchmark
// ============================================================

fn runTernaryBenchmark() void {
    std.debug.print("\n" ++ separator ++ "\n", .{});
    std.debug.print("         TERNARY CSD BENCHMARK\n", .{});
    std.debug.print(separator ++ "\n\n", .{});

    var w_bench: [34]u8 align(32) = undefined;
    var x_bench: [32]i8 align(32) = undefined;
    w_bench[0] = 0x00;
    w_bench[1] = 0x3C;

    for (0..32) |i| {
        const v: i32 = @as(i32, @intCast(i % 17)) - 8;
        w_bench[2 + i] = @bitCast(@as(i8, @intCast(v)));
        x_bench[i] = @intCast(@as(i32, @intCast(i % 13)) - 6);
    }

    const ITERS = 1_000_000;

    // Shift-Add
    var t0 = std.time.Timer.start() catch {
        std.debug.print("Timer failed\n", .{});
        return;
    };
    var acc0: i32 = 0;
    for (0..ITERS) |_| {
        acc0 +%= lut.dotI8x32_shiftadd(w_bench[2..].ptr, &x_bench);
    }
    const ns_shiftadd = t0.read();
    std.mem.doNotOptimizeAway(acc0);

    // Native i16
    var t5 = std.time.Timer.start() catch return;
    var acc5: i32 = 0;
    for (0..ITERS) |_| {
        acc5 +%= lut.dotI8x32_native(w_bench[2..].ptr, &x_bench);
    }
    const ns_native = t5.read();
    std.mem.doNotOptimizeAway(acc5);

    // Ternary OTF
    var t1 = std.time.Timer.start() catch return;
    var acc1: i32 = 0;
    for (0..ITERS) |_| {
        acc1 +%= ternary.dotTernaryRaw(w_bench[2..].ptr, &x_bench);
    }
    const ns_tern_otf = t1.read();
    std.mem.doNotOptimizeAway(acc1);

    // Ternary Grouped OTF
    var t2 = std.time.Timer.start() catch return;
    var acc2: i32 = 0;
    for (0..ITERS) |_| {
        const gblk = ternary.TernaryBlockGrouped.fromRawWeights(w_bench[2..].ptr);
        acc2 +%= ternary.dotTernaryGrouped32(&gblk, &x_bench);
    }
    const ns_gtern_otf = t2.read();
    std.mem.doNotOptimizeAway(acc2);

    // Ternary Grouped Cached
    const gblk_cached = ternary.TernaryBlockGrouped.fromRawWeights(w_bench[2..].ptr);
    var t3 = std.time.Timer.start() catch return;
    var acc3: i32 = 0;
    for (0..ITERS) |_| {
        acc3 +%= ternary.dotTernaryGrouped32(&gblk_cached, &x_bench);
    }
    const ns_cached = t3.read();
    std.mem.doNotOptimizeAway(acc3);

    // Ternary Basic Cached
    const tblk_cached = ternary.TernaryBlock.fromRawWeights(w_bench[2..].ptr);
    var t4 = std.time.Timer.start() catch return;
    var acc4: i32 = 0;
    for (0..ITERS) |_| {
        acc4 +%= ternary.dotTernary32(&tblk_cached, &x_bench);
    }
    const ns_basic_cached = t4.read();
    std.mem.doNotOptimizeAway(acc4);

    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        }
    }.f;

    const ops_per_sec = struct {
        fn f(ns: u64, iters: usize) f64 {
            if (ns == 0) return 0;
            return @as(f64, @floatFromInt(iters)) / (@as(f64, @floatFromInt(ns)) / 1_000_000_000.0);
        }
    }.f;

    std.debug.print("Iterations: {}\n\n", .{ITERS});

    std.debug.print("┌" ++ dash42 ++ "┬" ++ dash12 ++ "┬" ++ dash14 ++ "┬" ++ dash10 ++ "┐\n", .{});
    std.debug.print("│ {s:<42}│ {s:<12}│ {s:<14}│ {s:<10}│\n", .{ "Method", "Time (ms)", "Mops/s", "Speedup" });
    std.debug.print("├" ++ dash42 ++ "┼" ++ dash12 ++ "┼" ++ dash14 ++ "┼" ++ dash10 ++ "┤\n", .{});

    const methods = [_]struct { name: []const u8, ns: u64 }{
        .{ .name = "Shift-Add (7-bit, matmul-free)", .ns = ns_shiftadd },
        .{ .name = "Native i16 multiply (vpmullw)", .ns = ns_native },
        .{ .name = "Ternary basic (on-the-fly)", .ns = ns_tern_otf },
        .{ .name = "Ternary grouped (on-the-fly)", .ns = ns_gtern_otf },
        .{ .name = "Ternary basic (CACHED)", .ns = ns_basic_cached },
        .{ .name = "Ternary grouped (CACHED)", .ns = ns_cached },
    };

    for (methods) |m_val| {
        const speedup = if (m_val.ns > 0) @as(f64, @floatFromInt(ns_shiftadd)) / @as(f64, @floatFromInt(m_val.ns)) else 0;
        std.debug.print("│ {s:<42}│ {d:>10.2} │ {d:>12.1} │ {d:>7.2}x │\n", .{
            m_val.name, ms(m_val.ns), ops_per_sec(m_val.ns, ITERS) / 1_000_000.0, speedup,
        });
    }

    std.debug.print("└" ++ dash42 ++ "┴" ++ dash12 ++ "┴" ++ dash14 ++ "┴" ++ dash10 ++ "┘\n", .{});

    std.debug.print("\nAccuracy:\n", .{});
    std.debug.print("  Shift-Add result:           {}\n", .{acc0});
    std.debug.print("  Native i16 result:          {}\n", .{acc5});
    std.debug.print("  Ternary OTF result:         {}\n", .{acc1});
    std.debug.print("  Ternary grouped OTF result: {}\n", .{acc2});
    std.debug.print("  Ternary basic cached:       {}\n", .{acc4});
    std.debug.print("  Ternary grouped cached:     {}\n", .{acc3});

    if (acc0 == acc5) {
        std.debug.print("  [NATIVE EXACT MATCH with shift-add]\n", .{});
    }
    if (acc0 == acc1 and acc0 == acc3) {
        std.debug.print("  [ALL EXACT MATCH]\n", .{});
    } else {
        std.debug.print("  [APPROXIMATE — expected with {} CSD terms]\n", .{ternary.MAX_TERMS});
    }

    std.debug.print("\n" ++ separator ++ "\n\n", .{});
}
