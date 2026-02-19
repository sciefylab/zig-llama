// src/inference.zig — SIMD-optimized ops
const std = @import("std");
const LlamaModel = @import("model.zig").LlamaModel;
const Tensor = @import("tensor.zig").Tensor;
const KVCache = @import("kv_cache.zig").KVCache;
const tensor_ops = @import("tensor.zig");
const validation = @import("validation.zig");

pub const ProfileStats = struct {
    matmul_ns: u64 = 0,
    attention_ns: u64 = 0,
    rmsnorm_ns: u64 = 0,
    ffn_ns: u64 = 0,
    embed_ns: u64 = 0,
    other_ns: u64 = 0,
    total_calls: u64 = 0,
    logits_calls: u64 = 0,

    pub fn print(self: *const ProfileStats) void {
        const total = self.matmul_ns + self.attention_ns + self.rmsnorm_ns +
            self.ffn_ns + self.embed_ns + self.other_ns;
        if (total == 0) return;

        std.debug.print("\n=== PROFILING BREAKDOWN ===\n", .{});
        printLine("Embed", self.embed_ns, total);
        printLine("MatMul", self.matmul_ns, total);
        printLine("Attention", self.attention_ns, total);
        printLine("RMSNorm", self.rmsnorm_ns, total);
        printLine("FFN", self.ffn_ns, total);
        printLine("Other", self.other_ns, total);
        std.debug.print("Logits calls:     {}\n", .{self.logits_calls});
        std.debug.print("Total:           {:>10.2} ms ({} forward calls)\n", .{
            @as(f64, @floatFromInt(total)) / 1_000_000, self.total_calls,
        });
    }

    fn printLine(name: []const u8, ns: u64, total: u64) void {
        std.debug.print("{s:<17}{:>10.2} ms ({:>5.1}%)\n", .{
            name,
            @as(f64, @floatFromInt(ns)) / 1_000_000,
            @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
    }

    pub fn reset(self: *ProfileStats) void {
        self.* = .{};
    }
};

pub const InferenceError = error{
    TokenOutOfRange,
    ContextLengthExceeded,
    CacheOutOfBounds,
};

// ============================================================
// SIMD types for vectorized ops
// ============================================================

const SimdF32 = @Vector(8, f32);

inline fn loadF32x8(ptr: [*]const f32) SimdF32 {
    return @as(*align(1) const SimdF32, @ptrCast(ptr)).*;
}

inline fn storeF32x8(ptr: [*]f32, v: SimdF32) void {
    @as(*align(1) SimdF32, @ptrCast(ptr)).* = v;
}

// ============================================================
// State
// ============================================================

pub const InferenceState = struct {
    model: *LlamaModel,
    kv_cache: KVCache,

    x: Tensor,
    xb: Tensor,
    xb2: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn: Tensor,
    logits: Tensor,
    ffn_hidden: Tensor,
    ffn_hidden2: Tensor,

    profile_stats: ProfileStats,
    profiling_enabled: bool,
    validation_enabled: bool,
    validator: validation.Validator,
    n_threads: usize,
    mt_wo_enabled: bool,
    mt_w2_enabled: bool,
    allocator: std.mem.Allocator,

    pub fn init(model_ptr: *LlamaModel, allocator: std.mem.Allocator) !InferenceState {
        const cfg = model_ptr.config;
        const kv_dim = cfg.dim / cfg.n_heads * cfg.n_kv_heads;

        var kv_cache = try KVCache.init(allocator, cfg.n_layers, cfg.max_seq_len, cfg.n_kv_heads, cfg.dim / cfg.n_heads);
        errdefer kv_cache.deinit();

        var x = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer x.deinit();
        var xb = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer xb.deinit();
        var xb2 = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer xb2.deinit();
        var q_t = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer q_t.deinit();
        var k_t = try Tensor.zeros(allocator, &.{kv_dim});
        errdefer k_t.deinit();
        var v_t = try Tensor.zeros(allocator, &.{kv_dim});
        errdefer v_t.deinit();
        var attn_t = try Tensor.zeros(allocator, &.{ cfg.n_heads, cfg.max_seq_len });
        errdefer attn_t.deinit();
        var logits_t = try Tensor.zeros(allocator, &.{cfg.vocab_size});
        errdefer logits_t.deinit();
        var ffn1 = try Tensor.zeros(allocator, &.{cfg.ffn_hidden_dim});
        errdefer ffn1.deinit();
        var ffn2 = try Tensor.zeros(allocator, &.{cfg.ffn_hidden_dim});
        errdefer ffn2.deinit();

        const cpu_threads = std.Thread.getCpuCount() catch 1;

        return .{
            .model = model_ptr,
            .kv_cache = kv_cache,
            .x = x,
            .xb = xb,
            .xb2 = xb2,
            .q = q_t,
            .k = k_t,
            .v = v_t,
            .attn = attn_t,
            .logits = logits_t,
            .ffn_hidden = ffn1,
            .ffn_hidden2 = ffn2,
            .profile_stats = .{},
            .profiling_enabled = false,
            .validation_enabled = true,
            .validator = validation.Validator.initDefault(),
            .n_threads = @min(cpu_threads, 8),
            .mt_wo_enabled = true,
            .mt_w2_enabled = true,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *InferenceState) void {
        self.kv_cache.deinit();
        self.x.deinit();
        self.xb.deinit();
        self.xb2.deinit();
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.attn.deinit();
        self.logits.deinit();
        self.ffn_hidden.deinit();
        self.ffn_hidden2.deinit();
    }

    pub fn reset(self: *InferenceState) void {
        for (self.kv_cache.key_cache) |*cache| @memset(cache.data, 0);
        for (self.kv_cache.value_cache) |*cache| @memset(cache.data, 0);
        self.profile_stats.reset();
    }

    pub fn forward(self: *InferenceState, token: u32, pos: usize) !Tensor {
        return self.forwardInternal(token, pos, true);
    }

    pub fn forwardNoLogits(self: *InferenceState, token: u32, pos: usize) !void {
        _ = try self.forwardInternal(token, pos, false);
    }

    fn forwardInternal(self: *InferenceState, token: u32, pos: usize, compute_logits: bool) !Tensor {
        const m = self.model;
        const cfg = m.config;
        const head_dim = cfg.dim / cfg.n_heads;
        const kv_dim = (cfg.dim / cfg.n_heads) * cfg.n_kv_heads;
        const dim = cfg.dim;
        const nt = self.n_threads;

        if (self.validation_enabled) {
            self.validator.validateToken(token, cfg.vocab_size) catch return InferenceError.TokenOutOfRange;
            self.validator.validatePosition(pos, cfg.max_seq_len) catch return InferenceError.ContextLengthExceeded;
        }

        var timer: ?std.time.Timer = null;
        if (self.profiling_enabled) timer = std.time.Timer.start() catch null;

        // 1) embedding
        switch (m.tok_embeddings.quant_type) {
            .Q8_0 => tensor_ops.embedTokenQ8_0(self.x.data, m.tok_embeddings.data, token, dim),
            .Q4_0, .Q4_1 => tensor_ops.embedTokenQ4_0(self.x.data, m.tok_embeddings.data, token, dim),
        }
        if (timer) |*t| self.profile_stats.embed_ns += t.lap();

        // 2) layers
        for (m.layers, 0..) |*layer, layer_idx| {
            rmsNorm(self.xb.data, self.x.data, layer.attn_norm.data, cfg.rms_norm_eps);
            if (timer) |*t| self.profile_stats.rmsnorm_ns += t.lap();

            // QKV — all MT
            tensor_ops.quantizedMatVecMt(self.q.data, &layer.wq, self.xb.data, cfg.dim, cfg.dim, nt);
            tensor_ops.quantizedMatVecMt(self.k.data, &layer.wk, self.xb.data, kv_dim, cfg.dim, nt);
            tensor_ops.quantizedMatVecMt(self.v.data, &layer.wv, self.xb.data, kv_dim, cfg.dim, nt);

            if (layer.bq) |bq| addBias(self.q.data, bq.data);
            if (layer.bk) |bk| addBias(self.k.data, bk.data);
            if (layer.bv) |bv| addBias(self.v.data, bv.data);
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            tensor_ops.applyRoPE(self.q.data, self.k.data, pos, head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.rope_theta);

            self.kv_cache.update(layer_idx, self.k, self.v, pos) catch return InferenceError.CacheOutOfBounds;
            if (timer) |*t| self.profile_stats.other_ns += t.lap();

            self.attention(layer_idx, pos);
            if (timer) |*t| self.profile_stats.attention_ns += t.lap();

            // WO — MT
            tensor_ops.quantizedMatVecMt(self.xb2.data, &layer.wo, self.xb.data, cfg.dim, cfg.dim, nt);
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            addVec(self.x.data, self.xb2.data);

            rmsNorm(self.xb.data, self.x.data, layer.ffn_norm.data, cfg.rms_norm_eps);
            if (timer) |*t| self.profile_stats.rmsnorm_ns += t.lap();

            // FFN w1+w3 fused MT
            tensor_ops.quantizedMatVec2Mt(self.ffn_hidden.data, &layer.w1, self.ffn_hidden2.data, &layer.w3, self.xb.data, cfg.ffn_hidden_dim, cfg.dim, nt);
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            siluMul(self.ffn_hidden.data, self.ffn_hidden2.data);
            if (timer) |*t| self.profile_stats.ffn_ns += t.lap();

            // W2 — MT
            tensor_ops.quantizedMatVecMt(self.xb.data, &layer.w2, self.ffn_hidden.data, cfg.dim, cfg.ffn_hidden_dim, nt);
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            addVec(self.x.data, self.xb.data);
            if (timer) |*t| self.profile_stats.other_ns += t.lap();
        }

        // 3) final norm + logits
        if (compute_logits) {
            self.profile_stats.logits_calls += 1;
            rmsNorm(self.x.data, self.x.data, m.norm.data, cfg.rms_norm_eps);
            if (timer) |*t| self.profile_stats.rmsnorm_ns += t.lap();

            const output_weight = if (m.use_tied_embeddings) &m.tok_embeddings else &m.output;
            tensor_ops.quantizedMatVecMt(self.logits.data, output_weight, self.x.data, cfg.vocab_size, cfg.dim, nt);
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();
        }

        self.profile_stats.total_calls += 1;
        return self.logits;
    }

    fn attention(self: *InferenceState, layer_idx: usize, pos: usize) void {
        @setRuntimeSafety(false);
        const cfg = self.model.config;
        const head_dim = cfg.dim / cfg.n_heads;
        const seq_len = pos + 1;
        const n_kv_heads = cfg.n_kv_heads;
        const kv_head_ratio = cfg.n_heads / n_kv_heads;

        @memset(self.xb.data, 0);

        const k_cache = self.kv_cache.key_cache[layer_idx].data.ptr;
        const v_cache = self.kv_cache.value_cache[layer_idx].data.ptr;
        const kv_stride = n_kv_heads * head_dim;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..cfg.n_heads) |h| {
            const q_head = self.q.data[h * head_dim ..][0..head_dim];
            const attn_scores = self.attn.data[h * cfg.max_seq_len ..][0..seq_len];
            const kv_head_idx = h / kv_head_ratio;
            const kv_offset = kv_head_idx * head_dim;

            // Q @ K^T — SIMD dot product
            for (0..seq_len) |t| {
                const k_head = k_cache + t * kv_stride + kv_offset;
                attn_scores[t] = dotF32(q_head.ptr, k_head, head_dim) * scale;
            }

            softmax(attn_scores);

            // attn @ V — SIMD scatter-add
            const xb_head = self.xb.data[h * head_dim ..][0..head_dim];
            for (0..seq_len) |t| {
                const v_head = v_cache + t * kv_stride + kv_offset;
                const w = attn_scores[t];
                axpy(xb_head.ptr, v_head, w, head_dim);
            }
        }
    }
};

// ============================================================
// SIMD-optimized ops
// ============================================================

/// SIMD dot product f32
fn dotF32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    @setRuntimeSafety(false);
    var acc0: SimdF32 = @splat(0);
    var acc1: SimdF32 = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        acc0 += loadF32x8(a + i) * loadF32x8(b + i);
        acc1 += loadF32x8(a + i + 8) * loadF32x8(b + i + 8);
    }
    while (i + 8 <= n) : (i += 8) {
        acc0 += loadF32x8(a + i) * loadF32x8(b + i);
    }
    var sum = @reduce(.Add, acc0 + acc1);
    while (i < n) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// SIMD axpy: y += a * x
fn axpy(y: [*]f32, x: [*]const f32, a: f32, n: usize) void {
    @setRuntimeSafety(false);
    const va: SimdF32 = @splat(a);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        storeF32x8(y + i, loadF32x8(y + i) + va * loadF32x8(x + i));
        storeF32x8(y + i + 8, loadF32x8(y + i + 8) + va * loadF32x8(x + i + 8));
    }
    while (i + 8 <= n) : (i += 8) {
        storeF32x8(y + i, loadF32x8(y + i) + va * loadF32x8(x + i));
    }
    while (i < n) : (i += 1) y[i] += a * x[i];
}

fn addBias(out: []f32, bias: []const f32) void {
    @setRuntimeSafety(false);
    const n = @min(out.len, bias.len);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        storeF32x8(out.ptr + i, loadF32x8(out.ptr + i) + loadF32x8(bias.ptr + i));
    }
    while (i < n) : (i += 1) out[i] += bias[i];
}

fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    @setRuntimeSafety(false);
    const n = x.len;
    var acc: SimdF32 = @splat(0);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const v = loadF32x8(x.ptr + i);
        acc += v * v;
    }
    var ss = @reduce(.Add, acc);
    while (i < n) : (i += 1) ss += x[i] * x[i];

    const s = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
    i = 0;
    while (i + 8 <= n) : (i += 8) {
        storeF32x8(out.ptr + i, loadF32x8(x.ptr + i) * loadF32x8(w.ptr + i) * @as(SimdF32, @splat(s)));
    }
    while (i < n) : (i += 1) out[i] = x[i] * s * w[i];
}

fn softmax(x: []f32) void {
    @setRuntimeSafety(false);
    if (x.len == 0) return;
    const n = x.len;

    // Find max — SIMD
    var max_vec: SimdF32 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        max_vec = @max(max_vec, loadF32x8(x.ptr + i));
    }
    var max_val = @reduce(.Max, max_vec);
    while (i < n) : (i += 1) max_val = @max(max_val, x[i]);

    // Exp + sum
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    // Normalize — SIMD
    if (sum > 0) {
        const inv_sum: SimdF32 = @splat(1.0 / sum);
        i = 0;
        while (i + 8 <= n) : (i += 8) {
            storeF32x8(x.ptr + i, loadF32x8(x.ptr + i) * inv_sum);
        }
        while (i < n) : (i += 1) x[i] /= sum;
    }
}

fn addVec(a: []f32, b: []const f32) void {
    @setRuntimeSafety(false);
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        storeF32x8(a.ptr + i, loadF32x8(a.ptr + i) + loadF32x8(b.ptr + i));
    }
    while (i < n) : (i += 1) a[i] += b[i];
}

fn siluMul(gate: []f32, up: []const f32) void {
    @setRuntimeSafety(false);
    const n = @min(gate.len, up.len);
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    // Process scalar — exp not easily SIMD-able portably
    for (0..n) |i| {
        const g = gate[i];
        const sigmoid = 1.0 / (1.0 + @exp(-g));
        gate[i] = g * sigmoid * up[i];
    }
}
