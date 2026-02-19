const std = @import("std");
const LlamaModel = @import("model.zig").LlamaModel;
const Tensor = @import("tensor.zig").Tensor;
const KVCache = @import("kv_cache.zig").KVCache;
const tensor_ops = @import("tensor.zig");
const validation = @import("validation.zig");

// =============================================================================
// Profiling
// =============================================================================

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
        std.debug.print("Embed:           {:>10.2} ms ({:>5.1}%)\n", .{
            @as(f64, @floatFromInt(self.embed_ns)) / 1_000_000,
            @as(f64, @floatFromInt(self.embed_ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
        std.debug.print("MatMul:          {:>10.2} ms ({:>5.1}%)\n", .{
            @as(f64, @floatFromInt(self.matmul_ns)) / 1_000_000,
            @as(f64, @floatFromInt(self.matmul_ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
        std.debug.print("Attention:       {:>10.2} ms ({:>5.1}%)\n", .{
            @as(f64, @floatFromInt(self.attention_ns)) / 1_000_000,
            @as(f64, @floatFromInt(self.attention_ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
        std.debug.print("RMSNorm:         {:>10.2} ms ({:>5.1}%)\n", .{
            @as(f64, @floatFromInt(self.rmsnorm_ns)) / 1_000_000,
            @as(f64, @floatFromInt(self.rmsnorm_ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
        std.debug.print("FFN:             {:>10.2} ms ({:>5.1}%)\n", .{
            @as(f64, @floatFromInt(self.ffn_ns)) / 1_000_000,
            @as(f64, @floatFromInt(self.ffn_ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
        std.debug.print("Other:           {:>10.2} ms ({:>5.1}%)\n", .{
            @as(f64, @floatFromInt(self.other_ns)) / 1_000_000,
            @as(f64, @floatFromInt(self.other_ns)) / @as(f64, @floatFromInt(total)) * 100,
        });
        std.debug.print("Logits calls:     {}\n", .{self.logits_calls});
        std.debug.print("Total:           {:>10.2} ms ({} forward calls)\n", .{
            @as(f64, @floatFromInt(total)) / 1_000_000,
            self.total_calls,
        });
    }

    pub fn reset(self: *ProfileStats) void {
        self.* = .{};
    }
};

// =============================================================================
// Errors
// =============================================================================

pub const InferenceError = error{
    TokenOutOfRange,
    ContextLengthExceeded,
    CacheOutOfBounds,
};

// =============================================================================
// State
// =============================================================================

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

    // Validasi (boleh dimatikan buat benchmark)
    validation_enabled: bool,
    validator: validation.Validator,

    // Threading
    n_threads: usize,

    // Optional toggles: MT untuk WO/W2 (default OFF karena run kamu tidak membaik)
    mt_wo_enabled: bool,
    mt_w2_enabled: bool,

    allocator: std.mem.Allocator,

    pub fn init(model_ptr: *LlamaModel, allocator: std.mem.Allocator) !InferenceState {
        const cfg = model_ptr.config;
        const kv_dim = cfg.dim / cfg.n_heads * cfg.n_kv_heads;

        var kv_cache = try KVCache.init(
            allocator,
            cfg.n_layers,
            cfg.max_seq_len,
            cfg.n_kv_heads,
            cfg.dim / cfg.n_heads,
        );
        errdefer kv_cache.deinit();

        var x = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer x.deinit();

        var xb = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer xb.deinit();

        var xb2 = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer xb2.deinit();

        var q = try Tensor.zeros(allocator, &.{cfg.dim});
        errdefer q.deinit();

        var k_tensor = try Tensor.zeros(allocator, &.{kv_dim});
        errdefer k_tensor.deinit();

        var v_tensor = try Tensor.zeros(allocator, &.{kv_dim});
        errdefer v_tensor.deinit();

        var attn = try Tensor.zeros(allocator, &.{ cfg.n_heads, cfg.max_seq_len });
        errdefer attn.deinit();

        var logits_tensor = try Tensor.zeros(allocator, &.{cfg.vocab_size});
        errdefer logits_tensor.deinit();

        var ffn_hidden = try Tensor.zeros(allocator, &.{cfg.ffn_hidden_dim});
        errdefer ffn_hidden.deinit();

        var ffn_hidden2 = try Tensor.zeros(allocator, &.{cfg.ffn_hidden_dim});
        errdefer ffn_hidden2.deinit();

        const cpu_threads = std.Thread.getCpuCount() catch 1;
        const use_threads: usize = @min(cpu_threads, 8);

        return .{
            .model = model_ptr,
            .kv_cache = kv_cache,
            .x = x,
            .xb = xb,
            .xb2 = xb2,
            .q = q,
            .k = k_tensor,
            .v = v_tensor,
            .attn = attn,
            .logits = logits_tensor,
            .ffn_hidden = ffn_hidden,
            .ffn_hidden2 = ffn_hidden2,
            .profile_stats = .{},
            // default OFF untuk speed; aktifkan dari main pakai --profile
            .profiling_enabled = false,

            .validation_enabled = true,
            .validator = validation.Validator.initDefault(),

            .n_threads = use_threads,

            // default OFF (berdasarkan hasil run kamu)
            .mt_wo_enabled = false,
            .mt_w2_enabled = false,

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

            // QKV (single-thread; biasanya tidak worth MT kalau MT impl-nya berat)
            tensor_ops.quantizedMatVec(self.q.data, &layer.wq, self.xb.data, cfg.dim, cfg.dim);
            tensor_ops.quantizedMatVec(self.k.data, &layer.wk, self.xb.data, kv_dim, cfg.dim);
            tensor_ops.quantizedMatVec(self.v.data, &layer.wv, self.xb.data, kv_dim, cfg.dim);

            if (layer.bq) |bq| addBias(self.q.data, bq.data);
            if (layer.bk) |bk| addBias(self.k.data, bk.data);
            if (layer.bv) |bv| addBias(self.v.data, bv.data);
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            // RoPE
            tensor_ops.applyRoPE(self.q.data, self.k.data, pos, head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.rope_theta);

            // KV cache
            self.kv_cache.update(layer_idx, self.k, self.v, pos) catch return InferenceError.CacheOutOfBounds;
            if (timer) |*t| self.profile_stats.other_ns += t.lap();

            // attention
            self.attention(layer_idx, pos);
            if (timer) |*t| self.profile_stats.attention_ns += t.lap();

            // WO (default single-thread; bisa MT via flag)
            if (self.mt_wo_enabled and self.n_threads > 1) {
                tensor_ops.quantizedMatVecMt(self.xb2.data, &layer.wo, self.xb.data, cfg.dim, cfg.dim, self.n_threads);
            } else {
                tensor_ops.quantizedMatVec(self.xb2.data, &layer.wo, self.xb.data, cfg.dim, cfg.dim);
            }
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            add(self.x.data, self.xb2.data);

            // ffn norm
            rmsNorm(self.xb.data, self.x.data, layer.ffn_norm.data, cfg.rms_norm_eps);
            if (timer) |*t| self.profile_stats.rmsnorm_ns += t.lap();

            // FFN w1 + w3: FUSED + MT (sudah bagus)
            tensor_ops.quantizedMatVec2Mt(
                self.ffn_hidden.data,
                &layer.w1,
                self.ffn_hidden2.data,
                &layer.w3,
                self.xb.data,
                cfg.ffn_hidden_dim,
                cfg.dim,
                self.n_threads,
            );
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            siluMul(self.ffn_hidden.data, self.ffn_hidden2.data);
            if (timer) |*t| self.profile_stats.ffn_ns += t.lap();

            // W2 (default single-thread; bisa MT via flag)
            if (self.mt_w2_enabled and self.n_threads > 1) {
                tensor_ops.quantizedMatVecMt(
                    self.xb.data,
                    &layer.w2,
                    self.ffn_hidden.data,
                    cfg.dim,
                    cfg.ffn_hidden_dim,
                    self.n_threads,
                );
            } else {
                tensor_ops.quantizedMatVec(self.xb.data, &layer.w2, self.ffn_hidden.data, cfg.dim, cfg.ffn_hidden_dim);
            }
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();

            add(self.x.data, self.xb.data);
            if (timer) |*t| self.profile_stats.other_ns += t.lap();
        }

        // 3) final norm + logits
        if (compute_logits) {
            self.profile_stats.logits_calls += 1;

            rmsNorm(self.x.data, self.x.data, m.norm.data, cfg.rms_norm_eps);
            if (timer) |*t| self.profile_stats.rmsnorm_ns += t.lap();

            const output_weight = if (m.use_tied_embeddings) &m.tok_embeddings else &m.output;

            // lm_head: selalu MT (ini yang paling besar)
            tensor_ops.quantizedMatVecMt(
                self.logits.data,
                output_weight,
                self.x.data,
                cfg.vocab_size,
                cfg.dim,
                self.n_threads,
            );
            if (timer) |*t| self.profile_stats.matmul_ns += t.lap();
        }

        self.profile_stats.total_calls += 1;
        return self.logits;
    }

    fn attention(self: *InferenceState, layer_idx: usize, pos: usize) void {
        const cfg = self.model.config;
        const head_dim = cfg.dim / cfg.n_heads;
        const seq_len = pos + 1;
        const n_kv_heads = cfg.n_kv_heads;
        const kv_head_ratio = cfg.n_heads / n_kv_heads;

        @memset(self.xb.data, 0);

        const k_cache_base = self.kv_cache.key_cache[layer_idx].data.ptr;
        const v_cache_base = self.kv_cache.value_cache[layer_idx].data.ptr;
        const kv_stride = n_kv_heads * head_dim;

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..cfg.n_heads) |h| {
            const q_head = self.q.data[h * head_dim ..][0..head_dim];
            const attn_scores = self.attn.data[h * cfg.max_seq_len ..][0..seq_len];
            const kv_head_idx = h / kv_head_ratio;
            const kv_offset = kv_head_idx * head_dim;

            // Q @ K^T
            for (0..seq_len) |t| {
                const k_head = k_cache_base + t * kv_stride + kv_offset;
                var sum: f32 = 0;
                for (0..head_dim) |j| sum += q_head[j] * k_head[j];
                attn_scores[t] = sum * scale;
            }

            softmax(attn_scores);

            // attn @ V
            const xb_head = self.xb.data[h * head_dim ..][0..head_dim];
            for (0..seq_len) |t| {
                const v_head = v_cache_base + t * kv_stride + kv_offset;
                const w = attn_scores[t];
                for (0..head_dim) |j| xb_head[j] += w * v_head[j];
            }
        }
    }
};

// =============================================================================
// Ops
// =============================================================================

fn addBias(out: []f32, bias: []const f32) void {
    const n = @min(out.len, bias.len);
    for (0..n) |i| out[i] += bias[i];
}

fn rmsNorm(out: []f32, x: []const f32, w: []const f32, eps: f32) void {
    const n = x.len;

    var ss: f32 = 0;
    for (x) |v| ss += v * v;

    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);

    for (0..n) |i| out[i] = x[i] * scale * w[i];
}

fn softmax(x: []f32) void {
    if (x.len == 0) return;

    var max_val = x[0];
    for (x[1..]) |v| max_val = @max(max_val, v);

    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }

    if (sum > 0) {
        for (x) |*v| v.* /= sum;
    }
}

fn add(a: []f32, b: []const f32) void {
    const n = @min(a.len, b.len);
    for (0..n) |i| a[i] += b[i];
}

fn siluMul(gate: []f32, up: []const f32) void {
    const n = @min(gate.len, up.len);
    for (0..n) |i| {
        const g = gate[i];
        const sigmoid = 1.0 / (1.0 + @exp(-g));
        gate[i] = g * sigmoid * up[i];
    }
}
