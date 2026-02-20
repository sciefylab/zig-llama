// src/model.zig — Optimized weight loading (no double allocation)
const std = @import("std");
const builtin = @import("builtin");
const gguf = @import("gguf.zig");

const Tensor = @import("tensor.zig").Tensor;
const QuantizedTensor = @import("tensor.zig").QuantizedTensor;

const ALIGN_BYTES: usize = 64;
const ALIGNMENT: ?std.mem.Alignment = blk: {
    if (!std.math.isPowerOfTwo(ALIGN_BYTES)) @compileError("ALIGN_BYTES must be power-of-two");
    const log2: std.math.Log2Int(usize) = @intCast(@ctz(@as(usize, ALIGN_BYTES)));
    break :blk @as(std.mem.Alignment, @enumFromInt(log2));
};

pub const ModelConfig = struct {
    dim: u32,
    hidden_dim: u32,
    ffn_hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    max_seq_len: u32,
    rope_theta: f32,
    rms_norm_eps: f32,

    pub fn fromGGUF(g: *gguf.GGUFFile) ModelConfig {
        const arch = g.getMetadataString("general.architecture") orelse "llama";
        var key_buf: [128]u8 = undefined;
        const makeKey = struct {
            fn f(buf: *[128]u8, arch_name: []const u8, suffix: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "{s}.{s}", .{ arch_name, suffix }) catch "llama.embedding_length";
            }
        }.f;

        const dim = g.getMetadataU32(makeKey(&key_buf, arch, "embedding_length")) orelse 4096;
        return .{
            .dim = dim,
            .hidden_dim = dim,
            .ffn_hidden_dim = g.getMetadataU32(makeKey(&key_buf, arch, "feed_forward_length")) orelse 11008,
            .n_layers = g.getMetadataU32(makeKey(&key_buf, arch, "block_count")) orelse 32,
            .n_heads = g.getMetadataU32(makeKey(&key_buf, arch, "attention.head_count")) orelse 32,
            .n_kv_heads = g.getMetadataU32(makeKey(&key_buf, arch, "attention.head_count_kv")) orelse 32,
            .vocab_size = 151936,
            .max_seq_len = @min(g.getMetadataU32(makeKey(&key_buf, arch, "context_length")) orelse 2048, 4096),
            .rope_theta = g.getMetadataF32(makeKey(&key_buf, arch, "rope.freq_base")) orelse 10000.0,
            .rms_norm_eps = g.getMetadataF32(makeKey(&key_buf, arch, "attention.layer_norm_rms_epsilon")) orelse 1e-6,
        };
    }
};

pub const TransformerLayer = struct {
    wq: QuantizedTensor,
    wk: QuantizedTensor,
    wv: QuantizedTensor,
    wo: QuantizedTensor,
    bq: ?Tensor,
    bk: ?Tensor,
    bv: ?Tensor,
    w1: QuantizedTensor,
    w2: QuantizedTensor,
    w3: QuantizedTensor,
    attn_norm: Tensor,
    ffn_norm: Tensor,

    pub fn deinit(self: *TransformerLayer) void {
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.wo.deinit();
        if (self.bq) |*bq| bq.deinit();
        if (self.bk) |*bk| bk.deinit();
        if (self.bv) |*bv| bv.deinit();
        self.w1.deinit();
        self.w2.deinit();
        self.w3.deinit();
        self.attn_norm.deinit();
        self.ffn_norm.deinit();
    }
};

pub const LlamaModel = struct {
    config: ModelConfig,
    layers: []TransformerLayer,
    norm: Tensor,
    tok_embeddings: QuantizedTensor,
    output: QuantizedTensor,
    use_tied_embeddings: bool,
    allocator: std.mem.Allocator,

    pub fn fromGGUF(gguf_file: *gguf.GGUFFile, allocator: std.mem.Allocator) !LlamaModel {
        var config = ModelConfig.fromGGUF(gguf_file);

        // Open file for direct reading — no full file buffer
        const file = try std.fs.cwd().openFile(gguf_file.file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        std.debug.print("  File size: {d:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / 1048576.0});

        if (gguf_file.getTensorByName("token_embd.weight")) |emb_info| {
            if (emb_info.n_dims >= 2) config.vocab_size = @intCast(emb_info.dims[1]);
        }

        std.debug.print("Loading Q8_0 model: {} layers, dim={}, vocab={}\n", .{ config.n_layers, config.dim, config.vocab_size });

        const layers = try allocator.alloc(TransformerLayer, config.n_layers);
        errdefer allocator.free(layers);

        for (0..config.n_layers) |li| {
            if (li == 0 or (li + 1) % 10 == 0)
                std.debug.print("  Loading layer {}/{}...\n", .{ li + 1, config.n_layers });

            var nb: [128]u8 = undefined;

            layers[li] = .{
                .wq = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.attn_q.weight", li),
                .wk = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.attn_k.weight", li),
                .wv = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.attn_v.weight", li),
                .wo = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.attn_output.weight", li),
                .bq = loadFT(gguf_file, file, allocator, &nb, "blk.{}.attn_q.bias", li),
                .bk = loadFT(gguf_file, file, allocator, &nb, "blk.{}.attn_k.bias", li),
                .bv = loadFT(gguf_file, file, allocator, &nb, "blk.{}.attn_v.bias", li),
                .w1 = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.ffn_gate.weight", li),
                .w2 = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.ffn_down.weight", li),
                .w3 = try loadQT(gguf_file, file, allocator, &nb, "blk.{}.ffn_up.weight", li),
                .attn_norm = try loadFTReq(gguf_file, file, allocator, &nb, "blk.{}.attn_norm.weight", li),
                .ffn_norm = try loadFTReq(gguf_file, file, allocator, &nb, "blk.{}.ffn_norm.weight", li),
            };

            if (li == 0 and layers[0].bq != null)
                std.debug.print("  Note: Model has attention biases (Qwen2 style)\n", .{});
        }

        std.debug.print("  Loading token embeddings (Q8_0)...\n", .{});
        const tok_emb = try loadQTByName(gguf_file, file, allocator, "token_embd.weight");
        if (tok_emb.shape.len >= 2) config.vocab_size = tok_emb.shape[1];

        std.debug.print("  Loading output projection...\n", .{});
        var use_tied = false;
        const output = loadQTByName(gguf_file, file, allocator, "output.weight") catch |err| blk: {
            if (err == error.TensorNotFound) {
                std.debug.print("  Note: Using tied embeddings\n", .{});
                use_tied = true;
                break :blk QuantizedTensor{
                    .data = @constCast(@as([]const u8, &[_]u8{})),
                    .shape = @constCast(@as([]const u32, &[_]u32{})),
                    .n_blocks = 0,
                    .block_size = 32,
                    .quant_type = .Q8_0,
                    .allocator = allocator,
                    .owns_data = false,
                };
            }
            return err;
        };

        const norm = try loadFTByName(gguf_file, file, allocator, "output_norm.weight");

        // Memory stats
        var qb: usize = 0;
        var fb: usize = 0;
        for (layers) |*l| {
            qb += l.wq.data.len + l.wk.data.len + l.wv.data.len + l.wo.data.len + l.w1.data.len + l.w2.data.len + l.w3.data.len;
            fb += l.attn_norm.data.len * 4 + l.ffn_norm.data.len * 4;
            if (l.bq) |b| fb += b.data.len * 4;
            if (l.bk) |b| fb += b.data.len * 4;
            if (l.bv) |b| fb += b.data.len * 4;
        }
        qb += tok_emb.data.len;
        fb += norm.data.len * 4;
        if (!use_tied) qb += output.data.len;

        std.debug.print("\nMemory Usage:\n", .{});
        std.debug.print("  Quantized weights: {d:.2} MB\n", .{@as(f64, @floatFromInt(qb)) / 1048576.0});
        std.debug.print("  F32 tensors:       {d:.2} MB\n", .{@as(f64, @floatFromInt(fb)) / 1048576.0});
        std.debug.print("  Total:             {d:.2} MB\n", .{@as(f64, @floatFromInt(qb + fb)) / 1048576.0});

        return .{
            .config = config,
            .layers = layers,
            .norm = norm,
            .tok_embeddings = tok_emb,
            .output = output,
            .use_tied_embeddings = use_tied,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LlamaModel) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        self.norm.deinit();
        self.tok_embeddings.deinit();
        if (!self.use_tied_embeddings) self.output.deinit();
    }
};

// =============================================================================
// Tensor Loading — direct file read to aligned buffer (no double copy)
// =============================================================================

fn loadQT(gf: *gguf.GGUFFile, file: std.fs.File, a: std.mem.Allocator, nb: *[128]u8, comptime fmt: []const u8, li: usize) !QuantizedTensor {
    const name = std.fmt.bufPrint(nb, fmt, .{li}) catch unreachable;
    return loadQTByName(gf, file, a, name);
}

fn loadQTByName(gf: *gguf.GGUFFile, file: std.fs.File, a: std.mem.Allocator, name: []const u8) !QuantizedTensor {
    const ti = gf.getTensorByName(name) orelse return error.TensorNotFound;
    const off = gf.tensor_data_offset + ti.offset;
    const qt: QuantizedTensor.QuantType = switch (ti.dtype) {
        .Q4_0 => .Q4_0,
        .Q4_1 => .Q4_1,
        .Q8_0 => .Q8_0,
        .F32 => return error.F32NotQuantized,
        else => return error.UnsupportedQuantType,
    };
    const bpb: u32 = switch (qt) {
        .Q4_0 => 18,
        .Q4_1 => 20,
        .Q8_0 => 34,
    };
    var total: u64 = 1;
    for (0..ti.n_dims) |j| total *= ti.dims[j];
    const nb2 = @as(u32, @intCast(total / 32));
    const ds = @as(usize, nb2) * bpb;

    const shape = try a.alloc(u32, ti.n_dims);
    errdefer a.free(shape);
    for (0..ti.n_dims) |j| shape[j] = @intCast(ti.dims[j]);

    // Allocate aligned buffer and read directly from file
    const d = try a.alignedAlloc(u8, ALIGNMENT, ds);
    errdefer a.free(d);

    try file.seekTo(off);
    const bytes_read = try file.readAll(d);
    if (bytes_read != ds) return error.IncompleteRead;

    return .{
        .data = d,
        .shape = shape,
        .n_blocks = nb2,
        .block_size = 32,
        .quant_type = qt,
        .allocator = a,
        .owns_data = true,
    };
}

fn loadFT(gf: *gguf.GGUFFile, file: std.fs.File, a: std.mem.Allocator, nb: *[128]u8, comptime fmt: []const u8, li: usize) ?Tensor {
    const name = std.fmt.bufPrint(nb, fmt, .{li}) catch unreachable;
    return loadFTByName(gf, file, a, name) catch null;
}

fn loadFTReq(gf: *gguf.GGUFFile, file: std.fs.File, a: std.mem.Allocator, nb: *[128]u8, comptime fmt: []const u8, li: usize) !Tensor {
    const name = std.fmt.bufPrint(nb, fmt, .{li}) catch unreachable;
    return loadFTByName(gf, file, a, name);
}

fn loadFTByName(gf: *gguf.GGUFFile, file: std.fs.File, a: std.mem.Allocator, name: []const u8) !Tensor {
    const ti = gf.getTensorByName(name) orelse return error.TensorNotFound;
    const off = gf.tensor_data_offset + ti.offset;
    var total: usize = 1;
    for (0..ti.n_dims) |j| total *= @intCast(ti.dims[j]);

    // Allocate aligned buffer and read directly
    const data = try a.alignedAlloc(f32, ALIGNMENT, total);
    errdefer a.free(data);

    try file.seekTo(off);
    const bytes_read = try file.readAll(std.mem.sliceAsBytes(data));
    if (bytes_read != total * 4) return error.IncompleteRead;

    return .{ .data = data, .shape = &[_]u32{@intCast(total)}, .allocator = a };
}
