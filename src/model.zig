const std = @import("std");
const gguf = @import("gguf.zig");

const Tensor = @import("tensor.zig").Tensor;
const QuantizedTensor = @import("tensor.zig").QuantizedTensor;

// ============================
// Alignment for fast SIMD loads
// ============================

// must be power-of-two
const ALIGN_BYTES: usize = 64;

// Zig 0.15.x: alignedAlloc expects ?std.mem.Alignment (enum of log2(bytes))
const ALIGNMENT: ?std.mem.Alignment = blk: {
    if (!std.math.isPowerOfTwo(ALIGN_BYTES)) {
        @compileError("ALIGN_BYTES must be power-of-two");
    }
    const log2: std.math.Log2Int(usize) = @intCast(@ctz(@as(usize, ALIGN_BYTES)));
    break :blk @as(std.mem.Alignment, @enumFromInt(log2));
};

// ============================
// Model config
// ============================

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
        const ffn_dim = g.getMetadataU32(makeKey(&key_buf, arch, "feed_forward_length")) orelse 11008;
        const n_layers = g.getMetadataU32(makeKey(&key_buf, arch, "block_count")) orelse 32;
        const n_heads = g.getMetadataU32(makeKey(&key_buf, arch, "attention.head_count")) orelse 32;
        const n_kv_heads = g.getMetadataU32(makeKey(&key_buf, arch, "attention.head_count_kv")) orelse 32;
        const ctx_len = g.getMetadataU32(makeKey(&key_buf, arch, "context_length")) orelse 2048;
        const rope_theta = g.getMetadataF32(makeKey(&key_buf, arch, "rope.freq_base")) orelse 10000.0;
        const rms_eps = g.getMetadataF32(makeKey(&key_buf, arch, "attention.layer_norm_rms_epsilon")) orelse 1e-6;

        return ModelConfig{
            .dim = dim,
            .hidden_dim = dim,
            .ffn_hidden_dim = ffn_dim,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .vocab_size = 151936,
            .max_seq_len = @min(ctx_len, 4096),
            .rope_theta = rope_theta,
            .rms_norm_eps = rms_eps,
        };
    }
};

// ============================
// Layer + model structs
// ============================

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
    file_data: []u8,

    pub fn fromGGUF(gguf_file: *gguf.GGUFFile, allocator: std.mem.Allocator) !LlamaModel {
        var config = ModelConfig.fromGGUF(gguf_file);

        // Read file into memory (raw bytes)
        const file = try std.fs.cwd().openFile(gguf_file.file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const file_data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(file_data);

        const bytes_read = try file.readAll(file_data);
        if (bytes_read != file_size) {
            return error.IncompleteRead;
        }

        std.debug.print("  File loaded: {d:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / 1024.0 / 1024.0});

        // Get vocab size from token embeddings tensor
        if (gguf_file.getTensorByName("token_embd.weight")) |emb_info| {
            if (emb_info.n_dims >= 2) {
                config.vocab_size = @intCast(emb_info.dims[1]);
            }
        }

        std.debug.print("Loading Q8_0 model: {} layers, dim={}, vocab={}\n", .{
            config.n_layers,
            config.dim,
            config.vocab_size,
        });

        const layers = try allocator.alloc(TransformerLayer, config.n_layers);
        errdefer allocator.free(layers);

        for (0..config.n_layers) |layer_idx| {
            if (layer_idx == 0 or (layer_idx + 1) % 10 == 0) {
                std.debug.print("  Loading layer {}/{}...\n", .{ layer_idx + 1, config.n_layers });
            }

            var name_buf: [128]u8 = undefined;

            const wq = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_q.weight", layer_idx);
            const wk = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_k.weight", layer_idx);
            const wv = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_v.weight", layer_idx);
            const wo = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_output.weight", layer_idx);

            const bq = loadF32Tensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_q.bias", layer_idx);
            const bk = loadF32Tensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_k.bias", layer_idx);
            const bv = loadF32Tensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_v.bias", layer_idx);

            if (layer_idx == 0 and bq != null) {
                std.debug.print("  Note: Model has attention biases (Qwen2 style)\n", .{});
            }

            const w1 = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.ffn_gate.weight", layer_idx);
            const w2 = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.ffn_down.weight", layer_idx);
            const w3 = try loadQuantizedTensor(gguf_file, file_data, allocator, &name_buf, "blk.{}.ffn_up.weight", layer_idx);

            const attn_norm = try loadF32TensorRequired(gguf_file, file_data, allocator, &name_buf, "blk.{}.attn_norm.weight", layer_idx);
            const ffn_norm = try loadF32TensorRequired(gguf_file, file_data, allocator, &name_buf, "blk.{}.ffn_norm.weight", layer_idx);

            layers[layer_idx] = TransformerLayer{
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .wo = wo,
                .bq = bq,
                .bk = bk,
                .bv = bv,
                .w1 = w1,
                .w2 = w2,
                .w3 = w3,
                .attn_norm = attn_norm,
                .ffn_norm = ffn_norm,
            };
        }

        std.debug.print("  Loading token embeddings (Q8_0)...\n", .{});
        const tok_emb = try loadQuantizedTensorByName(gguf_file, file_data, allocator, "token_embd.weight");

        if (tok_emb.shape.len >= 2) {
            config.vocab_size = tok_emb.shape[1];
        }

        std.debug.print("  Loading output projection...\n", .{});
        var use_tied_embeddings = false;

        const output = loadQuantizedTensorByName(gguf_file, file_data, allocator, "output.weight") catch |err| blk: {
            if (err == error.TensorNotFound) {
                std.debug.print("  Note: Using tied embeddings (output = token_embd)\n", .{});
                use_tied_embeddings = true;

                const empty_u8: []u8 = @constCast(@as([]const u8, &[_]u8{}));
                const empty_u32: []u32 = @constCast(@as([]const u32, &[_]u32{}));

                break :blk QuantizedTensor{
                    .data = empty_u8,
                    .shape = empty_u32,
                    .n_blocks = 0,
                    .block_size = 32,
                    .quant_type = .Q8_0,
                    .allocator = allocator,
                };
            }
            return err;
        };

        const norm = try loadF32TensorByName(gguf_file, file_data, allocator, "output_norm.weight");

        // Memory report
        var quant_bytes: usize = 0;
        var f32_bytes: usize = 0;

        for (layers) |*layer| {
            quant_bytes += layer.wq.data.len + layer.wk.data.len + layer.wv.data.len + layer.wo.data.len;
            quant_bytes += layer.w1.data.len + layer.w2.data.len + layer.w3.data.len;

            f32_bytes += layer.attn_norm.data.len * 4 + layer.ffn_norm.data.len * 4;
            if (layer.bq) |bq0| f32_bytes += bq0.data.len * 4;
            if (layer.bk) |bk0| f32_bytes += bk0.data.len * 4;
            if (layer.bv) |bv0| f32_bytes += bv0.data.len * 4;
        }

        quant_bytes += tok_emb.data.len;
        f32_bytes += norm.data.len * 4;
        if (!use_tied_embeddings) quant_bytes += output.data.len;

        std.debug.print("\nMemory Usage:\n", .{});
        std.debug.print("  Quantized weights: {d:.2} MB\n", .{@as(f64, @floatFromInt(quant_bytes)) / 1024.0 / 1024.0});
        std.debug.print("  F32 tensors:       {d:.2} MB\n", .{@as(f64, @floatFromInt(f32_bytes)) / 1024.0 / 1024.0});
        std.debug.print("  Total:             {d:.2} MB\n", .{@as(f64, @floatFromInt(quant_bytes + f32_bytes)) / 1024.0 / 1024.0});

        return LlamaModel{
            .config = config,
            .layers = layers,
            .norm = norm,
            .tok_embeddings = tok_emb,
            .output = output,
            .use_tied_embeddings = use_tied_embeddings,
            .allocator = allocator,
            .file_data = file_data,
        };
    }

    pub fn deinit(self: *LlamaModel) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);

        self.norm.deinit();
        self.tok_embeddings.deinit();
        if (!self.use_tied_embeddings) {
            self.output.deinit();
        }

        self.allocator.free(self.file_data);
    }
};

// =============================================================================
// Tensor Loading Helpers
// =============================================================================

fn loadQuantizedTensor(
    gguf_file: *gguf.GGUFFile,
    file_data: []const u8,
    allocator: std.mem.Allocator,
    name_buf: *[128]u8,
    comptime fmt: []const u8,
    layer_idx: usize,
) !QuantizedTensor {
    const name = std.fmt.bufPrint(name_buf, fmt, .{layer_idx}) catch unreachable;
    return loadQuantizedTensorByName(gguf_file, file_data, allocator, name);
}

fn loadQuantizedTensorByName(
    gguf_file: *gguf.GGUFFile,
    file_data: []const u8,
    allocator: std.mem.Allocator,
    name: []const u8,
) !QuantizedTensor {
    const tensor_info = gguf_file.getTensorByName(name) orelse {
        return error.TensorNotFound;
    };

    const data_offset = gguf_file.tensor_data_offset + tensor_info.offset;

    const quant_type: QuantizedTensor.QuantType = switch (tensor_info.dtype) {
        .Q4_0 => .Q4_0,
        .Q4_1 => .Q4_1,
        .Q8_0 => .Q8_0,
        .F32 => return error.F32NotQuantized,
        else => {
            std.debug.print("Unsupported quant type {s} for {s}\n", .{ @tagName(tensor_info.dtype), name });
            return error.UnsupportedQuantType;
        },
    };

    const block_size: u32 = 32;

    const bytes_per_block: u32 = switch (quant_type) {
        .Q4_0 => 18,
        .Q4_1 => 20,
        .Q8_0 => 34,
    };

    var total_elements: u64 = 1;
    for (0..tensor_info.n_dims) |i| {
        total_elements *= tensor_info.dims[i];
    }

    const n_blocks = @as(u32, @intCast(total_elements / block_size));
    const data_size = @as(usize, n_blocks) * bytes_per_block;

    // NOTE: Zig 0.15 alignedAlloc expects ?std.mem.Alignment
    const quant_data = try allocator.alignedAlloc(u8, ALIGNMENT, data_size);
    errdefer allocator.free(quant_data);

    @memcpy(quant_data, file_data[data_offset..][0..data_size]);

    const shape = try allocator.alloc(u32, tensor_info.n_dims);
    for (0..tensor_info.n_dims) |i| {
        shape[i] = @intCast(tensor_info.dims[i]);
    }

    return QuantizedTensor{
        .data = quant_data,
        .shape = shape,
        .n_blocks = n_blocks,
        .block_size = block_size,
        .quant_type = quant_type,
        .allocator = allocator,
    };
}

fn loadF32Tensor(
    gguf_file: *gguf.GGUFFile,
    file_data: []const u8,
    allocator: std.mem.Allocator,
    name_buf: *[128]u8,
    comptime fmt: []const u8,
    layer_idx: usize,
) ?Tensor {
    const name = std.fmt.bufPrint(name_buf, fmt, .{layer_idx}) catch unreachable;
    return loadF32TensorByName(gguf_file, file_data, allocator, name) catch null;
}

fn loadF32TensorRequired(
    gguf_file: *gguf.GGUFFile,
    file_data: []const u8,
    allocator: std.mem.Allocator,
    name_buf: *[128]u8,
    comptime fmt: []const u8,
    layer_idx: usize,
) !Tensor {
    const name = std.fmt.bufPrint(name_buf, fmt, .{layer_idx}) catch unreachable;
    return loadF32TensorByName(gguf_file, file_data, allocator, name);
}

fn loadF32TensorByName(
    gguf_file: *gguf.GGUFFile,
    file_data: []const u8,
    allocator: std.mem.Allocator,
    name: []const u8,
) !Tensor {
    const tensor_info = gguf_file.getTensorByName(name) orelse return error.TensorNotFound;

    const data_offset = gguf_file.tensor_data_offset + tensor_info.offset;

    var total_elements: usize = 1;
    for (0..tensor_info.n_dims) |i| {
        total_elements *= @intCast(tensor_info.dims[i]);
    }

    // aligned allocation for faster SIMD
    const data = try allocator.alignedAlloc(f32, ALIGNMENT, total_elements);
    errdefer allocator.free(data);

    const src = file_data[data_offset..][0 .. total_elements * 4];
    @memcpy(std.mem.sliceAsBytes(data), src);

    return Tensor{
        .data = data,
        .shape = &[_]u32{@intCast(total_elements)},
        .allocator = allocator,
    };
}
