const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub const KVCache = struct {
    key_cache: []Tensor,
    value_cache: []Tensor,

    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,

    allocator: std.mem.Allocator,

    pub const MAX_SEQ_LEN_CAP: u32 = 4096;

    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: u32,
        max_seq_len: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) !KVCache {
        const capped_u32: u32 = if (MAX_SEQ_LEN_CAP != 0 and max_seq_len > MAX_SEQ_LEN_CAP)
            MAX_SEQ_LEN_CAP
        else
            max_seq_len;

        if (capped_u32 != max_seq_len) {
            std.debug.print(
                "KVCache: capping max_seq_len from {} to {} (to avoid huge memory)\n",
                .{ max_seq_len, capped_u32 },
            );
        }

        const layers: usize = @intCast(n_layers);
        const ms: usize = @intCast(capped_u32);
        const kvh: usize = @intCast(n_kv_heads);
        const hd: usize = @intCast(head_dim);

        if (layers == 0 or ms == 0 or kvh == 0 or hd == 0) {
            return error.InvalidKVConfig;
        }

        const kv_dim: usize = try mulNoOverflow(kvh, hd);
        const cache_size: usize = try mulNoOverflow(ms, kv_dim);

        const key_cache = try allocator.alloc(Tensor, layers);
        errdefer allocator.free(key_cache);

        const value_cache = try allocator.alloc(Tensor, layers);
        errdefer allocator.free(value_cache);

        for (0..layers) |i| {
            key_cache[i] = try allocTensor1DUninit(allocator, cache_size);
            errdefer key_cache[i].deinit();

            value_cache[i] = try allocTensor1DUninit(allocator, cache_size);
            errdefer value_cache[i].deinit();
        }

        return KVCache{
            .key_cache = key_cache,
            .value_cache = value_cache,
            .max_seq_len = ms,
            .n_kv_heads = kvh,
            .head_dim = hd,
            .kv_dim = kv_dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.key_cache) |*t| t.deinit();
        for (self.value_cache) |*t| t.deinit();
        self.allocator.free(self.key_cache);
        self.allocator.free(self.value_cache);
    }

    pub fn update(self: *KVCache, layer_idx: usize, k: Tensor, v: Tensor, pos: usize) !void {
        if (layer_idx >= self.key_cache.len) return error.LayerOutOfRange;
        if (pos >= self.max_seq_len) return error.ContextExceeded;

        if (k.data.len != self.kv_dim or v.data.len != self.kv_dim) {
            return error.InvalidKVSize;
        }

        const offset = pos * self.kv_dim;

        if (offset + self.kv_dim > self.key_cache[layer_idx].data.len) return error.CacheOutOfBounds;

        @memcpy(self.key_cache[layer_idx].data[offset..][0..self.kv_dim], k.data);
        @memcpy(self.value_cache[layer_idx].data[offset..][0..self.kv_dim], v.data);
    }
};

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

fn allocTensor1DUninit(allocator: std.mem.Allocator, len: usize) !Tensor {
    const data = try allocator.alloc(f32, len);
    errdefer allocator.free(data);

    // Tensor.shape expects []const u32, so we use a static shape reference
    // For 1D tensors in KVCache, we don't really need shape tracking
    return Tensor{
        .data = data,
        .shape = &[_]u32{@intCast(len)},
        .allocator = allocator,
    };
}

fn mulNoOverflow(a: usize, b: usize) !usize {
    const r = @mulWithOverflow(a, b);
    if (r[1] != 0) return error.Overflow;
    return r[0];
}
