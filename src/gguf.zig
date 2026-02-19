const std = @import("std");

const GGUF_MAGIC = 0x46554747;

pub const GGUFValueType = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    _,
};

pub const GGMLType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    _,
};

/// Metadata value - bisa berbagai tipe
pub const MetaValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    uint64: u64,
    int64: i64,
    float32: f32,
    float64: f64,
    bool_val: bool,
    string: []const u8,
    array_info: struct { item_type: u32, len: u64 },
};

/// Info tentang satu tensor
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    dtype: GGMLType,
    offset: u64,
};

// Helper: Baca integer langsung dari file
fn readInt(comptime T: type, file: std.fs.File) !T {
    var buf: [@sizeOf(T)]u8 = undefined;
    const n = try file.readAll(&buf);
    if (n != @sizeOf(T)) return error.EndOfStream;
    return std.mem.readInt(T, &buf, .little);
}

// Helper: Baca bytes dengan panjang tertentu
fn readBytes(allocator: std.mem.Allocator, file: std.fs.File, len: u64) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    const n = try file.readAll(buf);
    if (n != len) {
        return error.EndOfStream;
    }
    return buf;
}

pub const GGUFFile = struct {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,

    // Stored data
    metadata: std.StringHashMap(MetaValue),
    tensors: []TensorInfo, // Changed: gunakan slice biasa
    tensor_data_offset: u64,

    // Untuk vocab/tokenizer
    vocab_tokens: ?[][]const u8,
    vocab_scores: ?[]f32,

    allocator: std.mem.Allocator,
    file_path: []const u8,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !GGUFFile {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // === HEADER ===
        const magic = try readInt(u32, file);
        if (magic != GGUF_MAGIC) return error.InvalidFormat;

        const version = try readInt(u32, file);
        const tensor_count = try readInt(u64, file);
        const kv_count = try readInt(u64, file);

        std.debug.print("=== GGUF Header ===\n", .{});
        std.debug.print("Version: {d} | Tensors: {d} | Metadata: {d}\n\n", .{ version, tensor_count, kv_count });

        // Initialize metadata HashMap
        var metadata = std.StringHashMap(MetaValue).init(allocator);
        errdefer metadata.deinit();

        // Temporary storage for vocab
        var vocab_tokens: ?[][]const u8 = null;
        var vocab_scores: ?[]f32 = null;

        // === METADATA ===
        std.debug.print("=== Metadata ===\n", .{});
        var i: u64 = 0;
        while (i < kv_count) : (i += 1) {
            const key = try readStringAlloc(allocator, file);
            errdefer allocator.free(key);

            const raw_type = try readInt(u32, file);
            const val_type: GGUFValueType = @enumFromInt(raw_type);

            const result = try readValue(allocator, file, val_type, key);
            const value = result.value;

            // Update vocab if loaded
            if (result.vocab_tokens) |vt| vocab_tokens = vt;
            if (result.vocab_scores) |vs| vocab_scores = vs;

            try metadata.put(key, value);
            printMetadata(i, key, value);
        }

        // === TENSOR INFO ===
        std.debug.print("\n=== Tensor Info ({d} tensors) ===\n", .{tensor_count});

        // Allocate tensor array
        const tensors = try allocator.alloc(TensorInfo, tensor_count);
        errdefer allocator.free(tensors);

        var t: u64 = 0;
        while (t < tensor_count) : (t += 1) {
            const tensor_info = try readTensorInfo(allocator, file);
            tensors[t] = tensor_info;

            // Print first 5 and last 2
            if (t < 5 or t >= tensor_count - 2) {
                std.debug.print("[{d: >3}] {s: <40} shape: {any} type: {s}\n", .{
                    t,
                    tensor_info.name,
                    tensor_info.dims[0..tensor_info.n_dims],
                    @tagName(tensor_info.dtype),
                });
            } else if (t == 5) {
                std.debug.print("      ... ({d} more tensors) ...\n", .{tensor_count - 7});
            }
        }

        // Calculate tensor data offset (aligned to 32 bytes)
        const current_pos = try file.getPos();
        const tensor_data_offset = (current_pos + 31) & ~@as(u64, 31);

        std.debug.print("\n=== Data Section ===\n", .{});
        std.debug.print("Tensor data starts at offset: 0x{X} ({d} bytes)\n", .{ tensor_data_offset, tensor_data_offset });

        return GGUFFile{
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = kv_count,
            .metadata = metadata,
            .tensors = tensors,
            .tensor_data_offset = tensor_data_offset,
            .vocab_tokens = vocab_tokens,
            .vocab_scores = vocab_scores,
            .allocator = allocator,
            .file_path = try allocator.dupe(u8, path),
        };
    }

    fn readStringAlloc(allocator: std.mem.Allocator, file: std.fs.File) ![]u8 {
        const len = try readInt(u64, file);
        return try readBytes(allocator, file, len);
    }

    const ReadValueResult = struct {
        value: MetaValue,
        vocab_tokens: ?[][]const u8 = null,
        vocab_scores: ?[]f32 = null,
    };

    fn readValue(allocator: std.mem.Allocator, file: std.fs.File, val_type: GGUFValueType, key: []const u8) !ReadValueResult {
        var result = ReadValueResult{ .value = undefined };

        result.value = switch (val_type) {
            .UINT8 => .{ .uint8 = try readInt(u8, file) },
            .INT8 => .{ .int8 = try readInt(i8, file) },
            .UINT16 => .{ .uint16 = try readInt(u16, file) },
            .INT16 => .{ .int16 = try readInt(i16, file) },
            .UINT32 => .{ .uint32 = try readInt(u32, file) },
            .INT32 => .{ .int32 = try readInt(i32, file) },
            .UINT64 => .{ .uint64 = try readInt(u64, file) },
            .INT64 => .{ .int64 = try readInt(i64, file) },
            .FLOAT32 => blk: {
                const bits = try readInt(u32, file);
                break :blk .{ .float32 = @bitCast(bits) };
            },
            .FLOAT64 => blk: {
                const bits = try readInt(u64, file);
                break :blk .{ .float64 = @bitCast(bits) };
            },
            .BOOL => .{ .bool_val = (try readInt(u8, file)) != 0 },
            .STRING => .{ .string = try readStringAlloc(allocator, file) },
            .ARRAY => blk: {
                const item_type = try readInt(u32, file);
                const arr_len = try readInt(u64, file);

                // Special handling untuk vocab tokens
                if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
                    result.vocab_tokens = try readVocabTokens(allocator, file, arr_len);
                } else if (std.mem.eql(u8, key, "tokenizer.ggml.scores")) {
                    result.vocab_scores = try readVocabScores(allocator, file, arr_len);
                } else {
                    // Skip array data
                    try skipArray(file, @enumFromInt(item_type), arr_len);
                }

                break :blk .{ .array_info = .{ .item_type = item_type, .len = arr_len } };
            },
            _ => .{ .uint32 = 0 },
        };

        return result;
    }

    fn readVocabTokens(allocator: std.mem.Allocator, file: std.fs.File, count: u64) ![][]const u8 {
        const tokens = try allocator.alloc([]const u8, count);
        errdefer allocator.free(tokens);

        for (0..count) |i| {
            tokens[i] = try readStringAlloc(allocator, file);
        }
        std.debug.print("        [Loaded {d} vocab tokens]\n", .{count});
        return tokens;
    }

    fn readVocabScores(allocator: std.mem.Allocator, file: std.fs.File, count: u64) ![]f32 {
        const scores = try allocator.alloc(f32, count);
        errdefer allocator.free(scores);

        for (0..count) |i| {
            const bits = try readInt(u32, file);
            scores[i] = @bitCast(bits);
        }
        std.debug.print("        [Loaded {d} vocab scores]\n", .{count});
        return scores;
    }

    fn skipArray(file: std.fs.File, item_type: GGUFValueType, count: u64) !void {
        const item_size: u64 = switch (item_type) {
            .UINT8, .INT8, .BOOL => 1,
            .UINT16, .INT16 => 2,
            .UINT32, .INT32, .FLOAT32 => 4,
            .UINT64, .INT64, .FLOAT64 => 8,
            .STRING => {
                // Must read each string length and skip
                for (0..count) |_| {
                    const len = try readInt(u64, file);
                    try file.seekBy(@intCast(len));
                }
                return;
            },
            else => 4,
        };
        try file.seekBy(@intCast(count * item_size));
    }

    fn readTensorInfo(allocator: std.mem.Allocator, file: std.fs.File) !TensorInfo {
        const name = try readStringAlloc(allocator, file);
        const n_dims = try readInt(u32, file);

        var dims: [4]u64 = .{ 1, 1, 1, 1 };
        for (0..n_dims) |i| {
            dims[i] = try readInt(u64, file);
        }

        const dtype_raw = try readInt(u32, file);
        const offset = try readInt(u64, file);

        return TensorInfo{
            .name = name,
            .n_dims = n_dims,
            .dims = dims,
            .dtype = @enumFromInt(dtype_raw),
            .offset = offset,
        };
    }

    fn printMetadata(idx: u64, key: []const u8, value: MetaValue) void {
        std.debug.print("[{d: >2}] {s: <40} = ", .{ idx, key });
        switch (value) {
            .uint8 => |v| std.debug.print("{d} (u8)\n", .{v}),
            .int8 => |v| std.debug.print("{d} (i8)\n", .{v}),
            .uint16 => |v| std.debug.print("{d} (u16)\n", .{v}),
            .int16 => |v| std.debug.print("{d} (i16)\n", .{v}),
            .uint32 => |v| std.debug.print("{d} (u32)\n", .{v}),
            .int32 => |v| std.debug.print("{d} (i32)\n", .{v}),
            .uint64 => |v| std.debug.print("{d} (u64)\n", .{v}),
            .int64 => |v| std.debug.print("{d} (i64)\n", .{v}),
            .float32 => |v| std.debug.print("{d:.6} (f32)\n", .{v}),
            .float64 => |v| std.debug.print("{d:.6} (f64)\n", .{v}),
            .bool_val => |v| std.debug.print("{}\n", .{v}),
            .string => |v| {
                if (v.len > 50) {
                    std.debug.print("\"{s}...\" (len:{d})\n", .{ v[0..40], v.len });
                } else {
                    std.debug.print("\"{s}\"\n", .{v});
                }
            },
            .array_info => |v| std.debug.print("[Array type:{d} len:{d}]\n", .{ v.item_type, v.len }),
        }
    }

    // === ACCESSOR METHODS ===

    pub fn getMetadataU32(self: *const GGUFFile, key: []const u8) ?u32 {
        if (self.metadata.get(key)) |val| {
            return switch (val) {
                .uint32 => |v| v,
                else => null,
            };
        }
        return null;
    }

    pub fn getMetadataF32(self: *const GGUFFile, key: []const u8) ?f32 {
        if (self.metadata.get(key)) |val| {
            return switch (val) {
                .float32 => |v| v,
                else => null,
            };
        }
        return null;
    }

    pub fn getMetadataString(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        if (self.metadata.get(key)) |val| {
            return switch (val) {
                .string => |v| v,
                else => null,
            };
        }
        return null;
    }

    pub fn getTensorByName(self: *const GGUFFile, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) {
                return t;
            }
        }
        return null;
    }

    pub fn deinit(self: *GGUFFile) void {
        // Free stored strings in metadata
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            switch (entry.value_ptr.*) {
                .string => |s| self.allocator.free(s),
                else => {},
            }
        }
        self.metadata.deinit();

        // Free tensor names
        for (self.tensors) |t| {
            self.allocator.free(t.name);
        }
        self.allocator.free(self.tensors);

        // Free vocab
        if (self.vocab_tokens) |tokens| {
            for (tokens) |t| {
                self.allocator.free(t);
            }
            self.allocator.free(tokens);
        }
        if (self.vocab_scores) |scores| {
            self.allocator.free(scores);
        }

        self.allocator.free(self.file_path);
    }
};
