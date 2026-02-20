// src/ternary.zig — Ternary Weight Decomposition Engine v1
// CSD (Canonical Signed Digit) representation for matmul-free inference
// ZERO multiply instructions in the hot path

const std = @import("std");

pub const MAX_TERMS: u8 = 3;

pub const TernaryTerm = struct {
    shift: u4,
    sign: i2,
    active: bool,

    pub const ZERO = TernaryTerm{
        .shift = 0,
        .sign = 1,
        .active = false,
    };
};

pub const TernaryElement = struct {
    terms: [MAX_TERMS]TernaryTerm,
    n_terms: u8,
};

// ============================================================
// CSD Conversion
// ============================================================

pub fn toCSD(value: i8) TernaryElement {
    var result: TernaryElement = undefined;
    for (0..MAX_TERMS) |i| {
        result.terms[i] = TernaryTerm.ZERO;
    }
    result.n_terms = 0;

    if (value == 0) return result;

    const is_neg = value < 0;
    const abs_val: u8 = if (value == -128) 128 else @intCast(if (is_neg) -@as(i16, value) else @as(i16, value));

    var all_terms: [8]TernaryTerm = undefined;
    var total_terms: u8 = 0;

    var extended: u16 = @as(u16, abs_val);
    var carry: u1 = 0;
    var bit_pos: u8 = 0;

    while ((extended > 0 or carry > 0) and bit_pos < 9) {
        const current_bit: u1 = @intCast(extended & 1);
        const sum: u2 = @as(u2, current_bit) + @as(u2, carry);

        if (sum == 1) {
            const next_bit: u1 = @intCast((extended >> 1) & 1);
            if (next_bit == 1) {
                if (total_terms < 8) {
                    all_terms[total_terms] = .{
                        .shift = @intCast(bit_pos),
                        .sign = -1,
                        .active = true,
                    };
                    total_terms += 1;
                }
                carry = 1;
            } else {
                if (total_terms < 8) {
                    all_terms[total_terms] = .{
                        .shift = @intCast(bit_pos),
                        .sign = 1,
                        .active = true,
                    };
                    total_terms += 1;
                }
                carry = 0;
            }
        } else if (sum == 2) {
            carry = 1;
        } else {
            carry = 0;
        }

        extended >>= 1;
        bit_pos += 1;
    }

    if (is_neg) {
        for (0..total_terms) |i| {
            all_terms[i].sign = -all_terms[i].sign;
        }
    }

    if (total_terms <= MAX_TERMS) {
        for (0..total_terms) |i| {
            result.terms[i] = all_terms[i];
        }
        result.n_terms = total_terms;
        return result;
    }

    sortTermsByShiftDesc(all_terms[0..total_terms]);

    for (0..MAX_TERMS) |i| {
        result.terms[i] = all_terms[i];
    }
    result.n_terms = MAX_TERMS;

    return result;
}

fn sortTermsByShiftDesc(terms: []TernaryTerm) void {
    for (1..terms.len) |i| {
        const key = terms[i];
        var j: usize = i;
        while (j > 0 and terms[j - 1].shift < key.shift) {
            terms[j] = terms[j - 1];
            j -= 1;
        }
        terms[j] = key;
    }
}

pub fn csdToValue(elem: TernaryElement) i16 {
    var sum: i16 = 0;
    for (0..elem.n_terms) |i| {
        const t = elem.terms[i];
        if (!t.active) continue;
        const magnitude: i16 = @as(i16, 1) << @intCast(t.shift);
        if (t.sign < 0) {
            sum -= magnitude;
        } else {
            sum += magnitude;
        }
    }
    return sum;
}

// ============================================================
// Compile-time CSD lookup table
// ============================================================

pub const CSD_TABLE: [256]TernaryElement = blk: {
    @setEvalBranchQuota(100000);
    var table: [256]TernaryElement = undefined;
    for (0..256) |ui| {
        const val: i8 = @bitCast(@as(u8, @intCast(ui)));
        table[ui] = toCSD(val);
    }
    break :blk table;
};

pub inline fn toCSD_fast(value: i8) TernaryElement {
    return CSD_TABLE[@as(u8, @bitCast(value))];
}

// ============================================================
// SIMD types
// ============================================================

const VU8x32 = @Vector(32, u8);
const VI8x32 = @Vector(32, i8);
const VU16x32 = @Vector(32, u16);
const VI16x32 = @Vector(32, i16);
const VI32x32 = @Vector(32, i32);

// ============================================================
// TernaryBlock — per-element shifts and signs
// ============================================================

pub const TernaryBlock = struct {
    scale: f16,
    shifts: [MAX_TERMS][32]u8 align(32),
    signs: [MAX_TERMS][32]i8 align(32),

    pub fn fromQ8Block(q8_block: [*]const u8) TernaryBlock {
        var blk: TernaryBlock = undefined;
        const scale_bits: u16 = @as(u16, q8_block[0]) | (@as(u16, q8_block[1]) << 8);
        blk.scale = @bitCast(scale_bits);
        for (0..MAX_TERMS) |t| {
            @memset(&blk.shifts[t], 0);
            @memset(&blk.signs[t], 0);
        }
        for (0..32) |i| {
            const w_i8: i8 = @bitCast(q8_block[2 + i]);
            const csd = toCSD_fast(w_i8);
            for (0..csd.n_terms) |t| {
                if (t >= MAX_TERMS) break;
                const term = csd.terms[t];
                if (term.active) {
                    blk.shifts[t][i] = @intCast(term.shift);
                    blk.signs[t][i] = if (term.sign < 0) -1 else 1;
                }
            }
        }
        return blk;
    }

    pub fn fromRawWeights(raw_weights: [*]const u8) TernaryBlock {
        var blk: TernaryBlock = undefined;
        blk.scale = @bitCast(@as(u16, 0));
        for (0..MAX_TERMS) |t| {
            @memset(&blk.shifts[t], 0);
            @memset(&blk.signs[t], 0);
        }
        for (0..32) |i| {
            const w_i8: i8 = @bitCast(raw_weights[i]);
            const csd = toCSD_fast(w_i8);
            for (0..csd.n_terms) |t| {
                if (t >= MAX_TERMS) break;
                const term = csd.terms[t];
                if (term.active) {
                    blk.shifts[t][i] = @intCast(term.shift);
                    blk.signs[t][i] = if (term.sign < 0) -1 else 1;
                }
            }
        }
        return blk;
    }
};

// ============================================================
// TernaryBlockGrouped — bitmask per shift value
// ============================================================

pub const SHIFTS_USED = 8;

pub const TernaryBlockGrouped = struct {
    scale: f16,
    shift_masks: [MAX_TERMS][SHIFTS_USED]u32 align(32),
    signs: [MAX_TERMS][32]i8 align(32),

    pub fn fromQ8Block(q8_block: [*]const u8) TernaryBlockGrouped {
        var blk: TernaryBlockGrouped = undefined;
        const scale_bits: u16 = @as(u16, q8_block[0]) | (@as(u16, q8_block[1]) << 8);
        blk.scale = @bitCast(scale_bits);
        for (0..MAX_TERMS) |t| {
            @memset(&blk.shift_masks[t], 0);
            @memset(&blk.signs[t], 0);
        }
        for (0..32) |i| {
            const w_i8: i8 = @bitCast(q8_block[2 + i]);
            const csd = toCSD_fast(w_i8);
            for (0..csd.n_terms) |t| {
                if (t >= MAX_TERMS) break;
                const term = csd.terms[t];
                if (term.active) {
                    const shift_idx: usize = @intCast(term.shift);
                    blk.shift_masks[t][shift_idx] |= @as(u32, 1) << @intCast(i);
                    blk.signs[t][i] = if (term.sign < 0) -1 else 1;
                }
            }
        }
        return blk;
    }

    pub fn fromRawWeights(raw_weights: [*]const u8) TernaryBlockGrouped {
        var blk: TernaryBlockGrouped = undefined;
        blk.scale = @bitCast(@as(u16, 0));
        for (0..MAX_TERMS) |t| {
            @memset(&blk.shift_masks[t], 0);
            @memset(&blk.signs[t], 0);
        }
        for (0..32) |i| {
            const w_i8: i8 = @bitCast(raw_weights[i]);
            const csd = toCSD_fast(w_i8);
            for (0..csd.n_terms) |t| {
                if (t >= MAX_TERMS) break;
                const term = csd.terms[t];
                if (term.active) {
                    const shift_idx: usize = @intCast(term.shift);
                    blk.shift_masks[t][shift_idx] |= @as(u32, 1) << @intCast(i);
                    blk.signs[t][i] = if (term.sign < 0) -1 else 1;
                }
            }
        }
        return blk;
    }
};

// ============================================================
// CORE KERNEL: dotTernary32
// ============================================================

pub fn dotTernary32(blk: *const TernaryBlock, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);

    const x_i8: VI8x32 = @as(*align(1) const VI8x32, @ptrCast(x_ptr)).*;
    const x_i16: VI16x32 = @intCast(x_i8);
    const zero16: VI16x32 = @splat(0);

    var total_acc: VI16x32 = @splat(0);

    inline for (0..MAX_TERMS) |t| {
        const signs_i8: VI8x32 = @as(*align(32) const VI8x32, @ptrCast(&blk.signs[t])).*;
        const shifts_u8: VU8x32 = @as(*align(32) const VU8x32, @ptrCast(&blk.shifts[t])).*;

        const active: @Vector(32, bool) = signs_i8 != @as(VI8x32, @splat(0));
        const sign_neg: @Vector(32, bool) = signs_i8 < @as(VI8x32, @splat(0));
        const signed_x: VI16x32 = @select(i16, sign_neg, -x_i16, x_i16);
        const active_x: VI16x32 = @select(i16, active, signed_x, zero16);

        var term_acc: VI16x32 = @splat(0);

        inline for (0..8) |s| {
            const s_u8: u8 = @intCast(s);
            const shift_match: @Vector(32, bool) = shifts_u8 == @as(VU8x32, @splat(s_u8));
            const combined: @Vector(32, bool) = @as(@Vector(32, bool), shift_match) & @as(@Vector(32, bool), active);
            const masked: VI16x32 = @select(i16, combined, active_x, zero16);
            const shift_amt: @Vector(32, u4) = @splat(@intCast(s));
            term_acc += masked << shift_amt;
        }

        total_acc += term_acc;
    }

    const acc_i32: VI32x32 = @intCast(total_acc);
    return @reduce(.Add, acc_i32);
}

// ============================================================
// CORE KERNEL: dotTernaryGrouped32
// ============================================================

pub fn dotTernaryGrouped32(blk: *const TernaryBlockGrouped, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);

    const x_i8: VI8x32 = @as(*align(1) const VI8x32, @ptrCast(x_ptr)).*;
    const x_i16: VI16x32 = @intCast(x_i8);
    const zero16: VI16x32 = @splat(0);

    var total_acc: VI16x32 = @splat(0);

    inline for (0..MAX_TERMS) |t| {
        const signs_i8: VI8x32 = @as(*align(32) const VI8x32, @ptrCast(&blk.signs[t])).*;

        const sign_neg: @Vector(32, bool) = signs_i8 < @as(VI8x32, @splat(0));
        const sign_pos: @Vector(32, bool) = signs_i8 > @as(VI8x32, @splat(0));

        const signed_x: VI16x32 = @select(
            i16,
            sign_neg,
            -x_i16,
            @select(i16, sign_pos, x_i16, zero16),
        );

        var term_acc: VI16x32 = @splat(0);

        inline for (0..SHIFTS_USED) |s| {
            const mask = blk.shift_masks[t][s];
            if (mask != 0) {
                const mask_vec = bitmaskToVecBool(mask);
                const masked: VI16x32 = @select(i16, mask_vec, signed_x, zero16);
                const shift_amt: @Vector(32, u4) = @splat(@intCast(s));
                term_acc += masked << shift_amt;
            }
        }

        total_acc += term_acc;
    }

    const acc_i32: VI32x32 = @intCast(total_acc);
    return @reduce(.Add, acc_i32);
}

inline fn bitmaskToVecBool(mask: u32) @Vector(32, bool) {
    const bit_positions = comptime blk: {
        @setEvalBranchQuota(10000);
        var positions: @Vector(32, u32) = undefined;
        for (0..32) |i| {
            positions[i] = @as(u32, 1) << @intCast(i);
        }
        break :blk positions;
    };
    const mask_splat: @Vector(32, u32) = @splat(mask);
    return (mask_splat & bit_positions) != @as(@Vector(32, u32), @splat(0));
}

// ============================================================
// Raw-pointer dot products (for lut_mul.zig integration)
// ============================================================

pub fn dotTernaryRaw(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);
    const blk = TernaryBlock.fromRawWeights(w_ptr);
    return dotTernary32(&blk, x_ptr);
}

pub fn dotTernaryGroupedRaw(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);
    const blk = TernaryBlockGrouped.fromRawWeights(w_ptr);
    return dotTernaryGrouped32(&blk, x_ptr);
}

// ============================================================
// readF16 helper
// ============================================================

inline fn readF16(ptr: [*]const u8) f32 {
    const bits: u16 = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ============================================================
// TernaryWeightCache — precomputed at model load
// ============================================================

pub const TernaryWeightCache = struct {
    blocks: []TernaryBlockGrouped,
    n_blocks_per_row: usize,
    n_rows: usize,
    scale_cache: []f16,
    allocator: std.mem.Allocator,

    pub fn fromQ8Data(
        data: []const u8,
        rows: usize,
        cols: usize,
        allocator: std.mem.Allocator,
    ) !TernaryWeightCache {
        const QBYTES: usize = 34;
        const bpr = cols / 32;
        const total_blocks = rows * bpr;

        const blocks = try allocator.alloc(TernaryBlockGrouped, total_blocks);
        errdefer allocator.free(blocks);

        const scale_data = try allocator.alloc(f16, total_blocks);
        errdefer allocator.free(scale_data);

        for (0..rows) |r| {
            for (0..bpr) |b| {
                const idx = r * bpr + b;
                const offset = r * bpr * QBYTES + b * QBYTES;
                blocks[idx] = TernaryBlockGrouped.fromQ8Block(data.ptr + offset);
                const scale_bits: u16 = @as(u16, data[offset]) | (@as(u16, data[offset + 1]) << 8);
                scale_data[idx] = @bitCast(scale_bits);
            }
        }

        return .{
            .blocks = blocks,
            .n_blocks_per_row = bpr,
            .n_rows = rows,
            .scale_cache = scale_data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TernaryWeightCache) void {
        self.allocator.free(self.blocks);
        self.allocator.free(self.scale_cache);
    }
};

// ============================================================
// Cached MatVec — full rows
// ============================================================

pub fn matVecTernaryCached(
    out: []f32,
    cache: *const TernaryWeightCache,
    qbuf: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);
    const bpr = cols >> 5;

    var r: usize = 0;
    while (r + 8 <= rows) : (r += 8) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        var s4: f32 = 0;
        var s5: f32 = 0;
        var s6: f32 = 0;
        var s7: f32 = 0;

        for (0..bpr) |b| {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];

            const idx0 = (r + 0) * bpr + b;
            const idx1 = (r + 1) * bpr + b;
            const idx2 = (r + 2) * bpr + b;
            const idx3 = (r + 3) * bpr + b;
            const idx4 = (r + 4) * bpr + b;
            const idx5 = (r + 5) * bpr + b;
            const idx6 = (r + 6) * bpr + b;
            const idx7 = (r + 7) * bpr + b;

            const ws0: f32 = @floatCast(cache.scale_cache[idx0]);
            const ws1: f32 = @floatCast(cache.scale_cache[idx1]);
            const ws2: f32 = @floatCast(cache.scale_cache[idx2]);
            const ws3: f32 = @floatCast(cache.scale_cache[idx3]);
            const ws4: f32 = @floatCast(cache.scale_cache[idx4]);
            const ws5: f32 = @floatCast(cache.scale_cache[idx5]);
            const ws6: f32 = @floatCast(cache.scale_cache[idx6]);
            const ws7: f32 = @floatCast(cache.scale_cache[idx7]);

            s0 += (ws0 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx0], xp)));
            s1 += (ws1 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx1], xp)));
            s2 += (ws2 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx2], xp)));
            s3 += (ws3 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx3], xp)));
            s4 += (ws4 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx4], xp)));
            s5 += (ws5 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx5], xp)));
            s6 += (ws6 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx6], xp)));
            s7 += (ws7 * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx7], xp)));
        }

        out[r + 0] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
        out[r + 4] = s4;
        out[r + 5] = s5;
        out[r + 6] = s6;
        out[r + 7] = s7;
    }
    while (r < rows) : (r += 1) {
        var sum: f32 = 0;
        for (0..bpr) |b| {
            const idx = r * bpr + b;
            const ws: f32 = @floatCast(cache.scale_cache[idx]);
            const d = dotTernaryGrouped32(&cache.blocks[idx], qbuf.ptr + (b << 5));
            sum += (ws * scales[b]) * @as(f32, @floatFromInt(d));
        }
        out[r] = sum;
    }
}

// ============================================================
// Cached range kernels for MT
// ============================================================

pub fn matVecTernaryCached_range(
    out_range: []f32,
    cache: *const TernaryWeightCache,
    qbuf: []const i8,
    scales: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);
    const bpr = cols >> 5;
    const nr = row_end - row_start;

    var r: usize = 0;
    while (r + 8 <= nr) : (r += 8) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        var s4: f32 = 0;
        var s5: f32 = 0;
        var s6: f32 = 0;
        var s7: f32 = 0;

        const br = row_start + r;
        for (0..bpr) |b| {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];

            const idx0 = (br + 0) * bpr + b;
            const idx1 = (br + 1) * bpr + b;
            const idx2 = (br + 2) * bpr + b;
            const idx3 = (br + 3) * bpr + b;
            const idx4 = (br + 4) * bpr + b;
            const idx5 = (br + 5) * bpr + b;
            const idx6 = (br + 6) * bpr + b;
            const idx7 = (br + 7) * bpr + b;

            s0 += (@as(f32, @floatCast(cache.scale_cache[idx0])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx0], xp)));
            s1 += (@as(f32, @floatCast(cache.scale_cache[idx1])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx1], xp)));
            s2 += (@as(f32, @floatCast(cache.scale_cache[idx2])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx2], xp)));
            s3 += (@as(f32, @floatCast(cache.scale_cache[idx3])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx3], xp)));
            s4 += (@as(f32, @floatCast(cache.scale_cache[idx4])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx4], xp)));
            s5 += (@as(f32, @floatCast(cache.scale_cache[idx5])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx5], xp)));
            s6 += (@as(f32, @floatCast(cache.scale_cache[idx6])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx6], xp)));
            s7 += (@as(f32, @floatCast(cache.scale_cache[idx7])) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx7], xp)));
        }

        out_range[r + 0] = s0;
        out_range[r + 1] = s1;
        out_range[r + 2] = s2;
        out_range[r + 3] = s3;
        out_range[r + 4] = s4;
        out_range[r + 5] = s5;
        out_range[r + 6] = s6;
        out_range[r + 7] = s7;
    }
    while (r < nr) : (r += 1) {
        var sum: f32 = 0;
        const br = row_start + r;
        for (0..bpr) |b| {
            const idx = br * bpr + b;
            const ws: f32 = @floatCast(cache.scale_cache[idx]);
            sum += (ws * scales[b]) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache.blocks[idx], qbuf.ptr + (b << 5))));
        }
        out_range[r] = sum;
    }
}

pub fn matVecTernaryCached_range_pair(
    out_a: []f32,
    cache_a: *const TernaryWeightCache,
    out_b: []f32,
    cache_b: *const TernaryWeightCache,
    qbuf: []const i8,
    scales: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);
    const bpr = cols >> 5;
    const nr = row_end - row_start;

    for (0..nr) |r| {
        var sa: f32 = 0;
        var sb: f32 = 0;
        const br = row_start + r;

        for (0..bpr) |b| {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];
            const idx = br * bpr + b;

            const wsa: f32 = @floatCast(cache_a.scale_cache[idx]);
            const wsb: f32 = @floatCast(cache_b.scale_cache[idx]);

            sa += (wsa * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache_a.blocks[idx], xp)));
            sb += (wsb * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&cache_b.blocks[idx], xp)));
        }

        out_a[r] = sa;
        out_b[r] = sb;
    }
}

// ============================================================
// On-the-fly MatVec (no cache, zero extra memory)
// ============================================================

pub fn matVecQ8_0_ternary_otf_qin(
    out: []f32,
    data: []const u8,
    qbuf: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);
    const QBYTES: usize = 34;
    const bpr = cols >> 5;
    const rs = bpr * QBYTES;

    var r: usize = 0;
    while (r + 4 <= rows) : (r += 4) {
        var s0: f32 = 0;
        var s1: f32 = 0;
        var s2: f32 = 0;
        var s3: f32 = 0;
        var p0 = data.ptr + (r + 0) * rs;
        var p1 = data.ptr + (r + 1) * rs;
        var p2 = data.ptr + (r + 2) * rs;
        var p3 = data.ptr + (r + 3) * rs;

        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];

            const blk0 = TernaryBlockGrouped.fromQ8Block(p0);
            const blk1 = TernaryBlockGrouped.fromQ8Block(p1);
            const blk2 = TernaryBlockGrouped.fromQ8Block(p2);
            const blk3 = TernaryBlockGrouped.fromQ8Block(p3);

            s0 += (readF16(p0) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk0, xp)));
            s1 += (readF16(p1) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk1, xp)));
            s2 += (readF16(p2) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk2, xp)));
            s3 += (readF16(p3) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk3, xp)));

            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
        }
        out[r + 0] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    while (r < rows) : (r += 1) {
        var sum: f32 = 0;
        var w = data.ptr + r * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            const blk = TernaryBlockGrouped.fromQ8Block(w);
            sum += (readF16(w) * scales[b]) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk, qbuf.ptr + (b << 5))));
            w += QBYTES;
        }
        out[r] = sum;
    }
}

pub fn matVecQ8_0_ternary_otf_range_qin(
    out_range: []f32,
    data: []const u8,
    qbuf: []const i8,
    scales: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);
    const QBYTES: usize = 34;
    const bpr = cols >> 5;
    const rs = bpr * QBYTES;
    const bp = data.ptr + row_start * rs;
    const nr = row_end - row_start;

    for (0..nr) |r| {
        var sum: f32 = 0;
        var p = bp + r * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            const blk = TernaryBlockGrouped.fromQ8Block(p);
            sum += (readF16(p) * scales[b]) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk, qbuf.ptr + (b << 5))));
            p += QBYTES;
        }
        out_range[r] = sum;
    }
}

pub fn matVecQ8_0_ternary_otf_range_pair_qin(
    out_a: []f32,
    data_a: []const u8,
    out_b: []f32,
    data_b: []const u8,
    qbuf: []const i8,
    scales: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);
    const QBYTES: usize = 34;
    const bpr = cols >> 5;
    const rs = bpr * QBYTES;
    const ba = data_a.ptr + row_start * rs;
    const bb = data_b.ptr + row_start * rs;
    const nr = row_end - row_start;

    for (0..nr) |r| {
        var sa: f32 = 0;
        var sb: f32 = 0;
        var pa = ba + r * rs;
        var pb = bb + r * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];

            const blk_a = TernaryBlockGrouped.fromQ8Block(pa);
            const blk_b = TernaryBlockGrouped.fromQ8Block(pb);

            sa += (readF16(pa) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk_a, xp)));
            sb += (readF16(pb) * sx) * @as(f32, @floatFromInt(dotTernaryGrouped32(&blk_b, xp)));

            pa += QBYTES;
            pb += QBYTES;
        }
        out_a[r] = sa;
        out_b[r] = sb;
    }
}

// ============================================================
// Statistics
// ============================================================

pub const CSDStats = struct {
    total_values: usize,
    exact_count: usize,
    total_terms: usize,
    max_abs_error: u16,
    total_abs_error: u64,
    term_histogram: [MAX_TERMS + 2]usize,

    pub fn compute() CSDStats {
        var stats = CSDStats{
            .total_values = 256,
            .exact_count = 0,
            .total_terms = 0,
            .max_abs_error = 0,
            .total_abs_error = 0,
            .term_histogram = [_]usize{0} ** (MAX_TERMS + 2),
        };

        for (0..256) |ui| {
            const val: i8 = @bitCast(@as(u8, @intCast(ui)));
            const csd = toCSD_fast(val);
            const reconstructed = csdToValue(csd);
            const err = @as(i16, val) - reconstructed;
            const abs_err: u16 = @intCast(@abs(err));

            if (err == 0) stats.exact_count += 1;
            stats.total_terms += csd.n_terms;
            if (abs_err > stats.max_abs_error) stats.max_abs_error = abs_err;
            stats.total_abs_error += abs_err;

            const bin: usize = if (csd.n_terms > MAX_TERMS) MAX_TERMS + 1 else csd.n_terms;
            stats.term_histogram[bin] += 1;
        }

        return stats;
    }

    pub fn avgTerms(self: *const CSDStats) f64 {
        return @as(f64, @floatFromInt(self.total_terms)) / @as(f64, @floatFromInt(self.total_values));
    }

    pub fn avgError(self: *const CSDStats) f64 {
        return @as(f64, @floatFromInt(self.total_abs_error)) / @as(f64, @floatFromInt(self.total_values));
    }

    pub fn exactPercent(self: *const CSDStats) f64 {
        return @as(f64, @floatFromInt(self.exact_count)) / @as(f64, @floatFromInt(self.total_values)) * 100.0;
    }
};

// ============================================================
// Diagnostics
// ============================================================

pub fn printTernaryInfo() void {
    std.debug.print("\n=== TERNARY CSD ENGINE ===\n", .{});
    std.debug.print("Max terms per weight: {}\n", .{MAX_TERMS});
    std.debug.print("CSD table: compile-time (256 entries)\n", .{});

    const stats = CSDStats.compute();

    std.debug.print("Exact reconstructions: {}/{} ({d:.1}%)\n", .{
        stats.exact_count, stats.total_values, stats.exactPercent(),
    });
    std.debug.print("Avg terms/weight: {d:.2}\n", .{stats.avgTerms()});
    std.debug.print("Max abs error: {}\n", .{stats.max_abs_error});
    std.debug.print("Avg abs error: {d:.4}\n", .{stats.avgError()});

    std.debug.print("Term distribution:", .{});
    for (0..MAX_TERMS + 1) |i| {
        std.debug.print(" {}t={}", .{ i, stats.term_histogram[i] });
    }
    std.debug.print("\n", .{});

    const checks = [_]struct { val: i8, label: []const u8 }{
        .{ .val = 7, .label = "7" },
        .{ .val = -3, .label = "-3" },
        .{ .val = 127, .label = "127" },
        .{ .val = -128, .label = "-128" },
        .{ .val = 73, .label = "73" },
        .{ .val = 85, .label = "85" },
        .{ .val = 0, .label = "0" },
    };

    for (checks) |c| {
        const csd = toCSD_fast(c.val);
        const got = csdToValue(csd);
        const status: []const u8 = if (got == @as(i16, c.val)) "EXACT" else "APPROX";
        std.debug.print("  CSD({s:>4}): reconstructed={d:>4} [{s}] ({} terms)\n", .{
            c.label, got, status, csd.n_terms,
        });
    }

    var wt: [32]u8 align(32) = undefined;
    var xt: [32]i8 align(32) = undefined;

    @memset(&wt, 0);
    @memset(&xt, 0);
    wt[0] = @bitCast(@as(i8, 7));
    xt[0] = 8;
    const blk1 = TernaryBlock.fromRawWeights(&wt);
    const r1 = dotTernary32(&blk1, &xt);
    const gblk1 = TernaryBlockGrouped.fromRawWeights(&wt);
    const gr1 = dotTernaryGrouped32(&gblk1, &xt);
    std.debug.print("  dot(7,8): basic={} grouped={} {s}\n", .{
        r1, gr1, if (r1 == 56 and gr1 == 56) "OK" else "FAIL",
    });

    @memset(&wt, 0);
    @memset(&xt, 0);
    wt[0] = @bitCast(@as(i8, -5));
    xt[0] = -6;
    const blk2 = TernaryBlock.fromRawWeights(&wt);
    const r2 = dotTernary32(&blk2, &xt);
    const gblk2 = TernaryBlockGrouped.fromRawWeights(&wt);
    const gr2 = dotTernaryGrouped32(&gblk2, &xt);
    std.debug.print("  dot(-5,-6): basic={} grouped={} {s}\n", .{
        r2, gr2, if (r2 == 30 and gr2 == 30) "OK" else "FAIL",
    });

    std.debug.print("==============================\n\n", .{});
}
