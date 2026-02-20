// src/lut_mul.zig — Native SIMD MatVec v3 + Ternary Decomposition
// Comptime dispatch: zero overhead from global var checks
const std = @import("std");
const ternary = @import("ternary.zig");

// ============================================================
// Global mode switches
// ============================================================

pub var g_lut_enabled: bool = true;
pub var g_ternary_mode: enum { off, on_the_fly, cached } = .off;

// Comptime: native i16 multiply (vpmullw on AVX2)
// This replaces the old g_use_native_dot runtime global
pub const USE_NATIVE_DOT: bool = false; // TRUE matmul-free

// ============================================================
// SIMD types
// ============================================================

const VU8x32 = @Vector(32, u8);
const VI8x32 = @Vector(32, i8);
const VU16x32 = @Vector(32, u16);
const VI16x32 = @Vector(32, i16);
const VI32x32 = @Vector(32, i32);

// ============================================================
// FAST PATH: Native i16 multiply — uses vpmullw on AVX2
// Each i8 product max = 127*127 = 16129, fits in i16
// But sum of 32 could overflow, so widen to i32 for reduce
// ============================================================

pub inline fn dotI8x32_native(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);

    const w_i8: VI8x32 = @bitCast(@as(*align(1) const VU8x32, @ptrCast(w_ptr)).*);
    const x_i8: VI8x32 = @as(*align(1) const VI8x32, @ptrCast(x_ptr)).*;

    const w_i16: VI16x32 = @intCast(w_i8);
    const x_i16: VI16x32 = @intCast(x_i8);
    const prod_i16: VI16x32 = w_i16 * x_i16;
    const prod_i32: VI32x32 = @intCast(prod_i16);

    return @reduce(.Add, prod_i32);
}

// Same but takes pre-loaded x vector — zero copy
pub inline fn dotI8x32_native_prex(w_ptr: [*]const u8, x8: VI8x32) i32 {
    @setRuntimeSafety(false);

    const w_i8: VI8x32 = @bitCast(@as(*align(1) const VU8x32, @ptrCast(w_ptr)).*);
    const w_i16: VI16x32 = @intCast(w_i8);
    const x_i16: VI16x32 = @intCast(x8);
    const prod_i16: VI16x32 = w_i16 * x_i16;
    const prod_i32: VI32x32 = @intCast(prod_i16);

    return @reduce(.Add, prod_i32);
}

// ============================================================
// Original: SIMD Shift-Add Multiply — ZERO MUL instruction
// Kept as fallback for matmul-free research
// ============================================================

pub fn dotI8x32_shiftadd(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);

    const w_u8: VU8x32 = @as(*align(1) const VU8x32, @ptrCast(w_ptr)).*;
    const x_i8: VI8x32 = @as(*align(1) const VI8x32, @ptrCast(x_ptr)).*;

    const w_sign: VI8x32 = @bitCast(w_u8);
    const w_neg: @Vector(32, bool) = w_sign < @as(VI8x32, @splat(0));
    const x_neg: @Vector(32, bool) = x_i8 < @as(VI8x32, @splat(0));
    const result_neg: @Vector(32, bool) = w_neg != x_neg;

    const w_abs_i8: VI8x32 = @select(i8, w_neg, -%w_sign, w_sign);
    const x_abs_i8: VI8x32 = @select(i8, x_neg, -%x_i8, x_i8);

    const w_abs: VU16x32 = @intCast(@as(VU8x32, @bitCast(w_abs_i8)));
    const x_abs: @Vector(32, u8) = @bitCast(x_abs_i8);

    const zero16: VU16x32 = @splat(0);

    const b0_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(1))) != @as(@Vector(32, u8), @splat(0));
    var prod: VU16x32 = @select(u16, b0_mask, w_abs, zero16);

    const b1_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(2))) != @as(@Vector(32, u8), @splat(0));
    prod += @select(u16, b1_mask, w_abs << @splat(1), zero16);

    const b2_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(4))) != @as(@Vector(32, u8), @splat(0));
    prod += @select(u16, b2_mask, w_abs << @splat(2), zero16);

    const b3_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(8))) != @as(@Vector(32, u8), @splat(0));
    prod += @select(u16, b3_mask, w_abs << @splat(3), zero16);

    const b4_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(16))) != @as(@Vector(32, u8), @splat(0));
    prod += @select(u16, b4_mask, w_abs << @splat(4), zero16);

    const b5_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(32))) != @as(@Vector(32, u8), @splat(0));
    prod += @select(u16, b5_mask, w_abs << @splat(5), zero16);

    const b6_mask: @Vector(32, bool) = (x_abs & @as(@Vector(32, u8), @splat(64))) != @as(@Vector(32, u8), @splat(0));
    prod += @select(u16, b6_mask, w_abs << @splat(6), zero16);

    const prod_i16: VI16x32 = @bitCast(prod);
    const signed_prod: VI16x32 = @select(i16, result_neg, -%prod_i16, prod_i16);

    const prod_i32: VI32x32 = @intCast(signed_prod);
    return @reduce(.Add, prod_i32);
}

// ============================================================
// Unified dot dispatcher — comptime resolved, zero overhead
// ============================================================

pub inline fn dotI8x32_best(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    if (comptime USE_NATIVE_DOT) {
        return dotI8x32_native(w_ptr, x_ptr);
    } else {
        return dotI8x32_shiftadd(w_ptr, x_ptr);
    }
}

pub inline fn dotI8x32_best_prex(w_ptr: [*]const u8, x8: VI8x32) i32 {
    if (comptime USE_NATIVE_DOT) {
        return dotI8x32_native_prex(w_ptr, x8);
    } else {
        // Fallback: copy vector to buffer, use shift-add
        var x_buf: [32]i8 align(32) = undefined;
        @as(*align(32) VI8x32, @ptrCast(&x_buf)).* = x8;
        return dotI8x32_shiftadd(w_ptr, &x_buf);
    }
}

pub inline fn dotI8x32_lut(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    return dotI8x32_best(w_ptr, x_ptr);
}

// ============================================================
// readF16
// ============================================================

inline fn readF16(ptr: [*]const u8) f32 {
    const bits: u16 = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ============================================================
// Q8 Block with float input
// ============================================================

pub fn dotQ8Block_f32input_lut(w_block: [*]const u8, input: [*]const f32) f32 {
    @setRuntimeSafety(false);
    const w_scale: f32 = readF16(w_block);

    var max_abs: f32 = 0;
    for (0..32) |i| {
        const a = @abs(input[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0) return 0;

    const inv = 127.0 / max_abs;
    const x_scale = max_abs / 127.0;

    var qbuf: [32]i8 align(32) = undefined;
    for (0..32) |i| {
        var qi: i32 = @intFromFloat(@round(input[i] * inv));
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        qbuf[i] = @intCast(qi);
    }

    const isum = dotI8x32_best(w_block + 2, &qbuf);
    return (w_scale * x_scale) * @as(f32, @floatFromInt(isum));
}

// ============================================================
// Static buffers + quantize
// ============================================================

var g_lut_qbuf: [16384]i8 = undefined;
var g_lut_scales: [512]f32 = undefined;

fn quantizeLocal(q_out: []i8, s_out: []f32, input: []const f32, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const blocks = cols >> 5;
    var b: usize = 0;
    while (b < blocks) : (b += 1) {
        const base = b << 5;
        var max_abs: f32 = 0;
        for (0..QBLOCK) |ii| {
            const a = @abs(input[base + ii]);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs == 0) {
            s_out[b] = 0;
            @memset(q_out[base .. base + QBLOCK], 0);
            continue;
        }
        const inv_val = 127.0 / max_abs;
        s_out[b] = max_abs / 127.0;
        for (0..QBLOCK) |ii| {
            var qi: i32 = @intFromFloat(@round(input[base + ii] * inv_val));
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            q_out[base + ii] = @intCast(qi);
        }
    }
}

// ============================================================
// Ternary Cache Management
// ============================================================

pub const TernaryCacheEntry = struct {
    cache: ternary.TernaryWeightCache,
    data_ptr: [*]const u8,
    rows: usize,
    cols: usize,
};

const MAX_CACHED_TENSORS = 512;
var g_ternary_caches: [MAX_CACHED_TENSORS]?TernaryCacheEntry = [_]?TernaryCacheEntry{null} ** MAX_CACHED_TENSORS;
var g_ternary_cache_count: usize = 0;
var g_ternary_cache_allocator: ?std.mem.Allocator = null;

pub fn initTernaryCache(allocator: std.mem.Allocator) void {
    g_ternary_cache_allocator = allocator;
    g_ternary_cache_count = 0;
    for (&g_ternary_caches) |*c| c.* = null;
}

pub fn deinitTernaryCache() void {
    for (&g_ternary_caches) |*entry| {
        if (entry.*) |*e| {
            e.cache.deinit();
            entry.* = null;
        }
    }
    g_ternary_cache_count = 0;
}

pub fn getOrCreateTernaryCache(
    data: []const u8,
    rows: usize,
    cols: usize,
) ?*const ternary.TernaryWeightCache {
    for (&g_ternary_caches) |*entry| {
        if (entry.*) |*e| {
            if (e.data_ptr == data.ptr and e.rows == rows and e.cols == cols) {
                return &e.cache;
            }
        }
    }

    const allocator = g_ternary_cache_allocator orelse return null;
    if (g_ternary_cache_count >= MAX_CACHED_TENSORS) return null;

    const cache = ternary.TernaryWeightCache.fromQ8Data(data, rows, cols, allocator) catch return null;

    g_ternary_caches[g_ternary_cache_count] = .{
        .cache = cache,
        .data_ptr = data.ptr,
        .rows = rows,
        .cols = cols,
    };
    g_ternary_cache_count += 1;

    return &g_ternary_caches[g_ternary_cache_count - 1].?.cache;
}

// ============================================================
// MatVec Q8_0 — qin path, 8-row batched
// ============================================================

pub fn matVecQ8_0_lut_qin(out: []f32, data: []const u8, qbuf: []const i8, scales: []const f32, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);

    // Try cached ternary path first
    if (g_ternary_mode == .cached) {
        if (getOrCreateTernaryCache(data, rows, cols)) |cache| {
            ternary.matVecTernaryCached(out, cache, qbuf, scales, rows, cols);
            return;
        }
    }

    const QBYTES: usize = 34;
    const bpr = cols >> 5;
    const rs = bpr * QBYTES;

    // On-the-fly ternary uses different kernel
    if (g_ternary_mode == .on_the_fly) {
        ternary.matVecQ8_0_ternary_otf_qin(out, data, qbuf, scales, rows, cols);
        return;
    }

    // Native/shift-add path — comptime resolved
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
        var p0 = data.ptr + (r + 0) * rs;
        var p1 = data.ptr + (r + 1) * rs;
        var p2 = data.ptr + (r + 2) * rs;
        var p3 = data.ptr + (r + 3) * rs;
        var p4 = data.ptr + (r + 4) * rs;
        var p5 = data.ptr + (r + 5) * rs;
        var p6 = data.ptr + (r + 6) * rs;
        var p7 = data.ptr + (r + 7) * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];

            s0 += (readF16(p0) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p0 + 2, xp)));
            s1 += (readF16(p1) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p1 + 2, xp)));
            s2 += (readF16(p2) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p2 + 2, xp)));
            s3 += (readF16(p3) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p3 + 2, xp)));
            s4 += (readF16(p4) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p4 + 2, xp)));
            s5 += (readF16(p5) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p5 + 2, xp)));
            s6 += (readF16(p6) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p6 + 2, xp)));
            s7 += (readF16(p7) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p7 + 2, xp)));

            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
            p4 += QBYTES;
            p5 += QBYTES;
            p6 += QBYTES;
            p7 += QBYTES;
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
        var w = data.ptr + r * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            sum += (readF16(w) * scales[b]) * @as(f32, @floatFromInt(dotI8x32_best(w + 2, qbuf.ptr + (b << 5))));
            w += QBYTES;
        }
        out[r] = sum;
    }
}

// ============================================================
// MatVec Q8_0 — float input
// ============================================================

pub fn matVecQ8_0_lut_f32(out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const blocks = cols >> 5;

    if (cols <= 16384) {
        quantizeLocal(g_lut_qbuf[0..cols], g_lut_scales[0..blocks], input, cols);
        matVecQ8_0_lut_qin(out, data, g_lut_qbuf[0..cols], g_lut_scales[0..blocks], rows, cols);
    } else {
        const QBYTES: usize = 34;
        const rs = blocks * QBYTES;
        for (0..rows) |r| {
            var sum: f32 = 0;
            const w = data.ptr + r * rs;
            for (0..blocks) |b| {
                sum += dotQ8Block_f32input_lut(w + b * QBYTES, input.ptr + (b << 5));
            }
            out[r] = sum;
        }
    }
}

// ============================================================
// Range kernels for MT pool
// ============================================================

pub fn matVecQ8_0_range_lut_qin(out_range: []f32, data: []const u8, qbuf: []const i8, scales: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);

    if (g_ternary_mode == .on_the_fly) {
        ternary.matVecQ8_0_ternary_otf_range_qin(out_range, data, qbuf, scales, row_start, row_end, cols);
        return;
    }

    const QBYTES: usize = 34;
    const bpr = cols >> 5;
    const rs = bpr * QBYTES;
    const bp = data.ptr + row_start * rs;
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
        var p0 = bp + (r + 0) * rs;
        var p1 = bp + (r + 1) * rs;
        var p2 = bp + (r + 2) * rs;
        var p3 = bp + (r + 3) * rs;
        var p4 = bp + (r + 4) * rs;
        var p5 = bp + (r + 5) * rs;
        var p6 = bp + (r + 6) * rs;
        var p7 = bp + (r + 7) * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            const xp = qbuf.ptr + (b << 5);
            const sx = scales[b];

            s0 += (readF16(p0) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p0 + 2, xp)));
            s1 += (readF16(p1) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p1 + 2, xp)));
            s2 += (readF16(p2) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p2 + 2, xp)));
            s3 += (readF16(p3) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p3 + 2, xp)));
            s4 += (readF16(p4) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p4 + 2, xp)));
            s5 += (readF16(p5) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p5 + 2, xp)));
            s6 += (readF16(p6) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p6 + 2, xp)));
            s7 += (readF16(p7) * sx) * @as(f32, @floatFromInt(dotI8x32_best(p7 + 2, xp)));

            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
            p4 += QBYTES;
            p5 += QBYTES;
            p6 += QBYTES;
            p7 += QBYTES;
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
        var p = bp + r * rs;
        var b: usize = 0;
        while (b < bpr) : (b += 1) {
            sum += (readF16(p) * scales[b]) * @as(f32, @floatFromInt(dotI8x32_best(p + 2, qbuf.ptr + (b << 5))));
            p += QBYTES;
        }
        out_range[r] = sum;
    }
}

pub fn matVecQ8_0_range_lut_f32(out_range: []f32, data: []const u8, input: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBYTES: usize = 34;
    const bpr = cols >> 5;
    const rs = bpr * QBYTES;
    const bp = data.ptr + row_start * rs;
    const nr = row_end - row_start;
    for (0..nr) |r| {
        var sum: f32 = 0;
        const w = bp + r * rs;
        for (0..bpr) |b| {
            sum += dotQ8Block_f32input_lut(w + b * QBYTES, input.ptr + (b << 5));
        }
        out_range[r] = sum;
    }
}

pub fn matVecQ8_0_range_pair_lut_qin(out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, qbuf: []const i8, scales: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);

    if (g_ternary_mode == .on_the_fly) {
        ternary.matVecQ8_0_ternary_otf_range_pair_qin(out_a, data_a, out_b, data_b, qbuf, scales, row_start, row_end, cols);
        return;
    }

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

            sa += (readF16(pa) * sx) * @as(f32, @floatFromInt(dotI8x32_best(pa + 2, xp)));
            sb += (readF16(pb) * sx) * @as(f32, @floatFromInt(dotI8x32_best(pb + 2, xp)));

            pa += QBYTES;
            pb += QBYTES;
        }
        out_a[r] = sa;
        out_b[r] = sb;
    }
}

pub fn matVecQ8_0_range_pair_lut_f32(out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, input: []const f32, row_start: usize, row_end: usize, cols: usize) void {
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
        const wa = ba + r * rs;
        const wb = bb + r * rs;
        for (0..bpr) |b| {
            const ip = input.ptr + (b << 5);
            sa += dotQ8Block_f32input_lut(wa + b * QBYTES, ip);
            sb += dotQ8Block_f32input_lut(wb + b * QBYTES, ip);
        }
        out_a[r] = sa;
        out_b[r] = sb;
    }
}

// ============================================================
// Diagnostics
// ============================================================

pub fn printLutInfo() void {
    std.debug.print("\n=== MATMUL ENGINE ===\n", .{});
    std.debug.print("Status: {s}\n", .{if (g_lut_enabled) "ENABLED" else "DISABLED"});
    std.debug.print("Native dot: {s}\n", .{if (USE_NATIVE_DOT) "ENABLED (comptime, vpmullw)" else "DISABLED (shift-add)"});

    const mode_str: []const u8 = switch (g_ternary_mode) {
        .off => if (USE_NATIVE_DOT) "Native i16 multiply (hardware SIMD)" else "Shift-Add (7-bit decomposition)",
        .on_the_fly => "Ternary CSD (on-the-fly, ~2.8 terms avg)",
        .cached => "Ternary CSD (cached, ~2.8 terms avg)",
    };
    std.debug.print("Method: {s}\n", .{mode_str});
    std.debug.print("SIMD width: 32 elements (256-bit)\n", .{});

    // Sanity tests
    var wt: [32]u8 align(32) = undefined;
    var xt: [32]i8 align(32) = undefined;

    @memset(&wt, 0);
    @memset(&xt, 0);
    wt[0] = @bitCast(@as(i8, 7));
    xt[0] = 8;
    const r1 = dotI8x32_best(&wt, &xt);

    @memset(&wt, 0);
    @memset(&xt, 0);
    wt[0] = @bitCast(@as(i8, -3));
    xt[0] = 4;
    const r2 = dotI8x32_best(&wt, &xt);

    @memset(&wt, 0);
    @memset(&xt, 0);
    wt[0] = @bitCast(@as(i8, -5));
    xt[0] = -6;
    const r3 = dotI8x32_best(&wt, &xt);

    std.debug.print("Sanity: 7x8={} {s}\n", .{ r1, if (r1 == 56) "OK" else "FAIL" });
    std.debug.print("Sanity: -3x4={} {s}\n", .{ r2, if (r2 == -12) "OK" else "FAIL" });
    std.debug.print("Sanity: -5x-6={} {s}\n", .{ r3, if (r3 == 30) "OK" else "FAIL" });

    if (g_ternary_mode != .off) {
        ternary.printTernaryInfo();
    }

    std.debug.print("===========================\n\n", .{});
}
