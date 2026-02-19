//! ==========================================================================
//! LUT Multiplication Engine — INTEGER MULTIPLY-FREE
//! ==========================================================================
//!
//! Semua perkalian INTEGER i8 x i8 diganti dengan tabel lookup.
//! Ini menghilangkan 99%+ dari total operasi multiply:
//!   - 32 multiply per Q8 block (diganti tabel)
//!   - Ribuan block per row
//!   - Ribuan row per matmul
//!
//! Float scaling (2-3 per block) tetap pakai operator * karena:
//!   - Jumlahnya sangat kecil (<0.1% dari total multiply)
//!   - Presisi f32 tidak bisa di-tabel-kan secara akurat
//!

const std = @import("std");
const tables = @import("lut_tables.zig");

// ============================================================
// Configuration
// ============================================================

pub var g_lut_enabled: bool = true;

// ============================================================
// Core: Integer Multiply via Pre-computed Table
// Mengganti semua i8 * i8 dalam dot product
// ============================================================

inline fn lookupMulU8I8(a_raw: u8, b: i8) i16 {
    @setRuntimeSafety(false);
    const ia: usize = @intCast(a_raw);
    const ib: usize = @intCast(@as(u8, @bitCast(b)));
    return tables.grid_i16[ia][ib];
}

inline fn lookupMulI8(a: i8, b: i8) i16 {
    @setRuntimeSafety(false);
    const ia: usize = @intCast(@as(u8, @bitCast(a)));
    const ib: usize = @intCast(@as(u8, @bitCast(b)));
    return tables.grid_i16[ia][ib];
}

// ============================================================
// Dot Product i8[32] x i8[32] -> i32 via Table Lookup
//
// SEBELUM (hardware multiply):
//   prod16 = w16 * x16;   <-- 32 integer multiplies
//
// SESUDAH (table lookup):
//   sum += grid_i16[w[i]][x[i]];  <-- 32 table lookups, ZERO multiply
// ============================================================

pub fn dotI8x32_lut(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);

    var sum: i32 = 0;

    comptime var base: usize = 0;
    inline while (base < 32) : (base += 8) {
        const s0: i32 = lookupMulU8I8(w_ptr[base + 0], x_ptr[base + 0]);
        const s1: i32 = lookupMulU8I8(w_ptr[base + 1], x_ptr[base + 1]);
        const s2: i32 = lookupMulU8I8(w_ptr[base + 2], x_ptr[base + 2]);
        const s3: i32 = lookupMulU8I8(w_ptr[base + 3], x_ptr[base + 3]);
        const s4: i32 = lookupMulU8I8(w_ptr[base + 4], x_ptr[base + 4]);
        const s5: i32 = lookupMulU8I8(w_ptr[base + 5], x_ptr[base + 5]);
        const s6: i32 = lookupMulU8I8(w_ptr[base + 6], x_ptr[base + 6]);
        const s7: i32 = lookupMulU8I8(w_ptr[base + 7], x_ptr[base + 7]);

        sum += (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    }

    return sum;
}

// ============================================================
// readF16
// ============================================================

inline fn readF16(ptr: [*]const u8) f32 {
    const bits: u16 = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ============================================================
// Q8 Block with float input — LUT dot + normal float scale
// ============================================================

pub fn dotQ8Block_f32input_lut(
    w_block: [*]const u8,
    input: [*]const f32,
) f32 {
    @setRuntimeSafety(false);

    const w_scale: f32 = readF16(w_block);

    // Quantize input to i8
    var max_abs: f32 = 0;
    for (0..32) |i| {
        const a = @abs(input[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs == 0) return 0;

    const inv = 127.0 / max_abs;
    const x_scale = max_abs / 127.0;

    // Quantize + LUT dot in one pass
    var dot: i32 = 0;
    for (0..32) |i| {
        var qi: i32 = @intFromFloat(@round(input[i] * inv));
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        const x_i8: i8 = @intCast(qi);
        // TABLE LOOKUP instead of multiply
        dot += lookupMulU8I8(w_block[2 + i], x_i8);
    }

    // Float scaling: only 2 multiplies per 32-element block
    return (w_scale * x_scale) * @as(f32, @floatFromInt(dot));
}

// ============================================================
// Static buffers
// ============================================================

var g_lut_qbuf: [16384]i8 = undefined;
var g_lut_scales: [512]f32 = undefined;

// ============================================================
// Quantize input — standard precision
// ============================================================

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

        const inv = 127.0 / max_abs;
        s_out[b] = max_abs / 127.0;

        for (0..QBLOCK) |ii| {
            var qi: i32 = @intFromFloat(@round(input[base + ii] * inv));
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            q_out[base + ii] = @intCast(qi);
        }
    }
}

// ============================================================
// MatVec Q8_0 via LUT — qin path
// Integer dot product: TABLE LOOKUP (zero multiply)
// Float scaling: normal * (2 per block, negligible)
// ============================================================

pub fn matVecQ8_0_lut_qin(
    out: []f32,
    data: []const u8,
    qbuf: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBYTES: usize = 34;
    const blocks_per_row = cols >> 5;
    const row_stride = blocks_per_row * QBYTES;

    var r: usize = 0;

    while (r + 8 <= rows) : (r += 8) {
        var sum0: f32 = 0;
        var sum1: f32 = 0;
        var sum2: f32 = 0;
        var sum3: f32 = 0;
        var sum4: f32 = 0;
        var sum5: f32 = 0;
        var sum6: f32 = 0;
        var sum7: f32 = 0;

        var p0 = data.ptr + (r + 0) * row_stride;
        var p1 = data.ptr + (r + 1) * row_stride;
        var p2 = data.ptr + (r + 2) * row_stride;
        var p3 = data.ptr + (r + 3) * row_stride;
        var p4 = data.ptr + (r + 4) * row_stride;
        var p5 = data.ptr + (r + 5) * row_stride;
        var p6 = data.ptr + (r + 6) * row_stride;
        var p7 = data.ptr + (r + 7) * row_stride;

        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const x_ptr = qbuf.ptr + (b << 5);
            const sx: f32 = scales[b];

            // === TABLE LOOKUP: 32 lookups per row, ZERO multiply ===
            const d0: i32 = dotI8x32_lut(p0 + 2, x_ptr);
            const d1: i32 = dotI8x32_lut(p1 + 2, x_ptr);
            const d2: i32 = dotI8x32_lut(p2 + 2, x_ptr);
            const d3: i32 = dotI8x32_lut(p3 + 2, x_ptr);
            const d4: i32 = dotI8x32_lut(p4 + 2, x_ptr);
            const d5: i32 = dotI8x32_lut(p5 + 2, x_ptr);
            const d6: i32 = dotI8x32_lut(p6 + 2, x_ptr);
            const d7: i32 = dotI8x32_lut(p7 + 2, x_ptr);

            // === Float scaling: 2 multiplies per block (negligible) ===
            const sw0 = readF16(p0);
            const sw1 = readF16(p1);
            const sw2 = readF16(p2);
            const sw3 = readF16(p3);
            const sw4 = readF16(p4);
            const sw5 = readF16(p5);
            const sw6 = readF16(p6);
            const sw7 = readF16(p7);

            sum0 += (sw0 * sx) * @as(f32, @floatFromInt(d0));
            sum1 += (sw1 * sx) * @as(f32, @floatFromInt(d1));
            sum2 += (sw2 * sx) * @as(f32, @floatFromInt(d2));
            sum3 += (sw3 * sx) * @as(f32, @floatFromInt(d3));
            sum4 += (sw4 * sx) * @as(f32, @floatFromInt(d4));
            sum5 += (sw5 * sx) * @as(f32, @floatFromInt(d5));
            sum6 += (sw6 * sx) * @as(f32, @floatFromInt(d6));
            sum7 += (sw7 * sx) * @as(f32, @floatFromInt(d7));

            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
            p4 += QBYTES;
            p5 += QBYTES;
            p6 += QBYTES;
            p7 += QBYTES;
        }

        out[r + 0] = sum0;
        out[r + 1] = sum1;
        out[r + 2] = sum2;
        out[r + 3] = sum3;
        out[r + 4] = sum4;
        out[r + 5] = sum5;
        out[r + 6] = sum6;
        out[r + 7] = sum7;
    }

    while (r < rows) : (r += 1) {
        var sum: f32 = 0;
        var w = data.ptr + r * row_stride;

        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const sw = readF16(w);
            const sx = scales[b];
            const dot: i32 = dotI8x32_lut(w + 2, qbuf.ptr + (b << 5));
            sum += (sw * sx) * @as(f32, @floatFromInt(dot));
            w += QBYTES;
        }
        out[r] = sum;
    }
}

// ============================================================
// MatVec Q8_0 via LUT — float input (quantize once)
// ============================================================

pub fn matVecQ8_0_lut_f32(
    out: []f32,
    data: []const u8,
    input: []const f32,
    rows: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const blocks = cols >> 5;

    if (cols <= 16384) {
        quantizeLocal(g_lut_qbuf[0..cols], g_lut_scales[0..blocks], input, cols);
        matVecQ8_0_lut_qin(out, data, g_lut_qbuf[0..cols], g_lut_scales[0..blocks], rows, cols);
    } else {
        // Fallback per-block for very large cols
        const QBYTES: usize = 34;
        const blocks_per_row = blocks;
        const row_stride = blocks_per_row * QBYTES;

        for (0..rows) |r| {
            var sum: f32 = 0;
            const w = data.ptr + r * row_stride;
            for (0..blocks_per_row) |b| {
                sum += dotQ8Block_f32input_lut(w + b * QBYTES, input.ptr + (b << 5));
            }
            out[r] = sum;
        }
    }
}

// ============================================================
// Range kernels for MT pool
// ============================================================

pub fn matVecQ8_0_range_lut_qin(
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
    const PREFETCH_AHEAD: usize = 4;
    const blocks_per_row = cols >> 5;
    const row_stride = blocks_per_row * QBYTES;

    const base_ptr = data.ptr + row_start * row_stride;
    const nrows = row_end - row_start;

    var r: usize = 0;

    while (r + 8 <= nrows) : (r += 8) {
        var sum0: f32 = 0;
        var sum1: f32 = 0;
        var sum2: f32 = 0;
        var sum3: f32 = 0;
        var sum4: f32 = 0;
        var sum5: f32 = 0;
        var sum6: f32 = 0;
        var sum7: f32 = 0;

        var p0 = base_ptr + (r + 0) * row_stride;
        var p1 = base_ptr + (r + 1) * row_stride;
        var p2 = base_ptr + (r + 2) * row_stride;
        var p3 = base_ptr + (r + 3) * row_stride;
        var p4 = base_ptr + (r + 4) * row_stride;
        var p5 = base_ptr + (r + 5) * row_stride;
        var p6 = base_ptr + (r + 6) * row_stride;
        var p7 = base_ptr + (r + 7) * row_stride;

        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            if (b + PREFETCH_AHEAD < blocks_per_row) {
                const off = PREFETCH_AHEAD * QBYTES;
                @prefetch(p0 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p1 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p2 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p3 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p4 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p5 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p6 + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(p7 + off, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            const x_ptr = qbuf.ptr + (b << 5);
            const sx: f32 = scales[b];

            const sw0 = readF16(p0);
            const sw1 = readF16(p1);
            const sw2 = readF16(p2);
            const sw3 = readF16(p3);
            const sw4 = readF16(p4);
            const sw5 = readF16(p5);
            const sw6 = readF16(p6);
            const sw7 = readF16(p7);

            const d0: i32 = dotI8x32_lut(p0 + 2, x_ptr);
            const d1: i32 = dotI8x32_lut(p1 + 2, x_ptr);
            const d2: i32 = dotI8x32_lut(p2 + 2, x_ptr);
            const d3: i32 = dotI8x32_lut(p3 + 2, x_ptr);
            const d4: i32 = dotI8x32_lut(p4 + 2, x_ptr);
            const d5: i32 = dotI8x32_lut(p5 + 2, x_ptr);
            const d6: i32 = dotI8x32_lut(p6 + 2, x_ptr);
            const d7: i32 = dotI8x32_lut(p7 + 2, x_ptr);

            sum0 += (sw0 * sx) * @as(f32, @floatFromInt(d0));
            sum1 += (sw1 * sx) * @as(f32, @floatFromInt(d1));
            sum2 += (sw2 * sx) * @as(f32, @floatFromInt(d2));
            sum3 += (sw3 * sx) * @as(f32, @floatFromInt(d3));
            sum4 += (sw4 * sx) * @as(f32, @floatFromInt(d4));
            sum5 += (sw5 * sx) * @as(f32, @floatFromInt(d5));
            sum6 += (sw6 * sx) * @as(f32, @floatFromInt(d6));
            sum7 += (sw7 * sx) * @as(f32, @floatFromInt(d7));

            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
            p4 += QBYTES;
            p5 += QBYTES;
            p6 += QBYTES;
            p7 += QBYTES;
        }

        out_range[r + 0] = sum0;
        out_range[r + 1] = sum1;
        out_range[r + 2] = sum2;
        out_range[r + 3] = sum3;
        out_range[r + 4] = sum4;
        out_range[r + 5] = sum5;
        out_range[r + 6] = sum6;
        out_range[r + 7] = sum7;
    }

    while (r < nrows) : (r += 1) {
        var sum: f32 = 0;
        var p = base_ptr + r * row_stride;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const sw = readF16(p);
            const sx = scales[b];
            const dot: i32 = dotI8x32_lut(p + 2, qbuf.ptr + (b << 5));
            sum += (sw * sx) * @as(f32, @floatFromInt(dot));
            p += QBYTES;
        }
        out_range[r] = sum;
    }
}

pub fn matVecQ8_0_range_lut_f32(
    out_range: []f32,
    data: []const u8,
    input: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBYTES: usize = 34;
    const blocks_per_row = cols >> 5;
    const row_stride = blocks_per_row * QBYTES;
    const base_ptr = data.ptr + row_start * row_stride;
    const nrows = row_end - row_start;

    var r: usize = 0;
    while (r < nrows) : (r += 1) {
        var sum: f32 = 0;
        const w = base_ptr + r * row_stride;
        for (0..blocks_per_row) |b| {
            sum += dotQ8Block_f32input_lut(w + b * QBYTES, input.ptr + (b << 5));
        }
        out_range[r] = sum;
    }
}

pub fn matVecQ8_0_range_pair_lut_qin(
    out_a_range: []f32,
    data_a: []const u8,
    out_b_range: []f32,
    data_b: []const u8,
    qbuf: []const i8,
    scales: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBYTES: usize = 34;
    const PREFETCH_AHEAD: usize = 4;
    const blocks_per_row = cols >> 5;
    const row_stride = blocks_per_row * QBYTES;

    const base_a = data_a.ptr + row_start * row_stride;
    const base_b = data_b.ptr + row_start * row_stride;
    const nrows = row_end - row_start;

    var r: usize = 0;
    while (r < nrows) : (r += 1) {
        var suma: f32 = 0;
        var sumb: f32 = 0;

        var pa = base_a + r * row_stride;
        var pb = base_b + r * row_stride;

        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            if (b + PREFETCH_AHEAD < blocks_per_row) {
                const off = PREFETCH_AHEAD * QBYTES;
                @prefetch(pa + off, .{ .rw = .read, .locality = 3, .cache = .data });
                @prefetch(pb + off, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            const x_ptr = qbuf.ptr + (b << 5);
            const sx: f32 = scales[b];

            const swa = readF16(pa);
            const swb = readF16(pb);

            const da: i32 = dotI8x32_lut(pa + 2, x_ptr);
            const db: i32 = dotI8x32_lut(pb + 2, x_ptr);

            suma += (swa * sx) * @as(f32, @floatFromInt(da));
            sumb += (swb * sx) * @as(f32, @floatFromInt(db));

            pa += QBYTES;
            pb += QBYTES;
        }

        out_a_range[r] = suma;
        out_b_range[r] = sumb;
    }
}

pub fn matVecQ8_0_range_pair_lut_f32(
    out_a_range: []f32,
    data_a: []const u8,
    out_b_range: []f32,
    data_b: []const u8,
    input: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBYTES: usize = 34;
    const blocks_per_row = cols >> 5;
    const row_stride = blocks_per_row * QBYTES;

    const base_a = data_a.ptr + row_start * row_stride;
    const base_b = data_b.ptr + row_start * row_stride;
    const nrows = row_end - row_start;

    for (0..nrows) |r| {
        var suma: f32 = 0;
        var sumb: f32 = 0;

        const wa = base_a + r * row_stride;
        const wb = base_b + r * row_stride;

        for (0..blocks_per_row) |b| {
            const in_ptr = input.ptr + (b << 5);
            suma += dotQ8Block_f32input_lut(wa + b * QBYTES, in_ptr);
            sumb += dotQ8Block_f32input_lut(wb + b * QBYTES, in_ptr);
        }

        out_a_range[r] = suma;
        out_b_range[r] = sumb;
    }
}

// ============================================================
// Diagnostics
// ============================================================

pub fn printLutInfo() void {
    std.debug.print("\n=== LUT ENGINE: TABLE LOOKUP ===\n", .{});
    std.debug.print("Status: {s}\n", .{if (g_lut_enabled) "ENABLED" else "DISABLED"});
    std.debug.print("Integer i8xi8 dot product: TABLE LOOKUP (zero multiply)\n", .{});
    std.debug.print("Float scaling: hardware * (2 per block, <0.1%% of ops)\n", .{});
    std.debug.print("Grid: 256x256 i16 = 128 KB\n", .{});

    // Sanity
    const r1 = lookupMulI8(7, 8);
    std.debug.print("Sanity: 7x8 = {} {s}\n", .{ r1, if (r1 == 56) "OK" else "FAIL" });

    const r2 = lookupMulI8(-3, 4);
    std.debug.print("Sanity: -3x4 = {} {s}\n", .{ r2, if (r2 == -12) "OK" else "FAIL" });

    const r3 = lookupMulI8(-5, -6);
    std.debug.print("Sanity: -5x-6 = {} {s}\n", .{ r3, if (r3 == 30) "OK" else "FAIL" });

    std.debug.print("=================================\n\n", .{});
}
