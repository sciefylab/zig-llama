//! ==========================================================================
//! LUT-based Multiplication Engine (MatMul-Free)
//! ==========================================================================
//!
//! Mengganti operator perkalian (*) dengan pencarian tabel 2D.
//!
//! Teknik: "Vertical-Horizontal Grid Lookup"
//!   - Sumbu vertikal   = operand A (baris)
//!   - Sumbu horisontal = operand B (kolom)
//!   - Sel [A][B]       = hasil A × B (pre-computed)
//!
//! Untuk Q8_0 weights × Q8_0 activations:
//!   - Kedua operand adalah i8 (-128..127)
//!   - Tabel i8×i8→i16 = 256 × 256 × 2 bytes = 128 KB
//!   - Muat di L1/L2 Cache CPU modern
//!
//! Untuk float precision:
//!   - Input float di-quantize ke i8 (sudah ada di USE_Q8_ACTIVATIONS)
//!   - Perkalian i8×i8 via tabel → i16/i32
//!   - Rescale dengan (scale_weight × scale_input) untuk kembali ke float
//!   - Presisi setara Q8_0 standard (karena quantization loss sama)
//!

const std = @import("std");
const builtin = @import("builtin");

// ============================================================
// Configuration
// ============================================================

/// Enable/disable LUT globally. When false, falls back to normal multiply.
pub var g_lut_enabled: bool = true;

/// Ukuran grid: 256×256 untuk i8 range penuh (-128..127)
const GRID_SIZE: usize = 256;

/// Offset untuk mapping signed i8 ke unsigned index: i8 + 128 → 0..255
const GRID_OFFSET: i16 = 128;

// ============================================================
// The Multiplication Grid (Tabel Perkalian 2D)
// ============================================================
//
// Layout memori:
//   grid[a + 128][b + 128] = a × b
//
// Dimana a, b ∈ [-128, 127] (range i8)
// Hasil: i16 karena max |127 × 127| = 16129 < 32767 (i16 max)
//
// Total: 256 × 256 × 2 = 131,072 bytes = 128 KB
// → Muat di L2 Cache (biasanya 256KB-1MB per core)
// → Sebagian besar hot rows akan tetap di L1 Cache (64KB)
//

/// Tabel perkalian utama: grid[i][j] = (i-128) * (j-128)
/// Aligned ke 64 bytes untuk optimal cache line access.
var grid: [GRID_SIZE][GRID_SIZE]i16 align(64) = undefined;

/// Flag inisialisasi
var grid_initialized: bool = false;

// ============================================================
// Initialization (dipanggil sekali saat startup)
// ============================================================

/// Inisialisasi tabel perkalian.
/// Harus dipanggil sebelum operasi lookup apapun.
/// Thread-safe: boleh dipanggil berkali-kali (idempotent).
pub fn initGrid() void {
    if (grid_initialized) return;

    // Pre-compute seluruh 256×256 = 65,536 hasil perkalian
    for (0..GRID_SIZE) |ia| {
        const a: i16 = @as(i16, @intCast(ia)) - GRID_OFFSET;
        for (0..GRID_SIZE) |ib| {
            const b: i16 = @as(i16, @intCast(ib)) - GRID_OFFSET;
            grid[ia][ib] = a * b;
        }
    }

    grid_initialized = true;
}

// ============================================================
// Core Lookup Functions
// ============================================================

/// Lookup perkalian tunggal: a × b via tabel
/// Input: i8 × i8 → i16
inline fn lookupMul(a: i8, b: i8) i16 {
    @setRuntimeSafety(false);
    const ia: usize = @intCast(@as(i16, a) + GRID_OFFSET);
    const ib: usize = @intCast(@as(i16, b) + GRID_OFFSET);
    return grid[ia][ib];
}

/// Lookup perkalian u8-as-i8 × i8 → i16
/// Untuk weight bytes yang di-cast ke signed
inline fn lookupMulU8I8(a_raw: u8, b: i8) i16 {
    @setRuntimeSafety(false);
    const a: i8 = @bitCast(a_raw);
    const ia: usize = @intCast(@as(i16, a) + GRID_OFFSET);
    const ib: usize = @intCast(@as(i16, b) + GRID_OFFSET);
    return grid[ia][ib];
}

// ============================================================
// Dot Product: i8[32] × i8[32] → i32 (via LUT)
// ============================================================
//
// Pengganti langsung untuk dotI8x32 dan dotI8x32_prex di tensor.zig
//
// Original (SIMD multiply):
//   const prod16: VI16x32 = w16 * x16;  ← PERKALIAN HARDWARE
//   const prod32: VI32x32 = @intCast(prod16);
//   return @reduce(.Add, prod32);
//
// LUT version:
//   Setiap pasangan w[i] × x[i] diambil dari tabel grid[][]
//   Lalu dijumlahkan → i32
//

/// Dot product 32 elemen i8 via lookup table.
/// Pengganti dotI8x32(w_ptr, x_ptr) di tensor.zig
pub inline fn dotI8x32_lut(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    @setRuntimeSafety(false);

    var sum: i32 = 0;

    // Unroll manual 32 elemen, 8 per iterasi untuk ILP
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

/// Dot product 32 elemen dengan x8 pre-loaded (sebagai slice).
/// Pengganti dotI8x32_prex(w_ptr, x8_vector) di tensor.zig
pub inline fn dotI8x32_prex_lut(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    return dotI8x32_lut(w_ptr, x_ptr);
}

// ============================================================
// Q8 Block Dot Product (float output via LUT)
// ============================================================
//
// Format Q8_0 block: [f16 scale][32 × i8 quants]
//   - 2 bytes scale (f16) + 32 bytes data = 34 bytes total
//
// Original (tensor.zig dotQ8Block_vec):
//   scale = readF16(blk)
//   q_float = @floatFromInt(q_i8)   ← convert to float
//   sum = q_float * input_float      ← FLOAT MULTIPLY
//   return sum * scale
//
// LUT version (untuk qin path: kedua sisi sudah i8):
//   scale_w = readF16(w_block)
//   scale_x = scales[block_idx]
//   dot_i32 = dotI8x32_lut(w_quants, x_quants)  ← TABLE LOOKUP
//   return scale_w * scale_x * float(dot_i32)     ← hanya 2 multiply untuk scaling
//

/// Read f16 dari 2 bytes little-endian
inline fn readF16(ptr: [*]const u8) f32 {
    const bits: u16 = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// Dot product satu Q8_0 block vs quantized input, via LUT.
/// Return: float hasil (sudah di-scale)
pub inline fn dotQ8Block_lut(
    w_block: [*]const u8, // 34 bytes: [f16 scale][32 × i8]
    x_quants: [*]const i8, // 32 × i8 (quantized input block)
    x_scale: f32, // scale factor untuk input block ini
) f32 {
    const w_scale: f32 = readF16(w_block);
    const dot: i32 = dotI8x32_lut(w_block + 2, x_quants);
    return (w_scale * x_scale) * @as(f32, @floatFromInt(dot));
}

// ============================================================
// Float-path LUT: untuk dotQ8Block_vec pengganti
// ============================================================
//
// Ketika input BELUM di-quantize (masih f32), kita perlu:
// 1. Quantize input 32 elemen ke i8 on-the-fly
// 2. Lookup via tabel
// 3. Rescale ke float
//
// Ini lebih lambat dari qin-path, tapi tetap menghindari
// hardware multiply untuk dot product inti.
//

/// Quantize 32 float ke i8 + scale, lalu dot product via LUT.
/// Pengganti dotQ8Block_vec / dotQ8Block_scalar ketika input masih f32.
pub fn dotQ8Block_f32input_lut(
    w_block: [*]const u8, // 34 bytes Q8_0 block
    input: [*]const f32, // 32 × f32
) f32 {
    @setRuntimeSafety(false);

    const w_scale: f32 = readF16(w_block);

    // Quick quantize input 32 elemen
    var max_abs: f32 = 0;
    for (0..32) |i| {
        const a = @abs(input[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs == 0) return 0;

    const inv = 127.0 / max_abs;
    const x_scale = max_abs / 127.0;

    // Quantize + lookup dalam satu pass
    var dot: i32 = 0;
    for (0..32) |i| {
        var qi: i32 = @intFromFloat(@round(input[i] * inv));
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        const x_i8: i8 = @intCast(qi);
        dot += lookupMulU8I8(w_block[2 + i], x_i8);
    }

    return (w_scale * x_scale) * @as(f32, @floatFromInt(dot));
}

// ============================================================
// Row-level matmul functions (pengganti langsung)
// ============================================================

/// MatVec Q8_0 via LUT — single thread, qin path (pre-quantized input)
/// Pengganti langsung matVecQ8_0_simd_qin
pub fn matVecQ8_0_lut_qin(
    out: []f32,
    data: []const u8,
    qbuf: []const i8,
    scales: []const f32,
    rows: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBLOCK = 32;
    const QBYTES = 34;

    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;

    var r: usize = 0;

    // Batched 8 rows untuk amortisasi overhead
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
            const x_ptr = qbuf.ptr + b * QBLOCK;
            const sx: f32 = scales[b];

            // LUT dot products — ZERO hardware multiply di sini
            const d0: i32 = dotI8x32_lut(p0 + 2, x_ptr);
            const d1: i32 = dotI8x32_lut(p1 + 2, x_ptr);
            const d2: i32 = dotI8x32_lut(p2 + 2, x_ptr);
            const d3: i32 = dotI8x32_lut(p3 + 2, x_ptr);
            const d4: i32 = dotI8x32_lut(p4 + 2, x_ptr);
            const d5: i32 = dotI8x32_lut(p5 + 2, x_ptr);
            const d6: i32 = dotI8x32_lut(p6 + 2, x_ptr);
            const d7: i32 = dotI8x32_lut(p7 + 2, x_ptr);

            // Hanya 2 multiply per block untuk rescaling (bukan 32)
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

    // Tail rows
    while (r < rows) : (r += 1) {
        var sum: f32 = 0;
        var w = data.ptr + r * row_stride;

        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const sw = readF16(w);
            const sx = scales[b];
            const dot: i32 = dotI8x32_lut(w + 2, qbuf.ptr + b * QBLOCK);
            sum += (sw * sx) * @as(f32, @floatFromInt(dot));
            w += QBYTES;
        }
        out[r] = sum;
    }
}

/// MatVec Q8_0 via LUT — float input path (quantize on-the-fly)
/// Pengganti matVecQ8_0_simd_f32 dan matVecQ8_0_scalar
pub fn matVecQ8_0_lut_f32(
    out: []f32,
    data: []const u8,
    input: []const f32,
    rows: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBLOCK = 32;
    const QBYTES = 34;

    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;

    // Quantize input sekali (sama seperti qin path)
    const max_blocks = 512; // cols / 32, max 16384 cols
    var qbuf: [max_blocks * QBLOCK]i8 = undefined;
    var scales_buf: [max_blocks]f32 = undefined;

    const actual_blocks = blocks_per_row;
    if (actual_blocks > max_blocks) {
        // Fallback: per-block quantize
        for (0..rows) |r| {
            var sum: f32 = 0;
            const w = data.ptr + r * row_stride;
            for (0..blocks_per_row) |b| {
                sum += dotQ8Block_f32input_lut(w + b * QBYTES, input.ptr + b * QBLOCK);
            }
            out[r] = sum;
        }
        return;
    }

    // Quantize semua input blocks
    quantizeInputQ8_0_local(
        qbuf[0 .. actual_blocks * QBLOCK],
        scales_buf[0..actual_blocks],
        input,
        cols,
    );

    // Gunakan qin path
    matVecQ8_0_lut_qin(out, data, qbuf[0 .. actual_blocks * QBLOCK], scales_buf[0..actual_blocks], rows, cols);
}

/// MatVec Q8_0 via LUT — range kernel untuk MT pool (qin pre-quantized)
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

    const QBLOCK = 32;
    const QBYTES = 34;
    const PREFETCH_AHEAD = 4;

    const blocks_per_row = cols / QBLOCK;
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

            const x_ptr = qbuf.ptr + b * QBLOCK;
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
            const dot: i32 = dotI8x32_lut(p + 2, qbuf.ptr + b * QBLOCK);
            sum += (sw * sx) * @as(f32, @floatFromInt(dot));
            p += QBYTES;
        }
        out_range[r] = sum;
    }
}

/// MatVec Q8_0 via LUT — range kernel float input untuk MT pool
pub fn matVecQ8_0_range_lut_f32(
    out_range: []f32,
    data: []const u8,
    input: []const f32,
    row_start: usize,
    row_end: usize,
    cols: usize,
) void {
    @setRuntimeSafety(false);

    const QBLOCK = 32;
    const QBYTES = 34;

    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;

    const base_ptr = data.ptr + row_start * row_stride;
    const nrows = row_end - row_start;

    var r: usize = 0;
    while (r < nrows) : (r += 1) {
        var sum: f32 = 0;
        const w = base_ptr + r * row_stride;

        for (0..blocks_per_row) |b| {
            sum += dotQ8Block_f32input_lut(w + b * QBYTES, input.ptr + b * QBLOCK);
        }
        out_range[r] = sum;
    }
}

/// Pair range kernel via LUT — qin path
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

    const QBLOCK = 32;
    const QBYTES = 34;
    const PREFETCH_AHEAD = 4;

    const blocks_per_row = cols / QBLOCK;
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

            const x_ptr = qbuf.ptr + b * QBLOCK;
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

/// Pair range kernel via LUT — float input path
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

    const QBLOCK = 32;
    const QBYTES = 34;

    const blocks_per_row = cols / QBLOCK;
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
            const in_ptr = input.ptr + b * QBLOCK;
            suma += dotQ8Block_f32input_lut(wa + b * QBYTES, in_ptr);
            sumb += dotQ8Block_f32input_lut(wb + b * QBYTES, in_ptr);
        }

        out_a_range[r] = suma;
        out_b_range[r] = sumb;
    }
}

// ============================================================
// Local quantization helper (sama dengan tensor.zig tapi self-contained)
// ============================================================

fn quantizeInputQ8_0_local(q_out: []i8, s_out: []f32, input: []const f32, cols: usize) void {
    @setRuntimeSafety(false);

    const QBLOCK = 32;
    const blocks = cols / QBLOCK;

    var b: usize = 0;
    while (b < blocks) : (b += 1) {
        const base = b * QBLOCK;

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
// Diagnostics
// ============================================================

pub fn printLutInfo() void {
    std.debug.print("\n=== LUT MULTIPLICATION ENGINE ===\n", .{});
    std.debug.print("Status: {s}\n", .{if (g_lut_enabled) "ENABLED" else "DISABLED"});
    std.debug.print("Grid size: {}×{} = {} entries\n", .{ GRID_SIZE, GRID_SIZE, GRID_SIZE * GRID_SIZE });
    std.debug.print("Grid memory: {} KB\n", .{(GRID_SIZE * GRID_SIZE * @sizeOf(i16)) / 1024});
    std.debug.print("Entry type: i8 × i8 → i16\n", .{});
    std.debug.print("Initialized: {}\n", .{grid_initialized});

    // Sanity check
    if (grid_initialized) {
        const test_a: i8 = 7;
        const test_b: i8 = 8;
        const result = lookupMul(test_a, test_b);
        const expected: i16 = @as(i16, test_a) * @as(i16, test_b);
        std.debug.print("Sanity: {}×{} = {} (expected {}) {s}\n", .{
            test_a,                                         test_b, result, expected,
            if (result == expected) "✓" else "✗ ERROR",
        });
    }
    std.debug.print("=================================\n\n", .{});
}

// ============================================================
// Tests
// ============================================================

test "LUT grid correctness" {
    initGrid();

    // Test semua pasangan di range kecil
    var a: i16 = -128;
    while (a < 128) : (a += 1) {
        var b_val: i16 = -128;
        while (b_val < 128) : (b_val += 1) {
            const expected: i16 = a * b_val;
            const got = lookupMul(@intCast(a), @intCast(b_val));
            try std.testing.expectEqual(expected, got);
        }
    }
}

test "LUT dot product matches reference" {
    initGrid();

    // Buat data test
    var w_bytes: [32]u8 = undefined;
    var x_vals: [32]i8 = undefined;

    for (0..32) |idx| {
        const i_i32: i32 = @intCast(idx);
        w_bytes[idx] = @bitCast(@as(i8, @intCast(@mod(i_i32, 17) - 8)));
        x_vals[idx] = @intCast(@mod(i_i32, 13) - 6);
    }

    // Reference: manual dot product
    var expected: i32 = 0;
    for (0..32) |idx| {
        const w_i8: i8 = @bitCast(w_bytes[idx]);
        expected += @as(i32, w_i8) * @as(i32, x_vals[idx]);
    }

    const got = dotI8x32_lut(&w_bytes, &x_vals);
    try std.testing.expectEqual(expected, got);
}

test "LUT Q8 block matches scalar" {
    initGrid();

    // Buat Q8_0 block palsu
    var block: [34]u8 = undefined;
    block[0] = 0x00; // f16 1.0 = 0x3C00 LE
    block[1] = 0x3C;

    for (0..32) |idx| {
        const i_i32: i32 = @intCast(idx);
        block[2 + idx] = @bitCast(@as(i8, @intCast(@mod(i_i32, 11) - 5)));
    }

    // Buat input float
    var input: [32]f32 = undefined;
    for (0..32) |idx| {
        input[idx] = @as(f32, @floatFromInt(@as(i32, @intCast(idx)) - 16)) * 0.1;
    }

    // Reference: scalar dot
    var ref_sum: f32 = 0;
    const scale = readF16(&block);
    for (0..32) |idx| {
        const qi: i8 = @bitCast(block[2 + idx]);
        ref_sum += @as(f32, @floatFromInt(qi)) * input[idx];
    }
    ref_sum *= scale;

    // LUT version
    const lut_result = dotQ8Block_f32input_lut(&block, &input);

    // Allow small tolerance due to quantization of input
    const diff = @abs(lut_result - ref_sum);
    const tolerance = @abs(ref_sum) * 0.15 + 0.01; // 15% relative + absolute
    try std.testing.expect(diff < tolerance);
}
