const std = @import("std");
const builtin = @import("builtin");
const lut = @import("lut_mul.zig");

// ============================================================
// Config
// ============================================================

pub const USE_VEC_ACCUM = true;
pub const USE_Q8_ACTIVATIONS: bool = true;
pub const USE_LUT_MUL: bool = true;

pub const QIN_MAX_COLS: usize = 16384;
pub const QIN_MAX_BLOCKS: usize = QIN_MAX_COLS / 32;

pub const QIN_MIN_COLS: usize = 4096;
pub const QIN_SMALLCOLS_MIN_ROWS: usize = 65536;
pub const QIN_MIN_ROWS_ST: usize = 8192;

pub const ALIGN_BYTES: usize = 64;
pub const ALIGNMENT: ?std.mem.Alignment = blk: {
    if (!std.math.isPowerOfTwo(ALIGN_BYTES)) {
        @compileError("ALIGN_BYTES must be power-of-two");
    }
    const log2: std.math.Log2Int(usize) = @intCast(@ctz(@as(usize, ALIGN_BYTES)));
    break :blk @as(std.mem.Alignment, @enumFromInt(log2));
};

// ============================================================
// ISA dispatch
// ============================================================

pub const IsaMode = enum { scalar, simd };

var g_isa_mode: IsaMode = .simd;

var g_detected: bool = false;
var g_has_avx: bool = false;
var g_has_avx2: bool = false;
var g_has_fma: bool = false;

var g_no_fma: bool = false;
var g_force_fma: bool = false;
var g_use_fma: bool = false;

var g_use_qin_smallcols: bool = false;
var g_force_qin_smallcols: ?bool = null;

// ============================================================
// Public lifecycle
// ============================================================

pub fn initIsa(allocator: std.mem.Allocator) void {
    g_isa_mode = .simd;
    g_no_fma = false;
    g_force_fma = false;
    g_use_fma = false;

    g_use_qin_smallcols = false;
    g_force_qin_smallcols = null;

    if (std.process.getEnvVarOwned(allocator, "ZIGLLAMA_ISA")) |env| {
        defer allocator.free(env);
        if (std.ascii.eqlIgnoreCase(env, "scalar") or std.ascii.eqlIgnoreCase(env, "portable")) {
            g_isa_mode = .scalar;
        } else if (std.ascii.eqlIgnoreCase(env, "simd")) {
            g_isa_mode = .simd;
        }
    } else |_| {}

    if (std.process.getEnvVarOwned(allocator, "ZIGLLAMA_NOFMA")) |v| {
        defer allocator.free(v);
        if (v.len != 0) g_no_fma = true;
    } else |_| {}

    if (std.process.getEnvVarOwned(allocator, "ZIGLLAMA_FMA")) |v| {
        defer allocator.free(v);
        if (v.len != 0) g_force_fma = true;
    } else |_| {}

    if (std.process.getEnvVarOwned(allocator, "ZIGLLAMA_QIN_SMALLCOLS")) |v| {
        defer allocator.free(v);
        if (v.len != 0) {
            g_force_qin_smallcols = !(v.len == 1 and v[0] == '0');
        }
    } else |_| {}

    detectCpuX86();

    g_use_fma = chooseFma();
    g_use_qin_smallcols = chooseQinSmallCols();

    // LUT: tabel sudah pre-computed sebagai comptime constants di lut_tables.zig
    // Tidak perlu runtime initialization
}

pub fn shutdownKernels() void {
    MatVecPool.shutdownGlobal();
}

pub fn printIsaInfo() void {
    std.debug.print("\n=== ISA / PLATFORM INFO ===\n", .{});
    std.debug.print("OS: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("Arch: {s}\n", .{@tagName(builtin.cpu.arch)});
    std.debug.print("ISA mode: {s}\n", .{switch (g_isa_mode) {
        .scalar => "scalar (portable)",
        .simd => "simd",
    }});

    if (builtin.cpu.arch == .x86_64) {
        std.debug.print("CPU features (runtime): detected=true avx={} avx2={} fma={}\n", .{
            g_has_avx, g_has_avx2, g_has_fma,
        });
        std.debug.print("FMA override: force={} disable={}\n", .{ g_force_fma, g_no_fma });
        std.debug.print("FMA chosen: {}\n", .{g_use_fma});
        std.debug.print("QIN small-cols override: {?}\n", .{g_force_qin_smallcols});
        std.debug.print("QIN small-cols chosen: {}\n", .{g_use_qin_smallcols});
    }
    std.debug.print("===========================\n", .{});

    if (USE_LUT_MUL) {
        lut.printLutInfo();
    }
}

// ============================================================
// CPUID helpers
// ============================================================

const CpuidRegs = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

inline fn cpuid(eax_in: u32, ecx_in: u32) CpuidRegs {
    var a: u32 = undefined;
    var b: u32 = undefined;
    var c: u32 = undefined;
    var d: u32 = undefined;

    asm volatile ("cpuid"
        : [a] "={eax}" (a),
          [b] "={ebx}" (b),
          [c] "={ecx}" (c),
          [d] "={edx}" (d),
        : [in_a] "{eax}" (eax_in),
          [in_c] "{ecx}" (ecx_in),
        : .{ .memory = true });

    return .{ .eax = a, .ebx = b, .ecx = c, .edx = d };
}

inline fn xgetbv0() u64 {
    var a: u32 = undefined;
    var d: u32 = undefined;

    asm volatile ("xgetbv"
        : [a] "={eax}" (a),
          [d] "={edx}" (d),
        : [c] "{ecx}" (@as(u32, 0)),
        : .{ .memory = true });

    return (@as(u64, d) << 32) | @as(u64, a);
}

fn detectCpuX86() void {
    g_detected = false;
    g_has_avx = false;
    g_has_avx2 = false;
    g_has_fma = false;

    if (builtin.cpu.arch != .x86_64) return;

    const l1 = cpuid(1, 0);
    const osxsave = ((l1.ecx >> 27) & 1) == 1;
    const avx_hw = ((l1.ecx >> 28) & 1) == 1;
    const fma_hw = ((l1.ecx >> 12) & 1) == 1;

    var avx_os_ok = false;
    if (osxsave and avx_hw) {
        const xcr0 = xgetbv0();
        avx_os_ok = (xcr0 & 0x6) == 0x6;
    }

    const l7 = cpuid(7, 0);
    const avx2_hw = ((l7.ebx >> 5) & 1) == 1;

    g_has_avx = avx_hw and avx_os_ok;
    g_has_avx2 = avx2_hw and avx_os_ok;
    g_has_fma = fma_hw and g_has_avx;

    g_detected = true;
}

fn chooseFma() bool {
    if (!g_has_fma) return false;
    if (g_no_fma) return false;
    if (g_force_fma) return true;

    var blk: [34]u8 = undefined;
    blk[0] = 0x00;
    blk[1] = 0x3C;

    for (0..32) |i| {
        const ii: i32 = @intCast(i);
        const v_i32: i32 = @mod(ii, @as(i32, 17)) - 8;
        const v: i8 = @intCast(v_i32);
        blk[2 + i] = @bitCast(v);
    }

    var inp: [32]f32 = undefined;
    for (0..32) |i| {
        inp[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16)) * 0.01;
    }

    const in_ptr: [*]const f32 = inp[0..].ptr;
    const v0 = loadF32x8_unaligned(in_ptr + 0);
    const v1 = loadF32x8_unaligned(in_ptr + 8);
    const v2 = loadF32x8_unaligned(in_ptr + 16);
    const v3 = loadF32x8_unaligned(in_ptr + 24);
    const bptr: [*]const u8 = blk[0..].ptr;

    const iters: usize = 120_000;

    var t0 = std.time.Timer.start() catch return false;
    var acc0: SimdF32 = @splat(0);
    for (0..iters) |_| {
        acc0 += dotQ8Block_vec(bptr, v0, v1, v2, v3);
    }
    const ns0 = t0.read();
    std.mem.doNotOptimizeAway(acc0);

    var t1 = std.time.Timer.start() catch return false;
    var acc1: SimdF32 = @splat(0);
    for (0..iters) |_| {
        acc1 += dotQ8Block_vec_fma(bptr, v0, v1, v2, v3);
    }
    const ns1 = t1.read();
    std.mem.doNotOptimizeAway(acc1);

    return ns1 < ns0;
}

fn chooseQinSmallCols() bool {
    if (!USE_Q8_ACTIVATIONS) return false;
    if (g_isa_mode == .scalar) return false;

    if (g_force_qin_smallcols) |forced| return forced;

    const cols: usize = 2048;
    const QBLOCK: usize = 32;
    const blocks: usize = cols / QBLOCK;
    const QBYTES: usize = 34;

    var wrow: [blocks * QBYTES]u8 = undefined;
    for (0..blocks) |b| {
        const base = b * QBYTES;
        wrow[base + 0] = 0x00;
        wrow[base + 1] = 0x3C;
        for (0..32) |i| {
            const v_i32: i32 = @as(i32, @intCast((b * 7 + i) % 17)) - 8;
            wrow[base + 2 + i] = @bitCast(@as(i8, @intCast(v_i32)));
        }
    }

    var x: [cols]f32 = undefined;
    for (0..cols) |i| {
        x[i] = (@as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1024))) * 0.001;
    }

    var qbuf: [cols]i8 = undefined;
    var scales: [blocks]f32 = undefined;
    quantizeInputQ8_0(qbuf[0..], scales[0..], x[0..], cols);

    const iters: usize = 2000;

    var t0 = std.time.Timer.start() catch return false;
    var accf: SimdF32 = @splat(0);
    for (0..iters) |_| {
        var p = wrow[0..].ptr;
        var in_ptr = x[0..].ptr;

        var b: usize = 0;
        while (b < blocks) : (b += 1) {
            const fv0 = loadF32x8_unaligned(in_ptr + 0);
            const fv1 = loadF32x8_unaligned(in_ptr + 8);
            const fv2 = loadF32x8_unaligned(in_ptr + 16);
            const fv3 = loadF32x8_unaligned(in_ptr + 24);

            if (g_use_fma) {
                accf += dotQ8Block_vec_fma(p, fv0, fv1, fv2, fv3);
            } else {
                accf += dotQ8Block_vec(p, fv0, fv1, fv2, fv3);
            }

            p += QBYTES;
            in_ptr += QBLOCK;
        }
    }
    const ns_float = t0.read();
    std.mem.doNotOptimizeAway(accf);

    var t1 = std.time.Timer.start() catch return false;
    var accq: f32 = 0;
    for (0..iters) |_| {
        var p = wrow[0..].ptr;

        var b: usize = 0;
        while (b < blocks) : (b += 1) {
            const sw = readF16(p);
            const sx = scales[b];
            const dot: i32 = dotI8x32_raw(p + 2, qbuf[0..].ptr + b * QBLOCK);
            accq += (sw * sx) * @as(f32, @floatFromInt(dot));
            p += QBYTES;
        }
    }
    const ns_qin = t1.read();
    std.mem.doNotOptimizeAway(accq);

    return ns_qin < ns_float;
}

// ============================================================
// Tensor types
// ============================================================

pub const Tensor = struct {
    data: []f32,
    shape: []const u32,
    allocator: ?std.mem.Allocator,

    pub fn zeros(allocator: std.mem.Allocator, shape: []const u32) !Tensor {
        var total: usize = 1;
        for (shape) |s| total *= s;

        const data = try allocator.alignedAlloc(f32, ALIGNMENT, total);
        @memset(data, 0);

        return Tensor{ .data = data, .shape = shape, .allocator = allocator };
    }

    pub fn deinit(self: *Tensor) void {
        if (self.allocator) |alloc| alloc.free(self.data);
    }
};

pub const QuantizedTensor = struct {
    data: []u8,
    shape: []u32,
    n_blocks: u32,
    block_size: u32,
    quant_type: QuantType,
    allocator: std.mem.Allocator,

    pub const QuantType = enum { Q4_0, Q4_1, Q8_0 };

    pub fn deinit(self: *QuantizedTensor) void {
        if (self.data.len > 0) self.allocator.free(self.data);
        if (self.shape.len > 0) self.allocator.free(self.shape);
    }
};

// ============================================================
// SIMD helpers
// ============================================================

const SIMD_WIDTH = 8;
const SimdF32 = @Vector(SIMD_WIDTH, f32);
const SimdU8 = @Vector(SIMD_WIDTH, u8);
const SimdI8 = @Vector(SIMD_WIDTH, i8);

const VU8x32 = @Vector(32, u8);
const VI8x32 = @Vector(32, i8);
const VI16x32 = @Vector(32, i16);
const VI32x32 = @Vector(32, i32);

inline fn readF16(ptr: [*]const u8) f32 {
    const bits: u16 = @as(u16, ptr[0]) | (@as(u16, ptr[1]) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

inline fn loadF32x8_unaligned(ptr: [*]const f32) SimdF32 {
    return @as(*align(1) const SimdF32, @ptrCast(ptr)).*;
}

inline fn loadQ8(ptr: [*]const u8) SimdF32 {
    const u: SimdU8 = @as(*align(1) const SimdU8, @ptrCast(ptr)).*;
    const s: SimdI8 = @bitCast(u);
    return @floatFromInt(s);
}

inline fn loadI8x32_from_u8(ptr: [*]const u8) VI8x32 {
    const u: VU8x32 = @as(*align(1) const VU8x32, @ptrCast(ptr)).*;
    return @bitCast(u);
}

inline fn loadI8x32(ptr: [*]const i8) VI8x32 {
    return @as(*align(1) const VI8x32, @ptrCast(ptr)).*;
}

/// Raw SIMD dot — always hardware multiply (for autotune)
inline fn dotI8x32_raw(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    const w8: VI8x32 = loadI8x32_from_u8(w_ptr);
    const x8: VI8x32 = loadI8x32(x_ptr);
    const w16: VI16x32 = @intCast(w8);
    const x16: VI16x32 = @intCast(x8);
    const prod16: VI16x32 = w16 * x16;
    const prod32: VI32x32 = @intCast(prod16);
    return @reduce(.Add, prod32);
}

/// Dispatching dot — LUT or SIMD
inline fn dotI8x32(w_ptr: [*]const u8, x_ptr: [*]const i8) i32 {
    if (USE_LUT_MUL and lut.g_lut_enabled) {
        return lut.dotI8x32_lut(w_ptr, x_ptr);
    }
    return dotI8x32_raw(w_ptr, x_ptr);
}

inline fn dotI8x32_prex(w_ptr: [*]const u8, x8: VI8x32) i32 {
    if (USE_LUT_MUL and lut.g_lut_enabled) {
        return dotI8x32_prex_lut_bridge(w_ptr, x8);
    }

    const w8: VI8x32 = loadI8x32_from_u8(w_ptr);
    const w16: VI16x32 = @intCast(w8);
    const x16: VI16x32 = @intCast(x8);
    const prod16: VI16x32 = w16 * x16;
    const prod32: VI32x32 = @intCast(prod16);
    return @reduce(.Add, prod32);
}

noinline fn dotI8x32_prex_lut_bridge(w_ptr: [*]const u8, x8: VI8x32) i32 {
    var x_buf: [32]i8 align(32) = undefined;
    @as(*align(32) VI8x32, @ptrCast(&x_buf)).* = x8;
    return lut.dotI8x32_lut(w_ptr, &x_buf);
}

inline fn dotQ8Block_vec(blk: [*]const u8, v0: SimdF32, v1: SimdF32, v2: SimdF32, v3: SimdF32) SimdF32 {
    const scale: SimdF32 = @splat(readF16(blk));
    const q0 = loadQ8(blk + 2);
    const q1 = loadQ8(blk + 10);
    const q2 = loadQ8(blk + 18);
    const q3 = loadQ8(blk + 26);
    const sum = ((q0 * v0) + (q1 * v1)) + ((q2 * v2) + (q3 * v3));
    return sum * scale;
}

inline fn dotQ8Block_vec_fma(blk: [*]const u8, v0: SimdF32, v1: SimdF32, v2: SimdF32, v3: SimdF32) SimdF32 {
    const scale: SimdF32 = @splat(readF16(blk));
    const q0 = loadQ8(blk + 2);
    const q1 = loadQ8(blk + 10);
    const q2 = loadQ8(blk + 18);
    const q3 = loadQ8(blk + 26);

    var acc: SimdF32 = @splat(0);
    acc = @mulAdd(SimdF32, q0, v0, acc);
    acc = @mulAdd(SimdF32, q1, v1, acc);
    acc = @mulAdd(SimdF32, q2, v2, acc);
    acc = @mulAdd(SimdF32, q3, v3, acc);

    return acc * scale;
}

// ============================================================
// Q8 activation quantization helpers
// ============================================================

inline fn canUseQinRC(rows: usize, cols: usize) bool {
    if (!USE_Q8_ACTIVATIONS) return false;
    if (cols > QIN_MAX_COLS) return false;
    if ((cols % 32) != 0) return false;
    if (cols >= QIN_MIN_COLS) return true;
    return rows >= QIN_SMALLCOLS_MIN_ROWS and g_use_qin_smallcols;
}

inline fn canUseLutQin(cols: usize) bool {
    if (cols > QIN_MAX_COLS) return false;
    if ((cols % 32) != 0) return false;
    return true;
}

fn quantizeInputQ8_0(q_out: []i8, s_out: []f32, input: []const f32, cols: usize) void {
    @setRuntimeSafety(false);

    const QBLOCK = 32;
    std.debug.assert(cols % QBLOCK == 0);
    std.debug.assert(q_out.len == cols);
    const blocks = cols / QBLOCK;
    std.debug.assert(s_out.len == blocks);

    var b: usize = 0;
    while (b < blocks) : (b += 1) {
        const base = b * QBLOCK;

        var max_abs: f32 = 0;
        var ii: usize = 0;
        while (ii < QBLOCK) : (ii += 1) {
            const v = input[base + ii];
            const a = @abs(v);
            if (a > max_abs) max_abs = a;
        }

        if (max_abs == 0) {
            s_out[b] = 0;
            @memset(q_out[base .. base + QBLOCK], 0);
            continue;
        }

        const inv = 127.0 / max_abs;
        const scale = max_abs / 127.0;
        s_out[b] = scale;

        ii = 0;
        while (ii < QBLOCK) : (ii += 1) {
            const v = input[base + ii] * inv;
            var qi: i32 = @intFromFloat(@round(v));
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            q_out[base + ii] = @intCast(qi);
        }
    }
}

// ============================================================
// Public API
// ============================================================

pub const KernelError = error{
    InvalidDims,
    UnsupportedQuantType,
};

pub fn quantizedMatVecChecked(
    out: []f32,
    weight: *const QuantizedTensor,
    input: []const f32,
    rows: usize,
    cols: usize,
) KernelError!void {
    if (rows == 0 or cols == 0) return error.InvalidDims;
    if (out.len < rows) return error.InvalidDims;
    if (input.len < cols) return error.InvalidDims;
    if ((cols % 32) != 0) return error.InvalidDims;

    if (weight.quant_type != .Q8_0 and weight.quant_type != .Q4_0 and weight.quant_type != .Q4_1) {
        return error.UnsupportedQuantType;
    }

    quantizedMatVec(out[0..rows], weight, input[0..cols], rows, cols);
}

pub fn quantizedMatVec(
    out: []f32,
    weight: *const QuantizedTensor,
    input: []const f32,
    rows: usize,
    cols: usize,
) void {
    switch (weight.quant_type) {
        .Q8_0 => {
            if (USE_LUT_MUL and lut.g_lut_enabled) {
                lut.matVecQ8_0_lut_f32(out, weight.data, input, rows, cols);
                return;
            }

            if (g_isa_mode == .scalar) {
                matVecQ8_0_scalar(out, weight.data, input, rows, cols);
                return;
            }

            if (canUseQinRC(rows, cols) and rows >= QIN_MIN_ROWS_ST) {
                matVecQ8_0_simd_qin(out, weight.data, input, rows, cols);
            } else {
                matVecQ8_0_simd_f32(out, weight.data, input, rows, cols);
            }
        },
        .Q4_0, .Q4_1 => matVecQ4_0(out, weight.data, input, rows, cols),
    }
}

pub fn quantizedMatVecMt(
    out: []f32,
    weight: *const QuantizedTensor,
    input: []const f32,
    rows: usize,
    cols: usize,
    requested_threads: usize,
) void {
    if (g_isa_mode == .scalar and !(USE_LUT_MUL and lut.g_lut_enabled)) {
        quantizedMatVec(out, weight, input, rows, cols);
        return;
    }

    if (requested_threads <= 1 or weight.quant_type != .Q8_0 or rows < 4096) {
        quantizedMatVec(out, weight, input, rows, cols);
        return;
    }

    const total_threads: usize = @min(@max(requested_threads, 1), MatVecPool.MAX_THREADS);
    const work_per_thread: usize = 4096;
    const max_useful = (rows + work_per_thread - 1) / work_per_thread;
    const active_threads: usize = @max(1, @min(total_threads, max_useful));

    if (active_threads <= 1) {
        quantizedMatVec(out, weight, input, rows, cols);
        return;
    }

    if (MatVecPool.getOrInit(total_threads)) |pool| {
        pool.runQ8Single(out, weight.data, input, rows, cols, active_threads);
    } else {
        quantizedMatVec(out, weight, input, rows, cols);
    }
}

pub fn quantizedMatVec2Mt(
    out_a: []f32,
    weight_a: *const QuantizedTensor,
    out_b: []f32,
    weight_b: *const QuantizedTensor,
    input: []const f32,
    rows: usize,
    cols: usize,
    requested_threads: usize,
) void {
    if (g_isa_mode == .scalar and !(USE_LUT_MUL and lut.g_lut_enabled)) {
        quantizedMatVec(out_a, weight_a, input, rows, cols);
        quantizedMatVec(out_b, weight_b, input, rows, cols);
        return;
    }

    if (requested_threads <= 1 or rows < 4096 or weight_a.quant_type != .Q8_0 or weight_b.quant_type != .Q8_0) {
        quantizedMatVec(out_a, weight_a, input, rows, cols);
        quantizedMatVec(out_b, weight_b, input, rows, cols);
        return;
    }

    const total_threads: usize = @min(@max(requested_threads, 1), MatVecPool.MAX_THREADS);
    const work_per_thread: usize = 4096;
    const max_useful = (rows + work_per_thread - 1) / work_per_thread;
    const active_threads: usize = @max(1, @min(total_threads, max_useful));

    if (active_threads <= 1) {
        quantizedMatVec(out_a, weight_a, input, rows, cols);
        quantizedMatVec(out_b, weight_b, input, rows, cols);
        return;
    }

    if (MatVecPool.getOrInit(total_threads)) |pool| {
        pool.runQ8Pair(out_a, weight_a.data, out_b, weight_b.data, input, rows, cols, active_threads);
    } else {
        quantizedMatVec(out_a, weight_a, input, rows, cols);
        quantizedMatVec(out_b, weight_b, input, rows, cols);
    }
}

// ============================================================
// Thread pool
// ============================================================

const MatVecPool = struct {
    pub const MAX_THREADS = 8;

    pub var g_init_lock: std.Thread.Mutex = .{};
    pub var g_pool: ?*MatVecPool = null;

    const JobKind = enum(u8) { none, single, pair };

    exec_mutex: std.Thread.Mutex = .{},
    mutex: std.Thread.Mutex = .{},
    cv: std.Thread.Condition = .{},
    done_cv: std.Thread.Condition = .{},

    allocator: std.mem.Allocator = std.heap.page_allocator,
    total_threads: usize = 1,
    threads: []std.Thread = &[_]std.Thread{},
    running: bool = true,

    job_id: std.atomic.Value(u64) = .init(0),
    done: std.atomic.Value(u32) = .init(0),

    active_threads: usize = 1,
    kind: JobKind = .none,

    out: []f32 = &[_]f32{},
    data: []const u8 = &[_]u8{},
    out_b: []f32 = &[_]f32{},
    data_b: []const u8 = &[_]u8{},
    input: []const f32 = &[_]f32{},
    rows: usize = 0,
    cols: usize = 0,

    qin_enabled: bool = false,
    qin_cols: usize = 0,
    qin_blocks: usize = 0,
    qin_qbuf: [QIN_MAX_COLS]i8 = undefined,
    qin_scales: [QIN_MAX_BLOCKS]f32 = undefined,

    fn workerMain(pool: *MatVecPool, tid: usize) void {
        @setRuntimeSafety(false);
        var last_job: u64 = 0;

        while (true) {
            pool.mutex.lock();
            while (pool.running and pool.job_id.load(.acquire) == last_job) {
                pool.cv.wait(&pool.mutex);
            }
            if (!pool.running) {
                pool.mutex.unlock();
                return;
            }

            last_job = pool.job_id.load(.acquire);

            const active = pool.active_threads;
            const kind = pool.kind;
            const out = pool.out;
            const data = pool.data;
            const out_b = pool.out_b;
            const data_b = pool.data_b;
            const input = pool.input;
            const rows = pool.rows;
            const cols = pool.cols;
            const qin_enabled = pool.qin_enabled;
            const qin_cols = pool.qin_cols;
            const qin_blocks = pool.qin_blocks;

            const qbuf: []const i8 = if (qin_enabled) pool.qin_qbuf[0..qin_cols] else &[_]i8{};
            const scales: []const f32 = if (qin_enabled) pool.qin_scales[0..qin_blocks] else &[_]f32{};

            pool.mutex.unlock();

            if (tid >= active) continue;

            const chunk = (rows + active - 1) / active;
            const start = tid * chunk;
            const end = @min(rows, start + chunk);

            if (start < end) {
                const is_lut = USE_LUT_MUL and lut.g_lut_enabled;

                switch (kind) {
                    .single => {
                        if (is_lut) {
                            if (qin_enabled and cols == qin_cols) {
                                lut.matVecQ8_0_range_lut_qin(out[start..end], data, qbuf, scales, start, end, cols);
                            } else {
                                lut.matVecQ8_0_range_lut_f32(out[start..end], data, input, start, end, cols);
                            }
                        } else {
                            if (qin_enabled and cols == qin_cols) {
                                matVecQ8_0_range_batched8_qin_prequant(out[start..end], data, qbuf, scales, start, end, cols);
                            } else {
                                matVecQ8_0_range_batched8_f32(out[start..end], data, input, start, end, cols);
                            }
                        }
                    },
                    .pair => {
                        if (is_lut) {
                            if (qin_enabled and cols == qin_cols) {
                                lut.matVecQ8_0_range_pair_lut_qin(out[start..end], data, out_b[start..end], data_b, qbuf, scales, start, end, cols);
                            } else {
                                lut.matVecQ8_0_range_pair_lut_f32(out[start..end], data, out_b[start..end], data_b, input, start, end, cols);
                            }
                        } else {
                            if (qin_enabled and cols == qin_cols) {
                                matVecQ8_0_range_pair_batched2_qin_prequant(out[start..end], data, out_b[start..end], data_b, qbuf, scales, start, end, cols);
                            } else {
                                matVecQ8_0_range_pair_batched2_f32(out[start..end], data, out_b[start..end], data_b, input, start, end, cols);
                            }
                        }
                    },
                    else => {},
                }
            }

            _ = pool.done.fetchAdd(1, .acq_rel);
            pool.mutex.lock();
            pool.done_cv.signal();
            pool.mutex.unlock();
        }
    }

    fn computeQin(pool: *MatVecPool, input: []const f32, cols: usize, rows: usize) void {
        const is_lut = USE_LUT_MUL and lut.g_lut_enabled;
        const use_qin = if (is_lut) canUseLutQin(cols) else canUseQinRC(rows, cols);

        pool.qin_enabled = use_qin;
        if (pool.qin_enabled) {
            const blocks = cols / 32;
            quantizeInputQ8_0(pool.qin_qbuf[0..cols], pool.qin_scales[0..blocks], input, cols);
            pool.qin_cols = cols;
            pool.qin_blocks = blocks;
        } else {
            pool.qin_cols = 0;
            pool.qin_blocks = 0;
        }
    }

    fn publishSingle(pool: *MatVecPool, out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize, active_threads: usize) void {
        pool.mutex.lock();
        pool.kind = .single;
        pool.out = out;
        pool.data = data;
        pool.input = input;
        pool.rows = rows;
        pool.cols = cols;
        pool.active_threads = active_threads;
        pool.computeQin(input, cols, rows);
        pool.done.store(0, .release);
        _ = pool.job_id.fetchAdd(1, .acq_rel);
        pool.mutex.unlock();
        pool.cv.broadcast();
    }

    fn publishPair(pool: *MatVecPool, out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, input: []const f32, rows: usize, cols: usize, active_threads: usize) void {
        pool.mutex.lock();
        pool.kind = .pair;
        pool.out = out_a;
        pool.data = data_a;
        pool.out_b = out_b;
        pool.data_b = data_b;
        pool.input = input;
        pool.rows = rows;
        pool.cols = cols;
        pool.active_threads = active_threads;
        pool.computeQin(input, cols, rows);
        pool.done.store(0, .release);
        _ = pool.job_id.fetchAdd(1, .acq_rel);
        pool.mutex.unlock();
        pool.cv.broadcast();
    }

    fn doMainChunkSingle(pool: *MatVecPool, out: []f32, data: []const u8, input: []const f32, end0: usize, cols: usize) void {
        @setRuntimeSafety(false);
        const is_lut = USE_LUT_MUL and lut.g_lut_enabled;

        if (is_lut) {
            if (pool.qin_enabled) {
                lut.matVecQ8_0_range_lut_qin(out[0..end0], data, pool.qin_qbuf[0..cols], pool.qin_scales[0..(cols / 32)], 0, end0, cols);
            } else {
                lut.matVecQ8_0_range_lut_f32(out[0..end0], data, input, 0, end0, cols);
            }
        } else {
            if (pool.qin_enabled) {
                matVecQ8_0_range_batched8_qin_prequant(out[0..end0], data, pool.qin_qbuf[0..cols], pool.qin_scales[0..(cols / 32)], 0, end0, cols);
            } else {
                matVecQ8_0_range_batched8_f32(out[0..end0], data, input, 0, end0, cols);
            }
        }
    }

    fn doMainChunkPair(pool: *MatVecPool, out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, input: []const f32, end0: usize, cols: usize) void {
        @setRuntimeSafety(false);
        const is_lut = USE_LUT_MUL and lut.g_lut_enabled;

        if (is_lut) {
            if (pool.qin_enabled) {
                lut.matVecQ8_0_range_pair_lut_qin(out_a[0..end0], data_a, out_b[0..end0], data_b, pool.qin_qbuf[0..cols], pool.qin_scales[0..(cols / 32)], 0, end0, cols);
            } else {
                lut.matVecQ8_0_range_pair_lut_f32(out_a[0..end0], data_a, out_b[0..end0], data_b, input, 0, end0, cols);
            }
        } else {
            if (pool.qin_enabled) {
                matVecQ8_0_range_pair_batched2_qin_prequant(out_a[0..end0], data_a, out_b[0..end0], data_b, pool.qin_qbuf[0..cols], pool.qin_scales[0..(cols / 32)], 0, end0, cols);
            } else {
                matVecQ8_0_range_pair_batched2_f32(out_a[0..end0], data_a, out_b[0..end0], data_b, input, 0, end0, cols);
            }
        }
    }

    pub fn runQ8Single(pool: *MatVecPool, out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize, active_threads: usize) void {
        @setRuntimeSafety(false);
        pool.exec_mutex.lock();
        defer pool.exec_mutex.unlock();

        publishSingle(pool, out, data, input, rows, cols, active_threads);

        const chunk = (rows + active_threads - 1) / active_threads;
        const end0 = @min(rows, chunk);
        if (0 < end0) pool.doMainChunkSingle(out, data, input, end0, cols);

        const target: u32 = @intCast(active_threads - 1);
        pool.mutex.lock();
        while (pool.done.load(.acquire) != target) pool.done_cv.wait(&pool.mutex);
        pool.mutex.unlock();
    }

    pub fn runQ8Pair(pool: *MatVecPool, out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, input: []const f32, rows: usize, cols: usize, active_threads: usize) void {
        @setRuntimeSafety(false);
        pool.exec_mutex.lock();
        defer pool.exec_mutex.unlock();

        publishPair(pool, out_a, data_a, out_b, data_b, input, rows, cols, active_threads);

        const chunk = (rows + active_threads - 1) / active_threads;
        const end0 = @min(rows, chunk);
        if (0 < end0) pool.doMainChunkPair(out_a, data_a, out_b, data_b, input, end0, cols);

        const target: u32 = @intCast(active_threads - 1);
        pool.mutex.lock();
        while (pool.done.load(.acquire) != target) pool.done_cv.wait(&pool.mutex);
        pool.mutex.unlock();
    }

    fn init(allocator: std.mem.Allocator, total_threads: usize) !*MatVecPool {
        var pool = try allocator.create(MatVecPool);
        pool.* = .{};
        pool.allocator = allocator;
        pool.total_threads = total_threads;

        const n_workers = total_threads - 1;
        pool.threads = try allocator.alloc(std.Thread, n_workers);

        for (0..n_workers) |i| {
            const tid = i + 1;
            pool.threads[i] = try std.Thread.spawn(.{}, MatVecPool.workerMain, .{ pool, tid });
        }
        return pool;
    }

    fn shutdown(pool: *MatVecPool) void {
        pool.mutex.lock();
        pool.running = false;
        _ = pool.job_id.fetchAdd(1, .acq_rel);
        pool.mutex.unlock();
        pool.cv.broadcast();

        for (pool.threads) |t| t.join();

        const alloc = pool.allocator;
        if (pool.threads.len > 0) alloc.free(pool.threads);
        alloc.destroy(pool);
    }

    pub fn shutdownGlobal() void {
        g_init_lock.lock();
        defer g_init_lock.unlock();
        if (g_pool) |p| {
            g_pool = null;
            p.shutdown();
        }
    }

    pub fn getOrInit(total_threads: usize) ?*MatVecPool {
        g_init_lock.lock();
        defer g_init_lock.unlock();

        const want: usize = @min(@max(total_threads, 1), MAX_THREADS);

        if (g_pool) |p| {
            if (p.total_threads >= want) return p;
            g_pool = null;
            p.shutdown();
        }

        const alloc = std.heap.page_allocator;
        const pnew = init(alloc, want) catch return null;
        g_pool = pnew;
        return pnew;
    }
};

// ============================================================
// Original Q8_0 kernels (scalar + SIMD)
// ============================================================

fn dotQ8Block_scalar(blk: [*]const u8, in_ptr: [*]const f32) f32 {
    if (USE_LUT_MUL and lut.g_lut_enabled) {
        return lut.dotQ8Block_f32input_lut(blk, in_ptr);
    }

    const scale = readF16(blk);
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const qi: i8 = @bitCast(blk[2 + i]);
        sum += @as(f32, @floatFromInt(qi)) * in_ptr[i];
    }
    return sum * scale;
}

fn matVecQ8_0_scalar(out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
    std.debug.assert(cols % QBLOCK == 0);
    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var sum: f32 = 0;
        var w = data.ptr + r * row_stride;
        var in_ptr = input.ptr;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            sum += dotQ8Block_scalar(w, in_ptr);
            w += QBYTES;
            in_ptr += QBLOCK;
        }
        out[r] = sum;
    }
}

fn matVecQ8_0_simd_f32(out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
    std.debug.assert(cols % QBLOCK == 0);
    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;
    const use_fma = g_use_fma;

    var row: usize = 0;
    while (row + 8 <= rows) : (row += 8) {
        var acc0: SimdF32 = @splat(0);
        var acc1: SimdF32 = @splat(0);
        var acc2: SimdF32 = @splat(0);
        var acc3: SimdF32 = @splat(0);
        var acc4: SimdF32 = @splat(0);
        var acc5: SimdF32 = @splat(0);
        var acc6: SimdF32 = @splat(0);
        var acc7: SimdF32 = @splat(0);

        var p0 = data.ptr + (row + 0) * row_stride;
        var p1 = data.ptr + (row + 1) * row_stride;
        var p2 = data.ptr + (row + 2) * row_stride;
        var p3 = data.ptr + (row + 3) * row_stride;
        var p4 = data.ptr + (row + 4) * row_stride;
        var p5 = data.ptr + (row + 5) * row_stride;
        var p6 = data.ptr + (row + 6) * row_stride;
        var p7 = data.ptr + (row + 7) * row_stride;
        var in_ptr = input.ptr;

        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const v0 = loadF32x8_unaligned(in_ptr + 0);
            const v1 = loadF32x8_unaligned(in_ptr + 8);
            const v2 = loadF32x8_unaligned(in_ptr + 16);
            const v3 = loadF32x8_unaligned(in_ptr + 24);

            if (use_fma) {
                acc0 += dotQ8Block_vec_fma(p0, v0, v1, v2, v3);
                acc1 += dotQ8Block_vec_fma(p1, v0, v1, v2, v3);
                acc2 += dotQ8Block_vec_fma(p2, v0, v1, v2, v3);
                acc3 += dotQ8Block_vec_fma(p3, v0, v1, v2, v3);
                acc4 += dotQ8Block_vec_fma(p4, v0, v1, v2, v3);
                acc5 += dotQ8Block_vec_fma(p5, v0, v1, v2, v3);
                acc6 += dotQ8Block_vec_fma(p6, v0, v1, v2, v3);
                acc7 += dotQ8Block_vec_fma(p7, v0, v1, v2, v3);
            } else {
                acc0 += dotQ8Block_vec(p0, v0, v1, v2, v3);
                acc1 += dotQ8Block_vec(p1, v0, v1, v2, v3);
                acc2 += dotQ8Block_vec(p2, v0, v1, v2, v3);
                acc3 += dotQ8Block_vec(p3, v0, v1, v2, v3);
                acc4 += dotQ8Block_vec(p4, v0, v1, v2, v3);
                acc5 += dotQ8Block_vec(p5, v0, v1, v2, v3);
                acc6 += dotQ8Block_vec(p6, v0, v1, v2, v3);
                acc7 += dotQ8Block_vec(p7, v0, v1, v2, v3);
            }

            in_ptr += QBLOCK;
            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
            p4 += QBYTES;
            p5 += QBYTES;
            p6 += QBYTES;
            p7 += QBYTES;
        }

        out[row + 0] = @reduce(.Add, acc0);
        out[row + 1] = @reduce(.Add, acc1);
        out[row + 2] = @reduce(.Add, acc2);
        out[row + 3] = @reduce(.Add, acc3);
        out[row + 4] = @reduce(.Add, acc4);
        out[row + 5] = @reduce(.Add, acc5);
        out[row + 6] = @reduce(.Add, acc6);
        out[row + 7] = @reduce(.Add, acc7);
    }

    while (row < rows) : (row += 1) {
        var acc: SimdF32 = @splat(0);
        var p = data.ptr + row * row_stride;
        var in_ptr = input.ptr;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const v0 = loadF32x8_unaligned(in_ptr + 0);
            const v1 = loadF32x8_unaligned(in_ptr + 8);
            const v2 = loadF32x8_unaligned(in_ptr + 16);
            const v3 = loadF32x8_unaligned(in_ptr + 24);
            if (use_fma) {
                acc += dotQ8Block_vec_fma(p, v0, v1, v2, v3);
            } else {
                acc += dotQ8Block_vec(p, v0, v1, v2, v3);
            }
            in_ptr += QBLOCK;
            p += QBYTES;
        }
        out[row] = @reduce(.Add, acc);
    }
}

fn matVecQ8_0_simd_qin(out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
    std.debug.assert(cols % QBLOCK == 0);

    if (!canUseQinRC(rows, cols)) {
        matVecQ8_0_simd_f32(out, data, input, rows, cols);
        return;
    }

    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;

    var qbuf_storage: [QIN_MAX_COLS]i8 = undefined;
    var scales_storage: [QIN_MAX_BLOCKS]f32 = undefined;
    const qbuf: []i8 = qbuf_storage[0..cols];
    const scales: []f32 = scales_storage[0..blocks_per_row];
    quantizeInputQ8_0(qbuf, scales, input, cols);

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
            const x8: VI8x32 = loadI8x32(qbuf.ptr + b * QBLOCK);
            const sx: f32 = scales[b];
            const sw0 = readF16(p0);
            const sw1 = readF16(p1);
            const sw2 = readF16(p2);
            const sw3 = readF16(p3);
            const sw4 = readF16(p4);
            const sw5 = readF16(p5);
            const sw6 = readF16(p6);
            const sw7 = readF16(p7);

            sum0 += (sw0 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p0 + 2, x8)));
            sum1 += (sw1 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p1 + 2, x8)));
            sum2 += (sw2 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p2 + 2, x8)));
            sum3 += (sw3 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p3 + 2, x8)));
            sum4 += (sw4 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p4 + 2, x8)));
            sum5 += (sw5 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p5 + 2, x8)));
            sum6 += (sw6 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p6 + 2, x8)));
            sum7 += (sw7 * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p7 + 2, x8)));

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
            const x8: VI8x32 = loadI8x32(qbuf.ptr + b * QBLOCK);
            sum += (sw * sx) * @as(f32, @floatFromInt(dotI8x32_prex(w + 2, x8)));
            w += QBYTES;
        }
        out[r] = sum;
    }
}

// ============================================================
// Range kernels for MT pool — original paths
// ============================================================

const PREFETCH_AHEAD: usize = 4;

fn matVecQ8_0_range_batched8_f32(out_range: []f32, data: []const u8, input: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
    const use_fma = g_use_fma;
    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;
    const base_ptr = data.ptr + row_start * row_stride;
    const nrows = row_end - row_start;

    var r: usize = 0;
    while (r + 8 <= nrows) : (r += 8) {
        var acc0: SimdF32 = @splat(0);
        var acc1: SimdF32 = @splat(0);
        var acc2: SimdF32 = @splat(0);
        var acc3: SimdF32 = @splat(0);
        var acc4: SimdF32 = @splat(0);
        var acc5: SimdF32 = @splat(0);
        var acc6: SimdF32 = @splat(0);
        var acc7: SimdF32 = @splat(0);
        var p0 = base_ptr + (r + 0) * row_stride;
        var p1 = base_ptr + (r + 1) * row_stride;
        var p2 = base_ptr + (r + 2) * row_stride;
        var p3 = base_ptr + (r + 3) * row_stride;
        var p4 = base_ptr + (r + 4) * row_stride;
        var p5 = base_ptr + (r + 5) * row_stride;
        var p6 = base_ptr + (r + 6) * row_stride;
        var p7 = base_ptr + (r + 7) * row_stride;
        var in_ptr = input.ptr;
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
            const v0 = loadF32x8_unaligned(in_ptr + 0);
            const v1 = loadF32x8_unaligned(in_ptr + 8);
            const v2 = loadF32x8_unaligned(in_ptr + 16);
            const v3 = loadF32x8_unaligned(in_ptr + 24);
            if (use_fma) {
                acc0 += dotQ8Block_vec_fma(p0, v0, v1, v2, v3);
                acc1 += dotQ8Block_vec_fma(p1, v0, v1, v2, v3);
                acc2 += dotQ8Block_vec_fma(p2, v0, v1, v2, v3);
                acc3 += dotQ8Block_vec_fma(p3, v0, v1, v2, v3);
                acc4 += dotQ8Block_vec_fma(p4, v0, v1, v2, v3);
                acc5 += dotQ8Block_vec_fma(p5, v0, v1, v2, v3);
                acc6 += dotQ8Block_vec_fma(p6, v0, v1, v2, v3);
                acc7 += dotQ8Block_vec_fma(p7, v0, v1, v2, v3);
            } else {
                acc0 += dotQ8Block_vec(p0, v0, v1, v2, v3);
                acc1 += dotQ8Block_vec(p1, v0, v1, v2, v3);
                acc2 += dotQ8Block_vec(p2, v0, v1, v2, v3);
                acc3 += dotQ8Block_vec(p3, v0, v1, v2, v3);
                acc4 += dotQ8Block_vec(p4, v0, v1, v2, v3);
                acc5 += dotQ8Block_vec(p5, v0, v1, v2, v3);
                acc6 += dotQ8Block_vec(p6, v0, v1, v2, v3);
                acc7 += dotQ8Block_vec(p7, v0, v1, v2, v3);
            }
            in_ptr += QBLOCK;
            p0 += QBYTES;
            p1 += QBYTES;
            p2 += QBYTES;
            p3 += QBYTES;
            p4 += QBYTES;
            p5 += QBYTES;
            p6 += QBYTES;
            p7 += QBYTES;
        }
        out_range[r + 0] = @reduce(.Add, acc0);
        out_range[r + 1] = @reduce(.Add, acc1);
        out_range[r + 2] = @reduce(.Add, acc2);
        out_range[r + 3] = @reduce(.Add, acc3);
        out_range[r + 4] = @reduce(.Add, acc4);
        out_range[r + 5] = @reduce(.Add, acc5);
        out_range[r + 6] = @reduce(.Add, acc6);
        out_range[r + 7] = @reduce(.Add, acc7);
    }
    while (r < nrows) : (r += 1) {
        var p = base_ptr + r * row_stride;
        var in_ptr = input.ptr;
        var acc: SimdF32 = @splat(0);
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const v0 = loadF32x8_unaligned(in_ptr + 0);
            const v1 = loadF32x8_unaligned(in_ptr + 8);
            const v2 = loadF32x8_unaligned(in_ptr + 16);
            const v3 = loadF32x8_unaligned(in_ptr + 24);
            if (use_fma) {
                acc += dotQ8Block_vec_fma(p, v0, v1, v2, v3);
            } else {
                acc += dotQ8Block_vec(p, v0, v1, v2, v3);
            }
            in_ptr += QBLOCK;
            p += QBYTES;
        }
        out_range[r] = @reduce(.Add, acc);
    }
}

fn matVecQ8_0_range_batched8_qin_prequant(out_range: []f32, data: []const u8, qbuf: []const i8, scales: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
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
            const x8: VI8x32 = loadI8x32(qbuf.ptr + b * QBLOCK);
            const sx: f32 = scales[b];
            sum0 += (readF16(p0) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p0 + 2, x8)));
            sum1 += (readF16(p1) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p1 + 2, x8)));
            sum2 += (readF16(p2) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p2 + 2, x8)));
            sum3 += (readF16(p3) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p3 + 2, x8)));
            sum4 += (readF16(p4) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p4 + 2, x8)));
            sum5 += (readF16(p5) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p5 + 2, x8)));
            sum6 += (readF16(p6) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p6 + 2, x8)));
            sum7 += (readF16(p7) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(p7 + 2, x8)));
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
            const x8: VI8x32 = loadI8x32(qbuf.ptr + b * QBLOCK);
            sum += (readF16(p) * scales[b]) * @as(f32, @floatFromInt(dotI8x32_prex(p + 2, x8)));
            p += QBYTES;
        }
        out_range[r] = sum;
    }
}

fn matVecQ8_0_range_pair_batched2_f32(out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, input: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
    const use_fma = g_use_fma;
    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;
    const base_a = data_a.ptr + row_start * row_stride;
    const base_b = data_b.ptr + row_start * row_stride;
    const nrows = row_end - row_start;
    var r: usize = 0;
    while (r + 2 <= nrows) : (r += 2) {
        var pa0 = base_a + (r + 0) * row_stride;
        var pb0 = base_b + (r + 0) * row_stride;
        var pa1 = base_a + (r + 1) * row_stride;
        var pb1 = base_b + (r + 1) * row_stride;
        var in_ptr = input.ptr;
        var aa0: SimdF32 = @splat(0);
        var ab0: SimdF32 = @splat(0);
        var aa1: SimdF32 = @splat(0);
        var ab1: SimdF32 = @splat(0);
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const v0 = loadF32x8_unaligned(in_ptr + 0);
            const v1 = loadF32x8_unaligned(in_ptr + 8);
            const v2 = loadF32x8_unaligned(in_ptr + 16);
            const v3 = loadF32x8_unaligned(in_ptr + 24);
            if (use_fma) {
                aa0 += dotQ8Block_vec_fma(pa0, v0, v1, v2, v3);
                ab0 += dotQ8Block_vec_fma(pb0, v0, v1, v2, v3);
                aa1 += dotQ8Block_vec_fma(pa1, v0, v1, v2, v3);
                ab1 += dotQ8Block_vec_fma(pb1, v0, v1, v2, v3);
            } else {
                aa0 += dotQ8Block_vec(pa0, v0, v1, v2, v3);
                ab0 += dotQ8Block_vec(pb0, v0, v1, v2, v3);
                aa1 += dotQ8Block_vec(pa1, v0, v1, v2, v3);
                ab1 += dotQ8Block_vec(pb1, v0, v1, v2, v3);
            }
            in_ptr += QBLOCK;
            pa0 += QBYTES;
            pb0 += QBYTES;
            pa1 += QBYTES;
            pb1 += QBYTES;
        }
        out_a[r + 0] = @reduce(.Add, aa0);
        out_b[r + 0] = @reduce(.Add, ab0);
        out_a[r + 1] = @reduce(.Add, aa1);
        out_b[r + 1] = @reduce(.Add, ab1);
    }
    while (r < nrows) : (r += 1) {
        var pa = base_a + r * row_stride;
        var pb = base_b + r * row_stride;
        var in_ptr = input.ptr;
        var aa: SimdF32 = @splat(0);
        var ab2: SimdF32 = @splat(0);
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const v0 = loadF32x8_unaligned(in_ptr + 0);
            const v1 = loadF32x8_unaligned(in_ptr + 8);
            const v2 = loadF32x8_unaligned(in_ptr + 16);
            const v3 = loadF32x8_unaligned(in_ptr + 24);
            if (use_fma) {
                aa += dotQ8Block_vec_fma(pa, v0, v1, v2, v3);
                ab2 += dotQ8Block_vec_fma(pb, v0, v1, v2, v3);
            } else {
                aa += dotQ8Block_vec(pa, v0, v1, v2, v3);
                ab2 += dotQ8Block_vec(pb, v0, v1, v2, v3);
            }
            in_ptr += QBLOCK;
            pa += QBYTES;
            pb += QBYTES;
        }
        out_a[r] = @reduce(.Add, aa);
        out_b[r] = @reduce(.Add, ab2);
    }
}

fn matVecQ8_0_range_pair_batched2_qin_prequant(out_a: []f32, data_a: []const u8, out_b: []f32, data_b: []const u8, qbuf: []const i8, scales: []const f32, row_start: usize, row_end: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
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
            const x8: VI8x32 = loadI8x32(qbuf.ptr + b * QBLOCK);
            const sx = scales[b];
            suma += (readF16(pa) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(pa + 2, x8)));
            sumb += (readF16(pb) * sx) * @as(f32, @floatFromInt(dotI8x32_prex(pb + 2, x8)));
            pa += QBYTES;
            pb += QBYTES;
        }
        out_a[r] = suma;
        out_b[r] = sumb;
    }
}

// ============================================================
// Q4_0
// ============================================================

fn matVecQ4_0(out: []f32, data: []const u8, input: []const f32, rows: usize, cols: usize) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 18;
    std.debug.assert(cols % QBLOCK == 0);
    const blocks_per_row = cols / QBLOCK;
    const row_stride = blocks_per_row * QBYTES;
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        var sum: f32 = 0;
        const ptr = data.ptr + row * row_stride;
        var in_ptr = input.ptr;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            sum += dotQ4Block(ptr + b * QBYTES, in_ptr);
            in_ptr += QBLOCK;
        }
        out[row] = sum;
    }
}

inline fn dotQ4Block(blk: [*]const u8, in_ptr: [*]const f32) f32 {
    const scale = readF16(blk);
    var sum: f32 = 0;
    comptime var i: usize = 0;
    inline while (i < 16) : (i += 1) {
        const byte = blk[2 + i];
        const lo: f32 = @floatFromInt(@as(i8, @intCast(byte & 0x0F)) - 8);
        const hi: f32 = @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8);
        sum += lo * in_ptr[i * 2] + hi * in_ptr[i * 2 + 1];
    }
    return sum * scale;
}

// ============================================================
// Embedding
// ============================================================

pub fn embedTokenQ8_0(out: []f32, data: []const u8, token: u32, dim: u32) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 34;
    std.debug.assert(dim % QBLOCK == 0);
    const blocks = dim / QBLOCK;
    const base = data.ptr + @as(usize, token) * @as(usize, blocks) * QBYTES;
    for (0..blocks) |b| {
        const blk = base + b * QBYTES;
        const scale: SimdF32 = @splat(readF16(blk));
        const out_off = b * QBLOCK;
        out[out_off + 0 ..][0..8].* = loadQ8(blk + 2) * scale;
        out[out_off + 8 ..][0..8].* = loadQ8(blk + 10) * scale;
        out[out_off + 16 ..][0..8].* = loadQ8(blk + 18) * scale;
        out[out_off + 24 ..][0..8].* = loadQ8(blk + 26) * scale;
    }
}

pub fn embedTokenQ4_0(out: []f32, data: []const u8, token: u32, dim: u32) void {
    @setRuntimeSafety(false);
    const QBLOCK = 32;
    const QBYTES = 18;
    std.debug.assert(dim % QBLOCK == 0);
    const blocks = dim / QBLOCK;
    const base = data.ptr + @as(usize, token) * @as(usize, blocks) * QBYTES;
    for (0..blocks) |b| {
        const blk = base + b * QBYTES;
        const scale = readF16(blk);
        const out_off = b * QBLOCK;
        comptime var i: usize = 0;
        inline while (i < 16) : (i += 1) {
            const byte = blk[2 + i];
            out[out_off + i * 2] = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0x0F)) - 8)) * scale;
            out[out_off + i * 2 + 1] = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8)) * scale;
        }
    }
}

// ============================================================
// RoPE
// ============================================================

pub fn applyRoPE(q: []f32, k: []f32, pos: usize, head_dim: u32, n_heads: u32, n_kv_heads: u32, rope_theta: f32) void {
    const hd: usize = head_dim;
    const half = hd / 2;
    const pos_f: f32 = @floatFromInt(pos);
    const hd_f: f32 = @floatFromInt(head_dim);
    const step = std.math.pow(f32, rope_theta, -2.0 / hd_f);
    const need: usize = half * 2;

    if (need > 4096) {
        var inv_freq: f32 = 1.0;
        for (0..n_heads) |h| {
            const head = q[h * hd ..][0..hd];
            inv_freq = 1.0;
            for (0..half) |ii| {
                const angle = pos_f * inv_freq;
                const c = @cos(angle);
                const s = @sin(angle);
                const x0 = head[ii];
                const x1 = head[ii + half];
                head[ii] = x0 * c - x1 * s;
                head[ii + half] = x0 * s + x1 * c;
                inv_freq *= step;
            }
        }
        for (0..n_kv_heads) |h| {
            const head = k[h * hd ..][0..hd];
            inv_freq = 1.0;
            for (0..half) |ii| {
                const angle = pos_f * inv_freq;
                const c = @cos(angle);
                const s = @sin(angle);
                const x0 = head[ii];
                const x1 = head[ii + half];
                head[ii] = x0 * c - x1 * s;
                head[ii + half] = x0 * s + x1 * c;
                inv_freq *= step;
            }
        }
        return;
    }

    var cs_storage: [4096]f32 = undefined;
    const cs = cs_storage[0..need];
    var inv_freq: f32 = 1.0;
    for (0..half) |ii| {
        const angle = pos_f * inv_freq;
        cs[ii * 2] = @cos(angle);
        cs[ii * 2 + 1] = @sin(angle);
        inv_freq *= step;
    }
    for (0..n_heads) |h| {
        const head = q[h * hd ..][0..hd];
        for (0..half) |ii| {
            const c = cs[ii * 2];
            const s = cs[ii * 2 + 1];
            const x0 = head[ii];
            const x1 = head[ii + half];
            head[ii] = x0 * c - x1 * s;
            head[ii + half] = x0 * s + x1 * c;
        }
    }
    for (0..n_kv_heads) |h| {
        const head = k[h * hd ..][0..hd];
        for (0..half) |ii| {
            const c = cs[ii * 2];
            const s = cs[ii * 2 + 1];
            const x0 = head[ii];
            const x1 = head[ii + half];
            head[ii] = x0 * c - x1 * s;
            head[ii + half] = x0 * s + x1 * c;
        }
    }
}
