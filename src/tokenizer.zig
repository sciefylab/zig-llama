const std = @import("std");
const builtin = @import("builtin");
const GGUFFile = @import("gguf.zig").GGUFFile;

/// Initialize console for proper UTF-8 output
pub fn initConsole() void {
    if (builtin.os.tag == .windows) {
        const kernel32 = struct {
            extern "kernel32" fn SetConsoleOutputCP(wCodePageID: c_uint) callconv(.winapi) c_int;
        };
        _ = kernel32.SetConsoleOutputCP(65001);
    }
}

pub const Tokenizer = struct {
    vocab: [][]const u8,
    vocab_scores: []f32,
    token_to_id: std.StringHashMap(u32),
    max_token_len: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
    allocator: std.mem.Allocator,
    owns_vocab_scores: bool,

    pub fn init(allocator: std.mem.Allocator, gguf: *const GGUFFile) !Tokenizer {
        initConsole();

        const vocab = gguf.vocab_tokens orelse return error.MissingVocab;

        var owns_scores = false;
        const vocab_scores: []f32 = gguf.vocab_scores orelse blk: {
            owns_scores = true;
            const tmp = try allocator.alloc(f32, vocab.len);
            @memset(tmp, 0.0);
            break :blk tmp;
        };

        var token_to_id = std.StringHashMap(u32).init(allocator);
        errdefer token_to_id.deinit();

        var max_token_len: usize = 0;
        for (vocab, 0..) |token_str, id| {
            try token_to_id.put(token_str, @intCast(id));
            if (token_str.len > max_token_len) max_token_len = token_str.len;
        }

        const bos_id = gguf.getMetadataU32("tokenizer.ggml.bos_token_id") orelse 1;
        const eos_id = gguf.getMetadataU32("tokenizer.ggml.eos_token_id") orelse 2;
        const unk_id =
            gguf.getMetadataU32("tokenizer.ggml.unknown_token_id") orelse
            gguf.getMetadataU32("tokenizer.ggml.unk_token_id") orelse
            0;

        return Tokenizer{
            .vocab = vocab,
            .vocab_scores = vocab_scores,
            .token_to_id = token_to_id,
            .max_token_len = max_token_len,
            .bos_token_id = bos_id,
            .eos_token_id = eos_id,
            .unk_token_id = unk_id,
            .allocator = allocator,
            .owns_vocab_scores = owns_scores,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.token_to_id.deinit();
        if (self.owns_vocab_scores) {
            self.allocator.free(self.vocab_scores);
        }
    }

    pub fn encode(self: *const Tokenizer, text: []const u8, bos: bool, allocator: std.mem.Allocator) ![]u32 {
        var tokens = std.ArrayList(u32){};
        errdefer tokens.deinit(allocator);

        if (bos) {
            try tokens.append(allocator, self.bos_token_id);
        }

        const prefix = "\xe2\x96\x81"; // ▁
        const gpt2_space = "\xc4\xa0"; // Ġ

        var offset: usize = 0;
        var is_word_start = true;

        while (offset < text.len) {
            var best_match_id: u32 = 0;
            var best_match_len: usize = 0;

            if (text[offset] == ' ') {
                is_word_start = true;
                offset += 1;
                continue;
            }

            // Try with SentencePiece prefix (▁)
            if (is_word_start) {
                var prefixed_buf: [128]u8 = undefined;
                @memcpy(prefixed_buf[0..3], prefix);

                const max_check = @min(self.max_token_len, text.len - offset + 3, 125);
                for (1..max_check) |len| {
                    if (offset + len > text.len) break;
                    @memcpy(prefixed_buf[3..][0..len], text[offset..][0..len]);

                    if (self.token_to_id.get(prefixed_buf[0 .. 3 + len])) |id| {
                        best_match_id = id;
                        best_match_len = len;
                    }
                }

                // Try with GPT-2 prefix (Ġ)
                @memcpy(prefixed_buf[0..2], gpt2_space);
                for (1..@min(self.max_token_len, text.len - offset + 2, 126)) |len| {
                    if (offset + len > text.len) break;
                    @memcpy(prefixed_buf[2..][0..len], text[offset..][0..len]);

                    if (self.token_to_id.get(prefixed_buf[0 .. 2 + len])) |id| {
                        if (len > best_match_len) {
                            best_match_id = id;
                            best_match_len = len;
                        }
                    }
                }
            }

            // Try without prefix
            for (1..self.max_token_len + 1) |len| {
                if (offset + len > text.len) break;
                const sub = text[offset..][0..len];
                if (self.token_to_id.get(sub)) |id| {
                    if (len > best_match_len) {
                        best_match_id = id;
                        best_match_len = len;
                    }
                }
            }

            if (best_match_len > 0) {
                try tokens.append(allocator, best_match_id);
                offset += best_match_len;
                is_word_start = false;
            } else {
                const byte = text[offset];
                var hex_buf: [6]u8 = undefined;
                const hex_token = std.fmt.bufPrint(&hex_buf, "<0x{X:0>2}>", .{byte}) catch {
                    offset += 1;
                    continue;
                };

                if (self.token_to_id.get(hex_token)) |id| {
                    try tokens.append(allocator, id);
                } else {
                    try tokens.append(allocator, self.unk_token_id);
                }
                offset += 1;
                is_word_start = false;
            }
        }

        return tokens.toOwnedSlice(allocator);
    }

    pub fn decodeRaw(self: *const Tokenizer, token_id: u32) []const u8 {
        if (token_id < self.vocab.len) return self.vocab[token_id];
        return "<unk>";
    }

    pub fn decode(self: *const Tokenizer, token_id: u32) ?[]const u8 {
        if (token_id == self.bos_token_id) return null;
        if (token_id == self.eos_token_id) return null;
        if (token_id < self.vocab.len) return self.vocab[token_id];
        return null;
    }

    pub fn isSpecialToken(self: *const Tokenizer, token_id: u32) bool {
        return token_id == self.bos_token_id or
            token_id == self.eos_token_id or
            token_id == self.unk_token_id;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.vocab.len;
    }
};

/// Print token with proper character handling for GPT-2/Qwen style tokens
pub fn printToken(token_opt: ?[]const u8) void {
    const token = token_opt orelse return;

    var i: usize = 0;
    while (i < token.len) {
        // =====================================================
        // GPT-2 / Qwen2 Style (2-byte sequences starting with 0xC4)
        // =====================================================

        // Ġ (U+0120, 0xC4 0xA0) -> space
        if (i + 1 < token.len and token[i] == 0xC4 and token[i + 1] == 0xA0) {
            std.debug.print(" ", .{});
            i += 2;
            continue;
        }

        // Ċ (U+010A, 0xC4 0x8A) -> newline
        if (i + 1 < token.len and token[i] == 0xC4 and token[i + 1] == 0x8A) {
            std.debug.print("\n", .{});
            i += 2;
            continue;
        }

        // ĉ (U+0109, 0xC4 0x89) -> tab
        if (i + 1 < token.len and token[i] == 0xC4 and token[i + 1] == 0x89) {
            std.debug.print("\t", .{});
            i += 2;
            continue;
        }

        // Ď (U+010E, 0xC4 0x8E) -> carriage return (sometimes used)
        if (i + 1 < token.len and token[i] == 0xC4 and token[i + 1] == 0x8E) {
            i += 2;
            continue;
        }

        // =====================================================
        // SentencePiece Style: ▁ (U+2581, 0xE2 0x96 0x81) -> space
        // =====================================================
        if (i + 2 < token.len and
            token[i] == 0xE2 and token[i + 1] == 0x96 and token[i + 2] == 0x81)
        {
            std.debug.print(" ", .{});
            i += 3;
            continue;
        }

        // =====================================================
        // Smart quotes and dashes (E2 80 xx)
        // =====================================================
        if (i + 2 < token.len and token[i] == 0xE2 and token[i + 1] == 0x80) {
            const b2 = token[i + 2];

            // ' ' (left/right single quotes)
            if (b2 == 0x98 or b2 == 0x99) {
                std.debug.print("'", .{});
                i += 3;
                continue;
            }
            // " " (left/right double quotes)
            if (b2 == 0x9C or b2 == 0x9D) {
                std.debug.print("\"", .{});
                i += 3;
                continue;
            }
            // — (em dash)
            if (b2 == 0x94) {
                std.debug.print("--", .{});
                i += 3;
                continue;
            }
            // – (en dash)
            if (b2 == 0x93) {
                std.debug.print("-", .{});
                i += 3;
                continue;
            }
            // … (ellipsis)
            if (b2 == 0xA6) {
                std.debug.print("...", .{});
                i += 3;
                continue;
            }

            // Other E2 80 xx - print as-is
            std.debug.print("{c}{c}{c}", .{ token[i], token[i + 1], token[i + 2] });
            i += 3;
            continue;
        }

        // =====================================================
        // General UTF-8 handling
        // =====================================================

        // 4-byte UTF-8 (F0-F4 xx xx xx) - emojis, etc.
        if (i + 3 < token.len and token[i] >= 0xF0 and token[i] <= 0xF4) {
            std.debug.print("{c}{c}{c}{c}", .{ token[i], token[i + 1], token[i + 2], token[i + 3] });
            i += 4;
            continue;
        }

        // 3-byte UTF-8 (E0-EF xx xx)
        if (i + 2 < token.len and token[i] >= 0xE0 and token[i] <= 0xEF) {
            std.debug.print("{c}{c}{c}", .{ token[i], token[i + 1], token[i + 2] });
            i += 3;
            continue;
        }

        // 2-byte UTF-8 (C2-DF xx) - but NOT our special cases above
        if (i + 1 < token.len and token[i] >= 0xC2 and token[i] <= 0xDF) {
            std.debug.print("{c}{c}", .{ token[i], token[i + 1] });
            i += 2;
            continue;
        }

        // =====================================================
        // Byte tokens <0xXX>
        // =====================================================
        if (i + 5 < token.len and
            token[i] == '<' and token[i + 1] == '0' and token[i + 2] == 'x')
        {
            var end_idx = i + 3;
            while (end_idx < token.len and token[end_idx] != '>') {
                end_idx += 1;
            }

            if (end_idx < token.len and end_idx > i + 3) {
                const hex_str = token[i + 3 .. end_idx];
                if (std.fmt.parseInt(u8, hex_str, 16)) |byte_val| {
                    if (byte_val >= 32 and byte_val < 127) {
                        std.debug.print("{c}", .{byte_val});
                    } else if (byte_val == '\n') {
                        std.debug.print("\n", .{});
                    } else if (byte_val == '\t') {
                        std.debug.print("\t", .{});
                    }
                    i = end_idx + 1;
                    continue;
                } else |_| {}
            }
        }

        // =====================================================
        // Regular ASCII
        // =====================================================
        if (token[i] >= 32 and token[i] < 127) {
            std.debug.print("{c}", .{token[i]});
        } else if (token[i] == '\n') {
            std.debug.print("\n", .{});
        } else if (token[i] == '\t') {
            std.debug.print("\t", .{});
        }
        // Skip other control characters

        i += 1;
    }
}

/// Decode and print multiple tokens
pub fn printTokens(tokenizer_inst: *const Tokenizer, token_ids: []const u32) void {
    for (token_ids) |id| {
        printToken(tokenizer_inst.decode(id));
    }
}
