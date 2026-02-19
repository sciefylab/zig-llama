const std = @import("std");

pub const ValidationError = error{
    EmptyPrompt,
    PromptTooLong,
    InvalidUtf8,
    InvalidToken,
    TokenOutOfRange,
    ContextLengthExceeded,
    InvalidModelPath,
    FileNotFound,
    InvalidTemperature,
    InvalidTopP,
    InvalidTopK,
    InvalidRepetitionPenalty,
    InvalidMaxTokens,
    NullPointer,
};

pub const ValidationConfig = struct {
    max_prompt_length: usize = 100_000, // 100KB max prompt
    max_prompt_tokens: usize = 32_000, // Max tokens in prompt
    min_temperature: f32 = 0.0,
    max_temperature: f32 = 2.0,
    min_top_p: f32 = 0.0,
    max_top_p: f32 = 1.0,
    min_top_k: u32 = 1,
    max_top_k: u32 = 100_000,
    min_repetition_penalty: f32 = 0.0,
    max_repetition_penalty: f32 = 10.0,
    max_generate_tokens: u32 = 32_000,
};

pub const Validator = struct {
    config: ValidationConfig,

    pub fn init(config: ValidationConfig) Validator {
        return .{ .config = config };
    }

    pub fn initDefault() Validator {
        return .{ .config = .{} };
    }

    // =========================================================================
    // Prompt Validation
    // =========================================================================

    pub fn validatePrompt(self: *const Validator, prompt: []const u8) ValidationError!void {
        // Check null/empty
        if (prompt.len == 0) {
            return ValidationError.EmptyPrompt;
        }

        // Check length
        if (prompt.len > self.config.max_prompt_length) {
            return ValidationError.PromptTooLong;
        }

        // Check UTF-8 validity
        if (!std.unicode.utf8ValidateSlice(prompt)) {
            return ValidationError.InvalidUtf8;
        }
    }

    pub fn validatePromptTokens(self: *const Validator, token_count: usize, max_context: usize) ValidationError!void {
        if (token_count == 0) {
            return ValidationError.EmptyPrompt;
        }

        if (token_count > self.config.max_prompt_tokens) {
            return ValidationError.PromptTooLong;
        }

        if (token_count >= max_context) {
            return ValidationError.ContextLengthExceeded;
        }
    }

    // =========================================================================
    // Token Validation
    // =========================================================================

    pub fn validateToken(self: *const Validator, token: u32, vocab_size: u32) ValidationError!void {
        _ = self;
        if (token >= vocab_size) {
            return ValidationError.TokenOutOfRange;
        }
    }

    pub fn validatePosition(self: *const Validator, pos: usize, max_seq_len: usize) ValidationError!void {
        _ = self;
        if (pos >= max_seq_len) {
            return ValidationError.ContextLengthExceeded;
        }
    }

    // =========================================================================
    // Sampler Config Validation
    // =========================================================================

    pub fn validateTemperature(self: *const Validator, temp: f32) ValidationError!void {
        if (std.math.isNan(temp) or std.math.isInf(temp)) {
            return ValidationError.InvalidTemperature;
        }
        if (temp < self.config.min_temperature or temp > self.config.max_temperature) {
            return ValidationError.InvalidTemperature;
        }
    }

    pub fn validateTopP(self: *const Validator, top_p: f32) ValidationError!void {
        if (std.math.isNan(top_p) or std.math.isInf(top_p)) {
            return ValidationError.InvalidTopP;
        }
        if (top_p < self.config.min_top_p or top_p > self.config.max_top_p) {
            return ValidationError.InvalidTopP;
        }
    }

    pub fn validateTopK(self: *const Validator, top_k: u32) ValidationError!void {
        if (top_k < self.config.min_top_k or top_k > self.config.max_top_k) {
            return ValidationError.InvalidTopK;
        }
    }

    pub fn validateRepetitionPenalty(self: *const Validator, penalty: f32) ValidationError!void {
        if (std.math.isNan(penalty) or std.math.isInf(penalty)) {
            return ValidationError.InvalidRepetitionPenalty;
        }
        if (penalty < self.config.min_repetition_penalty or penalty > self.config.max_repetition_penalty) {
            return ValidationError.InvalidRepetitionPenalty;
        }
    }

    pub fn validateMaxTokens(self: *const Validator, max_tokens: u32) ValidationError!void {
        if (max_tokens == 0 or max_tokens > self.config.max_generate_tokens) {
            return ValidationError.InvalidMaxTokens;
        }
    }

    // =========================================================================
    // File Path Validation
    // =========================================================================

    pub fn validateModelPath(self: *const Validator, path: []const u8) ValidationError!void {
        _ = self;

        if (path.len == 0) {
            return ValidationError.InvalidModelPath;
        }

        // Check if file exists
        std.fs.cwd().access(path, .{}) catch {
            return ValidationError.FileNotFound;
        };

        // Check extension
        if (!std.mem.endsWith(u8, path, ".gguf")) {
            return ValidationError.InvalidModelPath;
        }
    }

    // =========================================================================
    // Pointer Validation
    // =========================================================================

    pub fn validatePtr(self: *const Validator, ptr: anytype) ValidationError!void {
        _ = self;
        const T = @TypeOf(ptr);
        const info = @typeInfo(T);

        if (info == .Pointer) {
            if (@intFromPtr(ptr) == 0) {
                return ValidationError.NullPointer;
            }
        } else if (info == .Optional) {
            if (ptr == null) {
                return ValidationError.NullPointer;
            }
        }
    }

    // =========================================================================
    // Batch Validation
    // =========================================================================

    pub const SamplerParams = struct {
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        max_tokens: u32,
    };

    pub fn validateSamplerParams(self: *const Validator, params: SamplerParams) ValidationError!void {
        try self.validateTemperature(params.temperature);
        try self.validateTopP(params.top_p);
        try self.validateTopK(params.top_k);
        try self.validateRepetitionPenalty(params.repetition_penalty);
        try self.validateMaxTokens(params.max_tokens);
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Validate and sanitize prompt - returns sanitized version
pub fn sanitizePrompt(allocator: std.mem.Allocator, prompt: []const u8, max_len: usize) ![]u8 {
    const validator = Validator.initDefault();
    try validator.validatePrompt(prompt);

    // Truncate if needed
    const len = @min(prompt.len, max_len);

    // Find valid UTF-8 boundary
    var valid_len = len;
    while (valid_len > 0 and !isUtf8Start(prompt[valid_len - 1])) {
        valid_len -= 1;
    }

    const result = try allocator.alloc(u8, valid_len);
    @memcpy(result, prompt[0..valid_len]);

    return result;
}

fn isUtf8Start(byte: u8) bool {
    // UTF-8 continuation bytes start with 10xxxxxx
    return (byte & 0xC0) != 0x80;
}

/// Get human-readable error message
pub fn errorMessage(err: ValidationError) []const u8 {
    return switch (err) {
        ValidationError.EmptyPrompt => "Prompt cannot be empty",
        ValidationError.PromptTooLong => "Prompt exceeds maximum length",
        ValidationError.InvalidUtf8 => "Prompt contains invalid UTF-8 characters",
        ValidationError.InvalidToken => "Invalid token encountered",
        ValidationError.TokenOutOfRange => "Token ID exceeds vocabulary size",
        ValidationError.ContextLengthExceeded => "Context length exceeded",
        ValidationError.InvalidModelPath => "Invalid model file path",
        ValidationError.FileNotFound => "Model file not found",
        ValidationError.InvalidTemperature => "Temperature must be between 0.0 and 2.0",
        ValidationError.InvalidTopP => "top_p must be between 0.0 and 1.0",
        ValidationError.InvalidTopK => "top_k must be at least 1",
        ValidationError.InvalidRepetitionPenalty => "Repetition penalty out of valid range",
        ValidationError.InvalidMaxTokens => "max_tokens must be greater than 0",
        ValidationError.NullPointer => "Null pointer encountered",
    };
}

// =============================================================================
// Tests
// =============================================================================

test "validate empty prompt" {
    const validator = Validator.initDefault();
    try std.testing.expectError(ValidationError.EmptyPrompt, validator.validatePrompt(""));
}

test "validate valid prompt" {
    const validator = Validator.initDefault();
    try validator.validatePrompt("Hello, world!");
}

test "validate invalid utf8" {
    const validator = Validator.initDefault();
    const invalid_utf8 = &[_]u8{ 0xFF, 0xFE, 0x00 };
    try std.testing.expectError(ValidationError.InvalidUtf8, validator.validatePrompt(invalid_utf8));
}

test "validate temperature" {
    const validator = Validator.initDefault();

    try validator.validateTemperature(0.7);
    try validator.validateTemperature(0.0);
    try validator.validateTemperature(2.0);

    try std.testing.expectError(ValidationError.InvalidTemperature, validator.validateTemperature(-0.1));
    try std.testing.expectError(ValidationError.InvalidTemperature, validator.validateTemperature(2.1));
    try std.testing.expectError(ValidationError.InvalidTemperature, validator.validateTemperature(std.math.nan(f32)));
}

test "validate top_p" {
    const validator = Validator.initDefault();

    try validator.validateTopP(0.9);
    try validator.validateTopP(0.0);
    try validator.validateTopP(1.0);

    try std.testing.expectError(ValidationError.InvalidTopP, validator.validateTopP(-0.1));
    try std.testing.expectError(ValidationError.InvalidTopP, validator.validateTopP(1.1));
}

test "validate token" {
    const validator = Validator.initDefault();
    const vocab_size: u32 = 32000;

    try validator.validateToken(0, vocab_size);
    try validator.validateToken(31999, vocab_size);

    try std.testing.expectError(ValidationError.TokenOutOfRange, validator.validateToken(32000, vocab_size));
    try std.testing.expectError(ValidationError.TokenOutOfRange, validator.validateToken(50000, vocab_size));
}
