const std = @import("std");

pub const SamplerConfig = struct {
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.1,
    repetition_window: u32 = 64,
    seed: u64 = 0,
};

pub const Sampler = struct {
    config: SamplerConfig,
    vocab_size: u32,
    rng: std.Random.DefaultPrng,
    indices: []u32,
    allocator: std.mem.Allocator,

    // Track recent tokens for repetition penalty
    recent_tokens: []u32,
    recent_count: usize,

    pub fn init(config: SamplerConfig, vocab_size: u32, allocator: std.mem.Allocator) !Sampler {
        const seed = if (config.seed == 0)
            @as(u64, @intCast(std.time.timestamp()))
        else
            config.seed;

        const indices = try allocator.alloc(u32, vocab_size);
        for (indices, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        const recent_tokens = try allocator.alloc(u32, config.repetition_window);
        @memset(recent_tokens, 0);

        return Sampler{
            .config = config,
            .vocab_size = vocab_size,
            .rng = std.Random.DefaultPrng.init(seed),
            .indices = indices,
            .allocator = allocator,
            .recent_tokens = recent_tokens,
            .recent_count = 0,
        };
    }

    pub fn deinit(self: *Sampler) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.recent_tokens);
    }

    /// Reset the sampler state (call between different generations)
    pub fn reset(self: *Sampler) void {
        self.recent_count = 0;
        @memset(self.recent_tokens, 0);
    }

    /// Add token to recent history (for repetition penalty)
    pub fn addToRecent(self: *Sampler, token: u32) void {
        const window = self.config.repetition_window;

        if (self.recent_count < window) {
            self.recent_tokens[self.recent_count] = token;
            self.recent_count += 1;
        } else {
            // Shift left and add new token at end
            for (0..window - 1) |i| {
                self.recent_tokens[i] = self.recent_tokens[i + 1];
            }
            self.recent_tokens[window - 1] = token;
        }
    }

    pub fn sample(self: *Sampler, logits: []f32) u32 {
        // Apply repetition penalty BEFORE temperature
        if (self.config.repetition_penalty != 1.0 and self.recent_count > 0) {
            self.applyRepetitionPenalty(logits);
        }

        const temperature = self.config.temperature;

        // Apply temperature
        if (temperature != 1.0 and temperature > 0) {
            for (logits) |*l| {
                l.* /= temperature;
            }
        }

        // Convert to probabilities (softmax)
        var max_logit: f32 = logits[0];
        for (logits[1..]) |l| {
            max_logit = @max(max_logit, l);
        }

        var sum: f32 = 0.0;
        for (logits) |*l| {
            l.* = @exp(l.* - max_logit);
            sum += l.*;
        }

        if (sum > 0) {
            for (logits) |*l| {
                l.* /= sum;
            }
        }

        // Sample token
        var token: u32 = undefined;
        if (self.config.top_p < 1.0 or self.config.top_k < self.vocab_size) {
            token = self.sampleTopPK(logits);
        } else {
            token = self.sampleFromProbs(logits);
        }

        // Track this token for future repetition penalty
        self.addToRecent(token);

        return token;
    }

    /// Apply repetition penalty to logits
    fn applyRepetitionPenalty(self: *Sampler, logits: []f32) void {
        const penalty = self.config.repetition_penalty;
        const window = @min(self.recent_count, self.config.repetition_window);

        // Use a simple array to track seen tokens and counts
        // This avoids HashMap allocation issues
        for (self.recent_tokens[0..window]) |token| {
            if (token < logits.len) {
                if (logits[token] > 0) {
                    logits[token] /= penalty;
                } else {
                    logits[token] *= penalty;
                }
            }
        }
    }

    /// Top-p and Top-k sampling combined
    fn sampleTopPK(self: *Sampler, probs: []f32) u32 {
        const n = @min(self.config.top_k, @as(u32, @intCast(probs.len)));

        // Reset indices
        for (self.indices[0..n], 0..) |*idx, i| {
            idx.* = @intCast(i);
        }

        // Partial selection sort to find top-k
        for (0..n) |i| {
            var max_idx = i;
            var max_prob = probs[self.indices[i]];

            for (i + 1..probs.len) |j| {
                const j_idx: u32 = @intCast(j);
                if (probs[j_idx] > max_prob) {
                    max_idx = j;
                    max_prob = probs[j_idx];
                }
            }

            if (max_idx != i) {
                // Swap in indices
                const tmp = self.indices[i];
                self.indices[i] = @intCast(max_idx);
                if (max_idx < n) {
                    self.indices[max_idx] = tmp;
                }
            }
        }

        // Find cutoff for top-p
        var cumsum: f32 = 0.0;
        var cutoff: usize = 0;

        for (0..n) |i| {
            cumsum += probs[self.indices[i]];
            cutoff = i + 1;
            if (cumsum >= self.config.top_p) break;
        }

        // Ensure at least one token
        if (cutoff == 0) cutoff = 1;

        // Renormalize
        var new_sum: f32 = 0.0;
        for (0..cutoff) |i| {
            new_sum += probs[self.indices[i]];
        }

        if (new_sum == 0) return self.indices[0];

        // Sample from truncated distribution
        const r = self.rng.random().float(f32) * new_sum;
        cumsum = 0.0;

        for (0..cutoff) |i| {
            cumsum += probs[self.indices[i]];
            if (cumsum >= r) {
                return self.indices[i];
            }
        }

        return self.indices[0];
    }

    fn sampleFromProbs(self: *Sampler, probs: []const f32) u32 {
        const r = self.rng.random().float(f32);
        var cumsum: f32 = 0.0;

        for (probs, 0..) |p, i| {
            cumsum += p;
            if (cumsum >= r) {
                return @intCast(i);
            }
        }

        return @intCast(probs.len - 1);
    }

    /// Greedy sampling (argmax) - for deterministic output
    pub fn sampleGreedy(self: *Sampler, logits: []const f32) u32 {
        _ = self;
        var max_idx: u32 = 0;
        var max_val: f32 = logits[0];

        for (logits[1..], 1..) |l, i| {
            if (l > max_val) {
                max_val = l;
                max_idx = @intCast(i);
            }
        }

        return max_idx;
    }
};
