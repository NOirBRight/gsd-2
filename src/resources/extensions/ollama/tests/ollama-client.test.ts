// GSD2 — Tests for Ollama HTTP client
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { envPositiveInt, getOllamaHost } from "../ollama-client.js";

// ─── getOllamaHost ──────────────────────────────────────────────────────────

describe("getOllamaHost", () => {
	const originalHost = process.env.OLLAMA_HOST;

	afterEach(() => {
		if (originalHost === undefined) {
			delete process.env.OLLAMA_HOST;
		} else {
			process.env.OLLAMA_HOST = originalHost;
		}
	});

	it("returns default when OLLAMA_HOST is not set", () => {
		delete process.env.OLLAMA_HOST;
		assert.equal(getOllamaHost(), "http://localhost:11434");
	});

	it("returns OLLAMA_HOST when set with scheme", () => {
		process.env.OLLAMA_HOST = "http://myhost:12345";
		assert.equal(getOllamaHost(), "http://myhost:12345");
	});

	it("adds http:// when OLLAMA_HOST has no scheme", () => {
		process.env.OLLAMA_HOST = "myhost:12345";
		assert.equal(getOllamaHost(), "http://myhost:12345");
	});

	it("preserves https:// scheme", () => {
		process.env.OLLAMA_HOST = "https://secure-ollama.example.com";
		assert.equal(getOllamaHost(), "https://secure-ollama.example.com");
	});
});

// ─── envPositiveInt ─────────────────────────────────────────────────────────

describe("envPositiveInt", () => {
	const KEY = "TEST_OLLAMA_ENV_POSITIVE_INT";

	beforeEach(() => {
		delete process.env[KEY];
	});

	afterEach(() => {
		delete process.env[KEY];
	});

	it("returns fallback when env var is unset", () => {
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("returns fallback when env var is empty string", () => {
		process.env[KEY] = "";
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("returns parsed positive integer when valid", () => {
		process.env[KEY] = "5000";
		assert.equal(envPositiveInt(KEY, 1500), 5000);
	});

	it("floors fractional values", () => {
		process.env[KEY] = "5000.9";
		assert.equal(envPositiveInt(KEY, 1500), 5000);
	});

	it("returns fallback for non-numeric input", () => {
		process.env[KEY] = "abc";
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("returns fallback for zero", () => {
		process.env[KEY] = "0";
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("returns fallback for negative numbers", () => {
		process.env[KEY] = "-100";
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("returns fallback for Infinity", () => {
		process.env[KEY] = "Infinity";
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("returns fallback for NaN string", () => {
		process.env[KEY] = "NaN";
		assert.equal(envPositiveInt(KEY, 1500), 1500);
	});

	it("handles whitespace by parsing it as numeric", () => {
		// Number("  5000  ") returns 5000 — current behavior, document it.
		process.env[KEY] = "  5000  ";
		assert.equal(envPositiveInt(KEY, 1500), 5000);
	});
});
