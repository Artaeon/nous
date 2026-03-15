# Testing Nous

Nous has a comprehensive test suite covering every subsystem with unit tests, integration tests, benchmarks, fuzz tests, and race condition detection.

## Quick Start

```bash
make test            # Run all tests
make test-v          # Verbose output
make test-race       # With race detector
make test-short      # Quick subset
```

## Test Statistics

| Metric | Value |
|--------|-------|
| Total test cases | **1,161** |
| Test files | **73** |
| Test lines of code | **19,643+** |
| Race conditions | **0** |
| External dependencies | **0** |
| Fuzz targets | **7** |
| Benchmark functions | **25** |

## Test Architecture

### Unit Tests (1,100+ tests)

Every package has unit tests covering:
- **Cognitive pipeline**: Pipeline compression, step management, context building
- **Grounding system**: Token budget estimation, smart truncation, validation, reflection gate
- **Query classifier**: 3-tier routing (fast/medium/full), pattern matching, edge cases
- **Diff algorithm**: LCS edit script, hunk building, ANSI coloring, truncation
- **Tool safety**: Dangerous tool classification, confirmation flow, auto-approve
- **Persona & identity**: Core traits, privacy guarantees, self-knowledge formatting
- **Predictor**: Cache hits/misses, expiry, eviction, concurrent access, prediction strategies
- **Recipes**: Recording, deduplication, matching, replay, parameter substitution, pruning
- **Memory systems**: Working memory decay, long-term persistence, episodic search
- **Tools**: All 18 built-in tools, browser automation, HTML parsing
- **Training**: Data collection, quality filtering, export formats, auto-tuning
- **Server**: HTTP endpoints, fast path routing, job management
- **Sentinel**: inotify watcher, event debouncing, recursive monitoring
- **Blackboard**: Pub/sub messaging, goal management
- **Sandbox**: Policy enforcement, resource limits, audit logging

### Benchmarks (25 functions)

Run with: `go test ./internal/cognitive/ -bench=. -benchmem`

Key results (AMD Ryzen 7 5800H):

| Operation | Time | Allocs |
|-----------|------|--------|
| Token estimation | 0.5 ns/op | 0 |
| isReadOnly check | 1.7 ns/op | 0 |
| looksLikeFile check | 9.5 ns/op | 0 |
| SmartTruncate (short) | 55 ns/op | 1 |
| Reflection gate check | 418 ns/op | 5 |
| Cache key generation | 531 ns/op | 7 |
| Predictor lookup | 519 ns/op | 6 |
| LCS diff (5 lines) | 655 ns/op | 11 |
| Keyword overlap | 120 ns/op | 0 |
| Glob compression | 1.1 μs/op | 8 |
| Keyword extraction | 1.7 μs/op | 7 |
| Pipeline context build | 1.9 μs/op | 25 |
| Diff preview (small) | 2.5 μs/op | 33 |
| Query classify (fast) | 2.7 μs/op | 0 |
| Query classify (full) | 4.3 μs/op | 0 |
| Read compression | 8.1 μs/op | 5 |
| LCS diff (50 lines) | 17.6 μs/op | 59 |
| Pipeline 5 steps | 43 μs/op | 24 |
| Query classify (worst) | 97 μs/op | 1 |
| Recipe match (40 recipes) | 80 μs/op | 13 |
| LCS diff (200 lines) | 195 μs/op | 211 |

**Key insight**: All hot-path operations (token estimation, tool classification, cache lookups) are sub-microsecond. The cognitive pipeline adds <50μs overhead per step — negligible compared to LLM inference time.

### Fuzz Tests (7 targets)

Run with: `go test -fuzz=FuzzName -fuzztime=30s`

| Target | What it tests | Invariants checked |
|--------|--------------|-------------------|
| `FuzzCompressStep` | Pipeline compression | No panics on arbitrary tool/result pairs |
| `FuzzSmartTruncate` | Result truncation | Hard limit (2048 chars) always enforced |
| `FuzzClassifyQuery` | Query routing | Always returns valid path; IsSimple consistency |
| `FuzzExtractKeywords` | Keyword extraction | All keywords ≥ 3 characters |
| `FuzzDiffPreview` | Diff generation | Identical content → empty diff |
| `FuzzKeywordOverlap` | Similarity metric | Always in [0.0, 1.0] range |
| `FuzzSplitLines` | Line splitting | No embedded newlines in output |

Verified: **200,000+ executions, zero crashes** across all fuzz targets.

### Race Condition Detection

All tests pass with Go's race detector (`-race` flag):

```bash
go test ./... -race -count=1
# All 19 packages pass, 0 data races
```

The predictor specifically has concurrent access tests that exercise simultaneous read/write operations on the prediction cache with the race detector enabled.

### Integration Tests

- `cmd/nous/main_test.go` — REPL command parsing, flag handling
- `cmd/nous/integration_test.go` — Full system initialization
- `internal/compress/atoms_test.go` — Mock Ollama server integration
- `internal/server/server_test.go` — HTTP endpoint integration

## Coverage by Package

| Package | Test Coverage | Key Tests |
|---------|--------------|-----------|
| cognitive/ | High | 28 test files, benchmarks, fuzz tests |
| tools/ | High | 1,478 lines of tool tests |
| memory/ | High | Working, long-term, project, episodic |
| ollama/ | High | 707 lines, mock server tests |
| training/ | High | Collection, auto-tune, modelfile |
| server/ | Good | HTTP endpoints, CORS, jobs |
| hands/ | Good | 11 test files, runners, webhooks |
| channels/ | Good | Discord, Telegram, Matrix mocks |
| sentinel/ | Good | inotify, event processing |
| sandbox/ | Good | Policy, limits, audit |
| index/ | Good | AST parsing, incremental updates |
| blackboard/ | Good | Pub/sub, goals |
| compress/ | Good | Atom parsing, relevance scoring |

## Writing New Tests

Follow these patterns used throughout the codebase:

```go
// Table-driven tests
func TestFoo(t *testing.T) {
    tests := []struct {
        name  string
        input string
        want  string
    }{
        {"basic", "hello", "HELLO"},
        {"empty", "", ""},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Foo(tt.input)
            if got != tt.want {
                t.Errorf("Foo(%q) = %q, want %q", tt.input, got, tt.want)
            }
        })
    }
}
```

- Use `t.TempDir()` for filesystem tests
- Use `httptest.NewServer` for Ollama mock servers
- Use `t.Run` for sub-tests
- Test both success and error paths
- Include edge cases: empty input, nil values, boundary conditions

## CI/CD

GitHub Actions runs on every push:
1. `go fmt` — formatting check
2. `go vet` — static analysis
3. `go test ./...` — all unit tests with coverage
4. `go test -race` — race condition detection
5. Coverage artifact upload
