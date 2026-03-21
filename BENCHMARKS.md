# Nous Performance Benchmarks

All benchmarks measured on AMD Ryzen 7 5800H (16 threads), Go 1.22, Linux.

## Cognitive Pipeline

| Operation | Time/op | Allocs/op | Memory/op |
|-----------|---------|-----------|-----------|
| CompressStep (read, 100 lines) | 8.06 μs | 5 | 3,763 B |
| CompressStep (grep, 50 matches) | 7.79 μs | 25 | 8,154 B |
| CompressStep (glob, 30 files) | 1.14 μs | 8 | 1,521 B |
| Pipeline: add 5 steps | 43.1 μs | 24 | 19,104 B |
| Pipeline: build context (10 steps) | 1.88 μs | 25 | 2,361 B |

**Takeaway**: The entire pipeline overhead per reasoning step is <50μs — negligible compared to even the fastest cognitive generation paths.

## Anti-Hallucination (Grounding)

| Operation | Time/op | Allocs/op | Memory/op |
|-----------|---------|-----------|-----------|
| Token estimation | 0.54 ns | 0 | 0 B |
| SmartTruncate (short file) | 54.7 ns | 1 | 48 B |
| SmartTruncate (200-line file) | 7.48 μs | 31 | 16,285 B |
| Reflection gate check | 418 ns | 5 | 144 B |

**Takeaway**: Token estimation is essentially free (0.5ns, zero allocation). The reflection gate adds 418ns per tool call — invisible to the user.

## Query Classification

| Query Type | Time/op | Allocs/op | Memory/op |
|------------|---------|-----------|-----------|
| Fast path ("hello!") | 2.67 μs | 0 | 0 B |
| Full path ("read file main.go") | 4.30 μs | 0 | 0 B |
| Medium path ("explain GC") | 41.7 μs | 1 | 113 B |
| Worst case (no pattern match) | 96.8 μs | 1 | 291 B |

**Takeaway**: Fast queries are classified in <3μs with zero allocations. Even the worst case (checking all 30+ regex patterns) takes <100μs. The classifier routes simple queries to instant responses.

## Diff Algorithm (LCS)

| Input Size | Time/op | Allocs/op | Memory/op |
|------------|---------|-----------|-----------|
| 5 lines | 655 ns | 11 | 1,040 B |
| 50 lines (10 changes) | 17.6 μs | 59 | 28,288 B |
| 200 lines (20 changes) | 195 μs | 211 | 389,281 B |
| Full diff preview (small) | 2.55 μs | 33 | 2,249 B |

**Takeaway**: The O(nm) LCS algorithm handles typical diffs (50 lines) in 18μs. For very large files (200+ lines), the diff preview auto-truncates at 30 lines to keep display fast.

## Recipe Matching

| Operation | Time/op | Allocs/op | Memory/op |
|-----------|---------|-----------|-----------|
| Keyword extraction | 1.73 μs | 7 | 1,245 B |
| Keyword overlap | 120 ns | 0 | 0 B |
| Recipe match (40 recipes) | 79.9 μs | 13 | 24,718 B |

**Takeaway**: Matching against 40 learned recipes takes 80μs. With the 50-recipe cap and pruning, this stays well under 1ms.

## Prediction Cache

| Operation | Time/op | Allocs/op | Memory/op |
|-----------|---------|-----------|-----------|
| Cache key generation | 531 ns | 7 | 280 B |
| Cache lookup (20 entries) | 519 ns | 6 | 116 B |
| SHA256 short hash | 728 ns | 4 | 768 B |

**Takeaway**: Predictor lookups take 519ns. On a cache hit, this saves 50-500ms of actual tool execution. Even with a modest 30% hit rate, the predictor provides meaningful speedup.

## Utility Functions

| Operation | Time/op | Allocs/op | Memory/op |
|-----------|---------|-----------|-----------|
| isReadOnly (tool check) | 1.70 ns | 0 | 0 B |
| looksLikeFile (path check) | 9.46 ns | 0 | 0 B |

**Takeaway**: These hot-path helpers are essentially free, measured in single-digit nanoseconds.

## Summary

| Category | Typical Latency |
|----------|----------------|
| Token estimation | <1 ns |
| Tool classification | <10 ns |
| Cache lookup | ~500 ns |
| Grounding check | ~400 ns |
| Query classification | ~3 μs |
| Step compression | ~8 μs |
| Pipeline overhead | ~50 μs |
| Recipe matching | ~80 μs |
| Diff (50 lines) | ~18 μs |
| **Total overhead per step** | **~150 μs** |

The cognitive architecture overhead is approximately **150 μs per step**. All the innovation — grounding, prediction, recipes, classification — runs in microseconds, enabling pure cognitive processing without any external dependencies.

## Running Benchmarks

```bash
# All benchmarks
go test ./internal/cognitive/ -bench=. -benchmem

# Specific benchmark
go test ./internal/cognitive/ -bench=BenchmarkPipelineAddStep -benchmem -count=5

# With CPU profiling
go test ./internal/cognitive/ -bench=. -cpuprofile=cpu.prof
go tool pprof cpu.prof

# With memory profiling
go test ./internal/cognitive/ -bench=. -memprofile=mem.prof
go tool pprof mem.prof
```
