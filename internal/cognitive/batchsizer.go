package cognitive

import (
	"runtime"
	"strings"
	"sync"
	"time"
)

// AdaptiveBatchSizer dynamically adjusts LLM call parameters based on:
//   - Query complexity (simple vs complex)
//   - Available system memory
//   - Recent response times (to detect GPU pressure)
//   - Token budget from VirtualContext
//
// This prevents wasted compute on simple queries (small batches)
// and ensures complex queries get enough tokens (large batches).
type AdaptiveBatchSizer struct {
	mu sync.Mutex

	// Latency tracking: exponential moving average of response times
	avgLatencyMs float64
	latencySamples int

	// Memory pressure tracking
	lastMemCheck   time.Time
	memPressure    float64 // 0.0 = free, 1.0 = full
	memCheckPeriod time.Duration

	// Configuration
	baseNumPredict int
	baseNumCtx     int
	minNumPredict  int
	maxNumPredict  int
}

// BatchParams are the adjusted parameters for an LLM call.
type BatchParams struct {
	NumPredict int
	NumCtx     int
	Temperature float64
}

// NewAdaptiveBatchSizer creates a new adaptive batch sizer.
func NewAdaptiveBatchSizer() *AdaptiveBatchSizer {
	return &AdaptiveBatchSizer{
		baseNumPredict: 1024,
		baseNumCtx:     8192,
		minNumPredict:  128,
		maxNumPredict:  2048,
		memCheckPeriod: 10 * time.Second,
	}
}

// ParamsForQuery returns optimized batch parameters for a query.
func (abs *AdaptiveBatchSizer) ParamsForQuery(query string, path QueryPath) BatchParams {
	abs.mu.Lock()
	defer abs.mu.Unlock()

	params := BatchParams{
		NumPredict:  abs.baseNumPredict,
		NumCtx:      abs.baseNumCtx,
		Temperature: 0.7,
	}

	// Adjust based on query path
	switch path {
	case PathFast:
		params.NumPredict = 256
		params.NumCtx = 4096
		params.Temperature = 0.7
	case PathMedium:
		params.NumPredict = 768
		params.NumCtx = 8192
		params.Temperature = 0.6
	case PathFull:
		params.NumPredict = 1024
		params.NumCtx = 8192
		params.Temperature = 0.5
	}

	// Adjust for query complexity
	complexity := estimateComplexity(query)
	if complexity > 0.8 {
		params.NumPredict = min(params.NumPredict*2, abs.maxNumPredict)
	} else if complexity < 0.3 {
		params.NumPredict = max(params.NumPredict/2, abs.minNumPredict)
	}

	// Adjust for memory pressure
	abs.checkMemoryPressure()
	if abs.memPressure > 0.8 {
		params.NumPredict = max(params.NumPredict/2, abs.minNumPredict)
		params.NumCtx = max(params.NumCtx/2, 2048)
	}

	// Adjust for latency pressure: if recent calls are slow, reduce batch size
	if abs.avgLatencyMs > 5000 && abs.latencySamples > 3 {
		params.NumPredict = max(params.NumPredict*3/4, abs.minNumPredict)
	}

	return params
}

// RecordLatency records the latency of an LLM call for adaptive tuning.
func (abs *AdaptiveBatchSizer) RecordLatency(d time.Duration) {
	abs.mu.Lock()
	defer abs.mu.Unlock()

	ms := float64(d.Milliseconds())
	const alpha = 0.3
	if abs.latencySamples == 0 {
		abs.avgLatencyMs = ms
	} else {
		abs.avgLatencyMs = alpha*ms + (1-alpha)*abs.avgLatencyMs
	}
	abs.latencySamples++
}

// Stats returns current adaptive batch sizing stats.
func (abs *AdaptiveBatchSizer) Stats() (avgLatencyMs float64, memPressure float64, samples int) {
	abs.mu.Lock()
	defer abs.mu.Unlock()
	return abs.avgLatencyMs, abs.memPressure, abs.latencySamples
}

// checkMemoryPressure samples system memory usage.
func (abs *AdaptiveBatchSizer) checkMemoryPressure() {
	if time.Since(abs.lastMemCheck) < abs.memCheckPeriod {
		return
	}
	abs.lastMemCheck = time.Now()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Estimate pressure from Go heap (rough proxy)
	// Consider high pressure if Go heap > 500MB
	heapMB := float64(m.HeapAlloc) / (1024 * 1024)
	if heapMB > 500 {
		abs.memPressure = heapMB / 1000
		if abs.memPressure > 1.0 {
			abs.memPressure = 1.0
		}
	} else {
		abs.memPressure = 0
	}
}

// estimateComplexity returns a rough complexity score for a query (0.0-1.0).
func estimateComplexity(query string) float64 {
	score := 0.0
	words := strings.Fields(query)
	wordCount := len(words)

	// Length-based
	if wordCount > 30 {
		score += 0.3
	} else if wordCount > 15 {
		score += 0.2
	} else if wordCount > 8 {
		score += 0.1
	}

	lower := strings.ToLower(query)

	// Complexity markers
	complexMarkers := []string{
		"explain", "analyze", "compare", "implement", "refactor",
		"debug", "optimize", "design", "architecture", "step by step",
		"pros and cons", "trade-offs", "differences between",
	}
	for _, marker := range complexMarkers {
		if strings.Contains(lower, marker) {
			score += 0.2
			break
		}
	}

	// Multiple questions
	if strings.Count(query, "?") > 1 {
		score += 0.2
	}

	// Code-related
	codeMarkers := []string{".go", ".py", ".js", ".ts", "function", "class", "struct", "interface"}
	for _, marker := range codeMarkers {
		if strings.Contains(lower, marker) {
			score += 0.1
			break
		}
	}

	// Multi-step indicators
	multiStepMarkers := []string{"first", "then", "after that", "finally", "and also", "additionally"}
	for _, marker := range multiStepMarkers {
		if strings.Contains(lower, marker) {
			score += 0.15
			break
		}
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}
