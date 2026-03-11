package cognitive

import (
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// Prediction represents a speculative pre-computed result.
type Prediction struct {
	ToolName  string
	Args      map[string]string
	Result    string
	CreatedAt time.Time
}

// Predictor speculatively pre-computes likely follow-up tool calls during idle time.
// When the user asks a question, the predictor guesses what tools the reasoner
// will need next and caches their results. If the prediction hits, the reasoner
// gets an instant response instead of waiting for tool execution.
//
// Prediction strategies:
// 1. Adjacent files: if user reads X.go, pre-read X_test.go
// 2. Directory context: if user reads a file, pre-list its directory
// 3. Definition lookup: if user greps for a symbol, pre-read the defining file
// 4. Recent patterns: repeat the last tool sequence on related inputs
type Predictor struct {
	mu      sync.RWMutex
	cache   map[string]Prediction // key: "tool:args_hash"
	tools   *tools.Registry
	maxSize int

	// Stats
	hits   int
	misses int
}

// NewPredictor creates a predictor backed by the tool registry.
func NewPredictor(toolReg *tools.Registry) *Predictor {
	return &Predictor{
		cache:   make(map[string]Prediction),
		tools:   toolReg,
		maxSize: 20,
	}
}

// Predict analyzes the current context and speculatively executes likely follow-up tools.
// This is called after each tool execution to warm the cache for the next step.
// Non-blocking: runs predictions concurrently.
func (p *Predictor) Predict(lastTool string, lastArgs map[string]string, lastResult string) {
	predictions := p.generatePredictions(lastTool, lastArgs, lastResult)

	for _, pred := range predictions {
		pred := pred
		go func() {
			tool, err := p.tools.Get(pred.ToolName)
			if err != nil {
				return
			}

			// Only pre-compute read-only tools
			if !isReadOnly(pred.ToolName) {
				return
			}

			result, err := tool.Execute(pred.Args)
			if err != nil {
				return
			}

			key := cacheKey(pred.ToolName, pred.Args)
			p.mu.Lock()
			// Evict oldest if at capacity
			if len(p.cache) >= p.maxSize {
				p.evictOldest()
			}
			p.cache[key] = Prediction{
				ToolName:  pred.ToolName,
				Args:      pred.Args,
				Result:    result,
				CreatedAt: time.Now(),
			}
			p.mu.Unlock()
		}()
	}
}

// Lookup checks if a tool call has been pre-computed.
// Returns the cached result and true if found, empty string and false otherwise.
func (p *Predictor) Lookup(toolName string, args map[string]string) (string, bool) {
	key := cacheKey(toolName, args)

	p.mu.RLock()
	pred, ok := p.cache[key]
	p.mu.RUnlock()

	if !ok {
		p.mu.Lock()
		p.misses++
		p.mu.Unlock()
		return "", false
	}

	// Expire predictions older than 30 seconds
	if time.Since(pred.CreatedAt) > 30*time.Second {
		p.mu.Lock()
		delete(p.cache, key)
		p.misses++
		p.mu.Unlock()
		return "", false
	}

	p.mu.Lock()
	p.hits++
	delete(p.cache, key) // consume prediction
	p.mu.Unlock()

	return pred.Result, true
}

// Stats returns hit/miss counts.
func (p *Predictor) Stats() (hits, misses int) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.hits, p.misses
}

// HitRate returns the cache hit rate as 0.0-1.0.
func (p *Predictor) HitRate() float64 {
	p.mu.RLock()
	total := p.hits + p.misses
	h := p.hits
	p.mu.RUnlock()
	if total == 0 {
		return 0
	}
	return float64(h) / float64(total)
}

// CacheSize returns the number of cached predictions.
func (p *Predictor) CacheSize() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.cache)
}

// Clear removes all cached predictions.
func (p *Predictor) Clear() {
	p.mu.Lock()
	p.cache = make(map[string]Prediction)
	p.mu.Unlock()
}

// generatePredictions creates speculative tool calls based on the last action.
type predSpec struct {
	ToolName string
	Args     map[string]string
}

func (p *Predictor) generatePredictions(lastTool string, lastArgs map[string]string, lastResult string) []predSpec {
	var predictions []predSpec

	path := lastArgs["path"]

	switch lastTool {
	case "read":
		// Strategy 1: Pre-read the test file
		if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
			testFile := strings.TrimSuffix(path, ".go") + "_test.go"
			predictions = append(predictions, predSpec{
				ToolName: "read",
				Args:     map[string]string{"path": testFile},
			})
		}

		// Strategy 2: Pre-list the directory
		if path != "" {
			dir := dirOf(path)
			if dir != "" && dir != "." {
				predictions = append(predictions, predSpec{
					ToolName: "ls",
					Args:     map[string]string{"path": dir},
				})
			}
		}

	case "grep":
		// Strategy 3: Pre-read the first file from grep results
		if lastResult != "" {
			lines := strings.Split(lastResult, "\n")
			for _, line := range lines {
				if idx := strings.IndexByte(line, ':'); idx > 0 {
					candidate := line[:idx]
					if looksLikeFile(candidate) {
						predictions = append(predictions, predSpec{
							ToolName: "read",
							Args:     map[string]string{"path": candidate},
						})
						break // only pre-read the first match
					}
				}
			}
		}

	case "ls":
		// Strategy 4: Pre-read README or main entry point
		if lastResult != "" {
			lines := strings.Split(lastResult, "\n")
			for _, line := range lines {
				name := strings.TrimSpace(line)
				lower := strings.ToLower(name)
				if lower == "readme.md" || lower == "main.go" || lower == "index.ts" || lower == "index.js" {
					filePath := name
					if path != "" && path != "." {
						filePath = path + "/" + name
					}
					predictions = append(predictions, predSpec{
						ToolName: "read",
						Args:     map[string]string{"path": filePath},
					})
				}
			}
		}

	case "glob":
		// Strategy 5: Pre-read the first few matched files
		if lastResult != "" {
			lines := strings.Split(lastResult, "\n")
			count := 0
			for _, line := range lines {
				trimmed := strings.TrimSpace(line)
				if trimmed != "" && looksLikeFile(trimmed) {
					predictions = append(predictions, predSpec{
						ToolName: "read",
						Args:     map[string]string{"path": trimmed},
					})
					count++
					if count >= 2 {
						break
					}
				}
			}
		}
	}

	// Cap predictions
	if len(predictions) > 3 {
		predictions = predictions[:3]
	}

	return predictions
}

// evictOldest removes the oldest prediction from the cache. Must hold mu.
func (p *Predictor) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, pred := range p.cache {
		if oldestKey == "" || pred.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = pred.CreatedAt
		}
	}

	if oldestKey != "" {
		delete(p.cache, oldestKey)
	}
}

// isReadOnly returns true for tools that don't modify state.
func isReadOnly(toolName string) bool {
	switch toolName {
	case "read", "ls", "tree", "glob", "grep", "sysinfo", "diff":
		return true
	default:
		return false
	}
}

// cacheKey generates a unique key for a tool call.
func cacheKey(toolName string, args map[string]string) string {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	parts = append(parts, toolName)
	for _, k := range keys {
		parts = append(parts, k+"="+args[k])
	}
	return strings.Join(parts, "|")
}

// dirOf returns the directory portion of a path.
func dirOf(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			return path[:i]
		}
	}
	return ""
}

// looksLikeFile checks if a string looks like a file path.
func looksLikeFile(s string) bool {
	s = strings.TrimSpace(s)
	return len(s) > 0 && !strings.HasPrefix(s, "-") &&
		(strings.Contains(s, ".") || strings.Contains(s, "/"))
}
