package cognitive

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"
)

// PhantomReasoner pre-computes entire reasoning chains from tool results,
// so the LLM only generates the conclusion — not the reasoning itself.
//
// Innovation: No AI agent pre-computes the chain-of-thought. Every other
// system asks the model to THINK, then ACT. Phantom reasoning does the
// THINKING in deterministic code, building step-by-step chains from tool
// evidence, and the model just writes the final sentence.
//
// This is "Mad Libs for AI reasoning" — the blanks are trivial, the
// structure is bulletproof.
//
// Example:
//   Query: "How many Go files in internal/cognitive?"
//   Tool:  glob("internal/cognitive/**/*.go") → 31 files
//
//   Phantom chain (sent as response prefix):
//     "Step 1: I searched for Go files in internal/cognitive/ using glob.
//      Step 2: The search found 31 files including reasoner.go, pipeline.go,
//              synthesizer.go, exocortex.go, and others.
//      Step 3: Counting the results: exactly 31 .go files.
//      Therefore, "
//
//   LLM completes: "there are 31 Go files in internal/cognitive/."
//
// The model cannot hallucinate the count because it's pre-filled. It
// cannot claim different files because they're listed. Its ONLY job is
// to write a grammatically correct concluding sentence.
//
// Chain Caching: Deterministic chains from identical inputs produce identical
// outputs. The cache stores recent chains keyed on query+tool_results hash,
// so repeated/similar queries return instantly without rebuilding.
type PhantomReasoner struct {
	cache    map[string]*phantomCacheEntry
	cacheOrder []string // LRU order (oldest first)
	maxCache int
	mu       sync.RWMutex
}

type phantomCacheEntry struct {
	chain   *PhantomChain
	created time.Time
}

// PhantomChain holds a pre-computed reasoning chain.
type PhantomChain struct {
	// Steps contains the deterministic reasoning steps.
	Steps []PhantomStep

	// Conclusion is the partial text the LLM continues from (e.g., "Therefore, ").
	Conclusion string

	// FullContext is the entire chain as a single string.
	FullContext string

	// CanBypass is true if the chain already contains a complete answer
	// and the LLM isn't needed at all.
	CanBypass bool

	// DirectAnswer holds the bypass answer when CanBypass is true.
	DirectAnswer string
}

// PhantomStep is one step in the pre-computed reasoning chain.
type PhantomStep struct {
	StepNum     int
	Action      string // what was done
	Observation string // what was found
	Fact        string // the extracted fact
}

// NewPhantomReasoner creates a new phantom reasoner with LRU chain cache.
func NewPhantomReasoner() *PhantomReasoner {
	return &PhantomReasoner{
		cache:    make(map[string]*phantomCacheEntry),
		maxCache: 200,
	}
}

// BuildChain constructs a phantom reasoning chain from synthesis steps.
func (pr *PhantomReasoner) BuildChain(query string, steps []synthStep) *PhantomChain {
	if len(steps) == 0 {
		return &PhantomChain{}
	}

	chain := &PhantomChain{}
	var sb strings.Builder
	sb.WriteString("Think step by step:\n")

	stepNum := 1
	for _, step := range steps {
		if step.Err != nil {
			continue
		}
		if strings.TrimSpace(step.Result) == "" {
			continue
		}

		ps := pr.buildPhantomStep(stepNum, step)
		chain.Steps = append(chain.Steps, ps)

		sb.WriteString(fmt.Sprintf("Step %d: %s\n", ps.StepNum, ps.Action))
		sb.WriteString(fmt.Sprintf("Result: %s\n", ps.Observation))
		if ps.Fact != "" {
			sb.WriteString(fmt.Sprintf("Fact: %s\n", ps.Fact))
		}
		sb.WriteString("\n")

		stepNum++
	}

	// Check if we can bypass entirely
	if len(chain.Steps) > 0 {
		lastStep := chain.Steps[len(chain.Steps)-1]
		if lastStep.Fact != "" && pr.isCompleteFact(query, lastStep.Fact) {
			chain.CanBypass = true
			chain.DirectAnswer = lastStep.Fact
		}
	}

	if !chain.CanBypass {
		sb.WriteString("Therefore, ")
		chain.Conclusion = "Therefore, "
	}

	chain.FullContext = sb.String()
	return chain
}

// BuildChainFromPipeline constructs a chain from a Pipeline's steps.
func (pr *PhantomReasoner) BuildChainFromPipeline(query string, pipe *Pipeline) *PhantomChain {
	if pipe == nil || len(pipe.steps) == 0 {
		return &PhantomChain{}
	}

	var steps []synthStep
	for _, ps := range pipe.steps {
		steps = append(steps, synthStep{
			Tool:   ps.ToolName,
			Result: ps.RawResult,
		})
	}
	return pr.BuildChain(query, steps)
}

// BuildChainCached returns a cached chain if available, otherwise builds and caches.
// Cache key is hash(query + all tool results). TTL is 60 seconds.
func (pr *PhantomReasoner) BuildChainCached(query string, steps []synthStep) *PhantomChain {
	key := pr.cacheKey(query, steps)

	// Check cache
	pr.mu.RLock()
	if entry, ok := pr.cache[key]; ok && time.Since(entry.created) < 60*time.Second {
		pr.mu.RUnlock()
		return entry.chain
	}
	pr.mu.RUnlock()

	// Build and cache
	chain := pr.BuildChain(query, steps)
	pr.putCache(key, chain)
	return chain
}

// CacheStats returns cache size and hit-rate info.
func (pr *PhantomReasoner) CacheStats() (size int, maxSize int) {
	pr.mu.RLock()
	defer pr.mu.RUnlock()
	return len(pr.cache), pr.maxCache
}

// InvalidateCache clears all cached chains (e.g. after filesystem changes).
func (pr *PhantomReasoner) InvalidateCache() {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.cache = make(map[string]*phantomCacheEntry)
	pr.cacheOrder = nil
}

func (pr *PhantomReasoner) cacheKey(query string, steps []synthStep) string {
	h := sha256.New()
	h.Write([]byte(query))
	for _, s := range steps {
		h.Write([]byte(s.Tool))
		h.Write([]byte(s.Result))
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func (pr *PhantomReasoner) putCache(key string, chain *PhantomChain) {
	pr.mu.Lock()
	defer pr.mu.Unlock()

	if _, exists := pr.cache[key]; !exists {
		pr.cacheOrder = append(pr.cacheOrder, key)
	}
	pr.cache[key] = &phantomCacheEntry{chain: chain, created: time.Now()}

	// LRU eviction
	for len(pr.cache) > pr.maxCache && len(pr.cacheOrder) > 0 {
		oldest := pr.cacheOrder[0]
		pr.cacheOrder = pr.cacheOrder[1:]
		delete(pr.cache, oldest)
	}
}

// buildPhantomStep creates one reasoning step from a tool execution.
func (pr *PhantomReasoner) buildPhantomStep(num int, step synthStep) PhantomStep {
	ps := PhantomStep{StepNum: num}

	result := strings.TrimSpace(step.Result)
	lines := strings.Split(result, "\n")
	nonEmpty := countNonEmptyLines(lines)

	switch step.Tool {
	case "grep":
		pattern := step.Args["pattern"]
		ps.Action = fmt.Sprintf("I searched for `%s` using grep.", pattern)

		// Count matches and group by file
		files := make(map[string]int)
		for _, line := range lines {
			if parts := strings.SplitN(line, ":", 3); len(parts) >= 2 {
				files[parts[0]]++
			}
		}

		if nonEmpty == 1 {
			ps.Observation = fmt.Sprintf("Found 1 match: %s", truncatePhantom(lines[0], 120))
		} else {
			fileList := make([]string, 0, len(files))
			for f := range files {
				fileList = append(fileList, f)
			}
			if len(fileList) > 5 {
				ps.Observation = fmt.Sprintf("Found %d matches across %d files: %s, and %d more.",
					nonEmpty, len(files), strings.Join(fileList[:5], ", "), len(fileList)-5)
			} else {
				ps.Observation = fmt.Sprintf("Found %d matches across %d files: %s.",
					nonEmpty, len(files), strings.Join(fileList, ", "))
			}
		}
		ps.Fact = fmt.Sprintf("There are exactly %d matches for `%s`.", nonEmpty, pattern)

	case "glob":
		pattern := step.Args["pattern"]
		ps.Action = fmt.Sprintf("I searched for files matching `%s`.", pattern)

		if nonEmpty > 5 {
			first5 := make([]string, 0, 5)
			count := 0
			for _, l := range lines {
				if strings.TrimSpace(l) != "" && count < 5 {
					first5 = append(first5, strings.TrimSpace(l))
					count++
				}
			}
			ps.Observation = fmt.Sprintf("Found %d files: %s, and %d more.",
				nonEmpty, strings.Join(first5, ", "), nonEmpty-5)
		} else {
			ps.Observation = fmt.Sprintf("Found %d files.", nonEmpty)
		}
		ps.Fact = fmt.Sprintf("There are exactly %d files matching `%s`.", nonEmpty, pattern)

	case "read":
		path := step.Args["path"]
		ps.Action = fmt.Sprintf("I read the file `%s`.", path)
		ps.Observation = fmt.Sprintf("The file contains %d lines.", nonEmpty)

		// Extract key facts: package name, main types/functions
		var facts []string
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "package ") {
				facts = append(facts, trimmed)
			}
			if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, "struct") {
				facts = append(facts, trimmed)
			}
			if strings.HasPrefix(trimmed, "func ") && len(facts) < 5 {
				name := extractFuncName(trimmed)
				if name != "" {
					facts = append(facts, "func "+name)
				}
			}
		}
		if len(facts) > 0 {
			ps.Fact = fmt.Sprintf("File `%s` has %d lines. Key elements: %s.",
				path, nonEmpty, strings.Join(facts, ", "))
		} else {
			ps.Fact = fmt.Sprintf("File `%s` has %d lines.", path, nonEmpty)
		}

	case "ls":
		path := step.Args["path"]
		if path == "" {
			path = "."
		}
		ps.Action = fmt.Sprintf("I listed the contents of `%s`.", path)

		dirs, files := 0, 0
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			if strings.HasSuffix(trimmed, "/") {
				dirs++
			} else {
				files++
			}
		}
		ps.Observation = fmt.Sprintf("Found %d entries (%d directories, %d files).", nonEmpty, dirs, files)
		ps.Fact = fmt.Sprintf("Directory `%s` contains %d entries.", path, nonEmpty)

	case "tree":
		path := step.Args["path"]
		if path == "" {
			path = "."
		}
		ps.Action = fmt.Sprintf("I displayed the tree structure of `%s`.", path)
		ps.Observation = fmt.Sprintf("The tree shows %d entries.", nonEmpty)
		ps.Fact = fmt.Sprintf("The directory structure of `%s` has %d entries.", path, nonEmpty)

	case "git":
		cmd := step.Args["command"]
		ps.Action = fmt.Sprintf("I ran `git %s`.", cmd)
		if strings.Contains(cmd, "log") {
			ps.Observation = fmt.Sprintf("The log shows %d entries.", nonEmpty)
			ps.Fact = fmt.Sprintf("There are %d recent commits.", nonEmpty)
		} else if strings.Contains(cmd, "status") {
			if strings.Contains(result, "clean") || strings.Contains(result, "nichts") {
				ps.Fact = "The working tree is clean — no uncommitted changes."
			} else {
				ps.Fact = fmt.Sprintf("There are uncommitted changes: %s", truncatePhantom(result, 200))
			}
			ps.Observation = truncatePhantom(result, 200)
		} else if strings.Contains(cmd, "diff") {
			ps.Observation = fmt.Sprintf("The diff shows %d lines of changes.", nonEmpty)
			ps.Fact = fmt.Sprintf("There are %d lines of diff output.", nonEmpty)
		} else {
			ps.Observation = truncatePhantom(result, 200)
		}

	case "write":
		path := step.Args["path"]
		ps.Action = fmt.Sprintf("I wrote to file `%s`.", path)
		ps.Observation = "The write operation completed successfully."
		ps.Fact = fmt.Sprintf("File `%s` has been written successfully.", path)

	case "edit":
		path := step.Args["path"]
		ps.Action = fmt.Sprintf("I edited file `%s`.", path)
		ps.Observation = "The edit operation completed successfully."
		ps.Fact = fmt.Sprintf("File `%s` has been edited successfully.", path)

	default:
		ps.Action = fmt.Sprintf("I executed the `%s` tool.", step.Tool)
		ps.Observation = truncatePhantom(result, 200)
	}

	return ps
}

// isCompleteFact checks if a fact alone answers the query sufficiently.
func (pr *PhantomReasoner) isCompleteFact(query, fact string) bool {
	lower := strings.ToLower(query)

	// Quantitative questions: "how many", "count", "number of"
	quantitative := []string{"how many", "count", "number of", "how much"}
	for _, q := range quantitative {
		if strings.Contains(lower, q) {
			return true
		}
	}

	// Status questions answered by status facts
	if strings.Contains(lower, "status") && strings.Contains(fact, "clean") {
		return true
	}

	// Simple existence: "is there", "does X exist"
	if strings.Contains(lower, "is there") || strings.Contains(lower, "does") && strings.Contains(lower, "exist") {
		return true
	}

	return false
}

// extractFuncName pulls the function name from a Go func declaration.
func extractFuncName(line string) string {
	// "func (r *Reasoner) Run(...)" → "Run"
	// "func NewReasoner(...)" → "NewReasoner"
	line = strings.TrimPrefix(line, "func ")

	// Method with receiver
	if strings.HasPrefix(line, "(") {
		idx := strings.Index(line, ")")
		if idx < 0 {
			return ""
		}
		line = strings.TrimSpace(line[idx+1:])
	}

	// Get function name up to "("
	idx := strings.Index(line, "(")
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(line[:idx])
}

// countNonEmptyLines counts non-empty lines.
func countNonEmptyLines(lines []string) int {
	count := 0
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			count++
		}
	}
	return count
}

// truncatePhantom shortens text for phantom chain display.
func truncatePhantom(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
