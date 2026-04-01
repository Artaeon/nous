package cognitive

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// ---------------------------------------------------------------------------
// Cognitive Compiler
//
// Compiles neural generation responses into deterministic handlers. When a
// neural model generates a response, the compiler extracts a pattern template
// from the query and a slot-filling function from the response. Future
// queries matching the pattern bypass neural generation entirely — they
// execute the compiled handler deterministically.
//
// This extends the Crystal and ResponseCrystal systems:
//   - ResponseCrystalStore caches full responses by semantic similarity
//   - CrystalBook caches successful reasoning chains as reusable patterns
//   - CognitiveCompiler compiles query->response into template+slots
//
// The progression: cache -> pattern -> compiled handler.
// ---------------------------------------------------------------------------

// CompiledHandler is a deterministic response handler compiled from a neural
// generation. It pairs a query pattern (with named slots) to a response
// template whose slots are filled at execution time without any LLM call.
type CompiledHandler struct {
	ID       string        `json:"id"`
	Pattern  *QueryPattern `json:"pattern"`
	Template string        `json:"template"`  // response template with {slot} placeholders
	Slots    []SlotDef     `json:"slots"`
	Source   string        `json:"source"`    // "neural", "crystal", "manual"
	Quality  float64       `json:"quality"`   // 0.0-1.0, based on user acceptance
	Uses     int           `json:"uses"`      // times executed
	Compiled time.Time     `json:"compiled"`  // when compiled
	LastUsed time.Time     `json:"last_used"` // last execution
}

// QueryPattern represents an extracted pattern from a user query.
// The Template field holds a human-readable form ("what is {topic}") while
// the Regex field holds the compiled matcher with named capture groups.
type QueryPattern struct {
	Template   string  `json:"template"`   // "what is {topic}"
	Slots      []string `json:"slots"`     // slot names in order
	RegexStr   string  `json:"regex"`      // serialisable regex string
	Intent     string  `json:"intent"`     // NLU intent category
	Confidence float64 `json:"confidence"` // pattern confidence

	compiled *regexp.Regexp // transient, rebuilt on Load
}

// SlotDef defines how a slot gets filled during execution.
type SlotDef struct {
	Name     string `json:"name"`     // "topic", "location", etc.
	Type     string `json:"type"`     // "entity", "literal", "knowledge_lookup", "tool_result"
	Fallback string `json:"fallback"` // default value if extraction fails
}

// CompilerStats holds aggregated statistics about the cognitive compiler.
type CompilerStats struct {
	TotalHandlers   int     // number of compiled handlers
	TotalExecutions int     // sum of all handler Uses
	AvgQuality      float64 // mean quality across handlers
	HitRate         float64 // fraction of handlers with Uses > 0
	HighQuality     int     // handlers with Quality >= 0.8
	Stale           int     // handlers unused for > 30 days
}

// CognitiveCompiler compiles neural responses into deterministic handlers.
type CognitiveCompiler struct {
	Handlers    []*CompiledHandler `json:"handlers"`
	MaxHandlers int                `json:"-"`
	savePath    string
	mu          sync.RWMutex
}

// NewCognitiveCompiler creates a compiler, loading any previously saved
// handlers from the JSON file at savePath.
func NewCognitiveCompiler(savePath string) *CognitiveCompiler {
	cc := &CognitiveCompiler{
		MaxHandlers: 500,
		savePath:    savePath,
	}
	cc.Load() // best-effort; ignore errors on first run
	return cc
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

// Save persists all compiled handlers to the JSON file.
func (cc *CognitiveCompiler) Save() error {
	if cc.savePath == "" {
		return nil
	}
	cc.mu.RLock()
	data, err := json.MarshalIndent(cc.Handlers, "", "  ")
	cc.mu.RUnlock()
	if err != nil {
		return err
	}
	return safefile.WriteAtomic(cc.savePath, data, 0o644)
}

// Load reads compiled handlers from the JSON file. Regex patterns are
// recompiled from their string representation.
func (cc *CognitiveCompiler) Load() error {
	if cc.savePath == "" {
		return nil
	}
	data, err := os.ReadFile(cc.savePath)
	if err != nil {
		return err
	}
	var handlers []*CompiledHandler
	if err := json.Unmarshal(data, &handlers); err != nil {
		return err
	}
	// Recompile regex patterns from stored strings.
	for _, h := range handlers {
		if h.Pattern != nil && h.Pattern.RegexStr != "" {
			h.Pattern.compiled, _ = regexp.Compile(h.Pattern.RegexStr)
		}
	}
	cc.mu.Lock()
	cc.Handlers = handlers
	cc.mu.Unlock()
	return nil
}

// Ensure unused imports don't cause build failures. These will be used in
// subsequent methods (Compile, Match, Execute, etc.).
var (
	_ = strings.ToLower
	_ = time.Now
)
