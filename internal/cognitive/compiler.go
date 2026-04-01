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

// ---------------------------------------------------------------------------
// Pattern extraction (unexported helpers)
// ---------------------------------------------------------------------------

// extractPattern builds a QueryPattern from a raw input and a set of
// recognised entities. Each entity value found in the input is replaced
// with a named slot, and a regex with named capture groups is compiled.
func extractPattern(input string, entities map[string]string, intent string) *QueryPattern {
	if len(entities) == 0 || input == "" {
		return nil
	}

	lower := strings.ToLower(input)
	template := lower

	// Collect slot names in a stable order (sorted by position in input).
	type slotPos struct {
		name string
		pos  int
	}
	var slots []slotPos
	for key, val := range entities {
		valLower := strings.ToLower(val)
		idx := strings.Index(lower, valLower)
		if idx < 0 {
			continue
		}
		slots = append(slots, slotPos{name: key, pos: idx})
	}
	if len(slots) == 0 {
		return nil
	}
	// Sort by position so replacements don't shift offsets unexpectedly.
	for i := 1; i < len(slots); i++ {
		for j := i; j > 0 && slots[j].pos < slots[j-1].pos; j-- {
			slots[j], slots[j-1] = slots[j-1], slots[j]
		}
	}

	// Replace entity values with {slot_name} placeholders.
	var slotNames []string
	for _, s := range slots {
		valLower := strings.ToLower(entities[s.name])
		template = strings.Replace(template, valLower, "{"+s.name+"}", 1)
		slotNames = append(slotNames, s.name)
	}

	// Build regex: escape fixed text, swap placeholders for named groups.
	regexStr := buildPatternRegex(template, slotNames)
	compiled, err := regexp.Compile(regexStr)
	if err != nil {
		return nil
	}

	// Confidence heuristic: more fixed text relative to total length means
	// higher confidence. Many slots lower it.
	fixedLen := len(template)
	for _, name := range slotNames {
		fixedLen -= len("{" + name + "}")
	}
	confidence := float64(fixedLen) / float64(len(template)+1)
	if confidence < 0.2 {
		return nil // pattern is too generic
	}
	if len(slotNames) > 3 {
		confidence *= 0.7
	}

	return &QueryPattern{
		Template:   template,
		Slots:      slotNames,
		RegexStr:   regexStr,
		Intent:     intent,
		Confidence: confidence,
		compiled:   compiled,
	}
}

// buildPatternRegex converts a template like "what is {topic}" into a regex
// string: `^what is (?P<topic>.+?)$`.
func buildPatternRegex(template string, slotNames []string) string {
	// Temporarily replace slot placeholders with unique tokens so they
	// survive regexp.QuoteMeta.
	const tokenPrefix = "\x00SLOT_"
	const tokenSuffix = "\x00"
	tmp := template
	for _, name := range slotNames {
		tmp = strings.Replace(tmp, "{"+name+"}", tokenPrefix+name+tokenSuffix, 1)
	}

	// Escape everything for a literal match.
	escaped := regexp.QuoteMeta(tmp)

	// Restore slot placeholders as named capture groups.
	for _, name := range slotNames {
		escapedToken := regexp.QuoteMeta(tokenPrefix + name + tokenSuffix)
		escaped = strings.Replace(escaped, escapedToken, "(?P<"+name+">.+?)", 1)
	}

	return "^" + escaped + "$"
}

// ---------------------------------------------------------------------------
// Template extraction (unexported helper)
// ---------------------------------------------------------------------------

// extractTemplate builds a response template by replacing entity values in
// the response with their slot names.
func extractTemplate(response string, entities map[string]string) string {
	if len(entities) == 0 {
		return response
	}
	tmpl := response
	for key, val := range entities {
		if val == "" {
			continue
		}
		// Case-insensitive replacement: find the value in the response
		// preserving surrounding text.
		lower := strings.ToLower(tmpl)
		valLower := strings.ToLower(val)
		for {
			idx := strings.Index(lower, valLower)
			if idx < 0 {
				break
			}
			tmpl = tmpl[:idx] + "{" + key + "}" + tmpl[idx+len(val):]
			lower = strings.ToLower(tmpl)
		}
	}
	return tmpl
}

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------

// Compile extracts a pattern from input and a response template, producing
// a CompiledHandler that can answer similar queries deterministically.
// Returns nil if the pattern is too specific or the response too variable.
func (cc *CognitiveCompiler) Compile(input string, response string, intent string, entities map[string]string) *CompiledHandler {
	if input == "" || response == "" {
		return nil
	}

	pattern := extractPattern(input, entities, intent)
	if pattern == nil {
		return nil
	}

	tmpl := extractTemplate(response, entities)

	// Build slot definitions from the pattern's slot names.
	var slotDefs []SlotDef
	for _, name := range pattern.Slots {
		st := "entity"
		if strings.HasSuffix(name, "_lookup") || name == "knowledge" {
			st = "knowledge_lookup"
		}
		slotDefs = append(slotDefs, SlotDef{
			Name: name,
			Type: st,
		})
	}

	now := time.Now()
	handler := &CompiledHandler{
		ID:       compilerHandlerID(pattern.Template),
		Pattern:  pattern,
		Template: tmpl,
		Slots:    slotDefs,
		Source:   "neural",
		Quality:  0.6, // initial quality; needs observation to increase
		Compiled: now,
		LastUsed: now,
	}

	cc.mu.Lock()
	// Deduplicate: replace existing handler with the same pattern template.
	replaced := false
	for i, h := range cc.Handlers {
		if h.Pattern.Template == pattern.Template {
			cc.Handlers[i] = handler
			replaced = true
			break
		}
	}
	if !replaced {
		cc.Handlers = append(cc.Handlers, handler)
	}
	// Enforce capacity.
	if len(cc.Handlers) > cc.MaxHandlers {
		cc.pruneUnlocked()
	}
	cc.mu.Unlock()

	cc.Save()
	return handler
}

// compilerHandlerID builds a short deterministic ID from a pattern template.
func compilerHandlerID(template string) string {
	return "ch_" + shortHash(template)
}

// pruneUnlocked removes stale/low-quality handlers. Caller must hold cc.mu.
func (cc *CognitiveCompiler) pruneUnlocked() {
	cutoff := time.Now().Add(-30 * 24 * time.Hour)
	kept := make([]*CompiledHandler, 0, len(cc.Handlers))
	for _, h := range cc.Handlers {
		if h.Quality < 0.3 {
			continue
		}
		if h.Uses == 0 && h.LastUsed.Before(cutoff) {
			continue
		}
		if h.Uses > 0 && h.LastUsed.Before(cutoff) && h.Quality < 0.5 {
			continue
		}
		kept = append(kept, h)
	}
	cc.Handlers = kept
}

// ---------------------------------------------------------------------------
// Match
// ---------------------------------------------------------------------------

// Match finds the best compiled handler for an input query. It returns the
// handler and a map of extracted slot values, or (nil, nil) when no handler
// matches with sufficient confidence.
func (cc *CognitiveCompiler) Match(input string) (*CompiledHandler, map[string]string) {
	if input == "" {
		return nil, nil
	}

	lower := strings.ToLower(strings.TrimSpace(input))

	cc.mu.RLock()
	defer cc.mu.RUnlock()

	var bestHandler *CompiledHandler
	var bestSlots map[string]string
	bestScore := 0.0

	for _, h := range cc.Handlers {
		if h.Pattern == nil || h.Pattern.compiled == nil {
			continue
		}

		m := h.Pattern.compiled.FindStringSubmatch(lower)
		if m == nil {
			continue
		}

		// Extract named slots.
		extracted := make(map[string]string)
		for i, name := range h.Pattern.compiled.SubexpNames() {
			if i > 0 && i < len(m) && name != "" {
				extracted[name] = m[i]
			}
		}

		// Score: pattern confidence * quality, with a small boost for usage.
		score := h.Pattern.Confidence * h.Quality
		if h.Uses > 0 {
			usageBoost := float64(h.Uses) / (float64(h.Uses) + 10.0) * 0.2
			score += usageBoost
		}

		if score > bestScore {
			bestScore = score
			bestHandler = h
			bestSlots = extracted
		}
	}

	// Minimum acceptable score: the product of the lowest acceptable
	// confidence (0.7 from extractPattern) and the initial quality (0.6).
	if bestScore < 0.7*0.6 {
		return nil, nil
	}

	return bestHandler, bestSlots
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------

// Execute fills a compiled handler's template with the provided slot values
// and returns the deterministic response.
func (cc *CognitiveCompiler) Execute(handler *CompiledHandler, slots map[string]string) string {
	if handler == nil {
		return ""
	}

	result := handler.Template
	for _, sd := range handler.Slots {
		placeholder := "{" + sd.Name + "}"
		val, ok := slots[sd.Name]
		if !ok || val == "" {
			val = sd.Fallback
		}
		result = strings.ReplaceAll(result, placeholder, val)
	}

	// Update usage counters.
	cc.mu.Lock()
	handler.Uses++
	handler.LastUsed = time.Now()
	cc.mu.Unlock()

	return result
}
