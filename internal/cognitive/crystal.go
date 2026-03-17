package cognitive

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// Crystal represents a "thought crystallization" — a complete reasoning chain
// that has been compiled into a deterministic rule. When a new query matches
// a crystal's trigger pattern, the system executes the tool chain and generates
// the response WITHOUT any LLM inference. This is JIT compilation for thoughts.
//
// Innovation: Recipes record tool sequences. Crystals go further — they record
// the COMPLETE chain including result interpretation and response templates.
// Over time, common interactions crystallize into instant, 100% accurate paths.
type Crystal struct {
	ID           string        `json:"id"`
	Trigger      *CrystalTrigger `json:"trigger"`
	Steps        []CrystalStep `json:"steps"`
	ResponseTmpl string        `json:"response_template"` // template with {result_N} placeholders
	Embedding    []float64     `json:"embedding,omitempty"` // semantic embedding for similarity matching
	Uses         int           `json:"uses"`
	Successes    int           `json:"successes"`
	CreatedAt    time.Time     `json:"created_at"`
	LastUsed     time.Time     `json:"last_used"`
}

// CrystalTrigger defines when a crystal fires.
type CrystalTrigger struct {
	Pattern    string   `json:"pattern"`     // regex pattern matching user queries
	Keywords   []string `json:"keywords"`    // required keywords (all must be present)
	MinWords   int      `json:"min_words"`   // minimum word count
	MaxWords   int      `json:"max_words"`   // maximum word count
	compiled   *regexp.Regexp
}

// CrystalStep is a deterministic tool call within a crystal.
type CrystalStep struct {
	Tool       string            `json:"tool"`
	Args       map[string]string `json:"args"`        // concrete or template args
	ExtractRe  string            `json:"extract_re"`  // regex to extract answer from result
	ResultVar  string            `json:"result_var"`   // variable name for response template
}

// CrystalBook manages a collection of thought crystals.
type CrystalBook struct {
	crystals  []Crystal
	storePath string
	mu        sync.RWMutex
	maxSize   int
	embedFunc func(string) ([]float64, error) // optional: for semantic matching
}

// CrystalMatch is a crystal matched to a query with a confidence score.
type CrystalMatch struct {
	Crystal    *Crystal
	Confidence float64
	Captures   map[string]string // regex capture groups from trigger
}

// NewCrystalBook creates a new crystal book, loading from disk if available.
func NewCrystalBook(storePath string) *CrystalBook {
	cb := &CrystalBook{
		storePath: storePath,
		maxSize:   100,
	}
	cb.load()
	return cb
}

// Match finds the best matching crystal for a query.
// Returns nil if no crystal matches with sufficient confidence.
func (cb *CrystalBook) Match(query string) *CrystalMatch {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}

	words := strings.Fields(strings.ToLower(query))

	// Compute query embedding for semantic matching (if available)
	var queryEmbed []float64
	if cb.embedFunc != nil {
		if vec, err := cb.embedFunc(query); err == nil {
			queryEmbed = vec
		}
	}

	var best *CrystalMatch

	for i := range cb.crystals {
		c := &cb.crystals[i]
		score := cb.matchScoreWithEmbed(c, query, words, queryEmbed)
		if score < 0.7 {
			continue
		}

		captures := cb.extractCaptures(c, query)

		if best == nil || score > best.Confidence {
			best = &CrystalMatch{
				Crystal:    c,
				Confidence: score,
				Captures:   captures,
			}
		}
	}

	return best
}

// Crystallize records a successful reasoning chain as a new crystal.
// Called after a multi-step reasoning succeeds to compile the pattern.
func (cb *CrystalBook) Crystallize(query string, pipe *Pipeline, finalAnswer string) {
	if pipe == nil || pipe.StepCount() == 0 {
		return
	}

	// Don't crystallize if answer is too short or generic
	if len(finalAnswer) < 10 {
		return
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Check if a similar crystal already exists
	words := strings.Fields(strings.ToLower(query))
	for i := range cb.crystals {
		score := cb.matchScore(&cb.crystals[i], query, words)
		if score >= 0.75 {
			// Update existing crystal's success count
			cb.crystals[i].Uses++
			cb.crystals[i].Successes++
			cb.crystals[i].LastUsed = time.Now()
			cb.save()
			return
		}
	}

	// Build crystal from pipeline steps
	var steps []CrystalStep
	for _, s := range pipe.steps {
		step := CrystalStep{
			Tool:      s.ToolName,
			Args:      parseCrystalStepArgs(s.RawResult),
			ResultVar: "result_" + s.ToolName,
		}

		// Build extraction regex from the final answer
		extractRe := buildExtractionPattern(s.RawResult, finalAnswer)
		if extractRe != "" {
			step.ExtractRe = extractRe
		}

		steps = append(steps, step)
	}

	// Build trigger pattern from query
	trigger := buildTrigger(query)

	// Build response template
	tmpl := buildResponseTemplate(finalAnswer, pipe)

	// Compute embedding for semantic matching
	var embedding []float64
	if cb.embedFunc != nil {
		if vec, err := cb.embedFunc(query); err == nil {
			embedding = vec
		}
	}

	crystal := Crystal{
		ID:           crystalID(query),
		Trigger:      trigger,
		Steps:        steps,
		ResponseTmpl: tmpl,
		Embedding:    embedding,
		Uses:         1,
		Successes:    1,
		CreatedAt:    time.Now(),
		LastUsed:     time.Now(),
	}

	cb.crystals = append(cb.crystals, crystal)

	// Prune if over capacity
	if len(cb.crystals) > cb.maxSize {
		cb.prune()
	}

	cb.save()
}

// ReportSuccess marks a crystal execution as successful.
func (cb *CrystalBook) ReportSuccess(id string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	for i := range cb.crystals {
		if cb.crystals[i].ID == id {
			cb.crystals[i].Successes++
			cb.crystals[i].Uses++
			cb.crystals[i].LastUsed = time.Now()
			cb.save()
			return
		}
	}
}

// ReportFailure marks a crystal execution as failed. Low success rate
// crystals are eventually pruned.
func (cb *CrystalBook) ReportFailure(id string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	for i := range cb.crystals {
		if cb.crystals[i].ID == id {
			cb.crystals[i].Uses++
			cb.crystals[i].LastUsed = time.Now()
			cb.save()
			return
		}
	}
}

// Size returns the number of crystals.
func (cb *CrystalBook) Size() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return len(cb.crystals)
}

// Stats returns crystal book statistics.
func (cb *CrystalBook) Stats() CrystalStats {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	stats := CrystalStats{
		Total: len(cb.crystals),
	}

	for _, c := range cb.crystals {
		stats.TotalUses += c.Uses
		stats.TotalSuccesses += c.Successes
		if c.Uses > 0 {
			rate := float64(c.Successes) / float64(c.Uses)
			if rate >= 0.8 {
				stats.HighConfidence++
			}
		}
	}

	if stats.TotalUses > 0 {
		stats.OverallSuccessRate = float64(stats.TotalSuccesses) / float64(stats.TotalUses)
	}

	return stats
}

// CrystalStats holds aggregated crystal statistics.
type CrystalStats struct {
	Total              int
	TotalUses          int
	TotalSuccesses     int
	HighConfidence     int
	OverallSuccessRate float64
}

// --- Internal ---

// SetEmbedFunc sets the embedding function for semantic crystal matching.
func (cb *CrystalBook) SetEmbedFunc(fn func(string) ([]float64, error)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.embedFunc = fn
}

func (cb *CrystalBook) matchScore(c *Crystal, query string, words []string) float64 {
	return cb.matchScoreWithEmbed(c, query, words, nil)
}

func (cb *CrystalBook) matchScoreWithEmbed(c *Crystal, query string, words []string, queryEmbed []float64) float64 {
	score := 0.0

	// Compute keyword overlap (used by multiple signals)
	keywordRatio := 0.0
	if len(c.Trigger.Keywords) > 0 {
		matched := 0
		for _, kw := range c.Trigger.Keywords {
			for _, w := range words {
				clean := strings.TrimRight(w, "?!.,;:")
				if clean == kw || w == kw {
					matched++
					break
				}
			}
		}
		keywordRatio = float64(matched) / float64(len(c.Trigger.Keywords))
	}

	// Scoring weights:
	//   With embeddings: 0.40 semantic + 0.30 keyword + 0.20 success + 0.10 recency
	//   Without:         0.70 keyword + 0.20 success + 0.10 recency
	if len(queryEmbed) > 0 && len(c.Embedding) > 0 {
		sim := cosineSim(queryEmbed, c.Embedding)
		score += sim * 0.40
		score += keywordRatio * 0.30
	} else {
		score += keywordRatio * 0.70
	}

	// Success rate bonus — log-scaled, capped at 0.20
	if c.Uses > 2 {
		rate := float64(c.Successes) / float64(c.Uses)
		if rate >= 0.8 {
			usageBonus := math.Log2(float64(c.Uses)+1) * 0.03
			if usageBonus > 0.20 {
				usageBonus = 0.20
			}
			score += usageBonus
		}
	}

	// Recency bonus
	if time.Since(c.LastUsed) < time.Hour {
		score += 0.10
	} else if time.Since(c.LastUsed) < 24*time.Hour {
		score += 0.05
	}

	return score
}

func (cb *CrystalBook) extractCaptures(c *Crystal, query string) map[string]string {
	captures := make(map[string]string)
	if c.Trigger.compiled != nil {
		m := c.Trigger.compiled.FindStringSubmatch(query)
		for i, name := range c.Trigger.compiled.SubexpNames() {
			if i > 0 && i < len(m) && name != "" {
				captures[name] = m[i]
			}
		}
	}
	return captures
}

func (cb *CrystalBook) prune() {
	// Remove lowest-value crystals
	sort.Slice(cb.crystals, func(i, j int) bool {
		return crystalValue(&cb.crystals[i]) > crystalValue(&cb.crystals[j])
	})
	if len(cb.crystals) > cb.maxSize {
		cb.crystals = cb.crystals[:cb.maxSize]
	}
}

func crystalValue(c *Crystal) float64 {
	value := 0.0

	// Success rate
	if c.Uses > 0 {
		value += float64(c.Successes) / float64(c.Uses) * 2.0
	}

	// Usage frequency — log-scaled so early uses matter most.
	// log2(1+1)*0.2 = 0.2, log2(10+1)*0.2 = 0.69, log2(100+1)*0.2 = 1.33
	// This prevents high-use crystals from dominating purely on volume.
	value += math.Log2(float64(c.Uses)+1) * 0.2

	// Temporal decay: crystals lose value exponentially over time.
	// Half-life of 14 days — unused crystals fade, frequently used ones stay.
	daysSinceUse := time.Since(c.LastUsed).Hours() / 24
	recencyFactor := 1.0 / (1.0 + daysSinceUse/14.0)
	value *= recencyFactor

	// Recency bonus (stacks with decay)
	if daysSinceUse < 1 {
		value += 1.0
	} else if daysSinceUse < 7 {
		value += 0.5
	}

	// Age penalty: very old crystals (>90 days) with low usage are likely stale
	daysSinceCreation := time.Since(c.CreatedAt).Hours() / 24
	if daysSinceCreation > 90 && c.Uses < 5 {
		value *= 0.5
	}

	return value
}

func crystalID(query string) string {
	return shortHash(strings.ToLower(strings.TrimSpace(query)))
}

func buildTrigger(query string) *CrystalTrigger {
	words := strings.Fields(strings.ToLower(query))
	keywords := filterCrystalKeywords(words)

	// Build a loose regex from the query structure
	pattern := buildLoosePattern(query)

	return &CrystalTrigger{
		Pattern:  pattern,
		Keywords: keywords,
		MinWords: max(1, len(words)-3),
		MaxWords: len(words) + 5,
	}
}

func filterCrystalKeywords(words []string) []string {
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "shall": true, "can": true,
		"and": true, "or": true, "but": true, "if": true, "then": true,
		"of": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "with": true, "from": true, "by": true, "about": true,
		"this": true, "that": true, "these": true, "those": true,
		"it": true, "its": true, "i": true, "me": true, "my": true,
		"you": true, "your": true, "we": true, "our": true,
		"what": true, "which": true, "who": true, "how": true, "where": true,
		"when": true, "why": true, "all": true, "each": true, "every": true,
	}

	var out []string
	for _, w := range words {
		clean := strings.TrimRight(w, "?!.,;:")
		if len(clean) >= 3 && !stopWords[clean] {
			out = append(out, clean)
		}
	}
	return out
}

func buildLoosePattern(query string) string {
	// Build a Go-compatible regex pattern from significant words.
	// Go's RE2 engine doesn't support lookaheads, so we match keywords
	// in their original order with flexible spacing between them.
	words := strings.Fields(strings.ToLower(query))
	significant := filterCrystalKeywords(words)

	if len(significant) == 0 {
		return ""
	}

	var parts []string
	for _, w := range significant {
		parts = append(parts, regexp.QuoteMeta(w))
	}
	return strings.Join(parts, `\b.*\b`)
}

func buildExtractionPattern(toolResult, finalAnswer string) string {
	// Find substrings from the tool result that appear in the final answer
	// These represent the data the model extracted from the tool result
	if len(toolResult) < 5 || len(finalAnswer) < 5 {
		return ""
	}

	// Look for version numbers, function names, file paths, etc.
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`\d+\.\d+(?:\.\d+)?`),            // version numbers
		regexp.MustCompile(`func\s+\w+`),                      // function names
		regexp.MustCompile(`[A-Z][a-z]+(?:[A-Z][a-z]+)+`),    // CamelCase identifiers
		regexp.MustCompile(`[a-z]+(?:_[a-z]+)+`),              // snake_case identifiers
	}

	for _, pat := range patterns {
		if m := pat.FindString(toolResult); m != "" {
			if strings.Contains(finalAnswer, m) {
				return pat.String()
			}
		}
	}
	return ""
}

func buildResponseTemplate(answer string, pipe *Pipeline) string {
	// For now, store the answer as a template hint
	// Future: replace concrete values with {result_N} placeholders
	if len(answer) > 500 {
		return answer[:500]
	}
	return answer
}

func parseCrystalStepArgs(rawResult string) map[string]string {
	// Placeholder: in a full implementation, this would extract the
	// arguments that were used for the tool call from the pipeline step.
	// For now, return empty (the crystal steps use the original args).
	return make(map[string]string)
}

// TopCrystals returns the top N crystals sorted by success rate and usage.
func (cb *CrystalBook) TopCrystals(n int) []Crystal {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if len(cb.crystals) == 0 {
		return nil
	}

	// Copy and sort by value
	sorted := make([]Crystal, len(cb.crystals))
	copy(sorted, cb.crystals)
	sort.Slice(sorted, func(i, j int) bool {
		return crystalValue(&sorted[i]) > crystalValue(&sorted[j])
	})

	if n > len(sorted) {
		n = len(sorted)
	}
	return sorted[:n]
}

// SeedDevWorkflows pre-populates the crystal book with common developer
// workflow patterns. These crystals provide instant-from-day-1 responses
// for frequent queries without needing the model to learn them first.
// Only seeds if the book is currently empty to avoid polluting learned crystals.
func (cb *CrystalBook) SeedDevWorkflows() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if len(cb.crystals) > 0 {
		return 0
	}

	seeds := devWorkflowSeeds()
	now := time.Now()

	for _, s := range seeds {
		trigger := &CrystalTrigger{
			Pattern:  buildLoosePattern(s.query),
			Keywords: filterCrystalKeywords(strings.Fields(strings.ToLower(s.query))),
			MinWords: 1,
			MaxWords: 20,
		}

		crystal := Crystal{
			ID:           crystalID(s.query),
			Trigger:      trigger,
			Steps:        s.steps,
			ResponseTmpl: s.responseTmpl,
			Uses:         1,
			Successes:    1,
			CreatedAt:    now,
			LastUsed:     now,
		}
		cb.crystals = append(cb.crystals, crystal)
	}

	cb.save()
	return len(seeds)
}

// seedEntry defines a pre-built crystal for common dev workflows.
type seedEntry struct {
	query        string
	steps        []CrystalStep
	responseTmpl string
}

// devWorkflowSeeds returns the pre-built crystal definitions.
func devWorkflowSeeds() []seedEntry {
	return []seedEntry{
		// --- File reading patterns ---
		{
			query: "read go.mod",
			steps: []CrystalStep{{Tool: "read", Args: map[string]string{"path": "go.mod"}, ResultVar: "result_read"}},
			responseTmpl: "{result_read}",
		},
		{
			query: "show me the readme",
			steps: []CrystalStep{{Tool: "read", Args: map[string]string{"path": "README.md"}, ResultVar: "result_read"}},
			responseTmpl: "{result_read}",
		},
		{
			query: "read the main file",
			steps: []CrystalStep{{Tool: "read", Args: map[string]string{"path": "cmd/nous/main.go"}, ResultVar: "result_read"}},
			responseTmpl: "{result_read}",
		},
		{
			query: "show the dockerfile",
			steps: []CrystalStep{{Tool: "read", Args: map[string]string{"path": "Dockerfile"}, ResultVar: "result_read"}},
			responseTmpl: "{result_read}",
		},
		{
			query: "read the makefile",
			steps: []CrystalStep{{Tool: "read", Args: map[string]string{"path": "Makefile"}, ResultVar: "result_read"}},
			responseTmpl: "{result_read}",
		},

		// --- Project structure ---
		{
			query: "list files in the project",
			steps: []CrystalStep{{Tool: "ls", Args: map[string]string{"path": "."}, ResultVar: "result_ls"}},
			responseTmpl: "{result_ls}",
		},
		{
			query: "show project structure",
			steps: []CrystalStep{{Tool: "tree", Args: map[string]string{"path": "."}, ResultVar: "result_tree"}},
			responseTmpl: "{result_tree}",
		},
		{
			query: "list go files",
			steps: []CrystalStep{{Tool: "glob", Args: map[string]string{"pattern": "**/*.go"}, ResultVar: "result_glob"}},
			responseTmpl: "{result_glob}",
		},
		{
			query: "find all test files",
			steps: []CrystalStep{{Tool: "glob", Args: map[string]string{"pattern": "**/*_test.go"}, ResultVar: "result_glob"}},
			responseTmpl: "{result_glob}",
		},

		// --- Code search patterns ---
		{
			query: "search for TODO comments",
			steps: []CrystalStep{{Tool: "grep", Args: map[string]string{"pattern": "TODO", "glob": "*.go"}, ResultVar: "result_grep"}},
			responseTmpl: "{result_grep}",
		},
		{
			query: "find error handling",
			steps: []CrystalStep{{Tool: "grep", Args: map[string]string{"pattern": "if err != nil", "glob": "*.go"}, ResultVar: "result_grep"}},
			responseTmpl: "{result_grep}",
		},
		{
			query: "search for main function",
			steps: []CrystalStep{{Tool: "grep", Args: map[string]string{"pattern": "func main", "glob": "*.go"}, ResultVar: "result_grep"}},
			responseTmpl: "{result_grep}",
		},

		// --- Git workflows ---
		{
			query: "show git status",
			steps: []CrystalStep{{Tool: "git", Args: map[string]string{"command": "status"}, ResultVar: "result_git"}},
			responseTmpl: "{result_git}",
		},
		{
			query: "show recent commits",
			steps: []CrystalStep{{Tool: "git", Args: map[string]string{"command": "log --oneline -10"}, ResultVar: "result_git"}},
			responseTmpl: "{result_git}",
		},
		{
			query: "show git diff",
			steps: []CrystalStep{{Tool: "git", Args: map[string]string{"command": "diff"}, ResultVar: "result_git"}},
			responseTmpl: "{result_git}",
		},
		{
			query: "what branch am i on",
			steps: []CrystalStep{{Tool: "git", Args: map[string]string{"command": "branch --show-current"}, ResultVar: "result_git"}},
			responseTmpl: "You are on branch {result_git}",
		},
		{
			query: "show all branches",
			steps: []CrystalStep{{Tool: "git", Args: map[string]string{"command": "branch -a"}, ResultVar: "result_git"}},
			responseTmpl: "{result_git}",
		},

		// --- Multi-step patterns ---
		{
			query: "what go version does this project use",
			steps: []CrystalStep{
				{Tool: "read", Args: map[string]string{"path": "go.mod"}, ExtractRe: `go\s+\d+\.\d+`, ResultVar: "result_read"},
			},
			responseTmpl: "This project uses Go {result_read}",
		},
		{
			query: "how many go files are there",
			steps: []CrystalStep{
				{Tool: "glob", Args: map[string]string{"pattern": "**/*.go"}, ResultVar: "result_glob"},
			},
			responseTmpl: "Go files in the project:\n{result_glob}",
		},
		{
			query: "show dependencies",
			steps: []CrystalStep{
				{Tool: "read", Args: map[string]string{"path": "go.mod"}, ResultVar: "result_read"},
			},
			responseTmpl: "Project dependencies from go.mod:\n{result_read}",
		},
	}
}

// --- Persistence ---

func (cb *CrystalBook) load() {
	if cb.storePath == "" {
		return
	}
	data, err := os.ReadFile(cb.storePath)
	if err != nil {
		return
	}
	var crystals []Crystal
	if err := json.Unmarshal(data, &crystals); err == nil {
		cb.crystals = crystals
	}
}

func (cb *CrystalBook) save() {
	if cb.storePath == "" {
		return
	}
	data, err := json.MarshalIndent(cb.crystals, "", "  ")
	if err != nil {
		return
	}
	safefile.WriteAtomic(cb.storePath, data, 0o644)
}
