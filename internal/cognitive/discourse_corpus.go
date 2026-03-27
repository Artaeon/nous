package cognitive

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Discourse Corpus — Layer 2b: sentences indexed by discourse function.
//
// While the SentenceCorpus indexes by RELATION (is_a, has, located_in),
// the DiscourseCorpus indexes by HOW a sentence communicates:
//
//   defines:      "X is a Y that Z"
//   explains_why: "X happens because Y"
//   gives_example:"For example, X"
//   compares:     "Unlike X, Y"
//   evaluates:    "X is considered one of the most Y"
//   describes:    "X works by Y", "X involves Y"
//   consequences: "This leads to Y", "As a result, Y"
//   context:      "X was founded in Y", "X originated in Y"
//
// These patterns are mined from Wikipedia — no handwriting.
// At response time, the composer picks a discourse plan (e.g., for
// "what's your opinion on X?" → [defines, evaluates, compares])
// and retrieves one real sentence per slot.
// -----------------------------------------------------------------------

// DiscourseFunc represents how a sentence communicates.
type DiscourseFunc int

const (
	DFDefines      DiscourseFunc = iota // "X is a Y", "X refers to Y"
	DFExplainsWhy                       // "because Y", "due to Y"
	DFGivesExample                      // "for example", "such as"
	DFCompares                          // "unlike X", "compared to"
	DFEvaluates                         // "is considered", "is known for"
	DFDescribes                         // "works by", "involves", "consists of"
	DFConsequence                       // "leads to", "results in", "therefore"
	DFContext                           // "was founded in", "originated in"
	DFQuantifies                        // "has N", "contains N", numbers
)

var dfNames = [...]string{
	"defines", "explains_why", "gives_example", "compares",
	"evaluates", "describes", "consequence", "context", "quantifies",
}

func (df DiscourseFunc) String() string {
	if int(df) < len(dfNames) {
		return dfNames[df]
	}
	return "unknown"
}

func parseDFString(s string) DiscourseFunc {
	for i, name := range dfNames {
		if name == s {
			return DiscourseFunc(i)
		}
	}
	return DFDefines
}

// DiscourseSentence is a real sentence tagged with its discourse function
// and the topic it's about.
type DiscourseSentence struct {
	Sentence string        `json:"s"`
	Topic    string        `json:"t"`    // what the sentence is about
	Function DiscourseFunc `json:"f"`    // how it communicates
	Quality  int           `json:"q"`    // 0-3: higher = cleaner sentence
}

// DiscourseCorpus stores sentences indexed by discourse function.
type DiscourseCorpus struct {
	mu        sync.RWMutex
	sentences map[DiscourseFunc][]DiscourseSentence
	byTopic   map[string][]int // topic → indices into all sentences
	allSents  []DiscourseSentence
	totalSize int
}

// NewDiscourseCorpus creates an empty discourse corpus.
func NewDiscourseCorpus() *DiscourseCorpus {
	return &DiscourseCorpus{
		sentences: make(map[DiscourseFunc][]DiscourseSentence),
		byTopic:   make(map[string][]int),
	}
}

// Add inserts a discourse sentence into the corpus.
func (dc *DiscourseCorpus) Add(ds DiscourseSentence) {
	dc.mu.Lock()
	defer dc.mu.Unlock()
	idx := len(dc.allSents)
	dc.allSents = append(dc.allSents, ds)
	dc.sentences[ds.Function] = append(dc.sentences[ds.Function], ds)
	topicLower := strings.ToLower(ds.Topic)
	dc.byTopic[topicLower] = append(dc.byTopic[topicLower], idx)
	dc.totalSize++
}

// Size returns total sentence count.
func (dc *DiscourseCorpus) Size() int {
	dc.mu.RLock()
	defer dc.mu.RUnlock()
	return dc.totalSize
}

// FunctionCounts returns sentence count per discourse function.
func (dc *DiscourseCorpus) FunctionCounts() map[DiscourseFunc]int {
	dc.mu.RLock()
	defer dc.mu.RUnlock()
	counts := make(map[DiscourseFunc]int)
	for f, sents := range dc.sentences {
		counts[f] = len(sents)
	}
	return counts
}

// -----------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------

type dcEntry struct {
	S string `json:"s"` // sentence
	T string `json:"t"` // topic
	F string `json:"f"` // function name
	Q int    `json:"q"` // quality
}

func (dc *DiscourseCorpus) Save(path string) error {
	dc.mu.RLock()
	defer dc.mu.RUnlock()
	var entries []dcEntry
	for _, ds := range dc.allSents {
		entries = append(entries, dcEntry{
			S: ds.Sentence,
			T: ds.Topic,
			F: ds.Function.String(),
			Q: ds.Quality,
		})
	}
	data, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (dc *DiscourseCorpus) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var entries []dcEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}
	dc.mu.Lock()
	defer dc.mu.Unlock()
	dc.sentences = make(map[DiscourseFunc][]DiscourseSentence)
	dc.byTopic = make(map[string][]int)
	dc.allSents = nil
	dc.totalSize = 0
	for _, e := range entries {
		ds := DiscourseSentence{
			Sentence: e.S,
			Topic:    e.T,
			Function: parseDFString(e.F),
			Quality:  e.Q,
		}
		idx := len(dc.allSents)
		dc.allSents = append(dc.allSents, ds)
		dc.sentences[ds.Function] = append(dc.sentences[ds.Function], ds)
		dc.byTopic[strings.ToLower(ds.Topic)] = append(dc.byTopic[strings.ToLower(ds.Topic)], idx)
		dc.totalSize++
	}
	return nil
}

// -----------------------------------------------------------------------
// Retrieval — find sentences by topic and discourse function.
// -----------------------------------------------------------------------

// Retrieve finds the best sentence for a topic with a specific discourse
// function. Strongly prefers sentences FROM the topic's own article over
// sentences that merely mention the topic word.
func (dc *DiscourseCorpus) Retrieve(topic string, fn DiscourseFunc) string {
	dc.mu.RLock()
	defer dc.mu.RUnlock()

	topicLower := strings.ToLower(topic)

	// Exact topic match: sentences from the topic's own Wikipedia article.
	// These are always the best quality — they're ABOUT the topic, not
	// tangentially mentioning it.
	if indices, ok := dc.byTopic[topicLower]; ok {
		var best string
		bestQ := -1
		for _, idx := range indices {
			ds := dc.allSents[idx]
			if ds.Function == fn && ds.Quality > bestQ {
				best = ds.Sentence
				bestQ = ds.Quality
			}
		}
		if best != "" {
			return best
		}
	}

	// No exact article match — don't fall back to word-mention matches.
	// Those produce wrong-entity results ("Einstein on the Beach" for "einstein").
	return ""
}

// RetrieveMulti finds sentences for a topic across multiple discourse
// functions, returning one sentence per function slot. This is the
// core composition function — it assembles a multi-sentence response
// from real human text.
func (dc *DiscourseCorpus) RetrieveMulti(topic string, functions []DiscourseFunc) []string {
	var result []string
	seen := make(map[string]bool) // dedup

	for _, fn := range functions {
		sent := dc.Retrieve(topic, fn)
		if sent != "" && !seen[sent] {
			seen[sent] = true
			result = append(result, sent)
		}
	}
	return result
}

// RetrieveVaried finds a sentence with time-based variation.
func (dc *DiscourseCorpus) RetrieveVaried(topic string, fn DiscourseFunc) string {
	dc.mu.RLock()
	defer dc.mu.RUnlock()

	topicLower := strings.ToLower(topic)

	// Collect all matching sentences.
	var matches []DiscourseSentence

	// Exact topic matches first.
	if indices, ok := dc.byTopic[topicLower]; ok {
		for _, idx := range indices {
			ds := dc.allSents[idx]
			if ds.Function == fn {
				matches = append(matches, ds)
			}
		}
	}

	// Also sentences that mention the topic.
	if len(matches) == 0 {
		for _, ds := range dc.sentences[fn] {
			if strings.Contains(strings.ToLower(ds.Sentence), topicLower) {
				matches = append(matches, ds)
				if len(matches) >= 10 {
					break
				}
			}
		}
	}

	if len(matches) == 0 {
		return ""
	}

	// Time-based pick for variety.
	pick := int(time.Now().UnixNano()/1000) % len(matches)
	return matches[pick].Sentence
}

// -----------------------------------------------------------------------
// Extraction — mine discourse-typed sentences from Wikipedia articles.
// -----------------------------------------------------------------------

// ExtractDiscourseSentences mines sentences from a Wikipedia article and
// tags each with its discourse function. No handwriting — pure pattern
// detection on real text.
func ExtractDiscourseSentences(title, plainText string) []DiscourseSentence {
	if plainText == "" {
		return nil
	}

	sentences := splitSentences(plainText)
	var result []DiscourseSentence
	seen := make(map[string]bool)
	titleLower := strings.ToLower(title)

	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)
		// Quality filters.
		if len(sent) < 25 || len(sent) > 250 {
			continue
		}
		if isBoilerplate(sent) {
			continue
		}
		// Must start with capital letter.
		if len(sent) > 0 && (sent[0] < 'A' || sent[0] > 'Z') {
			continue
		}
		// No wiki markup remnants.
		if strings.Contains(sent, "]]") || strings.Contains(sent, "[[") {
			continue
		}
		// Add period if missing.
		if !strings.HasSuffix(sent, ".") && !strings.HasSuffix(sent, "!") && !strings.HasSuffix(sent, "?") {
			sent += "."
		}
		// Dedup.
		if seen[sent] {
			continue
		}

		// Detect discourse function from sentence structure.
		fn, quality := classifyDiscourseFunction(sent, titleLower)
		if quality < 0 {
			continue // not a useful discourse pattern
		}

		seen[sent] = true
		result = append(result, DiscourseSentence{
			Sentence: sent,
			Topic:    title,
			Function: fn,
			Quality:  quality,
		})
	}
	return result
}

// classifyDiscourseFunction detects how a sentence communicates.
// Returns the discourse function and a quality score (0-3, -1 = skip).
func classifyDiscourseFunction(sent, topicLower string) (DiscourseFunc, int) {
	lower := strings.ToLower(sent)

	// Must be relevant to the article topic (mention it or be about it).
	if !strings.Contains(lower, topicLower) {
		// Allow if topic is multi-word and first word matches.
		words := strings.Fields(topicLower)
		if len(words) <= 1 || !strings.Contains(lower, words[0]) {
			return 0, -1
		}
	}

	// Explains why: causal patterns.
	if containsCausalPattern(lower) {
		return DFExplainsWhy, qualityScore(sent)
	}

	// Gives example: exemplification patterns.
	if containsExamplePattern(lower) {
		return DFGivesExample, qualityScore(sent)
	}

	// Compares: comparative patterns.
	if containsComparePattern(lower) {
		return DFCompares, qualityScore(sent)
	}

	// Evaluates: evaluative patterns.
	if containsEvalPattern(lower) {
		return DFEvaluates, qualityScore(sent)
	}

	// Consequences: result patterns.
	if containsConsequencePattern(lower) {
		return DFConsequence, qualityScore(sent)
	}

	// Describes process: procedural patterns.
	if containsProcessPattern(lower) {
		return DFDescribes, qualityScore(sent)
	}

	// Context: temporal/origin patterns.
	if containsContextPattern(lower) {
		return DFContext, qualityScore(sent)
	}

	// Quantifies: contains specific numbers.
	if containsQuantification(lower) {
		return DFQuantifies, qualityScore(sent)
	}

	// Defines: "X is a/an Y" at the start — most common, check last.
	if containsDefinitionPattern(lower) {
		return DFDefines, qualityScore(sent)
	}

	return 0, -1 // no recognized pattern
}

func qualityScore(sent string) int {
	q := 1 // base quality
	if len(sent) >= 40 && len(sent) <= 150 {
		q++ // ideal length
	}
	if strings.HasSuffix(sent, ".") {
		q++ // proper ending
	}
	return q
}

// -----------------------------------------------------------------------
// Pattern detectors — each checks for a specific discourse function.
// These are NOT templates — they're detectors that find existing patterns
// in human-written text.
// -----------------------------------------------------------------------

func containsCausalPattern(lower string) bool {
	patterns := []string{
		" because ", " due to ", " caused by ", " the reason ",
		" as a result of ", " owing to ", " thanks to ",
		" this is because ", " this is why ", " this means ",
		" so that ", " in order to ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsExamplePattern(lower string) bool {
	patterns := []string{
		"for example", "for instance", "such as ", " including ",
		" one example ", " examples include ", " an example of ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsComparePattern(lower string) bool {
	patterns := []string{
		"unlike ", "compared to ", "in contrast ", "whereas ",
		"on the other hand", " while ", "similar to ",
		" differ ", " different from ", " more than ", " less than ",
		" bigger than ", " smaller than ", " faster than ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsEvalPattern(lower string) bool {
	patterns := []string{
		" is considered ", " is regarded as ", " is known for ",
		" is famous for ", " is one of the most ", " is one of the ",
		" is widely ", " is often ", " is commonly ",
		" is believed to ", " is thought to ",
		" most important ", " most significant ", " most popular ",
		" best known ", " well known ", " widely known ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsConsequencePattern(lower string) bool {
	patterns := []string{
		" leads to ", " results in ", " therefore ",
		" consequently ", " as a result ", " this led to ",
		" this caused ", " this resulted ",
		" which meant ", " which means ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsProcessPattern(lower string) bool {
	patterns := []string{
		" works by ", " involves ", " consists of ",
		" is made by ", " is done by ", " is produced ",
		" is created by ", " is formed ",
		" the process ", " the method ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsContextPattern(lower string) bool {
	patterns := []string{
		" was founded in ", " was established in ",
		" originated in ", " dates back to ",
		" was created in ", " was built in ",
		" first appeared in ", " was introduced in ",
		" began in ", " started in ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsQuantification(lower string) bool {
	// Contains a number that's part of a measurement or statistic.
	patterns := []string{
		" million ", " billion ", " thousand ",
		" percent ", "%",
		" km ", " miles ", " metres ", " meters ",
		" population of ", " area of ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func containsDefinitionPattern(lower string) bool {
	patterns := []string{
		" is a ", " is an ", " is the ",
		" are a ", " are an ", " are the ",
		" was a ", " was an ", " was the ",
		" were a ", " were an ",
		" refers to ", " is defined as ",
		" means ", " is called ",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Response Composition — the core generative engine.
//
// Given a topic and a query type, select a discourse plan (sequence of
// discourse functions), retrieve one real sentence per slot, and
// assemble them into a coherent response.
// -----------------------------------------------------------------------

// DiscourseResponse plans which discourse functions to use for a given
// query type and retrieves real sentences for each.
func (dc *DiscourseCorpus) ComposeResponse(topic string, queryType string) string {
	// Select discourse plan based on query type.
	plan := selectDiscoursePlan(queryType)

	// Retrieve one sentence per slot.
	sentences := dc.RetrieveMulti(topic, plan)

	if len(sentences) == 0 {
		return ""
	}

	return strings.Join(sentences, " ")
}

// selectDiscoursePlan chooses which discourse functions to use.
func selectDiscoursePlan(queryType string) []DiscourseFunc {
	switch queryType {
	case "what_is", "define", "explain":
		return []DiscourseFunc{DFDefines, DFDescribes, DFContext}
	case "why":
		return []DiscourseFunc{DFDefines, DFExplainsWhy, DFConsequence}
	case "opinion":
		return []DiscourseFunc{DFDefines, DFEvaluates, DFCompares}
	case "how":
		return []DiscourseFunc{DFDefines, DFDescribes, DFGivesExample}
	case "compare":
		return []DiscourseFunc{DFDefines, DFCompares, DFEvaluates}
	case "example":
		return []DiscourseFunc{DFGivesExample, DFDescribes}
	default:
		return []DiscourseFunc{DFDefines, DFDescribes, DFEvaluates}
	}
}
