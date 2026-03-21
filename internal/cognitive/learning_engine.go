package cognitive

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Conversational Learning Engine — Nous learns from every interaction.
//
// Unlike LLM training which requires massive datasets and GPU hours,
// this engine learns incrementally in real-time from conversation:
//
//   1. Fact Extraction   — "Go was created by Google" → graph edge
//   2. Pattern Absorption — learns sentence structures from the user
//   3. Preference Learning — notices topics the user cares about
//   4. Confidence Growth  — facts mentioned repeatedly become stronger
//   5. Knowledge Decay    — unused facts slowly lose confidence
//
// Zero GPU. Zero datasets. Zero latency. Every conversation teaches.
// -----------------------------------------------------------------------

// LearnedFact is a fact extracted from conversation.
type LearnedFact struct {
	Subject    string    `json:"subject"`
	Relation   RelType   `json:"relation"`
	Object     string    `json:"object"`
	Confidence float64   `json:"confidence"`
	Source     string    `json:"source"` // "conversation", "teaching", "correction"
	LearnedAt  time.Time `json:"learned_at"`
	Mentions   int       `json:"mentions"` // how many times this was mentioned
}

// LearnedPattern is a sentence structure extracted from user speech.
type LearnedPattern struct {
	Template   string    `json:"template"`    // "I think {topic} is {quality}"
	Category   string    `json:"category"`    // "opinion", "fact", "preference"
	UsageCount int       `json:"usage_count"` // how many times this pattern appeared
	LearnedAt  time.Time `json:"learned_at"`
}

// TeachingMode tracks whether the user is actively teaching Nous.
type TeachingMode struct {
	Active    bool
	Topic     string
	Facts     []LearnedFact
	StartedAt time.Time
}

// LearningStats tracks what Nous has learned.
type LearningStats struct {
	TotalFacts         int     `json:"total_facts"`
	FactsFromChat      int     `json:"facts_from_chat"`
	FactsFromTeaching  int     `json:"facts_from_teaching"`
	PatternsLearned    int     `json:"patterns_learned"`
	ConfidenceAvg      float64 `json:"confidence_avg"`
	TopTopics          []string `json:"top_topics"`
	SessionsLearned    int     `json:"sessions_learned"`
	LastLearnedAt      time.Time `json:"last_learned_at"`
}

// LearningEngine is the real-time conversational training system.
type LearningEngine struct {
	graph     *CognitiveGraph
	composer  *Composer

	// Learned patterns — sentence structures absorbed from the user
	patterns  []LearnedPattern

	// Teaching mode — the user is explicitly teaching Nous
	teaching  TeachingMode

	// Statistics
	stats     LearningStats

	// Topic frequency — tracks what the user cares about most
	topicFreq map[string]int

	// Persistence
	dataDir   string
	mu        sync.Mutex
}

// NewLearningEngine creates the conversational training system.
func NewLearningEngine(graph *CognitiveGraph, composer *Composer, dataDir string) *LearningEngine {
	le := &LearningEngine{
		graph:     graph,
		composer:  composer,
		topicFreq: make(map[string]int),
		dataDir:   dataDir,
	}
	le.load()
	return le
}

// -----------------------------------------------------------------------
// Core Learning — extracts knowledge from every conversation turn
// -----------------------------------------------------------------------

// LearnFromConversation processes a user message and extracts any
// learnable knowledge. This is called on every turn automatically.
// Returns the number of new facts learned.
func (le *LearningEngine) LearnFromConversation(userInput string) int {
	le.mu.Lock()
	defer le.mu.Unlock()

	learned := 0

	// 1. Extract structured facts (triples) from the user's message
	learned += le.extractFacts(userInput, "conversation")

	// 2. Detect and absorb sentence patterns
	le.absorbPattern(userInput)

	// 3. Track topic frequency
	topics := extractKeywords(strings.ToLower(userInput))
	for _, topic := range topics {
		le.topicFreq[topic]++
	}

	// 4. Check for teaching signals — user explicitly teaching Nous
	if le.detectTeaching(userInput) {
		learned += le.handleTeaching(userInput)
	}

	// 5. Check for corrections — user correcting Nous
	if correction := le.detectCorrection(userInput); correction != nil {
		le.applyCorrection(correction)
		learned++
	}

	// 6. Check for preference statements
	learned += le.detectPreferences(userInput)

	if learned > 0 {
		le.stats.LastLearnedAt = time.Now()
		le.save()
	}

	return learned
}

// extractFacts pulls structured facts from natural language.
func (le *LearningEngine) extractFacts(text, source string) int {
	// Split into sentences for better extraction
	sentences := splitSentences(text)
	learned := 0

	for _, sentence := range sentences {
		triples := ExtractTriples(sentence)
		for _, triple := range triples {
			if le.addFact(triple.Subject, triple.Relation, triple.Object, source) {
				learned++
			}
		}

		// Also try conversational patterns that ExtractTriples might miss
		learned += le.extractConversational(sentence, source)
	}

	return learned
}

// extractConversational handles informal fact statements that triple
// extraction might miss.
var (
	// "X loves/likes/enjoys Y"
	likesRe = regexp.MustCompile(`(?i)(?:i|he|she|we|they)\s+(?:love|like|enjoy|prefer|adore)s?\s+(.+?)(?:\.|!|$)`)
	// "X hates/dislikes Y"
	dislikesRe = regexp.MustCompile(`(?i)(?:i|he|she|we|they)\s+(?:hate|dislike|can't stand|despise)s?\s+(.+?)(?:\.|!|$)`)
	// "My name is X" / "I'm X"
	nameRe = regexp.MustCompile(`(?i)(?:my name is|i'm|i am|call me)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)`)
	// "I work at/for X"
	workRe = regexp.MustCompile(`(?i)i\s+(?:work|am working)\s+(?:at|for|with)\s+(.+?)(?:\.|!|,|$)`)
	// "I live in X"
	liveRe = regexp.MustCompile(`(?i)i\s+live\s+in\s+(.+?)(?:\.|!|,|$)`)
	// "X means Y" / "X refers to Y"
	meansRe = regexp.MustCompile(`(?i)(.+?)\s+(?:means|refers to|stands for|is defined as)\s+(.+?)(?:\.|!|$)`)
	// "did you know X" / "fun fact: X"
	didYouKnowRe = regexp.MustCompile(`(?i)(?:did you know|fun fact|btw|by the way|fyi)[:\s]+(.+?)(?:\.|!|$)`)
)

func (le *LearningEngine) extractConversational(sentence, source string) int {
	learned := 0

	if m := nameRe.FindStringSubmatch(sentence); len(m) > 1 {
		name := strings.TrimSpace(m[1])
		if le.addFact("user", RelDescribedAs, name, source) {
			learned++
		}
	}

	if m := workRe.FindStringSubmatch(sentence); len(m) > 1 {
		place := strings.TrimSpace(m[1])
		if le.addFact("user", RelPartOf, place, source) {
			learned++
		}
	}

	if m := liveRe.FindStringSubmatch(sentence); len(m) > 1 {
		location := strings.TrimSpace(m[1])
		if le.addFact("user", RelLocatedIn, location, source) {
			learned++
		}
	}

	if m := meansRe.FindStringSubmatch(sentence); len(m) > 2 {
		subject := strings.TrimSpace(m[1])
		object := strings.TrimSpace(m[2])
		if le.addFact(subject, RelIsA, object, source) {
			learned++
		}
	}

	if m := likesRe.FindStringSubmatch(sentence); len(m) > 1 {
		thing := strings.TrimSpace(m[1])
		if le.addFact("user", RelPrefers, thing, source) {
			learned++
		}
	}

	if m := dislikesRe.FindStringSubmatch(sentence); len(m) > 1 {
		thing := strings.TrimSpace(m[1])
		if le.addFact("user", RelDislikes, thing, source) {
			learned++
		}
	}

	// "did you know..." — extract facts from educational statements
	if m := didYouKnowRe.FindStringSubmatch(sentence); len(m) > 1 {
		factText := strings.TrimSpace(m[1])
		learned += le.extractFacts(factText, "teaching")
	}

	return learned
}

// addFact adds a fact to the cognitive graph. Returns true if new.
func (le *LearningEngine) addFact(subject string, rel RelType, object, source string) bool {
	if le.graph == nil {
		return false
	}

	subject = strings.TrimSpace(subject)
	object = strings.TrimSpace(object)
	if subject == "" || object == "" || len(subject) > 100 || len(object) > 200 {
		return false
	}

	// Determine node types from content
	subjType := guessNodeType(subject)
	objType := guessNodeType(object)

	// Create nodes and edge
	fromID := le.graph.EnsureNode(subject, subjType)
	toID := le.graph.EnsureNode(object, objType)
	le.graph.AddEdge(fromID, toID, rel, source)

	// Feed new words to the generative engine's lexicon
	if le.composer != nil && le.composer.Generative != nil {
		if subjType == NodeEntity {
			le.composer.Generative.LearnWord(subject, POSNoun)
		}
		if objType == NodeConcept {
			le.composer.Generative.LearnWord(object, POSNoun)
		}
	}

	// Update stats
	if source == "teaching" {
		le.stats.FactsFromTeaching++
	} else {
		le.stats.FactsFromChat++
	}
	le.stats.TotalFacts = le.graph.NodeCount()

	return true
}

// guessNodeType is defined in triple_extract.go — reused here.

// -----------------------------------------------------------------------
// Teaching Mode — user explicitly teaching Nous
// -----------------------------------------------------------------------

var teachSignals = []string{
	"let me teach you", "i'll teach you", "learn this",
	"remember this", "remember that", "you should know",
	"here's something", "let me tell you about",
	"you need to know", "i want you to learn",
}

func (le *LearningEngine) detectTeaching(input string) bool {
	lower := strings.ToLower(input)
	for _, sig := range teachSignals {
		if strings.Contains(lower, sig) {
			return true
		}
	}
	return false
}

func (le *LearningEngine) handleTeaching(input string) int {
	le.teaching.Active = true
	le.teaching.StartedAt = time.Now()

	// Extract topic from teaching statement
	lower := strings.ToLower(input)
	for _, sig := range teachSignals {
		idx := strings.Index(lower, sig)
		if idx >= 0 {
			after := strings.TrimSpace(input[idx+len(sig):])
			if after != "" {
				le.teaching.Topic = after
				// Try to extract facts from what follows
				return le.extractFacts(after, "teaching")
			}
		}
	}
	return 0
}

// -----------------------------------------------------------------------
// Correction Detection — user correcting Nous
// -----------------------------------------------------------------------

type correction struct {
	wrong   string
	right   string
	subject string
}

var correctionPatterns = []struct {
	re       *regexp.Regexp
	wrongIdx int
	rightIdx int
}{
	// "actually X is Y" / "no, X is Y"
	{regexp.MustCompile(`(?i)(?:actually|no,?\s+)(.+?)\s+(?:is|are)\s+(.+?)(?:\.|!|$)`), 0, 0},
	// "that's wrong, X is Y"
	{regexp.MustCompile(`(?i)(?:that's wrong|that's not right|incorrect)[,.\s]+(.+?)\s+(?:is|are)\s+(.+?)(?:\.|!|$)`), 0, 0},
	// "it's not X, it's Y"
	{regexp.MustCompile(`(?i)it'?s\s+not\s+(.+?),?\s+it'?s\s+(.+?)(?:\.|!|$)`), 1, 2},
}

func (le *LearningEngine) detectCorrection(input string) *correction {
	for _, cp := range correctionPatterns {
		if m := cp.re.FindStringSubmatch(input); len(m) > 2 {
			return &correction{
				wrong: strings.TrimSpace(m[1]),
				right: strings.TrimSpace(m[2]),
			}
		}
	}
	return nil
}

func (le *LearningEngine) applyCorrection(c *correction) {
	if le.graph == nil {
		return
	}

	// Extract facts from the correction (the "right" version)
	rightText := c.wrong + " is " + c.right
	le.extractFacts(rightText, "correction")

	// Boost the confidence of the correction
	// The corrected fact gets higher confidence than conversation-learned ones
	nodeID := nodeID(c.wrong)
	if node := le.graph.GetNode(nodeID); node != nil {
		node.Confidence = math.Min(1.0, node.Confidence+0.2)
	}
}

// -----------------------------------------------------------------------
// Preference Detection — what the user likes, dislikes, values
// -----------------------------------------------------------------------

func (le *LearningEngine) detectPreferences(input string) int {
	lower := strings.ToLower(input)
	learned := 0

	// "I prefer X over Y"
	preferRe := regexp.MustCompile(`(?i)i\s+prefer\s+(.+?)\s+(?:over|to|rather than)\s+(.+?)(?:\.|!|$)`)
	if m := preferRe.FindStringSubmatch(input); len(m) > 2 {
		preferred := strings.TrimSpace(m[1])
		other := strings.TrimSpace(m[2])
		if le.addFact("user", RelPrefers, preferred, "preference") {
			learned++
		}
		// Also note what they don't prefer
		if le.addFact("user", RelDislikes, other, "preference") {
			learned++
		}
	}

	// "my favorite X is Y"
	favRe := regexp.MustCompile(`(?i)my\s+(?:favorite|favourite)\s+(.+?)\s+is\s+(.+?)(?:\.|!|$)`)
	if m := favRe.FindStringSubmatch(input); len(m) > 2 {
		category := strings.TrimSpace(m[1])
		value := strings.TrimSpace(m[2])
		if le.addFact(value, RelIsA, "favorite "+category, "preference") {
			learned++
		}
	}

	// "I'm interested in X" / "I'm passionate about X"
	for _, prefix := range []string{"interested in", "passionate about",
		"fascinated by", "curious about", "into"} {
		if idx := strings.Index(lower, "i'm "+prefix+" "); idx >= 0 {
			after := strings.TrimSpace(input[idx+len("i'm "+prefix+" "):])
			after = strings.TrimRight(after, ".!?,;")
			if after != "" && le.addFact("user", RelPrefers, after, "preference") {
				learned++
			}
		}
	}

	return learned
}

// -----------------------------------------------------------------------
// Pattern Absorption — learns sentence structures from user speech
// -----------------------------------------------------------------------

// absorbPattern extracts reusable sentence templates from user input.
func (le *LearningEngine) absorbPattern(input string) {
	// Only absorb patterns from longer, well-formed sentences
	if len(input) < 15 || !strings.ContainsAny(input, ".!?") && len(input) < 40 {
		return
	}

	category := classifyPatternCategory(input)
	template := abstractToTemplate(input)
	if template == "" {
		return
	}

	// Check if we already have this pattern
	for i, p := range le.patterns {
		if p.Template == template {
			le.patterns[i].UsageCount++
			return
		}
	}

	// New pattern
	le.patterns = append(le.patterns, LearnedPattern{
		Template:   template,
		Category:   category,
		UsageCount: 1,
		LearnedAt:  time.Now(),
	})
	le.stats.PatternsLearned = len(le.patterns)

	// Keep max 200 patterns, prune least-used
	if len(le.patterns) > 200 {
		le.prunePatterns()
	}
}

// abstractToTemplate converts a sentence into a reusable template.
// "I think Go is a great language" → "I think {topic} is a {quality} language"
// Only replaces the 1-3 most salient words (entities, quality words, or the
// first long noun) — keeps the rest as literal structure.
func abstractToTemplate(input string) string {
	words := strings.Fields(input)
	if len(words) < 4 {
		return ""
	}

	// First pass: identify which words are slottable and pick the best ones
	type slotCandidate struct {
		idx      int
		slotType string // "{entity}", "{quality}", "{topic}"
		priority int    // lower = more salient
	}
	var candidates []slotCandidate
	for i, word := range words {
		clean := strings.Trim(strings.ToLower(word), ".,!?;:")
		if isFunctionWord(clean) || i == 0 || isCommonVerb(clean) {
			continue
		}
		if isCapitalized(word) {
			candidates = append(candidates, slotCandidate{i, "{entity}", 0})
		} else if isQualityWord(clean) {
			candidates = append(candidates, slotCandidate{i, "{quality}", 1})
		} else if len(clean) > 4 {
			candidates = append(candidates, slotCandidate{i, "{topic}", 2})
		}
	}

	if len(candidates) == 0 {
		return ""
	}

	// Sort by priority (entities first, then quality, then topics)
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].priority < candidates[i].priority {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	// Take at most 3 slots
	maxSlots := 3
	if maxSlots > len(candidates) {
		maxSlots = len(candidates)
	}
	slotted := make(map[int]string)
	for i := 0; i < maxSlots; i++ {
		slotted[candidates[i].idx] = candidates[i].slotType
	}

	// Second pass: build template
	var template strings.Builder
	for i, word := range words {
		if slot, ok := slotted[i]; ok {
			template.WriteString(slot)
		} else {
			template.WriteString(word)
		}
		if i < len(words)-1 {
			template.WriteByte(' ')
		}
	}

	return template.String()
}

func isFunctionWord(w string) bool {
	fwords := map[string]bool{
		"i": true, "you": true, "he": true, "she": true, "it": true,
		"we": true, "they": true, "the": true, "a": true, "an": true,
		"is": true, "are": true, "was": true, "were": true, "be": true,
		"am": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "having": true,
		"do": true, "does": true, "did": true,
		"will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "can": true, "must": true,
		"in": true, "on": true, "at": true, "to": true, "for": true,
		"of": true, "with": true, "by": true, "from": true, "about": true,
		"into": true, "through": true, "during": true, "before": true,
		"after": true, "above": true, "below": true, "between": true,
		"and": true, "or": true, "but": true, "nor": true,
		"not": true, "no": true, "so": true, "yet": true,
		"this": true, "that": true, "these": true, "those": true,
		"my": true, "your": true, "his": true, "her": true, "its": true,
		"our": true, "their": true,
		"what": true, "which": true, "who": true, "whom": true,
		"how": true, "when": true, "where": true, "why": true,
		"if": true, "then": true, "than": true, "as": true,
		"think": true, "know": true, "like": true, "really": true,
		"very": true, "quite": true, "just": true, "also": true,
	}
	return fwords[w]
}

func isCapitalized(w string) bool {
	return len(w) > 0 && w[0] >= 'A' && w[0] <= 'Z'
}

func isQualityWord(w string) bool {
	qualities := map[string]bool{
		"great": true, "good": true, "bad": true, "amazing": true,
		"terrible": true, "awesome": true, "fantastic": true, "horrible": true,
		"excellent": true, "wonderful": true, "beautiful": true, "ugly": true,
		"fast": true, "slow": true, "efficient": true, "powerful": true,
		"simple": true, "complex": true, "easy": true, "hard": true,
		"innovative": true, "revolutionary": true, "brilliant": true,
	}
	return qualities[w]
}

func isCommonVerb(w string) bool {
	verbs := map[string]bool{
		"is": true, "are": true, "was": true, "were": true,
		"go": true, "get": true, "make": true, "take": true,
		"come": true, "see": true, "say": true, "give": true,
		"use": true, "find": true, "tell": true, "work": true,
		"call": true, "try": true, "need": true, "want": true,
		"look": true, "put": true, "keep": true, "let": true,
		"begin": true, "seem": true, "help": true, "show": true,
		"hear": true, "play": true, "run": true, "move": true,
	}
	return verbs[w]
}

func classifyPatternCategory(input string) string {
	lower := strings.ToLower(input)
	if strings.Contains(lower, "i think") || strings.Contains(lower, "i believe") ||
		strings.Contains(lower, "in my opinion") {
		return "opinion"
	}
	if strings.Contains(lower, "i like") || strings.Contains(lower, "i love") ||
		strings.Contains(lower, "i prefer") || strings.Contains(lower, "i enjoy") {
		return "preference"
	}
	if strings.HasSuffix(strings.TrimSpace(input), "?") {
		return "question"
	}
	return "statement"
}

func (le *LearningEngine) prunePatterns() {
	// Sort by usage count descending
	for i := 0; i < len(le.patterns); i++ {
		for j := i + 1; j < len(le.patterns); j++ {
			if le.patterns[j].UsageCount > le.patterns[i].UsageCount {
				le.patterns[i], le.patterns[j] = le.patterns[j], le.patterns[i]
			}
		}
	}
	// Keep top 150
	if len(le.patterns) > 150 {
		le.patterns = le.patterns[:150]
	}
}

// -----------------------------------------------------------------------
// Knowledge Decay — unused knowledge slowly fades
// -----------------------------------------------------------------------

// DecayKnowledge reduces confidence on facts not accessed recently.
// Call this periodically (e.g., once per session start).
func (le *LearningEngine) DecayKnowledge() int {
	if le.graph == nil {
		return 0
	}

	le.mu.Lock()
	defer le.mu.Unlock()

	decayed := 0
	threshold := time.Now().Add(-7 * 24 * time.Hour) // 1 week

	nodes := le.graph.FindNodes("")
	for _, node := range nodes {
		if node.LastActive.Before(threshold) && node.Confidence > 0.2 {
			// Slow decay: lose 5% confidence per decay cycle
			node.Confidence *= 0.95
			decayed++
		}
	}

	return decayed
}

// -----------------------------------------------------------------------
// Stats and Reporting
// -----------------------------------------------------------------------

// Stats returns current learning statistics.
func (le *LearningEngine) Stats() LearningStats {
	le.mu.Lock()
	defer le.mu.Unlock()

	// Calculate top topics
	type topicCount struct {
		topic string
		count int
	}
	var topics []topicCount
	for t, c := range le.topicFreq {
		topics = append(topics, topicCount{t, c})
	}
	for i := 0; i < len(topics); i++ {
		for j := i + 1; j < len(topics); j++ {
			if topics[j].count > topics[i].count {
				topics[i], topics[j] = topics[j], topics[i]
			}
		}
	}

	le.stats.TopTopics = nil
	for i := 0; i < 5 && i < len(topics); i++ {
		le.stats.TopTopics = append(le.stats.TopTopics, topics[i].topic)
	}

	le.stats.PatternsLearned = len(le.patterns)
	if le.graph != nil {
		le.stats.TotalFacts = le.graph.NodeCount()
	}

	return le.stats
}

// LearnedPatterns returns all learned sentence patterns.
func (le *LearningEngine) LearnedPatterns() []LearnedPattern {
	le.mu.Lock()
	defer le.mu.Unlock()
	result := make([]LearnedPattern, len(le.patterns))
	copy(result, le.patterns)
	return result
}

// TopicInterest returns how interested the user is in a topic (0-100).
func (le *LearningEngine) TopicInterest(topic string) int {
	le.mu.Lock()
	defer le.mu.Unlock()

	count := le.topicFreq[strings.ToLower(topic)]
	if count == 0 {
		return 0
	}

	// Find max frequency for normalization
	maxFreq := 1
	for _, c := range le.topicFreq {
		if c > maxFreq {
			maxFreq = c
		}
	}

	return int(math.Min(100, float64(count)/float64(maxFreq)*100))
}

// -----------------------------------------------------------------------
// Persistence — saves/loads learning data
// -----------------------------------------------------------------------

type learningData struct {
	Patterns  []LearnedPattern `json:"patterns"`
	TopicFreq map[string]int   `json:"topic_freq"`
	Stats     LearningStats    `json:"stats"`
}

func (le *LearningEngine) save() {
	if le.dataDir == "" {
		return
	}

	data := learningData{
		Patterns:  le.patterns,
		TopicFreq: le.topicFreq,
		Stats:     le.stats,
	}

	path := filepath.Join(le.dataDir, "learning_engine.json")
	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(path, b, 0644)
}

func (le *LearningEngine) load() {
	if le.dataDir == "" {
		return
	}

	path := filepath.Join(le.dataDir, "learning_engine.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var data learningData
	if err := json.Unmarshal(b, &data); err != nil {
		return
	}

	le.patterns = data.Patterns
	if data.TopicFreq != nil {
		le.topicFreq = data.TopicFreq
	}
	le.stats = data.Stats
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// splitSentences is defined in extractive.go — reused here.

// FormatLearningReport generates a human-readable report of what Nous has learned.
func (le *LearningEngine) FormatLearningReport() string {
	stats := le.Stats()

	var b strings.Builder
	b.WriteString("Learning Report\n")
	b.WriteString(strings.Repeat("─", 40) + "\n")
	b.WriteString(fmt.Sprintf("Knowledge nodes:    %d\n", stats.TotalFacts))
	b.WriteString(fmt.Sprintf("Facts from chat:    %d\n", stats.FactsFromChat))
	b.WriteString(fmt.Sprintf("Facts from teaching:%d\n", stats.FactsFromTeaching))
	b.WriteString(fmt.Sprintf("Patterns absorbed:  %d\n", stats.PatternsLearned))

	if len(stats.TopTopics) > 0 {
		b.WriteString(fmt.Sprintf("Top interests:      %s\n", strings.Join(stats.TopTopics, ", ")))
	}

	if !stats.LastLearnedAt.IsZero() {
		b.WriteString(fmt.Sprintf("Last learned:       %s\n", stats.LastLearnedAt.Format("2006-01-02 15:04")))
	}

	return b.String()
}
