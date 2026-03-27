package cognitive

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Opinion Formation Engine — accumulates evidence from conversations,
// episodic memory, and knowledge graph observations to form genuine
// opinions over time.
//
// Unlike an LLM that resets every turn, Nous can say "I think X because
// I've seen Y" and mean it. Opinions persist, accumulate, evolve, and
// decay. They are revisable: challenge one and it updates.
//
// Evidence flows in from four sources:
//   - Episodic memory (past conversations)
//   - Knowledge graph (structured facts)
//   - Behavioral observation (user patterns)
//   - Current conversation (evaluative language)
//
// Over weeks and months, opinions crystallize. A topic mentioned once
// yields "I don't have a strong opinion yet." A topic discussed twenty
// times with consistent sentiment yields a confident, articulated stance.
// -----------------------------------------------------------------------

// EvidenceType classifies the source of evidence.
type EvidenceType int

const (
	EvidenceEpisodic     EvidenceType = iota // from past conversations
	EvidenceGraph                            // from knowledge graph facts
	EvidenceObservation                      // from watching user behavior
	EvidenceConversation                     // from current conversation
)

// Evidence is a single piece of support for or against an opinion.
type Evidence struct {
	Type      EvidenceType `json:"type"`
	Source    string       `json:"source"`    // where it came from (episode ID, graph, etc.)
	Content   string       `json:"content"`   // what the evidence says
	Valence   float64      `json:"valence"`   // -1.0 (against) to +1.0 (for)
	Weight    float64      `json:"weight"`    // importance (0.0-1.0)
	Timestamp time.Time    `json:"timestamp"`
}

// Opinion is Nous's accumulated perspective on a topic.
type Opinion struct {
	Topic        string     `json:"topic"`
	Position     string     `json:"position"`      // "positive", "negative", "mixed", "uncertain"
	Stance       float64    `json:"stance"`         // -1.0 (strongly against) to +1.0 (strongly for)
	Confidence   float64    `json:"confidence"`     // 0.0-1.0
	Summary      string     `json:"summary"`        // articulated opinion (1-2 sentences)
	Evidence     []Evidence `json:"evidence"`
	ForCount     int        `json:"for_count"`      // evidence pieces supporting
	AgainstCount int        `json:"against_count"`  // evidence pieces opposing
	LastUpdated  time.Time  `json:"last_updated"`
	FormCount    int        `json:"form_count"`     // times opinion was updated
}

// OpinionEngine accumulates evidence and forms opinions over time.
type OpinionEngine struct {
	mu       sync.RWMutex
	opinions map[string]*Opinion // normalized topic → opinion
	savePath string
}

// NewOpinionEngine creates a new opinion engine with the given save path.
func NewOpinionEngine(savePath string) *OpinionEngine {
	oe := &OpinionEngine{
		opinions: make(map[string]*Opinion),
		savePath: savePath,
	}
	oe.Load()
	return oe
}

// maxEvidencePerOpinion caps evidence lists to keep them manageable.
const maxEvidencePerOpinion = 20

// normalizeTopic lowercases, trims, and does simple singularization.
func normalizeTopic(topic string) string {
	t := strings.ToLower(strings.TrimSpace(topic))
	// Simple singularization — strip trailing 's' for common plurals,
	// but avoid mangling words like "is", "was", "stress", "success".
	if len(t) > 3 && strings.HasSuffix(t, "s") && !strings.HasSuffix(t, "ss") && !strings.HasSuffix(t, "us") {
		t = t[:len(t)-1]
	}
	return t
}

// AccumulateEvidence adds evidence to an existing or new opinion and
// recomputes stance, confidence, and position.
func (oe *OpinionEngine) AccumulateEvidence(topic string, ev Evidence) {
	oe.mu.Lock()
	defer oe.mu.Unlock()

	key := normalizeTopic(topic)
	op, ok := oe.opinions[key]
	if !ok {
		op = &Opinion{
			Topic:       key,
			Position:    "uncertain",
			LastUpdated: ev.Timestamp,
		}
		oe.opinions[key] = op
	}

	// Clamp valence and weight.
	ev.Valence = clampF(ev.Valence, -1.0, 1.0)
	ev.Weight = clampF(ev.Weight, 0.0, 1.0)
	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now()
	}

	op.Evidence = append(op.Evidence, ev)
	op.FormCount++
	op.LastUpdated = ev.Timestamp

	// Cap evidence at maxEvidencePerOpinion, keeping the strongest.
	if len(op.Evidence) > maxEvidencePerOpinion {
		pruneEvidence(op)
	}

	recomputeOpinion(op)
	op.Summary = ArticulateOpinion(op)
}

// GetOpinion retrieves an opinion by normalized topic. Returns nil if none.
func (oe *OpinionEngine) GetOpinion(topic string) *Opinion {
	oe.mu.RLock()
	defer oe.mu.RUnlock()

	key := normalizeTopic(topic)
	op, ok := oe.opinions[key]
	if !ok {
		return nil
	}
	// Return a copy so callers can't mutate internal state.
	cp := *op
	cp.Evidence = make([]Evidence, len(op.Evidence))
	copy(cp.Evidence, op.Evidence)
	return &cp
}

// FormOpinion actively gathers evidence from the knowledge graph and
// episodic memory to form or update an opinion on a topic.
func (oe *OpinionEngine) FormOpinion(topic string, graph *CognitiveGraph, episodic *memory.EpisodicMemory) *Opinion {
	now := time.Now()

	// Gather facts from knowledge graph.
	if graph != nil {
		facts := graph.LookupFacts(topic, 10)
		for _, fact := range facts {
			valence := evaluateFactSentiment(fact)
			oe.AccumulateEvidence(topic, Evidence{
				Type:      EvidenceGraph,
				Source:    "knowledge_graph",
				Content:   fact,
				Valence:   valence,
				Weight:    0.6,
				Timestamp: now,
			})
		}
	}

	// Gather episodes from episodic memory.
	if episodic != nil {
		episodes := episodic.SearchKeyword(topic, 10)
		for _, ep := range episodes {
			text := ep.Input + " " + ep.Output
			valence := evaluateTextSentiment(text)
			weight := 0.5
			if ep.Success {
				weight = 0.7
			}
			oe.AccumulateEvidence(topic, Evidence{
				Type:      EvidenceEpisodic,
				Source:    ep.ID,
				Content:   opTruncate(ep.Input, 120),
				Valence:   valence,
				Weight:    weight,
				Timestamp: ep.Timestamp,
			})
		}
	}

	return oe.GetOpinion(topic)
}

// LearnFromConversation extracts evaluative language from user input and
// accumulates evidence for mentioned topics.
func (oe *OpinionEngine) LearnFromConversation(input string, topics []string) {
	evals := extractEvaluations(input)
	now := time.Now()

	for _, ev := range evals {
		oe.AccumulateEvidence(ev.topic, Evidence{
			Type:      EvidenceConversation,
			Source:    "conversation",
			Content:   opTruncate(input, 120),
			Valence:   ev.valence,
			Weight:    ev.weight,
			Timestamp: now,
		})
	}

	// Also note that the user brought up these topics — mild positive signal
	// (engaging with something suggests at least some interest).
	for _, t := range topics {
		found := false
		for _, ev := range evals {
			if normalizeTopic(ev.topic) == normalizeTopic(t) {
				found = true
				break
			}
		}
		if !found {
			oe.AccumulateEvidence(t, Evidence{
				Type:      EvidenceObservation,
				Source:    "topic_mention",
				Content:   "user discussed " + t,
				Valence:   0.1, // very mild positive — just engagement
				Weight:    0.2,
				Timestamp: now,
			})
		}
	}
}

// ArticulateOpinion turns a structured opinion into natural language.
func ArticulateOpinion(op *Opinion) string {
	if op == nil || len(op.Evidence) == 0 {
		return ""
	}

	strongest := strongestEvidence(op)
	n := len(op.Evidence)

	switch {
	case op.Confidence < 0.3:
		// Uncertain — not enough evidence.
		if n == 1 {
			return "I don't have a strong opinion on " + op.Topic + " yet — I've only seen one mention so far."
		}
		return "I haven't formed a clear opinion on " + op.Topic + " yet. I've seen a few things but nothing conclusive."

	case op.Position == "positive":
		if op.Confidence > 0.7 {
			return "Based on " + countWord(n) + " interactions, I think " + op.Topic + " is genuinely good. " + evidencePhrase(strongest)
		}
		return "From what I've gathered, " + op.Topic + " seems solid. " + evidencePhrase(strongest)

	case op.Position == "negative":
		if op.Confidence > 0.7 {
			return "From what I've seen, " + op.Topic + " has real issues. " + evidencePhrase(strongest)
		}
		return "I'm leaning negative on " + op.Topic + ". " + evidencePhrase(strongest)

	case op.Position == "mixed":
		forEv, againstEv := splitEvidence(op)
		forStr := "no clear positives"
		if forEv != nil {
			forStr = forEv.Content
		}
		againstStr := "no clear negatives"
		if againstEv != nil {
			againstStr = againstEv.Content
		}
		return op.Topic + " is a mixed bag — " + opTruncate(forStr, 60) + ", but also " + opTruncate(againstStr, 60) + "."

	default:
		return "I'm still forming my thoughts on " + op.Topic + "."
	}
}

// ChallengeOpinion adds counter-evidence when the user disagrees with
// an opinion, making opinions revisable rather than rigid.
func (oe *OpinionEngine) ChallengeOpinion(topic string, counterEvidence string) {
	oe.AccumulateEvidence(topic, Evidence{
		Type:      EvidenceConversation,
		Source:    "user_challenge",
		Content:   counterEvidence,
		Valence:   0.0, // neutral valence by default — the content matters
		Weight:    0.8, // high weight — direct user pushback is significant
		Timestamp: time.Now(),
	})

	// Determine counter-valence: push against the current stance.
	oe.mu.Lock()
	defer oe.mu.Unlock()

	key := normalizeTopic(topic)
	op, ok := oe.opinions[key]
	if !ok {
		return
	}

	// Set the last piece of evidence to push against current stance.
	if len(op.Evidence) > 0 {
		last := &op.Evidence[len(op.Evidence)-1]
		sentiment := evaluateTextSentiment(counterEvidence)
		if sentiment == 0.0 {
			// If we can't detect sentiment, push toward center.
			last.Valence = -op.Stance * 0.5
		} else {
			last.Valence = sentiment
		}
	}

	recomputeOpinion(op)
	op.Summary = ArticulateOpinion(op)
}

// DecayOldEvidence reduces weight of evidence older than 30 days by
// 10% per month. Very old evidence (>6 months) gets weight halved.
func (oe *OpinionEngine) DecayOldEvidence() {
	oe.mu.Lock()
	defer oe.mu.Unlock()

	now := time.Now()
	for _, op := range oe.opinions {
		changed := false
		for i := range op.Evidence {
			age := now.Sub(op.Evidence[i].Timestamp)
			if age > 6*30*24*time.Hour {
				// Very old: halve the weight.
				newW := op.Evidence[i].Weight * 0.5
				if newW != op.Evidence[i].Weight {
					op.Evidence[i].Weight = newW
					changed = true
				}
			} else if age > 30*24*time.Hour {
				// Months old: reduce by 10% per 30-day period.
				months := age.Hours() / (30 * 24)
				factor := math.Pow(0.9, months)
				newW := op.Evidence[i].Weight * factor
				if newW != op.Evidence[i].Weight {
					op.Evidence[i].Weight = newW
					changed = true
				}
			}
		}
		if changed {
			recomputeOpinion(op)
			op.Summary = ArticulateOpinion(op)
		}
	}
}

// TopOpinions returns the N most confident opinions, sorted descending.
func (oe *OpinionEngine) TopOpinions(n int) []*Opinion {
	oe.mu.RLock()
	defer oe.mu.RUnlock()

	all := make([]*Opinion, 0, len(oe.opinions))
	for _, op := range oe.opinions {
		cp := *op
		cp.Evidence = make([]Evidence, len(op.Evidence))
		copy(cp.Evidence, op.Evidence)
		all = append(all, &cp)
	}

	sort.Slice(all, func(i, j int) bool {
		return all[i].Confidence > all[j].Confidence
	})

	if n > len(all) {
		n = len(all)
	}
	return all[:n]
}

// Save persists opinions to disk.
func (oe *OpinionEngine) Save() error {
	oe.mu.RLock()
	defer oe.mu.RUnlock()

	if len(oe.opinions) == 0 {
		return nil
	}

	dir := filepath.Dir(oe.savePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(oe.opinions, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(oe.savePath, data, 0o644)
}

// Load reads opinions from disk.
func (oe *OpinionEngine) Load() error {
	data, err := os.ReadFile(oe.savePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	var opinions map[string]*Opinion
	if err := json.Unmarshal(data, &opinions); err != nil {
		return err
	}

	oe.mu.Lock()
	oe.opinions = opinions
	oe.mu.Unlock()
	return nil
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

// recomputeOpinion recalculates stance, confidence, position, and counts.
// Caller must hold oe.mu.Lock or otherwise ensure exclusive access to op.
func recomputeOpinion(op *Opinion) {
	if len(op.Evidence) == 0 {
		op.Stance = 0
		op.Confidence = 0
		op.Position = "uncertain"
		op.ForCount = 0
		op.AgainstCount = 0
		return
	}

	var totalWeightedValence float64
	var totalWeight float64
	forCount := 0
	againstCount := 0

	for _, ev := range op.Evidence {
		w := ev.Weight
		if w <= 0 {
			w = 0.1
		}
		totalWeightedValence += ev.Valence * w
		totalWeight += w
		if ev.Valence > 0.1 {
			forCount++
		} else if ev.Valence < -0.1 {
			againstCount++
		}
	}

	op.ForCount = forCount
	op.AgainstCount = againstCount

	// Stance = weighted average of valences.
	if totalWeight > 0 {
		op.Stance = clampF(totalWeightedValence/totalWeight, -1.0, 1.0)
	} else {
		op.Stance = 0
	}

	// Confidence = min(1.0, count * avgWeight * 0.15).
	n := float64(len(op.Evidence))
	avgW := totalWeight / n
	op.Confidence = math.Min(1.0, n*avgW*0.15)

	// Don't let confidence exceed 0.3 with fewer than 2 pieces of evidence.
	if len(op.Evidence) < 2 && op.Confidence > 0.3 {
		op.Confidence = 0.3
	}

	// Position classification.
	switch {
	case op.Confidence < 0.25:
		op.Position = "uncertain"
	case forCount > 0 && againstCount > 0 && float64(min(forCount, againstCount))/float64(max(forCount, againstCount)) > 0.4:
		op.Position = "mixed"
	case op.Stance > 0.2:
		op.Position = "positive"
	case op.Stance < -0.2:
		op.Position = "negative"
	default:
		op.Position = "mixed"
	}
}

// pruneEvidence caps evidence at maxEvidencePerOpinion, keeping strongest.
func pruneEvidence(op *Opinion) {
	sort.Slice(op.Evidence, func(i, j int) bool {
		// Sort by absolute valence * weight descending (strongest first).
		si := math.Abs(op.Evidence[i].Valence) * op.Evidence[i].Weight
		sj := math.Abs(op.Evidence[j].Valence) * op.Evidence[j].Weight
		return si > sj
	})
	op.Evidence = op.Evidence[:maxEvidencePerOpinion]
}

// strongestEvidence returns the evidence with highest |valence| * weight.
func strongestEvidence(op *Opinion) *Evidence {
	if len(op.Evidence) == 0 {
		return nil
	}
	best := &op.Evidence[0]
	bestScore := math.Abs(best.Valence) * best.Weight
	for i := 1; i < len(op.Evidence); i++ {
		s := math.Abs(op.Evidence[i].Valence) * op.Evidence[i].Weight
		if s > bestScore {
			best = &op.Evidence[i]
			bestScore = s
		}
	}
	return best
}

// splitEvidence returns the strongest positive and strongest negative evidence.
func splitEvidence(op *Opinion) (forEv, againstEv *Evidence) {
	var bestFor, bestAgainst float64
	for i := range op.Evidence {
		score := op.Evidence[i].Valence * op.Evidence[i].Weight
		if score > bestFor {
			bestFor = score
			forEv = &op.Evidence[i]
		}
		if score < bestAgainst {
			bestAgainst = score
			againstEv = &op.Evidence[i]
		}
	}
	return
}

// evidencePhrase formats the strongest evidence as a natural clause.
func evidencePhrase(ev *Evidence) string {
	if ev == nil || ev.Content == "" {
		return ""
	}
	switch ev.Type {
	case EvidenceEpisodic:
		if ev.Source != "" && ev.Source != "conversation" {
			return "You mentioned \"" + opTruncate(ev.Content, 80) + "\" in a past conversation."
		}
		return "I recall: " + opTruncate(ev.Content, 80) + "."
	case EvidenceGraph:
		return "The knowledge base says " + opTruncate(ev.Content, 80) + "."
	case EvidenceConversation:
		return "You said \"" + opTruncate(ev.Content, 80) + ".\""
	case EvidenceObservation:
		return "I've noticed " + opTruncate(ev.Content, 80) + "."
	default:
		return opTruncate(ev.Content, 80)
	}
}

// countWord formats a number as a natural word for small counts.
func countWord(n int) string {
	switch {
	case n <= 2:
		return "a couple of"
	case n <= 5:
		return "several"
	case n <= 10:
		return "around ten"
	default:
		return "many"
	}
}

// opTruncate cuts a string to maxLen, adding "..." if truncated.
func opTruncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen < 4 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

// clampF clamps a float64 to [lo, hi].
func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// -----------------------------------------------------------------------
// Evaluative language extraction
// -----------------------------------------------------------------------

type extractedEval struct {
	topic   string
	valence float64
	weight  float64
}

// Adjective and verb sentiment lists.
var positiveAdj = map[string]float64{
	"great":       0.8,
	"good":        0.6,
	"excellent":   0.9,
	"amazing":     0.9,
	"wonderful":   0.8,
	"fantastic":   0.9,
	"awesome":     0.8,
	"useful":      0.6,
	"helpful":     0.6,
	"powerful":    0.7,
	"fast":        0.5,
	"elegant":     0.7,
	"beautiful":   0.7,
	"reliable":    0.7,
	"solid":       0.6,
	"impressive":  0.8,
	"intuitive":   0.7,
	"clean":       0.5,
	"simple":      0.5,
	"brilliant":   0.9,
	"nice":        0.5,
	"fun":         0.6,
	"interesting": 0.5,
}

var negativeAdj = map[string]float64{
	"terrible":      -0.9,
	"bad":           -0.6,
	"awful":         -0.9,
	"horrible":      -0.9,
	"pointless":     -0.7,
	"useless":       -0.7,
	"broken":        -0.8,
	"slow":          -0.5,
	"ugly":          -0.7,
	"confusing":     -0.6,
	"frustrating":   -0.7,
	"annoying":      -0.6,
	"buggy":         -0.8,
	"unreliable":    -0.7,
	"overcomplicated": -0.6,
	"clunky":        -0.6,
	"painful":       -0.7,
	"boring":        -0.5,
	"mediocre":      -0.4,
	"disappointing": -0.6,
	"overrated":     -0.5,
	"garbage":       -0.9,
	"trash":         -0.9,
}

var positiveVerbs = map[string]float64{
	"love":      0.9,
	"enjoy":     0.7,
	"like":      0.6,
	"prefer":    0.6,
	"recommend": 0.7,
	"appreciate": 0.6,
	"admire":    0.7,
}

var negativeVerbs = map[string]float64{
	"hate":    -0.9,
	"dislike": -0.7,
	"despise": -0.9,
	"loathe":  -0.9,
	"avoid":   -0.6,
	"detest":  -0.9,
}

var adverbModifiers = map[string]float64{
	"well":     0.6,
	"great":    0.8,
	"poorly":   -0.7,
	"badly":    -0.7,
	"terribly": -0.9,
	"nicely":   0.6,
	"fine":     0.4,
	"horribly": -0.9,
}

// extractEvaluations parses evaluative language from user input.
func extractEvaluations(input string) []extractedEval {
	lower := strings.ToLower(input)
	words := strings.Fields(lower)
	var results []extractedEval

	// Pattern: [topic] + "is" + [adjective]
	for i, w := range words {
		if w == "is" && i > 0 && i+1 < len(words) {
			topic := extractTopicBefore(words, i)
			adj := strings.Trim(words[i+1], ".,!?")
			if val, ok := positiveAdj[adj]; ok {
				results = append(results, extractedEval{topic: topic, valence: val, weight: 0.7})
			} else if val, ok := negativeAdj[adj]; ok {
				results = append(results, extractedEval{topic: topic, valence: val, weight: 0.7})
			}
		}
	}

	// Pattern: "I" + [verb] + [topic]
	for i, w := range words {
		if w == "i" && i+2 < len(words) {
			verb := strings.Trim(words[i+1], ".,!?")
			topic := extractTopicAfter(words, i+2)
			if val, ok := positiveVerbs[verb]; ok {
				results = append(results, extractedEval{topic: topic, valence: val, weight: 0.8})
			} else if val, ok := negativeVerbs[verb]; ok {
				results = append(results, extractedEval{topic: topic, valence: val, weight: 0.8})
			}
		}
	}

	// Pattern: [topic] + "works" + [adverb]
	for i, w := range words {
		if w == "works" && i > 0 && i+1 < len(words) {
			topic := extractTopicBefore(words, i)
			adv := strings.Trim(words[i+1], ".,!?")
			if val, ok := adverbModifiers[adv]; ok {
				results = append(results, extractedEval{topic: topic, valence: val, weight: 0.6})
			}
		}
	}

	// Pattern: [topic] + "is better/worse than" + [other topic]
	for i, w := range words {
		if w == "is" && i > 0 && i+3 < len(words) {
			comp := strings.Trim(words[i+1], ".,!?")
			if comp == "better" && strings.Trim(words[i+2], ".,!?") == "than" {
				topicA := extractTopicBefore(words, i)
				topicB := extractTopicAfter(words, i+3)
				results = append(results, extractedEval{topic: topicA, valence: 0.6, weight: 0.7})
				results = append(results, extractedEval{topic: topicB, valence: -0.4, weight: 0.5})
			} else if comp == "worse" && strings.Trim(words[i+2], ".,!?") == "than" {
				topicA := extractTopicBefore(words, i)
				topicB := extractTopicAfter(words, i+3)
				results = append(results, extractedEval{topic: topicA, valence: -0.6, weight: 0.7})
				results = append(results, extractedEval{topic: topicB, valence: 0.4, weight: 0.5})
			}
		}
	}

	return results
}

// extractTopicBefore pulls the topic from words before the given index.
// Takes up to 2 words, skipping common articles/pronouns.
func extractTopicBefore(words []string, idx int) string {
	skip := map[string]bool{"the": true, "a": true, "an": true, "this": true, "that": true, "my": true, "your": true}
	start := idx - 1
	if start < 0 {
		return ""
	}
	// Take 1-2 words before, skipping articles.
	parts := []string{strings.Trim(words[start], ".,!?")}
	if start-1 >= 0 && !skip[words[start-1]] {
		word := strings.Trim(words[start-1], ".,!?")
		if len(word) > 1 {
			parts = append([]string{word}, parts...)
		}
	}
	result := strings.Join(parts, " ")
	// Strip leading articles if they snuck in.
	for prefix := range skip {
		result = strings.TrimPrefix(result, prefix+" ")
	}
	return strings.TrimSpace(result)
}

// extractTopicAfter pulls the topic from words after the given index.
// Takes up to 2 words, cleaning punctuation.
func extractTopicAfter(words []string, idx int) string {
	if idx >= len(words) {
		return ""
	}
	parts := []string{strings.Trim(words[idx], ".,!?")}
	if idx+1 < len(words) {
		next := strings.Trim(words[idx+1], ".,!?")
		// Stop at conjunctions and prepositions.
		stop := map[string]bool{"and": true, "or": true, "but": true, "because": true, "when": true, "if": true, "for": true, "so": true}
		if !stop[next] && len(next) > 1 {
			parts = append(parts, next)
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

// evaluateTextSentiment does a quick positive/negative scan of text.
func evaluateTextSentiment(text string) float64 {
	lower := strings.ToLower(text)
	words := strings.Fields(lower)

	var total float64
	var count int

	for _, w := range words {
		w = strings.Trim(w, ".,!?;:\"'()")
		if val, ok := positiveAdj[w]; ok {
			total += val
			count++
		} else if val, ok := negativeAdj[w]; ok {
			total += val
			count++
		} else if val, ok := positiveVerbs[w]; ok {
			total += val
			count++
		} else if val, ok := negativeVerbs[w]; ok {
			total += val
			count++
		} else if val, ok := adverbModifiers[w]; ok {
			total += val
			count++
		}
	}

	if count == 0 {
		return 0.0
	}
	return clampF(total/float64(count), -1.0, 1.0)
}

// evaluateFactSentiment evaluates a knowledge graph fact string.
// Facts are usually neutral, but some carry implicit valence.
func evaluateFactSentiment(fact string) float64 {
	// Most graph facts are neutral statements.
	sentiment := evaluateTextSentiment(fact)
	// Dampen — facts are less emotionally charged than conversation.
	return sentiment * 0.5
}
