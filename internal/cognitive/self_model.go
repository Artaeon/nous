package cognitive

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// SelfModel tracks the system's own capabilities, strengths, weaknesses,
// and learning trajectory. It builds an increasingly accurate picture
// of what the system is good at and where it struggles.
//
// Innovation: Most AI assistants have no self-awareness. They answer every
// question with the same confidence regardless of whether they are actually
// good at that domain. SelfModel creates a feedback-driven understanding
// of the system's own performance — tracking success rates per domain,
// detecting trends over time, and generating honest assessments that
// set appropriate expectations. The result is an AI that can say "I'm
// great at explaining science but my career coaching is mediocre — you
// might want a human perspective too."
type SelfModel struct {
	capabilities map[string]*CapabilityProfile
	interactions []InteractionRecord
	maxRecords   int // max interaction records (1000)
	mu           sync.RWMutex
}

// CapabilityProfile tracks performance in one domain or skill.
type CapabilityProfile struct {
	Domain      string    `json:"domain"`       // "science_explain", "career_coaching", etc.
	Successes   int       `json:"successes"`
	Failures    int       `json:"failures"`
	AvgQuality  float64   `json:"avg_quality"`  // running average quality score
	Trend       string    `json:"trend"`        // "improving", "stable", "declining"
	LastUpdated time.Time `json:"last_updated"`
	Strengths   []string  `json:"strengths"`    // specific things this capability does well
	Weaknesses  []string  `json:"weaknesses"`   // specific things it struggles with
	BestTopics  []string  `json:"best_topics"`  // topics where it performs best
	WorstTopics []string  `json:"worst_topics"` // topics where it performs worst
}

// InteractionRecord logs one interaction for self-modeling.
type InteractionRecord struct {
	Domain     string    `json:"domain"`
	Query      string    `json:"query"`
	Quality    float64   `json:"quality"`      // 0-1 quality assessment
	WasHelpful bool      `json:"was_helpful"`  // did the user find it helpful
	Timestamp  time.Time `json:"timestamp"`
	Source     string    `json:"source"`       // which subsystem produced the response
}

// SelfAssessment is the system's honest evaluation of its ability to handle a query.
type SelfAssessment struct {
	CanHandle   bool     `json:"can_handle"`
	Confidence  float64  `json:"confidence"`
	BestApproach string  `json:"best_approach"` // "knowledge_lookup", "reasoning", "socratic", "synthesis", "honest_limit"
	Disclaimer  string   `json:"disclaimer"`
	Strengths   []string `json:"strengths"`
	Limitations []string `json:"limitations"`
}

// PerformanceReport summarizes the system's overall capabilities.
type PerformanceReport struct {
	TotalInteractions int           `json:"total_interactions"`
	OverallQuality    float64       `json:"overall_quality"`
	StrongestDomains  []DomainScore `json:"strongest_domains"`
	WeakestDomains    []DomainScore `json:"weakest_domains"`
	ImprovingAreas    []string      `json:"improving_areas"`
	DecliningAreas    []string      `json:"declining_areas"`
	RecommendedFocus  []string      `json:"recommended_focus"`
}

// DomainScore captures a domain's performance summary.
type DomainScore struct {
	Domain string  `json:"domain"`
	Score  float64 `json:"score"`
	Count  int     `json:"count"`
	Trend  string  `json:"trend"`
}

// selfModelPersist is the JSON-serializable snapshot of a SelfModel.
type selfModelPersist struct {
	Capabilities map[string]*CapabilityProfile `json:"capabilities"`
	Interactions []InteractionRecord           `json:"interactions"`
}

// NewSelfModel creates a new self-model with default settings.
func NewSelfModel() *SelfModel {
	return &SelfModel{
		capabilities: make(map[string]*CapabilityProfile),
		maxRecords:   1000,
	}
}

// -----------------------------------------------------------------------
// Assessment — evaluate whether we can handle a query well
// -----------------------------------------------------------------------

// Assess evaluates the system's ability to handle a query in the given domain.
// It returns an honest assessment including confidence, recommended approach,
// and any disclaimers about limitations.
func (sm *SelfModel) Assess(query string, domain string) *SelfAssessment {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	profile, exists := sm.capabilities[domain]

	// No data for this domain — moderate confidence with disclaimer.
	if !exists || (profile.Successes+profile.Failures) == 0 {
		return &SelfAssessment{
			CanHandle:   true,
			Confidence:  0.5,
			BestApproach: sm.suggestApproachLocked(domain, 0.5),
			Disclaimer:  "I don't have much experience with this type of question yet, so I'll do my best.",
			Strengths:   nil,
			Limitations: []string{"limited experience in this domain"},
		}
	}

	total := profile.Successes + profile.Failures
	successRate := float64(profile.Successes) / float64(total)

	switch {
	case successRate > 0.7:
		// High confidence — recommend best approach.
		confidence := clampFloat(0.7+successRate*0.3, 0, 1)
		return &SelfAssessment{
			CanHandle:    true,
			Confidence:   confidence,
			BestApproach: sm.suggestApproachLocked(domain, confidence),
			Disclaimer:   "",
			Strengths:    profile.Strengths,
			Limitations:  nil,
		}

	case successRate >= 0.4:
		// Moderate confidence — add disclaimer.
		confidence := 0.3 + successRate*0.4
		return &SelfAssessment{
			CanHandle:    true,
			Confidence:   confidence,
			BestApproach: sm.suggestApproachLocked(domain, confidence),
			Disclaimer:   sm.honestDisclaimerLocked(domain),
			Strengths:    profile.Strengths,
			Limitations:  profile.Weaknesses,
		}

	default:
		// Low confidence — suggest Socratic mode or honest limitation.
		confidence := successRate * 0.5
		return &SelfAssessment{
			CanHandle:    false,
			Confidence:   confidence,
			BestApproach: sm.suggestApproachLocked(domain, confidence),
			Disclaimer:   sm.honestDisclaimerLocked(domain),
			Strengths:    profile.Strengths,
			Limitations:  profile.Weaknesses,
		}
	}
}

// -----------------------------------------------------------------------
// Outcome recording — learn from each interaction
// -----------------------------------------------------------------------

// RecordOutcome records the outcome of an interaction, updating running
// averages, success/failure counts, and trend detection.
func (sm *SelfModel) RecordOutcome(domain string, query string, quality float64, wasHelpful bool, source string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	quality = clampFloat(quality, 0, 1)
	now := time.Now()

	rec := InteractionRecord{
		Domain:     domain,
		Query:      query,
		Quality:    quality,
		WasHelpful: wasHelpful,
		Timestamp:  now,
		Source:     source,
	}

	sm.interactions = append(sm.interactions, rec)
	if len(sm.interactions) > sm.maxRecords {
		// Drop the oldest records.
		excess := len(sm.interactions) - sm.maxRecords
		sm.interactions = sm.interactions[excess:]
	}

	profile, exists := sm.capabilities[domain]
	if !exists {
		profile = &CapabilityProfile{
			Domain: domain,
			Trend:  "stable",
		}
		sm.capabilities[domain] = profile
	}

	// Update success / failure counts.
	if wasHelpful {
		profile.Successes++
	} else {
		profile.Failures++
	}

	// Update running average quality (exponential moving average).
	total := profile.Successes + profile.Failures
	if total == 1 {
		profile.AvgQuality = quality
	} else {
		// EMA with alpha = 0.2 (recent interactions weighted more).
		alpha := 0.2
		profile.AvgQuality = alpha*quality + (1-alpha)*profile.AvgQuality
	}

	profile.LastUpdated = now

	// Detect trend from recent interactions in this domain.
	profile.Trend = sm.detectTrend(domain)

	// Update best/worst topics based on the query.
	sm.updateTopics(profile, query, quality)
}

// detectTrend examines recent interactions to determine if a domain is
// improving, declining, or stable. Compares the average quality of the
// first half versus second half of the most recent records.
func (sm *SelfModel) detectTrend(domain string) string {
	var domainRecords []InteractionRecord
	for _, r := range sm.interactions {
		if r.Domain == domain {
			domainRecords = append(domainRecords, r)
		}
	}

	if len(domainRecords) < 6 {
		return "stable"
	}

	mid := len(domainRecords) / 2
	firstHalf := domainRecords[:mid]
	secondHalf := domainRecords[mid:]

	avgFirst := averageQuality(firstHalf)
	avgSecond := averageQuality(secondHalf)

	diff := avgSecond - avgFirst
	switch {
	case diff > 0.1:
		return "improving"
	case diff < -0.1:
		return "declining"
	default:
		return "stable"
	}
}

// updateTopics maintains the best and worst topics for a capability
// profile based on observed query quality.
func (sm *SelfModel) updateTopics(profile *CapabilityProfile, query string, quality float64) {
	topic := extractPrimaryTopic(query)
	if topic == "" {
		return
	}

	if quality >= 0.7 {
		if !containsString(profile.BestTopics, topic) {
			profile.BestTopics = append(profile.BestTopics, topic)
			if len(profile.BestTopics) > 10 {
				profile.BestTopics = profile.BestTopics[1:]
			}
		}
	}
	if quality < 0.4 {
		if !containsString(profile.WorstTopics, topic) {
			profile.WorstTopics = append(profile.WorstTopics, topic)
			if len(profile.WorstTopics) > 10 {
				profile.WorstTopics = profile.WorstTopics[1:]
			}
		}
	}
}

// -----------------------------------------------------------------------
// Domain classification — classify a query into a domain
// -----------------------------------------------------------------------

// domainKeywords maps domains to sets of keywords used for classification.
var domainKeywords = map[string][]string{
	"science_explain": {
		"science", "physics", "chemistry", "biology", "quantum", "atom",
		"molecule", "evolution", "gravity", "energy", "cell", "dna",
		"experiment", "hypothesis", "scientific", "electron", "neutron",
		"thermodynamic", "photon", "genome",
	},
	"history_explain": {
		"history", "historical", "war", "revolution", "empire", "ancient",
		"medieval", "century", "civilization", "dynasty", "colonialism",
		"independence", "treaty", "battle",
	},
	"philosophy_explain": {
		"philosophy", "ethical", "moral", "existential", "metaphysics",
		"epistemology", "consciousness", "free will", "meaning of life",
		"philosophical", "ontology", "logic", "ethics", "meaning",
	},
	"career_coaching": {
		"career", "job", "resume", "interview", "salary", "promotion",
		"workplace", "professional", "hiring", "negotiate", "mentor",
		"networking", "linkedin", "quit",
	},
	"decision_support": {
		"decide", "decision", "choose", "choice", "option", "tradeoff",
		"pros and cons", "should i", "which is better", "worth it",
		"alternative", "weigh",
	},
	"planning": {
		"plan", "schedule", "roadmap", "timeline", "milestone", "goal",
		"strategy", "project", "budget", "deadline", "prioritize",
	},
	"comparison": {
		"compare", "versus", "difference", "better", "worse", "vs",
		"similarities", "contrast", "prefer",
	},
	"factual_qa": {
		"what is", "who is", "when did", "where is", "how many",
		"define", "definition", "fact", "true or false",
	},
	"creative": {
		"write", "poem", "story", "creative", "imagine", "fiction",
		"compose", "lyric", "novel", "paint", "design", "brainstorm",
	},
	"technical": {
		"code", "programming", "debug", "algorithm", "software", "api",
		"database", "server", "deploy", "function", "compile", "error",
		"bug", "git", "docker", "kubernetes",
	},
	"interpersonal": {
		"relationship", "friend", "family", "conflict", "communicate",
		"empathy", "feeling", "emotion", "support", "listen",
		"apolog", "forgive",
	},
	"meta": {
		"yourself", "your capabilities", "what can you", "how do you work",
		"are you", "who are you", "self-aware", "limitations",
	},
}

// ClassifyDomain classifies a query into a domain using keyword matching.
// Multi-word keywords are matched as substrings; single-word keywords are
// matched on word boundaries to avoid false positives (e.g. "evolution"
// inside "revolution").
func (sm *SelfModel) ClassifyDomain(query string) string {
	lower := strings.ToLower(query)
	words := strings.Fields(lower)
	// Build a set of individual words (stripped of punctuation) for
	// word-boundary matching.
	wordSet := make(map[string]bool, len(words))
	for _, w := range words {
		wordSet[strings.TrimRight(w, "?!.,;:")] = true
	}

	bestDomain := "factual_qa" // default
	bestScore := 0

	for domain, keywords := range domainKeywords {
		score := 0
		for _, kw := range keywords {
			if strings.Contains(kw, " ") {
				// Multi-word keyword: substring match.
				if strings.Contains(lower, kw) {
					score++
				}
			} else {
				// Single-word keyword: exact word match.
				if wordSet[kw] {
					score++
				}
			}
		}
		if score > bestScore {
			bestScore = score
			bestDomain = domain
		}
	}

	return bestDomain
}

// -----------------------------------------------------------------------
// Reporting — generate a performance report
// -----------------------------------------------------------------------

// GenerateReport produces a performance report summarizing the system's
// capabilities, strengths, weaknesses, and improvement trajectory.
func (sm *SelfModel) GenerateReport() *PerformanceReport {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	report := &PerformanceReport{
		TotalInteractions: len(sm.interactions),
	}

	if report.TotalInteractions == 0 {
		return report
	}

	// Compute overall average quality.
	report.OverallQuality = averageQuality(sm.interactions)

	// Collect domain scores.
	var scores []DomainScore
	for _, profile := range sm.capabilities {
		total := profile.Successes + profile.Failures
		if total == 0 {
			continue
		}
		scores = append(scores, DomainScore{
			Domain: profile.Domain,
			Score:  profile.AvgQuality,
			Count:  total,
			Trend:  profile.Trend,
		})
	}

	// Sort by score descending.
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	// Strongest domains (top 3).
	limit := 3
	if len(scores) < limit {
		limit = len(scores)
	}
	report.StrongestDomains = scores[:limit]

	// Weakest domains (bottom 3, reversed so worst is first).
	weakLimit := 3
	if len(scores) < weakLimit {
		weakLimit = len(scores)
	}
	weakStart := len(scores) - weakLimit
	weak := make([]DomainScore, weakLimit)
	copy(weak, scores[weakStart:])
	// Reverse so worst is first.
	for i, j := 0, len(weak)-1; i < j; i, j = i+1, j-1 {
		weak[i], weak[j] = weak[j], weak[i]
	}
	report.WeakestDomains = weak

	// Improving / declining areas.
	for _, s := range scores {
		switch s.Trend {
		case "improving":
			report.ImprovingAreas = append(report.ImprovingAreas, s.Domain)
		case "declining":
			report.DecliningAreas = append(report.DecliningAreas, s.Domain)
		}
	}

	// Recommended focus: declining areas and weak domains with enough data.
	seen := make(map[string]bool)
	for _, d := range report.DecliningAreas {
		if !seen[d] {
			report.RecommendedFocus = append(report.RecommendedFocus, d)
			seen[d] = true
		}
	}
	for _, ds := range report.WeakestDomains {
		if ds.Count >= 3 && !seen[ds.Domain] {
			report.RecommendedFocus = append(report.RecommendedFocus, ds.Domain)
			seen[ds.Domain] = true
		}
	}

	return report
}

// -----------------------------------------------------------------------
// Honest disclaimers and approach suggestions
// -----------------------------------------------------------------------

// domainDisclaimers maps domains to specific, honest limitation statements.
var domainDisclaimers = map[string]string{
	"science_explain":    "I'm strongest at explaining factual topics where I have knowledge. If this touches cutting-edge research, my information may be outdated.",
	"history_explain":    "I can summarize well-documented historical events, but for niche or contested history, my coverage may be incomplete.",
	"philosophy_explain": "I can lay out philosophical frameworks, but philosophy is inherently subjective — my framing reflects common academic perspectives, not the only valid view.",
	"career_coaching":    "I'm strongest at explaining factual topics where I have knowledge. For career coaching, my advice is general — you might want a human perspective too.",
	"decision_support":   "I can help structure your thinking, but decisions involve personal values and context I may not fully grasp.",
	"planning":           "I can help outline plans and structure, but real-world constraints and priorities are something only you fully understand.",
	"comparison":         "My comparison skills are solid when I have data on both items. For this topic, my knowledge may be limited.",
	"factual_qa":         "I aim for accuracy on factual questions, but I can make mistakes — please verify critical facts.",
	"creative":           "Creative output is subjective. I can generate ideas and drafts, but the final creative voice should be yours.",
	"technical":          "I can help with code and technical questions, but I can't run or test code — always verify in your own environment.",
	"interpersonal":      "Interpersonal situations are deeply personal. I can offer general frameworks, but a counselor or trusted friend who knows you would give better advice.",
	"meta":               "I try to be honest about what I can and can't do, though my self-assessment is itself imperfect.",
}

// HonestDisclaimer generates an honest, specific limitation statement for
// the given domain. It uses observed performance data when available to
// make the disclaimer more precise.
func (sm *SelfModel) HonestDisclaimer(domain string) string {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.honestDisclaimerLocked(domain)
}

// honestDisclaimerLocked is the internal disclaimer generator (caller holds lock).
func (sm *SelfModel) honestDisclaimerLocked(domain string) string {
	profile, exists := sm.capabilities[domain]

	// Use the domain-specific template as the base.
	base, ok := domainDisclaimers[domain]
	if !ok {
		base = "I have limited experience with this type of question, so my answer may not be as strong as in other areas."
	}

	if !exists || (profile.Successes+profile.Failures) < 3 {
		return base
	}

	total := profile.Successes + profile.Failures
	successRate := float64(profile.Successes) / float64(total)

	// Augment with observed performance data.
	if successRate < 0.4 {
		base += " Based on past interactions, this is one of my weaker areas."
	} else if successRate > 0.7 && profile.Trend == "improving" {
		base = "I've been getting better at this, and my track record is solid here."
	}

	if len(profile.WorstTopics) > 0 {
		worst := strings.Join(profile.WorstTopics, ", ")
		base += " I particularly struggle with: " + worst + "."
	}

	return base
}

// SuggestApproach recommends the best approach for handling a query
// based on the domain and confidence level.
func (sm *SelfModel) SuggestApproach(domain string, confidence float64) string {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.suggestApproachLocked(domain, confidence)
}

// suggestApproachLocked is the internal approach selector (caller holds lock).
func (sm *SelfModel) suggestApproachLocked(domain string, confidence float64) string {
	// Check if this is a coaching/interpersonal domain where Socratic is
	// appropriate at lower confidence.
	isCoachingDomain := domain == "career_coaching" ||
		domain == "decision_support" ||
		domain == "interpersonal" ||
		domain == "planning"

	switch {
	case confidence >= 0.7:
		return "knowledge_lookup"
	case confidence >= 0.5:
		return "synthesis"
	case confidence >= 0.3 && isCoachingDomain:
		return "socratic"
	case confidence >= 0.3:
		return "reasoning"
	default:
		return "honest_limit"
	}
}

// -----------------------------------------------------------------------
// Persistence — save/load the self-model as JSON
// -----------------------------------------------------------------------

// Save persists the self-model to a JSON file at the given path.
func (sm *SelfModel) Save(path string) error {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	snap := selfModelPersist{
		Capabilities: sm.capabilities,
		Interactions: sm.interactions,
	}

	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// Load reads the self-model from a JSON file at the given path.
func (sm *SelfModel) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var snap selfModelPersist
	if err := json.Unmarshal(data, &snap); err != nil {
		return err
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	if snap.Capabilities == nil {
		snap.Capabilities = make(map[string]*CapabilityProfile)
	}
	sm.capabilities = snap.Capabilities
	sm.interactions = snap.Interactions
	return nil
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// clampFloat clamps v to [lo, hi].
func clampFloat(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}

// averageQuality computes the mean quality of a set of interaction records.
func averageQuality(records []InteractionRecord) float64 {
	if len(records) == 0 {
		return 0
	}
	sum := 0.0
	for _, r := range records {
		sum += r.Quality
	}
	return sum / float64(len(records))
}

// containsString checks if a slice contains a string.
func containsString(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}

// extractPrimaryTopic pulls the most salient noun phrase from a query.
// This is a simple heuristic: grab the longest word that isn't a stop word.
func extractPrimaryTopic(query string) string {
	stop := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "can": true, "shall": true, "must": true,
		"to": true, "of": true, "in": true, "for": true, "on": true,
		"with": true, "at": true, "by": true, "from": true, "about": true,
		"into": true, "through": true, "during": true, "before": true,
		"after": true, "above": true, "below": true, "between": true,
		"and": true, "but": true, "or": true, "nor": true, "not": true,
		"so": true, "yet": true, "both": true, "either": true, "neither": true,
		"i": true, "me": true, "my": true, "we": true, "you": true,
		"your": true, "it": true, "its": true, "this": true, "that": true,
		"these": true, "those": true, "what": true, "which": true,
		"who": true, "whom": true, "how": true, "when": true, "where": true,
		"why": true, "tell": true, "explain": true, "describe": true,
		"compare": true, "help": true, "give": true, "make": true,
	}

	words := strings.Fields(strings.ToLower(query))
	best := ""
	for _, w := range words {
		// Strip trailing punctuation.
		w = strings.TrimRight(w, "?!.,;:")
		if stop[w] || len(w) < 3 {
			continue
		}
		if len(w) > len(best) {
			best = w
		}
	}
	return best
}
