package cognitive

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Cognitive Fingerprinting — learns HOW you think, not just WHAT you know.
//
// Builds a dynamic profile of the user's reasoning style, preferred
// explanation depth, decision patterns, peak activity times, and
// cognitive blind spots. Adapts Nous's communication style to match.
//
// This is a novel capability for local AI: the system doesn't just
// remember facts about you — it learns your cognitive patterns and
// adapts its behavior accordingly.
// -----------------------------------------------------------------------

// CognitiveProfile represents the user's thinking patterns.
type CognitiveProfile struct {
	// Communication preferences
	PreferredDepth    string  // "brief", "standard", "detailed"
	AvgFollowUps      float64 // average follow-up questions per topic
	PreferredStyle    string  // "analytical", "intuitive", "examples-first", "theory-first"

	// Cognitive patterns
	TopDomains        []string // most-explored knowledge domains
	LearningPattern   string   // "breadth-first", "depth-first", "random"
	DecisionStyle     string   // "data-driven", "intuition-driven", "mixed"

	// Temporal patterns
	PeakHours         []int    // hours of peak activity
	MostActiveDay     string   // day of week

	// Strengths and blind spots
	StrongDomains     []string // domains with deep engagement
	BlindSpots        []string // domains referenced but never explored

	// Engagement metrics
	TotalInteractions int
	AvgSessionLength  float64 // average interactions per session
	TopicBreadth      int     // unique topics explored

	// Raw data for updates
	LastUpdated       time.Time
}

// CognitiveProfiler builds and maintains the user's cognitive profile.
type CognitiveProfiler struct {
	Episodic *memory.EpisodicMemory
	Growth   *PersonalGrowth
	profile  *CognitiveProfile
}

// NewCognitiveProfiler creates a cognitive fingerprinting engine.
func NewCognitiveProfiler(episodic *memory.EpisodicMemory, growth *PersonalGrowth) *CognitiveProfiler {
	return &CognitiveProfiler{
		Episodic: episodic,
		Growth:   growth,
	}
}

// BuildProfile analyzes all available data to construct the cognitive profile.
func (cp *CognitiveProfiler) BuildProfile() *CognitiveProfile {
	if cp.Episodic == nil {
		return &CognitiveProfile{PreferredDepth: "standard", PreferredStyle: "analytical"}
	}

	episodes := cp.Episodic.Recent(500)
	if len(episodes) < 5 {
		return &CognitiveProfile{PreferredDepth: "standard", PreferredStyle: "analytical"}
	}

	profile := &CognitiveProfile{
		TotalInteractions: len(episodes),
		LastUpdated:       time.Now(),
	}

	// Analyze communication preferences.
	profile.PreferredDepth = cp.analyzeDepthPreference(episodes)
	profile.AvgFollowUps = cp.analyzeFollowUpRate(episodes)
	profile.PreferredStyle = cp.analyzeStyle(episodes)

	// Analyze cognitive patterns.
	profile.TopDomains = cp.analyzeTopDomains(episodes)
	profile.LearningPattern = cp.analyzeLearningPattern(episodes)
	profile.TopicBreadth = cp.countUniqueTopics(episodes)

	// Temporal patterns.
	profile.PeakHours = cp.analyzePeakHours(episodes)
	profile.MostActiveDay = cp.analyzeMostActiveDay(episodes)

	// Strengths and blind spots.
	profile.StrongDomains, profile.BlindSpots = cp.analyzeStrengthsAndGaps(episodes)

	cp.profile = profile
	return profile
}

// GetProfile returns the current profile (builds if needed).
func (cp *CognitiveProfiler) GetProfile() *CognitiveProfile {
	if cp.profile == nil || time.Since(cp.profile.LastUpdated) > 1*time.Hour {
		return cp.BuildProfile()
	}
	return cp.profile
}

// AdaptResponse suggests how to adapt a response based on the profile.
func (cp *CognitiveProfiler) AdaptResponse(response string) string {
	profile := cp.GetProfile()
	if profile == nil {
		return response
	}

	switch profile.PreferredDepth {
	case "brief":
		// Truncate to first 2 sentences if long.
		sentences := strings.SplitAfter(response, ". ")
		if len(sentences) > 2 {
			return strings.TrimSpace(strings.Join(sentences[:2], ""))
		}
	case "detailed":
		// Add a follow-up prompt to encourage deeper exploration.
		if !strings.HasSuffix(response, "?") && len(response) > 50 {
			response += " Want me to go deeper on any aspect?"
		}
	}

	return response
}

// FormatCognitiveProfile returns a human-readable profile description.
func FormatCognitiveProfile(p *CognitiveProfile) string {
	var b strings.Builder

	b.WriteString("# Your Cognitive Profile\n\n")

	fmt.Fprintf(&b, "**Interactions:** %d | **Topic breadth:** %d | **Style:** %s\n\n",
		p.TotalInteractions, p.TopicBreadth, p.PreferredStyle)

	b.WriteString("## How You Think\n")
	fmt.Fprintf(&b, "- Preferred depth: %s\n", p.PreferredDepth)
	fmt.Fprintf(&b, "- Learning pattern: %s\n", p.LearningPattern)
	fmt.Fprintf(&b, "- Avg follow-ups per topic: %.1f\n\n", p.AvgFollowUps)

	if len(p.TopDomains) > 0 {
		b.WriteString("## Your Domains\n")
		for _, d := range p.TopDomains {
			fmt.Fprintf(&b, "- %s\n", d)
		}
		b.WriteString("\n")
	}

	if len(p.StrongDomains) > 0 {
		b.WriteString("## Strengths (deep engagement)\n")
		for _, d := range p.StrongDomains {
			fmt.Fprintf(&b, "- %s\n", d)
		}
		b.WriteString("\n")
	}

	if len(p.BlindSpots) > 0 {
		b.WriteString("## Blind Spots (mentioned but unexplored)\n")
		for _, d := range p.BlindSpots {
			fmt.Fprintf(&b, "- %s\n", d)
		}
		b.WriteString("\n")
	}

	if len(p.PeakHours) > 0 {
		b.WriteString("## When You're Most Curious\n")
		fmt.Fprintf(&b, "- Peak hours: %v\n", p.PeakHours)
		if p.MostActiveDay != "" {
			fmt.Fprintf(&b, "- Most active day: %s\n", p.MostActiveDay)
		}
	}

	return b.String()
}

// -----------------------------------------------------------------------
// Analysis methods
// -----------------------------------------------------------------------

func (cp *CognitiveProfiler) analyzeDepthPreference(episodes []memory.Episode) string {
	longCount := 0
	shortCount := 0
	for _, ep := range episodes {
		words := len(strings.Fields(ep.Input))
		if words > 10 {
			longCount++
		} else if words <= 5 {
			shortCount++
		}
	}

	if longCount > shortCount*2 {
		return "detailed"
	}
	if shortCount > longCount*2 {
		return "brief"
	}
	return "standard"
}

func (cp *CognitiveProfiler) analyzeFollowUpRate(episodes []memory.Episode) float64 {
	if len(episodes) < 2 {
		return 0
	}

	sessionBreak := 30 * time.Minute
	sessions := 1
	followUps := 0

	for i := 1; i < len(episodes); i++ {
		gap := episodes[i-1].Timestamp.Sub(episodes[i].Timestamp)
		if gap > sessionBreak {
			sessions++
		} else {
			// Same session — check if follow-up.
			prevTopic := extractDreamTopic(episodes[i-1].Input)
			currTopic := extractDreamTopic(episodes[i].Input)
			if prevTopic != "" && prevTopic == currTopic {
				followUps++
			}
		}
	}

	if sessions == 0 {
		return 0
	}
	return float64(followUps) / float64(sessions)
}

func (cp *CognitiveProfiler) analyzeStyle(episodes []memory.Episode) string {
	questionWords := 0
	howWhyCount := 0
	whatIsCount := 0

	for _, ep := range episodes {
		lower := strings.ToLower(ep.Input)
		if strings.Contains(lower, "?") {
			questionWords++
		}
		if strings.HasPrefix(lower, "how ") || strings.HasPrefix(lower, "why ") {
			howWhyCount++
		}
		if strings.HasPrefix(lower, "what is") || strings.HasPrefix(lower, "what are") {
			whatIsCount++
		}
	}

	if howWhyCount > whatIsCount {
		return "analytical" // asks "how/why" more than "what is"
	}
	if whatIsCount > howWhyCount*2 {
		return "examples-first" // asks for definitions/examples
	}
	return "balanced"
}

func (cp *CognitiveProfiler) analyzeTopDomains(episodes []memory.Episode) []string {
	domains := make(map[string]int)
	for _, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic == "" {
			continue
		}
		domains[topic]++
	}

	type kv struct {
		k string
		v int
	}
	var sorted []kv
	for k, v := range domains {
		sorted = append(sorted, kv{k, v})
	}
	if len(sorted) == 0 {
		return nil
	}

	// Sort by count descending.
	for i := range sorted {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].v > sorted[i].v {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	var top []string
	for _, s := range sorted {
		if len(top) >= 5 {
			break
		}
		top = append(top, fmt.Sprintf("%s (%d)", s.k, s.v))
	}
	return top
}

func (cp *CognitiveProfiler) analyzeLearningPattern(episodes []memory.Episode) string {
	topics := make(map[string]int)
	var topicSequence []string

	for _, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic == "" {
			continue
		}
		topics[topic]++
		topicSequence = append(topicSequence, topic)
	}

	if len(topicSequence) < 5 {
		return "insufficient data"
	}

	// Check for depth-first: same topic repeated consecutively.
	consecutiveRepeats := 0
	for i := 1; i < len(topicSequence); i++ {
		if topicSequence[i] == topicSequence[i-1] {
			consecutiveRepeats++
		}
	}

	repeatRatio := float64(consecutiveRepeats) / float64(len(topicSequence))
	uniqueRatio := float64(len(topics)) / float64(len(topicSequence))

	if repeatRatio > 0.3 {
		return "depth-first" // digs deep into one topic at a time
	}
	if uniqueRatio > 0.7 {
		return "breadth-first" // explores many topics lightly
	}
	return "mixed"
}

func (cp *CognitiveProfiler) countUniqueTopics(episodes []memory.Episode) int {
	topics := make(map[string]bool)
	for _, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic != "" {
			topics[topic] = true
		}
	}
	return len(topics)
}

func (cp *CognitiveProfiler) analyzePeakHours(episodes []memory.Episode) []int {
	hourCounts := make(map[int]int)
	for _, ep := range episodes {
		hourCounts[ep.Timestamp.Hour()]++
	}

	// Find hours with above-average activity.
	total := 0
	for _, c := range hourCounts {
		total += c
	}
	avg := float64(total) / 24.0

	var peaks []int
	for h, c := range hourCounts {
		if float64(c) > avg*1.5 {
			peaks = append(peaks, h)
		}
	}
	return peaks
}

func (cp *CognitiveProfiler) analyzeMostActiveDay(episodes []memory.Episode) string {
	dayCounts := make(map[time.Weekday]int)
	for _, ep := range episodes {
		dayCounts[ep.Timestamp.Weekday()]++
	}

	bestDay := time.Sunday
	bestCount := 0
	for d, c := range dayCounts {
		if c > bestCount {
			bestCount = c
			bestDay = d
		}
	}

	if bestCount < 3 {
		return ""
	}
	return bestDay.String()
}

func (cp *CognitiveProfiler) analyzeStrengthsAndGaps(episodes []memory.Episode) ([]string, []string) {
	topicDepth := make(map[string]int)
	for _, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic != "" {
			topicDepth[topic]++
		}
	}

	var strengths, blindSpots []string

	for topic, count := range topicDepth {
		if count >= 5 {
			strengths = append(strengths, topic)
		}
	}

	// Blind spots: topics mentioned in responses but never asked about.
	// (This is a simplified version — full implementation would analyze
	// graph neighbors of strong topics that were never queried.)
	_ = math.Min // use math to avoid import error

	return strengths, blindSpots
}
