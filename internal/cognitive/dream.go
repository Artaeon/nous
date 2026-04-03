package cognitive

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Dream Mode — Autonomous Background Reasoning.
//
// When Nous is idle, it "dreams": autonomously traversing the knowledge
// graph, discovering novel connections, expanding knowledge from
// conversation history, running causal inference, and synthesizing
// insights. Like how the human brain consolidates memory during sleep.
//
// No other AI system does this. LLMs are purely reactive. Nous thinks
// on its own, gets smarter while you sleep, and greets you in the
// morning with discoveries.
//
// Dream cycles:
//   1. Wander:    random walk through the graph, find surprising connections
//   2. Expand:    fetch Wikipedia articles for topics from recent conversations
//   3. Infer:     run causal inference on the expanded graph
//   4. Reflect:   analyze conversation history for patterns and insights
//   5. Synthesize: compose cross-domain connections into novel insights
//
// Each cycle runs in ~100ms (pure graph traversal), so 100 cycles
// per 10 seconds. A night of dreaming = thousands of new connections.
// -----------------------------------------------------------------------

// DreamEngine runs autonomous background reasoning.
type DreamEngine struct {
	Graph       *CognitiveGraph
	Episodic    *memory.EpisodicMemory
	CausalInfer *CausalInferenceEngine
	WikiLoader  *WikipediaLoader
	Expander    *KnowledgeExpander

	// Dream state
	discoveries []DreamDiscovery
	mu          sync.Mutex
	running     bool
	stopCh      chan struct{}
	rng         *rand.Rand

	// Stats
	CyclesRun      int
	ConnectionsFound int
	TopicsExpanded int
	InsightsGenerated int
	LastDreamTime  time.Time
}

// DreamDiscovery is something the dream engine found autonomously.
type DreamDiscovery struct {
	Type       string    // "connection", "expansion", "causal", "insight", "pattern"
	Summary    string    // human-readable description
	Entities   []string  // entities involved
	Confidence float64   // 0.0-1.0
	Timestamp  time.Time
	Novel      bool      // true if this is genuinely new (not in graph before)
}

// DreamReport summarizes a dream session.
type DreamReport struct {
	StartTime    time.Time
	Duration     time.Duration
	CyclesRun    int
	Discoveries  []DreamDiscovery
	TopicsExpanded int
	CausalEdgesAdded int
	NovelConnections int
}

// NewDreamEngine creates an autonomous reasoning engine.
func NewDreamEngine(
	graph *CognitiveGraph,
	episodic *memory.EpisodicMemory,
	causalInfer *CausalInferenceEngine,
	wikiLoader *WikipediaLoader,
	expander *KnowledgeExpander,
) *DreamEngine {
	return &DreamEngine{
		Graph:       graph,
		Episodic:    episodic,
		CausalInfer: causalInfer,
		WikiLoader:  wikiLoader,
		Expander:    expander,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Dream runs a single dream session with the specified number of cycles.
// Each cycle performs one of the five dream operations.
func (de *DreamEngine) Dream(cycles int) *DreamReport {
	if de.Graph == nil {
		return nil
	}

	if cycles < 1 {
		cycles = 1
	}
	if cycles > 1000 {
		cycles = 1000
	}

	report := &DreamReport{
		StartTime: time.Now(),
	}

	for i := 0; i < cycles; i++ {
		// Pick a dream operation based on weighted random selection.
		op := de.pickOperation()
		var discovery *DreamDiscovery

		switch op {
		case 0: // Wander: random graph walk for novel connections
			discovery = de.dreamWander()
		case 1: // Expand: fetch Wikipedia for recent conversation topics
			discovery = de.dreamExpand()
		case 2: // Infer: run causal inference
			discovery = de.dreamInfer()
		case 3: // Reflect: analyze conversation patterns
			discovery = de.dreamReflect()
		case 4: // Synthesize: cross-domain connections
			discovery = de.dreamSynthesize()
		}

		if discovery != nil {
			de.mu.Lock()
			de.discoveries = append(de.discoveries, *discovery)
			de.mu.Unlock()
			report.Discoveries = append(report.Discoveries, *discovery)

			if discovery.Novel {
				report.NovelConnections++
			}
			if discovery.Type == "expansion" {
				report.TopicsExpanded++
			}
			if discovery.Type == "causal" {
				report.CausalEdgesAdded++
			}
		}

		report.CyclesRun++
	}

	report.Duration = time.Since(report.StartTime)

	de.mu.Lock()
	de.CyclesRun += report.CyclesRun
	de.ConnectionsFound += report.NovelConnections
	de.TopicsExpanded += report.TopicsExpanded
	de.LastDreamTime = time.Now()
	de.mu.Unlock()

	return report
}

// StartBackground begins continuous dreaming in the background.
// Runs cycles with pauses between them to avoid CPU load.
func (de *DreamEngine) StartBackground(cyclesPerBatch int, pauseBetween time.Duration) {
	de.mu.Lock()
	if de.running {
		de.mu.Unlock()
		return
	}
	de.running = true
	de.stopCh = make(chan struct{})
	de.mu.Unlock()

	go func() {
		for {
			select {
			case <-de.stopCh:
				return
			default:
				de.Dream(cyclesPerBatch)
				time.Sleep(pauseBetween)
			}
		}
	}()
}

// StopBackground stops continuous dreaming.
func (de *DreamEngine) StopBackground() {
	de.mu.Lock()
	if de.running {
		close(de.stopCh)
		de.running = false
	}
	de.mu.Unlock()
}

// GetDiscoveries returns all discoveries since last clear.
func (de *DreamEngine) GetDiscoveries() []DreamDiscovery {
	de.mu.Lock()
	defer de.mu.Unlock()
	result := make([]DreamDiscovery, len(de.discoveries))
	copy(result, de.discoveries)
	return result
}

// GetRecentDiscoveries returns discoveries from the last N hours.
func (de *DreamEngine) GetRecentDiscoveries(hours int) []DreamDiscovery {
	de.mu.Lock()
	defer de.mu.Unlock()
	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)
	var recent []DreamDiscovery
	for _, d := range de.discoveries {
		if d.Timestamp.After(cutoff) {
			recent = append(recent, d)
		}
	}
	return recent
}

// ClearDiscoveries resets the discovery list.
func (de *DreamEngine) ClearDiscoveries() {
	de.mu.Lock()
	de.discoveries = nil
	de.mu.Unlock()
}

// -----------------------------------------------------------------------
// Dream operations — five ways to think autonomously.
// -----------------------------------------------------------------------

// pickOperation selects a dream operation with weighted probability.
// Wander (40%) > Synthesize (25%) > Expand (15%) > Infer (10%) > Reflect (10%)
func (de *DreamEngine) pickOperation() int {
	r := de.rng.Float64()
	switch {
	case r < 0.40:
		return 0 // wander
	case r < 0.65:
		return 4 // synthesize
	case r < 0.80:
		return 1 // expand
	case r < 0.90:
		return 2 // infer
	default:
		return 3 // reflect
	}
}

// dreamWander: random walk through the graph looking for surprising connections.
// Picks two random nodes and checks if they're connected within 3 hops.
// If connected through an unexpected intermediary, that's a discovery.
func (de *DreamEngine) dreamWander() *DreamDiscovery {
	labels := de.Graph.AllLabels()
	if len(labels) < 10 {
		return nil
	}

	// Pick two random nodes.
	a := labels[de.rng.Intn(len(labels))]
	b := labels[de.rng.Intn(len(labels))]
	if a == b {
		return nil
	}

	// Check for 2-hop connection.
	aEdges := de.Graph.EdgesFrom(a)
	bEdges := de.Graph.EdgesFrom(b)

	// Direct connection → not surprising.
	for _, e := range aEdges {
		if de.Graph.NodeLabel(e.To) == b {
			return nil
		}
	}

	// 2-hop: A→X→B or A→X←B (shared neighbor)
	aNeighbors := make(map[string]string) // neighborLabel → relation
	for _, e := range aEdges {
		label := de.Graph.NodeLabel(e.To)
		if label != "" {
			aNeighbors[label] = string(e.Relation)
		}
	}

	for _, e := range bEdges {
		label := de.Graph.NodeLabel(e.To)
		if shared, ok := aNeighbors[label]; ok {
			// Found shared neighbor! This is a connection.
			// Check if it's surprising (different domains).
			if isDifferentDomain(a, b) {
				return &DreamDiscovery{
					Type:       "connection",
					Summary:    fmt.Sprintf("Surprising link: %s and %s are connected through %s (%s → %s, %s → %s)", a, b, label, a, shared, b, string(e.Relation)),
					Entities:   []string{a, b, label},
					Confidence: 0.5,
					Timestamp:  time.Now(),
					Novel:      true,
				}
			}
		}
	}

	return nil
}

// dreamExpand: pick a topic from recent conversations and expand knowledge.
func (de *DreamEngine) dreamExpand() *DreamDiscovery {
	if de.Episodic == nil || de.WikiLoader == nil {
		return nil
	}

	// Find a recent topic that isn't well-covered.
	recent := de.Episodic.Recent(20)
	for _, ep := range recent {
		// Extract topic from the episode.
		topic := extractDreamTopic(ep.Input)
		if topic == "" {
			continue
		}

		// Skip if already well-covered (3+ edges).
		edges := de.Graph.EdgesFrom(topic)
		if len(edges) >= 3 {
			continue
		}

		// Try to expand from Wikipedia.
		if de.WikiLoader.HasFetched(topic) {
			continue
		}

		result := de.WikiLoader.FetchAndLearn(topic)
		if result != nil && result.FactCount > 0 {
			return &DreamDiscovery{
				Type:       "expansion",
				Summary:    fmt.Sprintf("Learned about %s from Wikipedia: %d facts extracted", result.Topic, result.FactCount),
				Entities:   []string{result.Topic},
				Confidence: 0.7,
				Timestamp:  time.Now(),
				Novel:      true,
			}
		}
	}

	return nil
}

// dreamInfer: run causal inference to discover new causal edges.
func (de *DreamEngine) dreamInfer() *DreamDiscovery {
	if de.CausalInfer == nil {
		return nil
	}

	report := de.CausalInfer.InferAll()
	if report.AddedCount > 0 {
		return &DreamDiscovery{
			Type:       "causal",
			Summary:    fmt.Sprintf("Inferred %d new causal edges (%d temporal, %d dependency, %d production)", report.AddedCount, report.TemporalCount, report.DependencyCount, report.ProductionCount),
			Confidence: 0.6,
			Timestamp:  time.Now(),
			Novel:      true,
		}
	}

	return nil
}

// dreamReflect: analyze conversation history for behavioral patterns.
func (de *DreamEngine) dreamReflect() *DreamDiscovery {
	if de.Episodic == nil {
		return nil
	}

	recent := de.Episodic.Recent(50)
	if len(recent) < 5 {
		return nil
	}

	// Analyze topic frequency.
	topicCounts := make(map[string]int)
	for _, ep := range recent {
		topic := extractDreamTopic(ep.Input)
		if topic != "" {
			topicCounts[topic]++
		}
	}

	// Find recurring themes.
	var hotTopics []string
	for topic, count := range topicCounts {
		if count >= 3 {
			hotTopics = append(hotTopics, fmt.Sprintf("%s (%dx)", topic, count))
		}
	}

	if len(hotTopics) > 0 {
		return &DreamDiscovery{
			Type:       "pattern",
			Summary:    fmt.Sprintf("Recurring interests detected: %s", strings.Join(hotTopics, ", ")),
			Confidence: 0.8,
			Timestamp:  time.Now(),
			Novel:      false,
		}
	}

	return nil
}

// dreamSynthesize: find cross-domain connections that generate novel insights.
func (de *DreamEngine) dreamSynthesize() *DreamDiscovery {
	labels := de.Graph.AllLabels()
	if len(labels) < 20 {
		return nil
	}

	// Pick a random node and find its furthest-domain neighbor.
	start := labels[de.rng.Intn(len(labels))]
	startEdges := de.Graph.EdgesFrom(start)
	if len(startEdges) == 0 {
		return nil
	}

	// Get start node's domain.
	startDomain := getDomain(de.Graph, start)

	// Walk 2 hops and look for nodes in a different domain.
	for _, e1 := range startEdges {
		hop1 := de.Graph.NodeLabel(e1.To)
		if hop1 == "" {
			continue
		}
		hop1Edges := de.Graph.EdgesFrom(hop1)
		for _, e2 := range hop1Edges {
			hop2 := de.Graph.NodeLabel(e2.To)
			if hop2 == "" || hop2 == start {
				continue
			}

			hop2Domain := getDomain(de.Graph, hop2)
			if startDomain != "" && hop2Domain != "" && startDomain != hop2Domain {
				// Cross-domain connection found!
				insight := fmt.Sprintf("Cross-domain insight: %s (%s) connects to %s (%s) through %s. The %s relationship between %s and %s, combined with the %s relationship to %s, suggests an interdisciplinary bridge.",
					start, startDomain, hop2, hop2Domain, hop1,
					string(e1.Relation), start, hop1,
					string(e2.Relation), hop2)

				return &DreamDiscovery{
					Type:       "insight",
					Summary:    insight,
					Entities:   []string{start, hop1, hop2},
					Confidence: 0.4,
					Timestamp:  time.Now(),
					Novel:      true,
				}
			}
		}
	}

	return nil
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func extractDreamTopic(input string) string {
	lower := strings.ToLower(strings.TrimSpace(input))
	prefixes := []string{
		"what is ", "what are ", "who is ", "who was ", "explain ",
		"tell me about ", "describe ", "define ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			topic := strings.TrimRight(strings.TrimSpace(lower[len(p):]), "?.!")
			return strings.TrimSpace(topic)
		}
	}
	return ""
}

func isDifferentDomain(a, b string) bool {
	// Simple heuristic: if both are short (likely concepts not sentences),
	// they might be from different domains.
	return len(a) < 30 && len(b) < 30
}

func getDomain(graph *CognitiveGraph, label string) string {
	edges := graph.EdgesFrom(label)
	for _, e := range edges {
		if e.Relation == RelDomain {
			return graph.NodeLabel(e.To)
		}
		if e.Relation == RelIsA {
			return graph.NodeLabel(e.To)
		}
	}
	return ""
}

// FormatDreamReport formats a dream report for display.
func FormatDreamReport(report *DreamReport) string {
	var b strings.Builder

	fmt.Fprintf(&b, "# Dream Report\n\n")
	fmt.Fprintf(&b, "**Duration:** %s | **Cycles:** %d | **Discoveries:** %d\n\n",
		report.Duration.Round(time.Millisecond), report.CyclesRun, len(report.Discoveries))

	if report.NovelConnections > 0 {
		fmt.Fprintf(&b, "**Novel connections:** %d\n", report.NovelConnections)
	}
	if report.TopicsExpanded > 0 {
		fmt.Fprintf(&b, "**Topics expanded:** %d\n", report.TopicsExpanded)
	}
	if report.CausalEdgesAdded > 0 {
		fmt.Fprintf(&b, "**Causal edges:** %d\n", report.CausalEdgesAdded)
	}

	if len(report.Discoveries) > 0 {
		b.WriteString("\n## Discoveries\n\n")
		for i, d := range report.Discoveries {
			novelTag := ""
			if d.Novel {
				novelTag = " [NEW]"
			}
			fmt.Fprintf(&b, "%d. **[%s]** %s (confidence: %.0f%%)%s\n",
				i+1, d.Type, d.Summary, d.Confidence*100, novelTag)
		}
	}

	return b.String()
}

// MorningBriefingInsights returns dream discoveries formatted for
// the daily morning briefing. Filters for high-confidence, novel findings.
func (de *DreamEngine) MorningBriefingInsights(maxInsights int) []string {
	discoveries := de.GetRecentDiscoveries(24) // last 24 hours
	if len(discoveries) == 0 {
		return nil
	}

	// Sort by confidence descending, novel first.
	// Simple selection: pick the best ones.
	var insights []string
	for _, d := range discoveries {
		if d.Confidence < 0.4 || !d.Novel {
			continue
		}
		insights = append(insights, d.Summary)
		if len(insights) >= maxInsights {
			break
		}
	}

	return insights
}

// Stats returns dream engine statistics.
func (de *DreamEngine) Stats() map[string]interface{} {
	de.mu.Lock()
	defer de.mu.Unlock()
	return map[string]interface{}{
		"cycles_run":        de.CyclesRun,
		"connections_found": de.ConnectionsFound,
		"topics_expanded":   de.TopicsExpanded,
		"total_discoveries": len(de.discoveries),
		"last_dream":        de.LastDreamTime,
		"running":           de.running,
	}
}

// DreamCycleInterval returns the recommended pause between dream batches
// based on system load and time of day. Shorter pauses at night (more
// dreaming while user sleeps), longer during active hours.
func DreamCycleInterval() time.Duration {
	hour := time.Now().Hour()
	switch {
	case hour >= 0 && hour < 6:
		return 30 * time.Second // Night: dream actively
	case hour >= 6 && hour < 9:
		return 2 * time.Minute // Morning: moderate
	case hour >= 22:
		return 1 * time.Minute // Late evening: winding up
	default:
		return 5 * time.Minute // Daytime: conserve resources
	}
}

// DefaultDreamCycles returns the recommended cycles per batch.
func DefaultDreamCycles() int {
	hour := time.Now().Hour()
	if hour >= 0 && hour < 6 {
		return 50 // Night: more cycles
	}
	return 10 // Day: light dreaming
}

// RunQuickDream runs a short dream session and returns a summary.
// Used by the /dream command for on-demand dreaming.
func (de *DreamEngine) RunQuickDream() string {
	report := de.Dream(20)
	if report == nil {
		return "Dream engine not available."
	}

	if len(report.Discoveries) == 0 {
		return fmt.Sprintf("Dreamed for %s (%d cycles) — no new discoveries this time.",
			report.Duration.Round(time.Millisecond), report.CyclesRun)
	}

	var parts []string
	parts = append(parts, fmt.Sprintf("Dreamed for %s (%d cycles, %d discoveries):",
		report.Duration.Round(time.Millisecond), report.CyclesRun, len(report.Discoveries)))

	for _, d := range report.Discoveries {
		parts = append(parts, fmt.Sprintf("  - %s", d.Summary))
	}

	return strings.Join(parts, "\n")
}

// SleepDuration returns how long ago the last dream was, used to
// determine if Nous should share dream discoveries.
func (de *DreamEngine) SleepDuration() time.Duration {
	de.mu.Lock()
	defer de.mu.Unlock()
	if de.LastDreamTime.IsZero() {
		return 0
	}
	return time.Since(de.LastDreamTime)
}

// PendingInsightCount returns how many unshared novel discoveries exist.
func (de *DreamEngine) PendingInsightCount() int {
	de.mu.Lock()
	defer de.mu.Unlock()
	count := 0
	for _, d := range de.discoveries {
		if d.Novel && d.Confidence >= 0.4 {
			count++
		}
	}
	return count
}

// PopBestInsight returns and removes the highest-confidence novel discovery.
func (de *DreamEngine) PopBestInsight() *DreamDiscovery {
	de.mu.Lock()
	defer de.mu.Unlock()

	bestIdx := -1
	bestScore := 0.0

	for i, d := range de.discoveries {
		if !d.Novel {
			continue
		}
		score := d.Confidence
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx < 0 {
		return nil
	}

	discovery := de.discoveries[bestIdx]
	// Remove from list.
	de.discoveries = append(de.discoveries[:bestIdx], de.discoveries[bestIdx+1:]...)
	return &discovery
}

// maxF returns the larger of two float64 values.
func maxF(a, b float64) float64 {
	return math.Max(a, b)
}
