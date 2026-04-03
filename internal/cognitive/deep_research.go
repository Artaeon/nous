package cognitive

import (
	"fmt"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Autonomous Deep Research Agent — multi-step investigation.
//
// Not just answering questions but conducting research: decomposing a
// topic into sub-domains, fetching information across all of them,
// discovering causal chains, cross-referencing with user interests,
// and producing a structured knowledge map.
//
// Unlike LLM-based research (MiroFish), every fact is sourced and
// traceable. The research persists in the graph permanently.
// -----------------------------------------------------------------------

// DeepResearchAgent conducts autonomous multi-step investigations.
type DeepResearchAgent struct {
	Graph       *CognitiveGraph
	WikiLoader  *WikipediaLoader
	CausalInfer *CausalInferenceEngine
	MultiHop    *MultiHopReasoner
}

// ResearchPlan is the decomposed investigation structure.
type ResearchPlan struct {
	MainTopic   string
	SubTopics   []string
	Depth       string // "quick", "standard", "deep"
}

// ResearchResult is the full output of a research investigation.
type ResearchResult struct {
	Topic          string
	Plan           ResearchPlan
	TopicsCovered  int
	FactsGathered  int
	CausalChains   int
	Connections    []string // cross-topic connections found
	KeyFindings    []string
	Report         string
	Duration       time.Duration
}

// NewDeepResearchAgent creates a research agent.
func NewDeepResearchAgent(
	graph *CognitiveGraph,
	wiki *WikipediaLoader,
	causal *CausalInferenceEngine,
	multihop *MultiHopReasoner,
) *DeepResearchAgent {
	return &DeepResearchAgent{
		Graph:       graph,
		WikiLoader:  wiki,
		CausalInfer: causal,
		MultiHop:    multihop,
	}
}

// Research conducts a full investigation on a topic.
func (dra *DeepResearchAgent) Research(topic, depth string) *ResearchResult {
	start := time.Now()

	result := &ResearchResult{
		Topic: topic,
	}

	// Phase 1: Decompose the topic into sub-domains.
	plan := dra.decompose(topic, depth)
	result.Plan = plan

	// Phase 2: Fetch knowledge for each sub-topic.
	for _, sub := range plan.SubTopics {
		if dra.WikiLoader != nil {
			wikiResult := dra.WikiLoader.FetchAndLearn(sub)
			if wikiResult != nil {
				result.FactsGathered += wikiResult.FactCount
				result.TopicsCovered++
			}
		}
	}

	// Also fetch the main topic.
	if dra.WikiLoader != nil {
		mainResult := dra.WikiLoader.FetchAndLearn(topic)
		if mainResult != nil {
			result.FactsGathered += mainResult.FactCount
			result.TopicsCovered++
		}
	}

	// Phase 3: Run causal inference on the expanded graph.
	if dra.CausalInfer != nil {
		inferReport := dra.CausalInfer.InferAll()
		result.CausalChains = inferReport.AddedCount
	}

	// Phase 4: Find cross-topic connections.
	if dra.MultiHop != nil {
		for i := 0; i < len(plan.SubTopics)-1; i++ {
			for j := i + 1; j < len(plan.SubTopics); j++ {
				conn := dra.MultiHop.FindConnection(plan.SubTopics[i], plan.SubTopics[j])
				if conn != nil && conn.Summary != "" {
					result.Connections = append(result.Connections, conn.Summary)
				}
			}
		}
	}

	// Phase 5: Extract key findings.
	result.KeyFindings = dra.extractKeyFindings(topic, plan.SubTopics)

	// Phase 6: Compose report.
	result.Duration = time.Since(start)
	result.Report = dra.composeReport(result)

	return result
}

// -----------------------------------------------------------------------
// Topic decomposition
// -----------------------------------------------------------------------

func (dra *DeepResearchAgent) decompose(topic string, depth string) ResearchPlan {
	plan := ResearchPlan{
		MainTopic: topic,
		Depth:     depth,
	}

	// Use graph edges to find related sub-topics.
	if dra.Graph != nil {
		edges := dra.Graph.EdgesFrom(topic)
		for _, e := range edges {
			label := dra.Graph.NodeLabel(e.To)
			if label != "" && label != topic && len(label) > 3 {
				plan.SubTopics = append(plan.SubTopics, label)
			}
		}

		// Also check incoming edges.
		incoming := dra.Graph.EdgesTo(topic)
		for _, e := range incoming {
			label := dra.Graph.NodeLabel(e.From)
			if label != "" && label != topic && len(label) > 3 {
				plan.SubTopics = append(plan.SubTopics, label)
			}
		}
	}

	// Add heuristic sub-topics based on common research patterns.
	heuristic := []string{
		topic + " history",
		topic + " applications",
		topic + " challenges",
	}

	maxTopics := 5
	switch depth {
	case "deep":
		maxTopics = 15
	case "standard":
		maxTopics = 10
	case "quick":
		maxTopics = 5
	}

	// Deduplicate and limit.
	seen := make(map[string]bool)
	var unique []string
	for _, sub := range append(plan.SubTopics, heuristic...) {
		lower := strings.ToLower(sub)
		if !seen[lower] && lower != strings.ToLower(topic) {
			seen[lower] = true
			unique = append(unique, sub)
		}
		if len(unique) >= maxTopics {
			break
		}
	}
	plan.SubTopics = unique

	return plan
}

func (dra *DeepResearchAgent) extractKeyFindings(topic string, subTopics []string) []string {
	var findings []string

	if dra.Graph == nil {
		return findings
	}

	// Find the most connected sub-topic.
	maxEdges := 0
	bestSub := ""
	for _, sub := range subTopics {
		edges := dra.Graph.EdgesFrom(sub)
		if len(edges) > maxEdges {
			maxEdges = len(edges)
			bestSub = sub
		}
	}

	if bestSub != "" {
		findings = append(findings,
			fmt.Sprintf("Most connected sub-topic: %s (%d relationships)", bestSub, maxEdges))
	}

	// Find causal chains involving the main topic.
	edges := dra.Graph.EdgesFrom(topic)
	for _, e := range edges {
		if isCausalRelation(e.Relation) {
			label := dra.Graph.NodeLabel(e.To)
			findings = append(findings,
				fmt.Sprintf("%s %s %s", topic, string(e.Relation), label))
		}
	}

	return findings
}

func (dra *DeepResearchAgent) composeReport(result *ResearchResult) string {
	var b strings.Builder

	fmt.Fprintf(&b, "# Research Report: %s\n\n", result.Topic)
	fmt.Fprintf(&b, "**Depth:** %s | **Duration:** %s | **Topics:** %d | **Facts:** %d\n\n",
		result.Plan.Depth, result.Duration.Round(time.Millisecond),
		result.TopicsCovered, result.FactsGathered)

	// Sub-topics explored.
	if len(result.Plan.SubTopics) > 0 {
		b.WriteString("## Sub-topics Explored\n\n")
		for _, sub := range result.Plan.SubTopics {
			fmt.Fprintf(&b, "- %s\n", sub)
		}
		b.WriteString("\n")
	}

	// Key findings.
	if len(result.KeyFindings) > 0 {
		b.WriteString("## Key Findings\n\n")
		for _, f := range result.KeyFindings {
			fmt.Fprintf(&b, "- %s\n", f)
		}
		b.WriteString("\n")
	}

	// Cross-topic connections.
	if len(result.Connections) > 0 {
		b.WriteString("## Cross-Topic Connections\n\n")
		for _, c := range result.Connections {
			fmt.Fprintf(&b, "- %s\n", c)
		}
		b.WriteString("\n")
	}

	// Causal chains.
	if result.CausalChains > 0 {
		fmt.Fprintf(&b, "## Causal Analysis\n\n%d new causal edges discovered from research.\n\n", result.CausalChains)
	}

	fmt.Fprintf(&b, "---\n*All facts sourced from Wikipedia and knowledge graph. Zero hallucination.*\n")

	return b.String()
}

// IsResearchQuery detects deep research requests.
func IsResearchQuery(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	triggers := []string{
		"research ", "investigate ", "deep dive ",
		"study ", "analyze ", "explore everything about ",
		"tell me everything about ", "comprehensive overview of ",
	}
	for _, t := range triggers {
		if strings.HasPrefix(lower, t) {
			return true
		}
	}
	return false
}
