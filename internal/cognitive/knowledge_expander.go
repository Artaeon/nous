package cognitive

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// Recursive Knowledge Expander — self-growing knowledge base.
//
// The core insight: every paragraph in the knowledge corpus implicitly
// references topics the system doesn't yet have full coverage for.
// "Quantum mechanics... semiconductors, and lasers" mentions topics
// that may not have their own paragraphs.
//
// Three-phase autonomous growth:
//
//   Phase 1: Frontier Discovery — scan existing text for mentioned-but-
//   unknown topics. Cross-reference against the graph to find gaps.
//
//   Phase 2: Gap Filling — for each frontier topic, check if any
//   existing paragraph covers it. Extract facts from context sentences.
//
//   Phase 3: Recursive Closure — after filling gaps, re-scan for new
//   frontier topics. Stop when the frontier shrinks below a threshold.
//
// This is a novel architecture: the system discovers what it needs to
// know by analyzing what it already knows, fills those gaps, then
// discovers more gaps from the new knowledge. No external API needed.
// -----------------------------------------------------------------------

// KnowledgeExpander discovers and fills knowledge gaps autonomously.
type KnowledgeExpander struct {
	Graph        *CognitiveGraph
	SelfTeacher  *SelfTeach
	FactExtract  *WikiFactExtractor
	KnowledgeDir string // path to knowledge/*.txt files
}

// FrontierTopic is a topic mentioned in the corpus but not well-covered.
type FrontierTopic struct {
	Name       string   // the topic name
	Mentions   int      // how many times it appears in existing text
	Sources    []string // which paragraphs mention it
	HasNode    bool     // does a graph node exist?
	EdgeCount  int      // how many edges from/to this node
	Priority   float64  // expansion priority (higher = more important)
}

// ExpansionReport summarizes what the expander discovered and filled.
type ExpansionReport struct {
	Generation       int
	FrontierSize     int
	TopicsExpanded   int
	FactsExtracted   int
	EdgesAdded       int
	TopFrontier      []FrontierTopic // top 20 by priority
	ConvergenceRatio float64         // frontier/total — stops when < 0.05
}

// NewKnowledgeExpander creates an expander wired to existing systems.
func NewKnowledgeExpander(graph *CognitiveGraph, st *SelfTeach, knowledgeDir string) *KnowledgeExpander {
	return &KnowledgeExpander{
		Graph:        graph,
		SelfTeacher:  st,
		KnowledgeDir: knowledgeDir,
	}
}

// DiscoverFrontier scans existing knowledge text for mentioned-but-unknown topics.
// Returns topics sorted by priority (most frequently mentioned first).
func (ke *KnowledgeExpander) DiscoverFrontier() []FrontierTopic {
	if ke.Graph == nil || ke.KnowledgeDir == "" {
		return nil
	}

	// Load all knowledge paragraphs.
	paragraphs := loadKnowledgeParagraphs(ke.KnowledgeDir)
	if len(paragraphs) == 0 {
		return nil
	}

	// Extract candidate topics from all paragraphs.
	candidates := make(map[string]*FrontierTopic)

	for _, para := range paragraphs {
		topics := extractCandidateTopics(para)
		for _, t := range topics {
			lower := strings.ToLower(t)
			if ft, ok := candidates[lower]; ok {
				ft.Mentions++
			} else {
				candidates[lower] = &FrontierTopic{
					Name:     t,
					Mentions: 1,
				}
			}
		}
	}

	// Cross-reference against the graph to find gaps.
	var frontier []FrontierTopic
	for _, ft := range candidates {
		ft.HasNode = ke.Graph.HasLabel(ft.Name)

		if ft.HasNode {
			// Node exists — check if it has sufficient edges.
			edges := ke.Graph.EdgesFrom(ft.Name)
			incoming := ke.Graph.EdgesTo(ft.Name)
			ft.EdgeCount = len(edges) + len(incoming)

			// Well-covered topics (3+ edges) are not frontier.
			if ft.EdgeCount >= 3 {
				continue
			}
		}

		// Priority: more mentions + fewer edges = higher priority.
		ft.Priority = float64(ft.Mentions) * (1.0 - float64(ft.EdgeCount)*0.15)
		if !ft.HasNode {
			ft.Priority *= 1.5 // boost completely unknown topics
		}

		frontier = append(frontier, *ft)
	}

	// Sort by priority descending.
	sort.Slice(frontier, func(i, j int) bool {
		return frontier[i].Priority > frontier[j].Priority
	})

	return frontier
}

// Expand runs the full recursive expansion loop.
// maxGenerations limits how many rounds of discover→fill→rediscover.
// Returns a report for each generation.
func (ke *KnowledgeExpander) Expand(maxGenerations int) []ExpansionReport {
	if maxGenerations < 1 {
		maxGenerations = 1
	}
	if maxGenerations > 5 {
		maxGenerations = 5
	}

	var reports []ExpansionReport
	totalTopics := float64(len(ke.Graph.AllLabels()))

	for gen := 0; gen < maxGenerations; gen++ {
		frontier := ke.DiscoverFrontier()

		report := ExpansionReport{
			Generation:   gen + 1,
			FrontierSize: len(frontier),
		}

		if totalTopics > 0 {
			report.ConvergenceRatio = float64(len(frontier)) / totalTopics
		}

		// Top 20 for the report.
		top := 20
		if len(frontier) < top {
			top = len(frontier)
		}
		report.TopFrontier = frontier[:top]

		// Fill gaps: use SelfTeacher to mine existing knowledge files.
		expanded := 0
		factsExtracted := 0
		edgesAdded := 0

		// Process top-priority frontier topics.
		limit := 100
		if len(frontier) < limit {
			limit = len(frontier)
		}

		for _, ft := range frontier[:limit] {
			if ke.SelfTeacher != nil && !ke.SelfTeacher.HasLearned(ft.Name) {
				learned, _ := ke.SelfTeacher.LearnAbout(ft.Name)
				if learned > 0 {
					expanded++
					factsExtracted += learned
				}
			}

			// Also try extracting facts from paragraphs that mention this topic.
			if ke.Graph != nil {
				newEdges := ke.extractFactsAbout(ft.Name)
				edgesAdded += newEdges
			}
		}

		report.TopicsExpanded = expanded
		report.FactsExtracted = factsExtracted
		report.EdgesAdded = edgesAdded

		reports = append(reports, report)

		// Convergence check: stop when frontier is < 5% of total topics.
		if report.ConvergenceRatio < 0.05 && gen > 0 {
			break
		}

		// Update total for next generation's convergence check.
		totalTopics = float64(len(ke.Graph.AllLabels()))
	}

	return reports
}

// extractFactsAbout finds paragraphs mentioning a topic and extracts
// typed facts from the surrounding context.
func (ke *KnowledgeExpander) extractFactsAbout(topic string) int {
	paragraphs := loadKnowledgeParagraphs(ke.KnowledgeDir)
	topicLower := strings.ToLower(topic)
	added := 0

	for _, para := range paragraphs {
		if !strings.Contains(strings.ToLower(para), topicLower) {
			continue
		}

		// Split paragraph into sentences and find ones mentioning the topic.
		sentences := expanderSplitSentences(para)
		for _, sent := range sentences {
			if !strings.Contains(strings.ToLower(sent), topicLower) {
				continue
			}

			// Extract typed relationships from this sentence.
			facts := extractSentenceFacts(sent, topic)
			for _, fact := range facts {
				ke.Graph.AddEdge(fact.from, fact.to, fact.rel, "expander:"+topic)
				added++
			}
		}
	}

	return added
}

// -----------------------------------------------------------------------
// Candidate topic extraction — noun phrase mining from prose.
// -----------------------------------------------------------------------

// Patterns for extracting candidate topics from text.
var (
	// Capitalized multi-word names: "Albert Einstein", "World War II"
	properNounRe = regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+(?:of|the|and|in|de|von|van)\s+)?(?:[A-Z][a-z]+)+)\b`)

	// Domain terms: "X theory", "Euler's theorem", "quantum effect"
	// Handles possessives ([\w']+) and up to 3 preceding words.
	domainTermRe = regexp.MustCompile(`(?i)\b([\w']+(?:\s+[\w']+){0,3})\s+(?:theory|principle|law|effect|equation|model|theorem|hypothesis|paradox|phenomenon|mechanism|algorithm|protocol|technique|method)\b`)

	// Terms after "such as", "including", "for example"
	exemplarRe = regexp.MustCompile(`(?i)(?:such as|including|for example|e\.g\.|like)\s+(.{5,100}?)(?:\.|;|$)`)

	// Paragraph-initial subjects: "Gravity is the...", "DNA is a..."
	paraSubjectRe = regexp.MustCompile(`(?:^|\n\n)([A-Z][\w\s'-]{2,50}?)\s+(?:is|are|was|were)\s+`)
)

func extractCandidateTopics(paragraph string) []string {
	seen := make(map[string]bool)
	var topics []string

	addTopic := func(t string) {
		t = strings.TrimSpace(t)
		t = strings.Trim(t, ".,;:!?()\"'")

		// Strip possessive fragments: "'s theorem" → discard
		if strings.HasPrefix(t, "s ") || strings.HasPrefix(t, "'s ") {
			return
		}

		// Strip leading articles/determiners/conjunctions.
		stripPrefixes := []string{
			"the ", "a ", "an ", "this ", "that ", "these ", "those ",
			"and ", "or ", "but ", "to ", "of ", "in ", "by ", "for ", "with ",
		}
		changed := true
		for changed {
			changed = false
			lower := strings.ToLower(t)
			for _, p := range stripPrefixes {
				if strings.HasPrefix(lower, p) {
					t = strings.TrimSpace(t[len(p):])
					changed = true
					break
				}
			}
		}

		if len(t) < 3 || len(t) > 60 {
			return
		}

		// Must start with a letter.
		if len(t) > 0 && !unicode.IsLetter(rune(t[0])) {
			return
		}

		// Reject topics that are just common words or sentence fragments.
		words := strings.Fields(strings.ToLower(t))
		if len(words) > 0 && isStopPhrase(words[0]) {
			return
		}

		lower := strings.ToLower(t)
		if seen[lower] {
			return
		}
		if isStopPhrase(lower) {
			return
		}
		seen[lower] = true
		topics = append(topics, t)
	}

	// Paragraph-initial subjects (highest quality).
	for _, m := range paraSubjectRe.FindAllStringSubmatch(paragraph, -1) {
		addTopic(strings.TrimSpace(m[1]))
	}

	// Proper nouns.
	for _, m := range properNounRe.FindAllStringSubmatch(paragraph, -1) {
		addTopic(m[1])
	}

	// Domain terms.
	for _, m := range domainTermRe.FindAllStringSubmatch(paragraph, -1) {
		full := m[0]
		// Clean possessive: "Euler's theorem" is fine, but strip leading noise.
		addTopic(full)
	}

	// Exemplar lists.
	for _, m := range exemplarRe.FindAllStringSubmatch(paragraph, -1) {
		items := strings.Split(m[1], ",")
		for _, item := range items {
			item = strings.TrimSpace(item)
			for _, sub := range strings.Split(item, " and ") {
				addTopic(strings.TrimSpace(sub))
			}
		}
	}

	return topics
}

func isStopPhrase(s string) bool {
	stops := map[string]bool{
		"the": true, "this": true, "that": true, "these": true,
		"which": true, "what": true, "where": true, "when": true,
		"how": true, "why": true, "such": true, "other": true,
		"many": true, "some": true, "most": true, "each": true,
		"more": true, "less": true, "than": true, "also": true,
		"both": true, "all": true, "any": true, "its": true,
		"their": true, "not": true, "can": true, "may": true,
		"s": true, "is": true, "was": true, "are": true,
		"were": true, "has": true, "had": true, "have": true,
		"a": true, "an": true, "and": true, "or": true,
		"for": true, "with": true, "by": true, "to": true,
		// Too generic to be useful as frontier topics.
		"technique": true, "theory": true, "principle": true,
		"method": true, "process": true, "system": true,
		"model": true, "approach": true, "concept": true,
		"substance": true, "material": true, "structure": true,
		"mechanism": true, "function": true, "property": true,
		"type": true, "form": true, "kind": true,
	}
	return stops[s]
}

// expanderSplitSentences splits text into sentences on ". " boundaries.
func expanderSplitSentences(text string) []string {
	var sentences []string
	remaining := text
	for {
		idx := strings.Index(remaining, ". ")
		if idx < 0 {
			if s := strings.TrimSpace(remaining); s != "" {
				sentences = append(sentences, s)
			}
			break
		}
		sent := strings.TrimSpace(remaining[:idx+1])
		if sent != "" {
			sentences = append(sentences, sent)
		}
		remaining = remaining[idx+2:]
	}
	return sentences
}

// sentenceFact is a typed relationship extracted from a sentence.
type sentenceFact struct {
	from string
	to   string
	rel  RelType
}

// extractSentenceFacts extracts simple typed facts from a sentence
// mentioning the given topic.
func extractSentenceFacts(sentence, topic string) []sentenceFact {
	var facts []sentenceFact
	lower := strings.ToLower(sentence)

	// "topic is a/an X" → is_a
	isAPatterns := []string{" is a ", " is an ", " are "}
	for _, pat := range isAPatterns {
		if idx := strings.Index(lower, strings.ToLower(topic)+pat); idx >= 0 {
			after := sentence[idx+len(topic)+len(pat):]
			obj := firstNounPhrase(after)
			if obj != "" {
				facts = append(facts, sentenceFact{topic, obj, RelIsA})
			}
		}
	}

	// "topic causes/enables/prevents/requires X"
	causalVerbs := map[string]RelType{
		" causes ": RelCauses, " enables ": RelEnables,
		" prevents ": RelPrevents, " requires ": RelRequires,
		" produces ": RelProduces, " leads to ": RelCauses,
		" inhibits ": RelPrevents, " facilitates ": RelEnables,
	}
	for verb, rel := range causalVerbs {
		if idx := strings.Index(lower, verb); idx >= 0 {
			after := sentence[idx+len(verb):]
			obj := firstNounPhrase(after)
			if obj != "" {
				// Figure out the subject — if topic is before the verb, it's the subject.
				topicIdx := strings.Index(lower, strings.ToLower(topic))
				if topicIdx >= 0 && topicIdx < idx {
					facts = append(facts, sentenceFact{topic, obj, rel})
				}
			}
		}
	}

	return facts
}

// firstNounPhrase extracts the first noun phrase from text
// (up to a comma, period, or conjunction).
func firstNounPhrase(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	// Take text up to the first delimiter.
	for i, r := range text {
		if r == ',' || r == '.' || r == ';' || r == ':' {
			text = text[:i]
			break
		}
	}

	// Trim trailing function words.
	text = strings.TrimSpace(text)
	words := strings.Fields(text)
	if len(words) > 6 {
		words = words[:6]
	}

	// Remove trailing articles/prepositions.
	for len(words) > 0 {
		last := strings.ToLower(words[len(words)-1])
		if last == "the" || last == "a" || last == "an" || last == "and" ||
			last == "or" || last == "of" || last == "in" || last == "by" {
			words = words[:len(words)-1]
		} else {
			break
		}
	}

	result := strings.Join(words, " ")
	// Must start with a letter.
	if len(result) > 0 && !unicode.IsLetter(rune(result[0])) {
		return ""
	}
	return result
}

// FormatExpansionReport formats a report for display.
func FormatExpansionReport(reports []ExpansionReport) string {
	var b strings.Builder

	b.WriteString("# Knowledge Expansion Report\n\n")

	totalExpanded := 0
	totalFacts := 0
	totalEdges := 0

	for _, r := range reports {
		fmt.Fprintf(&b, "## Generation %d\n\n", r.Generation)
		fmt.Fprintf(&b, "- Frontier size: %d topics\n", r.FrontierSize)
		fmt.Fprintf(&b, "- Topics expanded: %d\n", r.TopicsExpanded)
		fmt.Fprintf(&b, "- Facts extracted: %d\n", r.FactsExtracted)
		fmt.Fprintf(&b, "- Edges added: %d\n", r.EdgesAdded)
		fmt.Fprintf(&b, "- Convergence: %.1f%%\n\n", r.ConvergenceRatio*100)

		if len(r.TopFrontier) > 0 {
			b.WriteString("Top frontier topics:\n")
			for i, ft := range r.TopFrontier {
				status := "new"
				if ft.HasNode {
					status = fmt.Sprintf("%d edges", ft.EdgeCount)
				}
				fmt.Fprintf(&b, "  %d. %s (mentioned %dx, %s, priority %.1f)\n",
					i+1, ft.Name, ft.Mentions, status, ft.Priority)
			}
			b.WriteString("\n")
		}

		totalExpanded += r.TopicsExpanded
		totalFacts += r.FactsExtracted
		totalEdges += r.EdgesAdded
	}

	fmt.Fprintf(&b, "## Total: %d topics expanded, %d facts, %d edges across %d generations\n",
		totalExpanded, totalFacts, totalEdges, len(reports))

	return b.String()
}
