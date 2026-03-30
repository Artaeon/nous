package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// DocumentGenerator creates structured long-form documents from knowledge.
// It arranges REAL knowledge paragraphs into logical multi-section structures,
// never generating text — only organizing existing content.
type DocumentGenerator struct {
	graph        *CognitiveGraph
	knowledgeDir string

	// Paragraph cache — loaded once from knowledge files.
	paragraphs     []string
	paragraphsOnce sync.Once
}

// DocumentSection is one section of a generated document.
type DocumentSection struct {
	Heading string
	Content string
}

// GeneratedDocument is a complete multi-section document.
type GeneratedDocument struct {
	Title    string
	Sections []DocumentSection
	WordCount int
}

// NewDocumentGenerator creates a DocumentGenerator wired to the cognitive
// graph and knowledge text directory.
func NewDocumentGenerator(graph *CognitiveGraph, knowledgeDir string) *DocumentGenerator {
	return &DocumentGenerator{
		graph:        graph,
		knowledgeDir: knowledgeDir,
	}
}

// Generate creates a structured long-form document about topic in the given
// style. Supported styles: "overview", "report", "essay", "guide".
func (dg *DocumentGenerator) Generate(topic string, style string) *GeneratedDocument {
	if topic == "" {
		return &GeneratedDocument{Title: "Untitled"}
	}

	switch strings.ToLower(style) {
	case "report":
		return dg.generateReport(topic)
	case "essay":
		return dg.generateEssay(topic)
	case "guide":
		return dg.generateGuide(topic)
	default:
		return dg.generateOverview(topic)
	}
}

// generateOverview builds an overview document:
// Introduction, Background, Key Concepts, Applications, Current State, Conclusion.
func (dg *DocumentGenerator) generateOverview(topic string) *GeneratedDocument {
	doc := &GeneratedDocument{
		Title: "Overview: " + capitalizeFirstDoc(topic),
	}

	mainPara := dg.findParagraph(topic)

	// Introduction — the main paragraph about the topic.
	introContent := mainPara
	if introContent == "" {
		introContent = dg.descriptionFromGraph(topic)
	}
	if introContent != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Introduction",
			Content: introContent,
		})
	}

	// Background — origin/history facts from the graph.
	background := dg.buildBackgroundSection(topic)
	if background != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Background",
			Content: background,
		})
	}

	// Key Concepts — related topics and their paragraphs.
	keyConcepts := dg.buildKeyConceptsSection(topic)
	if keyConcepts != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Key Concepts",
			Content: keyConcepts,
		})
	}

	// Applications — UsedFor facts.
	applications := dg.buildApplicationsSection(topic)
	if applications != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Applications",
			Content: applications,
		})
	}

	// Current State — latest related info.
	currentState := dg.buildCurrentStateSection(topic)
	if currentState != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Current State",
			Content: currentState,
		})
	}

	// Conclusion — synthesis of main points.
	conclusion := dg.buildConclusion(topic, doc.Sections)
	if conclusion != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Conclusion",
			Content: conclusion,
		})
	}

	doc.WordCount = countDocWords(doc)
	return doc
}

// generateReport builds a formal report:
// Executive Summary, Background, Analysis, Findings, Recommendations.
// Uses sentence deduplication to ensure no content appears in multiple sections.
func (dg *DocumentGenerator) generateReport(topic string) *GeneratedDocument {
	doc := &GeneratedDocument{
		Title: "Report: " + capitalizeFirstDoc(topic),
	}

	used := make(map[string]bool) // tracks used sentences across sections
	mainPara := dg.findParagraph(topic)

	// Executive Summary — first 3 sentences of the main paragraph.
	if mainPara != "" {
		summary := firstSentencesOf(mainPara, 3)
		markUsed(used, summary)
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Executive Summary",
			Content: summary,
		})
	} else {
		desc := dg.descriptionFromGraph(topic)
		if desc != "" {
			summary := firstSentencesOf(desc, 3)
			markUsed(used, summary)
			doc.Sections = append(doc.Sections, DocumentSection{
				Heading: "Executive Summary",
				Content: summary,
			})
		}
	}

	// Background — origin/history facts. If no history edges exist, use
	// the REMAINING sentences of the main paragraph (not the whole thing).
	background := dg.buildBackgroundSection(topic)
	if background == "" && mainPara != "" {
		sentences := splitSentencesDoc(mainPara)
		var remaining []string
		for _, s := range sentences {
			if !used[normalizeForDedup(s)] {
				remaining = append(remaining, s)
			}
		}
		if len(remaining) > 0 {
			background = strings.Join(remaining, " ")
		}
	}
	background = removeUsedSentences(used, background)
	if background != "" {
		markUsed(used, background)
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Background",
			Content: background,
		})
	}

	// Analysis — related concepts expanded, deduped against what's already used.
	analysis := removeUsedSentences(used, dg.buildKeyConceptsSection(topic))
	if analysis != "" {
		markUsed(used, analysis)
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Analysis",
			Content: analysis,
		})
	}

	// Findings — structured facts from the graph (bullet points, less likely to repeat).
	findings := dg.buildFindingsSection(topic)
	if findings != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Findings",
			Content: findings,
		})
	}

	// Recommendations — based on applications and uses.
	recommendations := dg.buildRecommendationsSection(topic)
	if recommendations != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Recommendations",
			Content: recommendations,
		})
	}

	doc.WordCount = countDocWords(doc)
	return doc
}

// markUsed adds all sentences from text to the used set.
func markUsed(used map[string]bool, text string) {
	for _, s := range splitSentencesDoc(text) {
		used[normalizeForDedup(s)] = true
	}
}

// removeUsedSentences strips sentences that already appear in an earlier section.
func removeUsedSentences(used map[string]bool, text string) string {
	if text == "" || len(used) == 0 {
		return text
	}

	// Handle paragraph-separated text (from buildKeyConceptsSection)
	paragraphs := strings.Split(text, "\n\n")
	var keptParagraphs []string
	for _, para := range paragraphs {
		sentences := splitSentencesDoc(para)
		var kept []string
		for _, s := range sentences {
			if !used[normalizeForDedup(s)] {
				kept = append(kept, s)
			}
		}
		if len(kept) > 0 {
			keptParagraphs = append(keptParagraphs, strings.Join(kept, " "))
		}
	}
	return strings.Join(keptParagraphs, "\n\n")
}

// normalizeForDedup lowercases and trims a sentence for dedup comparison.
func normalizeForDedup(s string) string {
	return strings.ToLower(strings.TrimRight(strings.TrimSpace(s), ".!? "))
}

// generateEssay builds an essay structure:
// Thesis, Arguments, Counterpoints, Conclusion.
func (dg *DocumentGenerator) generateEssay(topic string) *GeneratedDocument {
	doc := &GeneratedDocument{
		Title: capitalizeFirstDoc(topic),
	}

	mainPara := dg.findParagraph(topic)

	// Thesis — the defining statement about the topic.
	thesis := ""
	if mainPara != "" {
		thesis = firstSentencesOf(mainPara, 2)
	}
	if thesis == "" {
		thesis = dg.descriptionFromGraph(topic)
	}
	if thesis != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Thesis",
			Content: thesis,
		})
	}

	// Arguments — supporting facts and related concepts.
	arguments := dg.buildArgumentsSection(topic, mainPara)
	if arguments != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Arguments",
			Content: arguments,
		})
	}

	// Counterpoints — contradictions or opposing views from the graph.
	counterpoints := dg.buildCounterpointsSection(topic)
	if counterpoints != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Counterpoints",
			Content: counterpoints,
		})
	}

	// Conclusion.
	conclusion := dg.buildConclusion(topic, doc.Sections)
	if conclusion != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Conclusion",
			Content: conclusion,
		})
	}

	doc.WordCount = countDocWords(doc)
	return doc
}

// generateGuide builds a how-to guide structure:
// Overview, Prerequisites, Steps, Tips, Next Steps.
func (dg *DocumentGenerator) generateGuide(topic string) *GeneratedDocument {
	doc := &GeneratedDocument{
		Title: "Guide: " + capitalizeFirstDoc(topic),
	}

	mainPara := dg.findParagraph(topic)

	// Overview — what this is.
	overview := mainPara
	if overview == "" {
		overview = dg.descriptionFromGraph(topic)
	}
	if overview != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Overview",
			Content: overview,
		})
	}

	// Prerequisites — what you need to know (related concepts).
	prereqs := dg.buildPrerequisitesSection(topic)
	if prereqs != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Prerequisites",
			Content: prereqs,
		})
	}

	// Steps — structured facts as procedural knowledge.
	steps := dg.buildStepsSection(topic)
	if steps != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Steps",
			Content: steps,
		})
	}

	// Tips — additional facts and applications.
	tips := dg.buildTipsSection(topic)
	if tips != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Tips",
			Content: tips,
		})
	}

	// Next Steps — related topics to explore.
	nextSteps := dg.buildNextStepsSection(topic)
	if nextSteps != "" {
		doc.Sections = append(doc.Sections, DocumentSection{
			Heading: "Next Steps",
			Content: nextSteps,
		})
	}

	doc.WordCount = countDocWords(doc)
	return doc
}

// -----------------------------------------------------------------------
// Section builders — each extracts real content from knowledge + graph.
// -----------------------------------------------------------------------

func (dg *DocumentGenerator) buildBackgroundSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	var parts []string

	// Look for origin/history facts: founded_in, founded_by, created_by.
	edges := dg.graph.EdgesFrom(topic)
	for _, e := range edges {
		switch e.Relation {
		case RelFoundedIn, RelFoundedBy, RelCreatedBy, RelDerivedFrom, RelInfluencedBy:
			node := dg.graph.GetNode(e.To)
			if node != nil {
				subj := capitalizeFirstDoc(topic)
				fact := edgeToNaturalLanguage(subj, e.Relation, node.Label)
				if fact != "" {
					parts = append(parts, fact)
				}
			}
		}
	}

	// Also check incoming edges for "who created this".
	inEdges := dg.graph.EdgesTo(topic)
	for _, e := range inEdges {
		if e.Relation == RelCreatedBy || e.Relation == RelFoundedBy || e.Relation == RelInfluencedBy {
			fromNode := dg.graph.GetNode(e.From)
			if fromNode != nil {
				fact := edgeToNaturalLanguage(capitalizeFirstDoc(fromNode.Label), e.Relation, topic)
				if fact != "" {
					parts = append(parts, fact)
				}
			}
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " ")
}

func (dg *DocumentGenerator) buildKeyConceptsSection(topic string) string {
	related := dg.relatedTopics(topic, 5)
	if len(related) == 0 {
		return ""
	}

	var parts []string
	for _, rel := range related {
		para := dg.findParagraph(rel)
		if para != "" {
			// Use just the first few sentences to keep it concise.
			parts = append(parts, firstSentencesOf(para, 2))
		} else {
			// Fall back to graph description.
			desc := dg.descriptionFromGraph(rel)
			if desc != "" {
				parts = append(parts, firstSentencesOf(desc, 2))
			}
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

func (dg *DocumentGenerator) buildApplicationsSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	var apps []string
	edges := dg.graph.EdgesFrom(topic)
	for _, e := range edges {
		if e.Relation == RelUsedFor || e.Relation == RelOffers {
			node := dg.graph.GetNode(e.To)
			if node != nil && !isFragmentObject(node.Label) {
				apps = append(apps, node.Label)
			}
		}
	}

	if len(apps) == 0 {
		return ""
	}

	subj := capitalizeFirstDoc(topic)
	var lines []string
	for _, app := range apps {
		lines = append(lines, fmt.Sprintf("- %s is used for %s.", subj, app))
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildCurrentStateSection(topic string) string {
	// Gather the newest related information: combine graph facts with
	// any related topic paragraphs that mention current/modern/recent.
	related := dg.relatedTopics(topic, 3)
	var parts []string

	for _, rel := range related {
		para := dg.findParagraph(rel)
		if para == "" {
			continue
		}
		lower := strings.ToLower(para)
		if strings.Contains(lower, "current") || strings.Contains(lower, "modern") ||
			strings.Contains(lower, "recent") || strings.Contains(lower, "today") ||
			strings.Contains(lower, "ongoing") || strings.Contains(lower, "emerging") {
			parts = append(parts, firstSentencesOf(para, 2))
		}
	}

	// Also pull any "current" info from the main paragraph.
	mainPara := dg.findParagraph(topic)
	if mainPara != "" {
		sentences := splitSentencesDoc(mainPara)
		for _, s := range sentences {
			lower := strings.ToLower(s)
			if strings.Contains(lower, "current") || strings.Contains(lower, "modern") ||
				strings.Contains(lower, "recent") || strings.Contains(lower, "today") ||
				strings.Contains(lower, "ongoing") {
				parts = append(parts, s)
				break
			}
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " ")
}

func (dg *DocumentGenerator) buildFindingsSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	facts := dg.graph.LookupFacts(topic, 10)
	if len(facts) == 0 {
		return ""
	}

	var lines []string
	for _, f := range facts {
		lines = append(lines, "- "+f)
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildRecommendationsSection(topic string) string {
	// Derive recommendations from applications and related topics.
	apps := dg.buildApplicationsSection(topic)
	if apps != "" {
		return "Based on the applications identified above:\n\n" + apps
	}

	related := dg.relatedTopics(topic, 3)
	if len(related) == 0 {
		return ""
	}

	var lines []string
	for _, rel := range related {
		lines = append(lines, fmt.Sprintf("- Further study of %s is recommended.", rel))
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildArgumentsSection(topic string, mainPara string) string {
	var parts []string

	// Use the body of the main paragraph (after the first sentence) as argumentation.
	if mainPara != "" {
		sentences := splitSentencesDoc(mainPara)
		if len(sentences) > 1 {
			rest := strings.Join(sentences[1:], " ")
			parts = append(parts, rest)
		}
	}

	// Add supporting facts from the graph.
	if dg.graph != nil {
		facts := dg.graph.LookupFacts(topic, 5)
		for _, f := range facts {
			parts = append(parts, f)
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

func (dg *DocumentGenerator) buildCounterpointsSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	var parts []string

	// Look for contradicts or opposite_of relations.
	edges := dg.graph.EdgesFrom(topic)
	for _, e := range edges {
		if e.Relation == RelContradicts || e.Relation == RelOppositeOf {
			node := dg.graph.GetNode(e.To)
			if node != nil {
				para := dg.findParagraph(node.Label)
				if para != "" {
					parts = append(parts, firstSentencesOf(para, 2))
				} else {
					desc := dg.descriptionFromGraph(node.Label)
					if desc != "" {
						parts = append(parts, desc)
					}
				}
			}
		}
	}

	// Also check incoming contradicts edges.
	inEdges := dg.graph.EdgesTo(topic)
	for _, e := range inEdges {
		if e.Relation == RelContradicts || e.Relation == RelOppositeOf {
			fromNode := dg.graph.GetNode(e.From)
			if fromNode != nil && len(parts) < 3 {
				desc := dg.descriptionFromGraph(fromNode.Label)
				if desc != "" {
					parts = append(parts, desc)
				}
			}
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

func (dg *DocumentGenerator) buildPrerequisitesSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	var prereqs []string

	// Prerequisites: things this topic is_a, part_of, or derived_from.
	edges := dg.graph.EdgesFrom(topic)
	for _, e := range edges {
		switch e.Relation {
		case RelIsA, RelPartOf, RelDerivedFrom:
			node := dg.graph.GetNode(e.To)
			if node != nil && !isFragmentObject(node.Label) {
				prereqs = append(prereqs, node.Label)
			}
		}
	}

	if len(prereqs) == 0 {
		return ""
	}

	var lines []string
	for _, p := range prereqs {
		desc := dg.descriptionFromGraph(p)
		if desc != "" {
			lines = append(lines, fmt.Sprintf("- %s: %s", capitalizeFirstDoc(p), firstSentencesOf(desc, 1)))
		} else {
			lines = append(lines, fmt.Sprintf("- %s", capitalizeFirstDoc(p)))
		}
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildStepsSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	// Build procedural steps from "has" and "used_for" relations.
	var steps []string
	edges := dg.graph.EdgesFrom(topic)
	for _, e := range edges {
		if e.Relation == RelHas || e.Relation == RelUsedFor {
			node := dg.graph.GetNode(e.To)
			if node != nil && !isFragmentObject(node.Label) {
				if e.Relation == RelHas {
					steps = append(steps, fmt.Sprintf("Understand %s.", node.Label))
				} else {
					steps = append(steps, fmt.Sprintf("Apply to %s.", node.Label))
				}
			}
		}
	}

	if len(steps) == 0 {
		return ""
	}

	var lines []string
	for i, s := range steps {
		lines = append(lines, fmt.Sprintf("%d. %s", i+1, s))
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildTipsSection(topic string) string {
	if dg.graph == nil {
		return ""
	}

	facts := dg.graph.LookupFacts(topic, 5)
	if len(facts) == 0 {
		return ""
	}

	var lines []string
	for _, f := range facts {
		lines = append(lines, "- "+f)
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildNextStepsSection(topic string) string {
	related := dg.relatedTopics(topic, 5)
	if len(related) == 0 {
		return ""
	}

	var lines []string
	for _, rel := range related {
		lines = append(lines, fmt.Sprintf("- Explore %s.", capitalizeFirstDoc(rel)))
	}
	return strings.Join(lines, "\n")
}

func (dg *DocumentGenerator) buildConclusion(topic string, sections []DocumentSection) string {
	if len(sections) == 0 {
		return ""
	}

	// Synthesize a conclusion by pulling the first sentence from each section.
	var points []string
	for _, sec := range sections {
		first := firstSentencesOf(sec.Content, 1)
		if first != "" {
			points = append(points, first)
		}
	}

	if len(points) == 0 {
		return ""
	}

	return fmt.Sprintf("In summary, %s encompasses several important dimensions. %s",
		strings.ToLower(capitalizeFirstDoc(topic)),
		strings.Join(points, " "))
}

// -----------------------------------------------------------------------
// Knowledge access helpers
// -----------------------------------------------------------------------

// findParagraph searches the knowledge text files for a paragraph about
// the given topic and returns it verbatim. Uses the same paragraph cache
// as the action router's findKnowledgeParagraph.
func (dg *DocumentGenerator) findParagraph(topic string) string {
	return findKnowledgeParagraph(dg.knowledgeDir, topic)
}

// descriptionFromGraph retrieves the described_as text from the cognitive graph.
func (dg *DocumentGenerator) descriptionFromGraph(topic string) string {
	if dg.graph == nil {
		return ""
	}
	return dg.graph.LookupDescription(topic)
}

// relatedTopics returns labels of topics connected to the given topic via
// any relation type in the graph, up to maxCount.
func (dg *DocumentGenerator) relatedTopics(topic string, maxCount int) []string {
	if dg.graph == nil {
		return nil
	}

	seen := make(map[string]bool)
	var related []string
	topicLower := strings.ToLower(strings.TrimSpace(topic))

	// Outgoing edges.
	edges := dg.graph.EdgesFrom(topic)
	for _, e := range edges {
		if e.Relation == RelDescribedAs {
			continue
		}
		node := dg.graph.GetNode(e.To)
		if node == nil || isFragmentObject(node.Label) {
			continue
		}
		label := strings.ToLower(node.Label)
		if label == topicLower || seen[label] {
			continue
		}
		seen[label] = true
		related = append(related, node.Label)
		if len(related) >= maxCount {
			return related
		}
	}

	// Incoming edges.
	inEdges := dg.graph.EdgesTo(topic)
	for _, e := range inEdges {
		if e.Relation == RelDescribedAs {
			continue
		}
		node := dg.graph.GetNode(e.From)
		if node == nil || isFragmentObject(node.Label) {
			continue
		}
		label := strings.ToLower(node.Label)
		if label == topicLower || seen[label] {
			continue
		}
		seen[label] = true
		related = append(related, node.Label)
		if len(related) >= maxCount {
			return related
		}
	}

	return related
}

// loadParagraphs loads and caches all paragraphs from the knowledge directory.
func (dg *DocumentGenerator) loadParagraphs() []string {
	dg.paragraphsOnce.Do(func() {
		if dg.knowledgeDir == "" {
			return
		}
		files, err := filepath.Glob(filepath.Join(dg.knowledgeDir, "*.txt"))
		if err != nil || len(files) == 0 {
			return
		}
		for _, f := range files {
			data, err := os.ReadFile(f)
			if err != nil {
				continue
			}
			for _, p := range splitParagraphs(string(data)) {
				if p = strings.TrimSpace(p); p != "" {
					dg.paragraphs = append(dg.paragraphs, p)
				}
			}
		}
	})
	return dg.paragraphs
}

// -----------------------------------------------------------------------
// Formatting — render documents as Markdown or plain text.
// -----------------------------------------------------------------------

// FormatAsMarkdown renders a GeneratedDocument with # headings and paragraphs.
func FormatAsMarkdown(doc *GeneratedDocument) string {
	if doc == nil {
		return ""
	}

	var b strings.Builder
	b.WriteString("# ")
	b.WriteString(doc.Title)
	b.WriteString("\n\n")

	for _, sec := range doc.Sections {
		b.WriteString("## ")
		b.WriteString(sec.Heading)
		b.WriteString("\n\n")
		b.WriteString(sec.Content)
		b.WriteString("\n\n")
	}

	return strings.TrimSpace(b.String())
}

// FormatAsPlainText renders a GeneratedDocument with === underlines and indentation.
func FormatAsPlainText(doc *GeneratedDocument) string {
	if doc == nil {
		return ""
	}

	var b strings.Builder
	b.WriteString(doc.Title)
	b.WriteString("\n")
	b.WriteString(strings.Repeat("=", len(doc.Title)))
	b.WriteString("\n\n")

	for _, sec := range doc.Sections {
		b.WriteString(sec.Heading)
		b.WriteString("\n")
		b.WriteString(strings.Repeat("-", len(sec.Heading)))
		b.WriteString("\n\n")

		// Indent content lines.
		lines := strings.Split(sec.Content, "\n")
		for _, line := range lines {
			if strings.TrimSpace(line) == "" {
				b.WriteString("\n")
			} else {
				b.WriteString("  ")
				b.WriteString(line)
				b.WriteString("\n")
			}
		}
		b.WriteString("\n")
	}

	return strings.TrimSpace(b.String())
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// capitalizeFirstDoc capitalizes the first letter of a string.
func capitalizeFirstDoc(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// splitSentencesDoc splits text into sentences on ". " boundaries.
func splitSentencesDoc(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	remaining := text
	for {
		idx := strings.Index(remaining, ". ")
		if idx < 0 {
			// Last sentence (may or may not end with period).
			s := strings.TrimSpace(remaining)
			if s != "" {
				sentences = append(sentences, s)
			}
			break
		}
		s := strings.TrimSpace(remaining[:idx+1])
		if s != "" {
			sentences = append(sentences, s)
		}
		remaining = remaining[idx+2:]
	}
	return sentences
}

// firstSentencesOf returns the first n sentences of text.
func firstSentencesOf(text string, n int) string {
	sentences := splitSentencesDoc(text)
	if len(sentences) == 0 {
		return ""
	}
	if n > len(sentences) {
		n = len(sentences)
	}
	return strings.Join(sentences[:n], " ")
}

// countDocWords counts all words in a document.
func countDocWords(doc *GeneratedDocument) int {
	total := 0
	for _, sec := range doc.Sections {
		total += len(strings.Fields(sec.Content))
	}
	return total
}
