package cognitive

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// -----------------------------------------------------------------------
// Self-Teach — runtime knowledge mining.
//
// When Nous encounters a topic with no graph coverage it can search its
// own knowledge corpus (plain-text files), extract simple relational
// facts, and inject them into the CognitiveGraph for the current session.
//
// This is intentionally lightweight: no embedding model, no LLM call.
// It reads .txt files, finds paragraphs that mention the topic, and
// pulls subject-relation-object triples out of common sentence patterns.
// -----------------------------------------------------------------------

// SelfTeach allows Nous to learn about unknown topics by mining its own
// knowledge files at query time, without requiring a restart.
type SelfTeach struct {
	knowledgeDir string
	graph        *CognitiveGraph
	learned      map[string]bool // topics already self-taught
	mu           sync.RWMutex
}

// NewSelfTeach creates a SelfTeach instance. knowledgeDir is the path to
// the directory of .txt knowledge files. graph is the live cognitive
// graph that new facts will be added to.
func NewSelfTeach(knowledgeDir string, graph *CognitiveGraph) *SelfTeach {
	return &SelfTeach{
		knowledgeDir: knowledgeDir,
		graph:        graph,
		learned:      make(map[string]bool),
	}
}

// HasLearned returns true if we already attempted to learn about topic,
// preventing redundant filesystem scans.
func (st *SelfTeach) HasLearned(topic string) bool {
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.learned[strings.ToLower(strings.TrimSpace(topic))]
}

// SearchKnowledge scans every .txt file in the knowledge directory and
// returns up to 3 paragraphs that mention the topic (case-insensitive).
func (st *SelfTeach) SearchKnowledge(topic string) []string {
	lower := strings.ToLower(strings.TrimSpace(topic))
	if lower == "" {
		return nil
	}

	files, err := filepath.Glob(filepath.Join(st.knowledgeDir, "*.txt"))
	if err != nil || len(files) == 0 {
		return nil
	}

	const maxParagraphs = 3
	var matches []string

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		paragraphs := splitParagraphs(string(data))
		for _, p := range paragraphs {
			if strings.Contains(strings.ToLower(p), lower) {
				matches = append(matches, strings.TrimSpace(p))
				if len(matches) >= maxParagraphs {
					return matches
				}
			}
		}
	}
	return matches
}

// LearnAbout searches the knowledge corpus for the topic, extracts
// relational facts from matching paragraphs, and adds them to the graph.
// Returns the number of new facts added. The topic is marked as learned
// regardless of whether any facts were found, so we don't rescan.
func (st *SelfTeach) LearnAbout(topic string) (int, error) {
	key := strings.ToLower(strings.TrimSpace(topic))
	if key == "" {
		return 0, nil
	}

	// Mark learned (even before extraction) to avoid re-entry.
	st.mu.Lock()
	if st.learned[key] {
		st.mu.Unlock()
		return 0, nil
	}
	st.learned[key] = true
	st.mu.Unlock()

	paragraphs := st.SearchKnowledge(topic)
	if len(paragraphs) == 0 {
		return 0, nil
	}

	count := 0
	for _, para := range paragraphs {
		facts := extractSimpleFacts(para, topic)
		for _, fact := range facts {
			st.graph.AddEdge(fact.subject, fact.object, fact.relation, "self-teach")
			count++
		}
	}
	return count, nil
}

// -----------------------------------------------------------------------
// Lightweight fact extraction from prose.
//
// This is a minimal pattern-based extractor that pulls subject-relation-
// object triples from common English constructions. It is NOT a full NLP
// pipeline; it covers the most frequent patterns found in the knowledge
// corpus (definitions, classifications, authorship, composition).
//
// A more powerful FactExtractor pipeline can replace this later.
// -----------------------------------------------------------------------

type simpleFact struct {
	subject  string
	relation RelType
	object   string
}

// Patterns matched against individual sentences in a paragraph.
var factPatternsSelfTeach = []struct {
	re  *regexp.Regexp
	rel RelType
	// subj and obj are sub-match indexes (1-based).
	subj, obj int
}{
	// "X is a Y" / "X is the Y"
	{
		re:   regexp.MustCompile(`(?i)^([A-Z][\w\s-]{1,40}?)\s+is\s+(?:a|an|the)\s+(.{3,80}?)(?:[\.,;]|$)`),
		rel:  RelIsA,
		subj: 1, obj: 2,
	},
	// "X was created by Y" / "X was developed by Y"
	{
		re:   regexp.MustCompile(`(?i)([A-Z][\w\s-]{1,40}?)\s+was\s+(?:created|developed|invented|discovered)\s+by\s+(.{3,60}?)(?:[\.,;]|$)`),
		rel:  RelCreatedBy,
		subj: 1, obj: 2,
	},
	// "X was founded by Y"
	{
		re:   regexp.MustCompile(`(?i)([A-Z][\w\s-]{1,40}?)\s+was\s+founded\s+by\s+(.{3,60}?)(?:[\.,;]|$)`),
		rel:  RelFoundedBy,
		subj: 1, obj: 2,
	},
	// "X is used for Y" / "X is used in Y"
	{
		re:   regexp.MustCompile(`(?i)([A-Z][\w\s-]{1,40}?)\s+is\s+used\s+(?:for|in)\s+(.{3,80}?)(?:[\.,;]|$)`),
		rel:  RelUsedFor,
		subj: 1, obj: 2,
	},
	// "X is part of Y"
	{
		re:   regexp.MustCompile(`(?i)([A-Z][\w\s-]{1,40}?)\s+is\s+part\s+of\s+(.{3,60}?)(?:[\.,;]|$)`),
		rel:  RelPartOf,
		subj: 1, obj: 2,
	},
	// "X is located in Y"
	{
		re:   regexp.MustCompile(`(?i)([A-Z][\w\s-]{1,40}?)\s+is\s+located\s+in\s+(.{3,60}?)(?:[\.,;]|$)`),
		rel:  RelLocatedIn,
		subj: 1, obj: 2,
	},
}

// extractSimpleFacts runs pattern-based extraction over a paragraph,
// returning facts that involve the given topic.
func extractSimpleFacts(paragraph, topic string) []simpleFact {
	lower := strings.ToLower(topic)
	sentences := splitSentences(paragraph)

	var facts []simpleFact
	for _, sent := range sentences {
		// Only consider sentences that actually mention the topic.
		if !strings.Contains(strings.ToLower(sent), lower) {
			continue
		}
		for _, pat := range factPatternsSelfTeach {
			m := pat.re.FindStringSubmatch(sent)
			if m == nil {
				continue
			}
			subj := strings.TrimSpace(m[pat.subj])
			obj := strings.TrimSpace(m[pat.obj])
			if subj == "" || obj == "" {
				continue
			}
			facts = append(facts, simpleFact{
				subject:  subj,
				relation: pat.rel,
				object:   obj,
			})
		}
	}
	return facts
}

// splitParagraphs splits text on blank lines (one or more consecutive
// newlines with optional whitespace between).
func splitParagraphs(text string) []string {
	raw := regexp.MustCompile(`\n\s*\n`).Split(text, -1)
	var out []string
	for _, p := range raw {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
