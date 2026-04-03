package cognitive

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// On-Demand Wikipedia Loader — infinite knowledge growth.
//
// When Nous doesn't know about a topic, instead of saying "I don't know",
// it fetches the Wikipedia article, extracts facts into the knowledge
// graph, stores the paragraph for future retrieval, and answers — all
// in the same turn.
//
// This is LAZY KNOWLEDGE LOADING: the knowledge base grows to cover
// whatever the user asks about. First question about "quantum tunneling"
// fetches Wikipedia, extracts 5-15 facts, and the topic is known forever.
//
// Key properties:
//   - First fetch: ~200ms (Wikipedia API call)
//   - Subsequent queries: 1ms (cached in graph + paragraph store)
//   - No bulk import needed — grows organically with usage
//   - Works offline for previously-fetched topics
//   - Each fetch adds 5-15 typed graph edges + description paragraph
// -----------------------------------------------------------------------

// WikipediaLoader fetches and ingests Wikipedia articles on-demand.
type WikipediaLoader struct {
	Graph          *CognitiveGraph
	fetched        map[string]bool // topics already fetched this session
	paragraphCache map[string]string // topic → lead paragraph
	mu             sync.Mutex
}

// WikiFetchResult describes what was loaded from a Wikipedia fetch.
type WikiFetchResult struct {
	Topic     string
	Paragraph string   // lead paragraph (Wikipedia-quality prose)
	FactCount int      // number of typed graph edges added
	Source    string   // "wikipedia:en"
	Duration  time.Duration
	Cached    bool     // true if served from cache
}

// NewWikipediaLoader creates a loader wired to the knowledge graph.
func NewWikipediaLoader(graph *CognitiveGraph) *WikipediaLoader {
	return &WikipediaLoader{
		Graph:          graph,
		fetched:        make(map[string]bool),
		paragraphCache: make(map[string]string),
	}
}

// FetchAndLearn fetches a Wikipedia article, extracts facts into the
// graph, and returns the lead paragraph for immediate response.
// Returns nil if the topic isn't found on Wikipedia.
func (wl *WikipediaLoader) FetchAndLearn(topic string) *WikiFetchResult {
	if topic == "" || wl.Graph == nil {
		return nil
	}

	topicLower := strings.ToLower(strings.TrimSpace(topic))

	wl.mu.Lock()
	// Check session cache.
	if wl.fetched[topicLower] {
		para := wl.paragraphCache[topicLower]
		wl.mu.Unlock()
		if para != "" {
			return &WikiFetchResult{
				Topic:     topic,
				Paragraph: para,
				Cached:    true,
			}
		}
		return nil
	}
	wl.fetched[topicLower] = true
	wl.mu.Unlock()

	start := time.Now()

	// Fetch from Wikipedia REST API.
	extract, title, err := wl.fetchWikipedia(topic)
	if err != nil || extract == "" {
		return nil
	}

	// Extract typed facts from the article text.
	facts := ArticleToFacts(title, extract)

	// Add facts to the knowledge graph.
	factCount := 0
	for _, fact := range facts {
		rel := wikiParseRel(fact.Relation)
		if rel == "" {
			rel = RelRelatedTo
		}
		wl.Graph.AddEdge(fact.Subject, fact.Object, rel, "wikipedia:en")
		factCount++
	}

	// Extract and store the lead paragraph (first 2-3 sentences).
	leadPara := extractLeadParagraph(extract, title)

	// Store the lead paragraph as a described_as edge too.
	if leadPara != "" {
		wl.Graph.AddEdge(title, leadPara, RelDescribedAs, "wikipedia:en")
	}

	// Cache the paragraph for this session.
	wl.mu.Lock()
	wl.paragraphCache[topicLower] = leadPara
	wl.mu.Unlock()

	return &WikiFetchResult{
		Topic:     title,
		Paragraph: leadPara,
		FactCount: factCount,
		Source:    "wikipedia:en",
		Duration:  time.Since(start),
	}
}

// HasFetched returns true if this topic was already fetched this session.
func (wl *WikipediaLoader) HasFetched(topic string) bool {
	wl.mu.Lock()
	defer wl.mu.Unlock()
	return wl.fetched[strings.ToLower(strings.TrimSpace(topic))]
}

// GetCachedParagraph returns the cached lead paragraph for a topic, if any.
func (wl *WikipediaLoader) GetCachedParagraph(topic string) string {
	wl.mu.Lock()
	defer wl.mu.Unlock()
	return wl.paragraphCache[strings.ToLower(strings.TrimSpace(topic))]
}

// -----------------------------------------------------------------------
// Wikipedia API
// -----------------------------------------------------------------------

// wikiSummaryResponse matches the Wikipedia REST API /page/summary response.
type wikiSummaryResponse struct {
	Title       string `json:"title"`
	DisplayName string `json:"displaytitle"`
	Extract     string `json:"extract"`
	Description string `json:"description"`
}

func (wl *WikipediaLoader) fetchWikipedia(topic string) (extract, title string, err error) {
	normalized := strings.ReplaceAll(strings.TrimSpace(topic), " ", "_")
	apiURL := "https://en.wikipedia.org/api/rest_v1/page/summary/" + url.PathEscape(normalized)

	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("User-Agent", "Nous/1.0 (local AI assistant; https://github.com/artaeon/nous)")

	resp, err := client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("wikipedia fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", "", fmt.Errorf("wikipedia: HTTP %d for %q", resp.StatusCode, topic)
	}

	var result wikiSummaryResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", fmt.Errorf("wikipedia: invalid JSON: %w", err)
	}

	if result.Extract == "" {
		return "", "", fmt.Errorf("wikipedia: no extract for %q", topic)
	}

	return result.Extract, result.Title, nil
}

// -----------------------------------------------------------------------
// Text processing helpers
// -----------------------------------------------------------------------

// extractLeadParagraph returns the first 3 sentences of text.
func extractLeadParagraph(text, title string) string {
	sentences := strings.SplitAfter(text, ". ")
	maxSentences := 3
	if len(sentences) < maxSentences {
		maxSentences = len(sentences)
	}

	lead := strings.TrimSpace(strings.Join(sentences[:maxSentences], ""))
	if lead == "" {
		return text
	}
	return lead
}

// wikiParseRel converts a string relation name to a RelType constant.
func wikiParseRel(rel string) RelType {
	switch strings.ToLower(rel) {
	case "is_a":
		return RelIsA
	case "located_in":
		return RelLocatedIn
	case "part_of":
		return RelPartOf
	case "created_by":
		return RelCreatedBy
	case "founded_by":
		return RelFoundedBy
	case "founded_in":
		return RelFoundedIn
	case "has":
		return RelHas
	case "offers":
		return RelOffers
	case "used_for":
		return RelUsedFor
	case "related_to":
		return RelRelatedTo
	case "similar_to":
		return RelSimilarTo
	case "causes":
		return RelCauses
	case "contradicts":
		return RelContradicts
	case "follows":
		return RelFollows
	case "described_as":
		return RelDescribedAs
	case "known_for":
		return RelKnownFor
	case "influenced_by":
		return RelInfluencedBy
	case "derived_from":
		return RelDerivedFrom
	case "domain":
		return RelDomain
	case "prevents":
		return RelPrevents
	case "enables":
		return RelEnables
	case "requires":
		return RelRequires
	case "produces":
		return RelProduces
	default:
		return RelRelatedTo
	}
}
