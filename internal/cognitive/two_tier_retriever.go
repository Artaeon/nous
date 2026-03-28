package cognitive

import (
	"math"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
)

// RetrieverTier identifies which retrieval method produced a result.
type RetrieverTier int

const (
	TierLexical  RetrieverTier = iota // BM25-like keyword matching
	TierSemantic                      // vector similarity
	TierGraph                         // knowledge graph traversal
)

// RetrievalResult is one retrieved evidence unit.
type RetrievalResult struct {
	Text      string
	Source    string
	Score     float64
	Tier      RetrieverTier
	Freshness time.Time // when the source was last verified
}

// TwoTierRetriever combines lexical, semantic, and graph retrieval into
// a unified ranking pipeline. Results are merged via weighted reciprocal
// rank fusion (RRF), which is robust to score-distribution mismatches
// across heterogeneous retrieval backends.
type TwoTierRetriever struct {
	KnowledgeVec *KnowledgeVec
	Graph        *CognitiveGraph
	Documents    []indexedDocument // lexical index
	mu           sync.RWMutex

	// Config
	LexicalWeight  float64 // default 0.35
	SemanticWeight float64 // default 0.45
	GraphWeight    float64 // default 0.20
	TopK           int     // default 10

	// Corpus-level stats for BM25 IDF
	avgDocLen float64
	docCount  int
	docFreq   map[string]int // term -> number of documents containing it
}

// indexedDocument stores pre-processed text for lexical retrieval.
type indexedDocument struct {
	Text      string
	Source    string
	Terms     map[string]int // term -> frequency
	TermCount int
	Timestamp time.Time
}

// NewTwoTierRetriever creates a retriever with sensible defaults.
func NewTwoTierRetriever(kv *KnowledgeVec, graph *CognitiveGraph) *TwoTierRetriever {
	return &TwoTierRetriever{
		KnowledgeVec:   kv,
		Graph:          graph,
		LexicalWeight:  0.35,
		SemanticWeight: 0.45,
		GraphWeight:    0.20,
		TopK:           10,
		docFreq:        make(map[string]int),
	}
}

// IndexDocument adds a document to the lexical index.
func (r *TwoTierRetriever) IndexDocument(text, source string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}

	terms := tokenizeForRetrieval(text)
	freq := make(map[string]int, len(terms))
	for _, t := range terms {
		freq[t]++
	}

	doc := indexedDocument{
		Text:      text,
		Source:    source,
		Terms:     freq,
		TermCount: len(terms),
		Timestamp: time.Now(),
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.Documents = append(r.Documents, doc)
	r.docCount++

	// Update corpus-level document frequency counts.
	for term := range freq {
		r.docFreq[term]++
	}

	// Recalculate average document length.
	totalLen := 0.0
	for _, d := range r.Documents {
		totalLen += float64(d.TermCount)
	}
	r.avgDocLen = totalLen / float64(r.docCount)
}

// Retrieve performs combined lexical + semantic + graph retrieval and
// merges the ranked lists using weighted reciprocal rank fusion.
func (r *TwoTierRetriever) Retrieve(query string, topK int) []RetrievalResult {
	if topK <= 0 {
		topK = r.TopK
	}
	if topK <= 0 {
		topK = 10
	}

	// Run all three tiers.
	lexical := r.retrieveLexical(query, topK*2)
	semantic := r.retrieveSemantic(query, topK*2)
	graph := r.retrieveGraph(query, topK*2)

	// Merge via weighted RRF.
	merged := r.reciprocalRankFusion(lexical, semantic, graph, topK)
	return merged
}

// ---------- Lexical retrieval (BM25) ----------

// BM25 parameters
const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

func (r *TwoTierRetriever) retrieveLexical(query string, topK int) []RetrievalResult {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.Documents) == 0 {
		return nil
	}

	queryTerms := tokenizeForRetrieval(query)
	if len(queryTerms) == 0 {
		return nil
	}

	type scored struct {
		idx   int
		score float64
	}
	scores := make([]scored, 0, len(r.Documents))

	for i, doc := range r.Documents {
		score := r.bm25Score(queryTerms, doc)
		if score > 0 {
			scores = append(scores, scored{idx: i, score: score})
		}
	}

	sort.Slice(scores, func(a, b int) bool {
		return scores[a].score > scores[b].score
	})

	if topK > len(scores) {
		topK = len(scores)
	}

	results := make([]RetrievalResult, topK)
	for i := 0; i < topK; i++ {
		doc := r.Documents[scores[i].idx]
		results[i] = RetrievalResult{
			Text:      doc.Text,
			Source:    doc.Source,
			Score:     scores[i].score,
			Tier:      TierLexical,
			Freshness: doc.Timestamp,
		}
	}
	return results
}

// bm25Score computes BM25 score for a document given query terms.
// Must be called under r.mu.RLock().
func (r *TwoTierRetriever) bm25Score(queryTerms []string, doc indexedDocument) float64 {
	score := 0.0
	for _, qt := range queryTerms {
		tf := float64(doc.Terms[qt])
		if tf == 0 {
			continue
		}

		// IDF: log((N - df + 0.5) / (df + 0.5) + 1)
		df := float64(r.docFreq[qt])
		n := float64(r.docCount)
		idf := math.Log((n-df+0.5)/(df+0.5) + 1.0)

		// TF normalization with document length
		dl := float64(doc.TermCount)
		avgdl := r.avgDocLen
		if avgdl == 0 {
			avgdl = 1
		}
		tfNorm := (tf * (bm25K1 + 1)) / (tf + bm25K1*(1-bm25B+bm25B*(dl/avgdl)))

		score += idf * tfNorm
	}
	return score
}

// ---------- Semantic retrieval ----------

func (r *TwoTierRetriever) retrieveSemantic(query string, topK int) []RetrievalResult {
	if r.KnowledgeVec == nil {
		return nil
	}

	results, err := r.KnowledgeVec.Search(query, topK)
	if err != nil || len(results) == 0 {
		return nil
	}

	out := make([]RetrievalResult, len(results))
	for i, kr := range results {
		out[i] = RetrievalResult{
			Text:      kr.Text,
			Source:    kr.Source,
			Score:     kr.Score,
			Tier:      TierSemantic,
			Freshness: time.Now(), // semantic store doesn't track timestamps
		}
	}
	return out
}

// ---------- Graph retrieval ----------

func (r *TwoTierRetriever) retrieveGraph(query string, topK int) []RetrievalResult {
	if r.Graph == nil {
		return nil
	}

	// Extract key entities from the query.
	entities := extractEntities(query)
	if len(entities) == 0 {
		return nil
	}

	type graphFact struct {
		text      string
		source    string
		relevance float64
	}

	seen := make(map[string]bool)
	var facts []graphFact

	for _, entity := range entities {
		// Hop 1: direct edges from entity.
		// Try exact label first; if no edges, use FindNodes for fuzzy match.
		edges := r.Graph.EdgesFrom(entity)
		if len(edges) == 0 {
			// Try individual words if entity is multi-word.
			for _, w := range strings.Fields(entity) {
				edges = append(edges, r.Graph.EdgesFrom(w)...)
			}
		}
		for _, edge := range edges {
			node := r.Graph.GetNode(edge.To)
			if node == nil {
				continue
			}
			fact := edgeToNaturalLanguage(entity, edge.Relation, node.Label)
			if fact == "" || seen[fact] {
				continue
			}
			seen[fact] = true
			facts = append(facts, graphFact{
				text:      fact,
				source:    "graph:" + entity,
				relevance: graphRelationRelevance(edge.Relation) * edge.Weight,
			})

			// Hop 2: one more level out.
			hop2Edges := r.Graph.EdgesFrom(node.Label)
			for _, e2 := range hop2Edges {
				n2 := r.Graph.GetNode(e2.To)
				if n2 == nil {
					continue
				}
				f2 := edgeToNaturalLanguage(node.Label, e2.Relation, n2.Label)
				if f2 == "" || seen[f2] {
					continue
				}
				seen[f2] = true
				// Second-hop facts score lower.
				facts = append(facts, graphFact{
					text:      f2,
					source:    "graph:" + node.Label,
					relevance: graphRelationRelevance(e2.Relation) * e2.Weight * 0.5,
				})
			}
		}

		// Also check incoming edges (things that point TO the entity).
		inEdges := r.Graph.EdgesTo(entity)
		if len(inEdges) == 0 {
			for _, w := range strings.Fields(entity) {
				inEdges = append(inEdges, r.Graph.EdgesTo(w)...)
			}
		}
		for _, edge := range inEdges {
			node := r.Graph.GetNode(edge.From)
			if node == nil {
				continue
			}
			fact := edgeToNaturalLanguage(node.Label, edge.Relation, entity)
			if fact == "" || seen[fact] {
				continue
			}
			seen[fact] = true
			facts = append(facts, graphFact{
				text:      fact,
				source:    "graph:" + entity,
				relevance: graphRelationRelevance(edge.Relation) * edge.Weight * 0.8,
			})
		}
	}

	// Sort by relevance and cap.
	sort.Slice(facts, func(i, j int) bool {
		return facts[i].relevance > facts[j].relevance
	})
	if topK > len(facts) {
		topK = len(facts)
	}

	results := make([]RetrievalResult, topK)
	for i := 0; i < topK; i++ {
		results[i] = RetrievalResult{
			Text:      facts[i].text,
			Source:    facts[i].source,
			Score:     facts[i].relevance,
			Tier:      TierGraph,
			Freshness: time.Now(),
		}
	}
	return results
}

// graphRelationRelevance scores how informative a relation type is.
func graphRelationRelevance(rel RelType) float64 {
	switch rel {
	case RelIsA:
		return 1.0
	case RelDescribedAs:
		return 0.95
	case RelKnownFor:
		return 0.9
	case RelUsedFor:
		return 0.85
	case RelPartOf:
		return 0.8
	case RelCreatedBy, RelFoundedBy:
		return 0.8
	case RelHas:
		return 0.75
	case RelCauses:
		return 0.85
	case RelSimilarTo:
		return 0.6
	case RelRelatedTo:
		return 0.5
	default:
		return 0.6
	}
}

// ---------- Reciprocal Rank Fusion ----------

// reciprocalRankFusion merges ranked lists from different tiers.
// RRF score for a document d = sum over all lists L: weight_L / (k + rank_L(d))
// where k is a constant (60 is standard) and rank starts at 1.
func (r *TwoTierRetriever) reciprocalRankFusion(
	lexical, semantic, graph []RetrievalResult,
	topK int,
) []RetrievalResult {
	const rrfK = 60.0

	type fusedEntry struct {
		result RetrievalResult
		score  float64
		tiers  int // bitmask of contributing tiers
	}

	// Index by (Text, Source) to deduplicate across tiers.
	byKey := make(map[string]*fusedEntry)

	addList := func(list []RetrievalResult, weight float64, tier RetrieverTier) {
		for rank, rr := range list {
			key := rr.Text + "\x00" + rr.Source
			rrfScore := weight / (rrfK + float64(rank+1))

			if existing, ok := byKey[key]; ok {
				existing.score += rrfScore
				existing.tiers |= 1 << int(tier)
				// Keep the highest original score.
				if rr.Score > existing.result.Score {
					existing.result.Score = rr.Score
				}
			} else {
				byKey[key] = &fusedEntry{
					result: rr,
					score:  rrfScore,
					tiers:  1 << int(tier),
				}
			}
		}
	}

	addList(lexical, r.LexicalWeight, TierLexical)
	addList(semantic, r.SemanticWeight, TierSemantic)
	addList(graph, r.GraphWeight, TierGraph)

	// Flatten and sort.
	entries := make([]*fusedEntry, 0, len(byKey))
	for _, e := range byKey {
		entries = append(entries, e)
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].score > entries[j].score
	})

	if topK > len(entries) {
		topK = len(entries)
	}

	results := make([]RetrievalResult, topK)
	for i := 0; i < topK; i++ {
		results[i] = entries[i].result
		results[i].Score = entries[i].score
	}
	return results
}

// ---------- Text utilities ----------

// retrieverStopWords is the stop-word set used by the retrieval pipeline.
var retrieverStopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true,
	"to": true, "of": true, "in": true, "for": true, "on": true,
	"with": true, "at": true, "by": true, "from": true, "as": true,
	"into": true, "through": true, "during": true, "before": true,
	"after": true, "above": true, "below": true, "between": true,
	"out": true, "off": true, "over": true, "under": true, "again": true,
	"further": true, "then": true, "once": true, "here": true, "there": true,
	"when": true, "where": true, "why": true, "how": true,
	"all": true, "each": true, "every": true, "both": true, "few": true,
	"more": true, "most": true, "other": true, "some": true, "such": true,
	"no": true, "not": true, "only": true, "own": true, "same": true,
	"so": true, "than": true, "too": true, "very": true, "just": true,
	"because": true, "but": true, "and": true, "or": true, "if": true,
	"while": true, "about": true, "up": true, "it": true, "its": true,
	"this": true, "that": true, "these": true, "those": true,
	"he": true, "she": true, "they": true, "we": true, "me": true,
	"him": true, "her": true, "us": true, "them": true, "my": true,
	"your": true, "his": true, "our": true, "their": true,
	"what": true, "which": true, "who": true, "whom": true,
}

// isRetrieverStop returns true for common English stop words (retrieval pipeline).
func isRetrieverStop(w string) bool {
	return retrieverStopWords[w]
}

// tokenizeForRetrieval splits text into lowercase terms, removing
// punctuation and stop words. This is the retrieval pipeline's
// tokenizer (distinct from the extractive QA tokenizer).
func tokenizeForRetrieval(text string) []string {
	lower := strings.ToLower(text)
	fields := strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if len(f) < 2 {
			continue
		}
		if isRetrieverStop(f) {
			continue
		}
		out = append(out, f)
	}
	return out
}

// extractEntities pulls candidate entity names from a query for graph
// lookup. It keeps multi-word spans that look like proper nouns and
// significant content words.
func extractEntities(query string) []string {
	// First remove question/filler words to isolate content.
	lower := strings.ToLower(strings.TrimSpace(query))
	for _, prefix := range []string{
		"what is ", "what are ", "who is ", "who are ",
		"how does ", "how do ", "how is ", "how are ",
		"why does ", "why do ", "why is ", "why are ",
		"where is ", "where are ",
		"when did ", "when does ", "when is ",
		"tell me about ", "explain ", "describe ",
		"compare ", "what about ",
	} {
		lower = strings.TrimPrefix(lower, prefix)
	}

	// Remove connectors.
	for _, conn := range []string{" vs ", " versus ", " and ", " or ", " with "} {
		lower = strings.ReplaceAll(lower, conn, "\x00")
	}

	parts := strings.Split(lower, "\x00")
	var entities []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		// Strip trailing question marks and punctuation.
		p = strings.TrimRight(p, "?.!,;:")
		p = strings.TrimSpace(p)
		if p == "" || isRetrieverStop(p) {
			continue
		}
		entities = append(entities, p)
	}

	// If we ended up with nothing, fall back to content words.
	if len(entities) == 0 {
		words := tokenizeForRetrieval(query)
		for _, w := range words {
			if len(w) > 2 && !isRetrieverStop(w) {
				entities = append(entities, w)
			}
		}
	}

	return entities
}
