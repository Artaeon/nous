package cognitive

import (
	"regexp"
	"strings"
)

// RewrittenQuery is one derived sub-query from the original.
type RewrittenQuery struct {
	Original  string
	Rewritten string
	Purpose   string  // "entity_a", "entity_b", "criteria", "definition", "comparison"
	Weight    float64 // importance of this sub-query (0-1)
}

// QueryDecomposition breaks a complex query into retrievable sub-queries.
type QueryDecomposition struct {
	Original   string
	SubQueries []RewrittenQuery
	IsComplex  bool   // needed decomposition
	QueryType  string // "simple", "comparison", "multi_entity", "temporal", "causal"
}

// QueryRewriter handles query analysis and rewriting.
type QueryRewriter struct {
	compareRe  *regexp.Regexp
	causalRe   *regexp.Regexp
	temporalRe *regexp.Regexp
	multiRe    *regexp.Regexp
}

// NewQueryRewriter creates a QueryRewriter with pre-compiled patterns.
func NewQueryRewriter() *QueryRewriter {
	return &QueryRewriter{
		compareRe: regexp.MustCompile(
			`(?i)(?:compare|contrast|difference between|` +
				`similarities between|(.+?)\s+(?:vs\.?|versus|or|compared to|against)\s+(.+))`,
		),
		causalRe: regexp.MustCompile(
			`(?i)(?:why (?:did|does|do|is|are|was|were|has|have)|` +
				`what (?:causes?|caused|leads? to|led to)|` +
				`how (?:did|does|do) .+ (?:cause|lead to|result in|affect))`,
		),
		temporalRe: regexp.MustCompile(
			`(?i)(?:how has .+ changed|history of|evolution of|` +
				`over time|timeline|when did|since when|` +
				`before .+ and after|progression of)`,
		),
		multiRe: regexp.MustCompile(
			`(?i)(?:(.+?)\s+and\s+(?:how|what|why)\s+|` +
				`what is (.+?) and (?:how|what|why)|` +
				`(.+?)\s+(?:also|as well as|together with)\s+)`,
		),
	}
}

// Decompose analyzes a query and breaks it into sub-queries for retrieval.
func (qr *QueryRewriter) Decompose(query string) *QueryDecomposition {
	query = strings.TrimSpace(query)
	if query == "" {
		return &QueryDecomposition{Original: query, QueryType: "simple"}
	}

	lower := strings.ToLower(query)

	// Check comparison patterns.
	if qr.isComparison(lower) {
		return qr.decomposeComparison(query, lower)
	}

	// Check causal patterns.
	if qr.causalRe.MatchString(lower) {
		return qr.decomposeCausal(query, lower)
	}

	// Check temporal patterns.
	if qr.temporalRe.MatchString(lower) {
		return qr.decomposeTemporal(query, lower)
	}

	// Check multi-entity patterns.
	if qr.multiRe.MatchString(lower) {
		return qr.decomposeMultiEntity(query, lower)
	}

	// Simple query — just rewrite for retrieval.
	return &QueryDecomposition{
		Original:  query,
		QueryType: "simple",
		IsComplex: false,
		SubQueries: []RewrittenQuery{
			{
				Original:  query,
				Rewritten: qr.RewriteForRetrieval(query),
				Purpose:   "definition",
				Weight:    1.0,
			},
		},
	}
}

// isComparison checks whether the query is a comparison.
func (qr *QueryRewriter) isComparison(lower string) bool {
	// Check the regex first.
	if qr.compareRe.MatchString(lower) {
		return true
	}
	// Also catch "X vs Y" patterns that the regex might embed.
	for _, sep := range []string{" vs ", " vs. ", " versus "} {
		if strings.Contains(lower, sep) {
			return true
		}
	}
	return false
}

// decomposeComparison splits "X vs Y" or "compare X and Y" into sub-queries.
func (qr *QueryRewriter) decomposeComparison(query, lower string) *QueryDecomposition {
	d := &QueryDecomposition{
		Original:  query,
		QueryType: "comparison",
		IsComplex: true,
	}

	entityA, entityB := extractComparisonEntities(lower)
	if entityA == "" && entityB == "" {
		// Fallback: treat entire query as a single retrieval unit.
		d.SubQueries = []RewrittenQuery{
			{Original: query, Rewritten: qr.RewriteForRetrieval(query), Purpose: "comparison", Weight: 1.0},
		}
		return d
	}

	d.SubQueries = []RewrittenQuery{
		{
			Original:  query,
			Rewritten: entityA,
			Purpose:   "entity_a",
			Weight:    0.4,
		},
		{
			Original:  query,
			Rewritten: entityB,
			Purpose:   "entity_b",
			Weight:    0.4,
		},
		{
			Original:  query,
			Rewritten: entityA + " " + entityB + " comparison differences similarities",
			Purpose:   "criteria",
			Weight:    0.2,
		},
	}
	return d
}

// extractComparisonEntities pulls the two sides of a comparison.
func extractComparisonEntities(lower string) (string, string) {
	// Try explicit separators.
	for _, sep := range []string{" vs. ", " vs ", " versus ", " compared to ", " against "} {
		idx := strings.Index(lower, sep)
		if idx >= 0 {
			a := strings.TrimSpace(lower[:idx])
			b := strings.TrimSpace(lower[idx+len(sep):])
			a = stripComparisonPrefix(a)
			b = strings.TrimRight(b, "?.!,;:")
			b = strings.TrimSpace(b)
			if a != "" && b != "" {
				return a, b
			}
		}
	}

	// Try "compare X and Y" or "difference between X and Y".
	for _, pat := range []string{
		"compare ", "contrast ",
		"difference between ", "differences between ",
		"similarities between ", "similarity between ",
	} {
		idx := strings.Index(lower, pat)
		if idx < 0 {
			continue
		}
		rest := lower[idx+len(pat):]
		rest = strings.TrimRight(rest, "?.!,;:")
		// Split on " and " or " with ".
		for _, conj := range []string{" and ", " with "} {
			if ci := strings.Index(rest, conj); ci >= 0 {
				a := strings.TrimSpace(rest[:ci])
				b := strings.TrimSpace(rest[ci+len(conj):])
				if a != "" && b != "" {
					return a, b
				}
			}
		}
	}

	return "", ""
}

// stripComparisonPrefix removes leading question/command words.
func stripComparisonPrefix(s string) string {
	for _, p := range []string{
		"compare ", "contrast ", "what is ",
		"how does ", "why is ", "explain ",
		"tell me about ", "describe ",
	} {
		if strings.HasPrefix(s, p) {
			return strings.TrimSpace(s[len(p):])
		}
	}
	return s
}

// decomposeCausal splits "why did X cause Y" into sub-queries.
func (qr *QueryRewriter) decomposeCausal(query, lower string) *QueryDecomposition {
	d := &QueryDecomposition{
		Original:  query,
		QueryType: "causal",
		IsComplex: true,
	}

	cause, effect := extractCausalEntities(lower)
	if cause == "" && effect == "" {
		// Single causal query.
		topic := stripCausalPrefix(lower)
		d.SubQueries = []RewrittenQuery{
			{Original: query, Rewritten: topic, Purpose: "entity_a", Weight: 0.5},
			{Original: query, Rewritten: topic + " cause reason mechanism", Purpose: "criteria", Weight: 0.5},
		}
		return d
	}

	d.SubQueries = []RewrittenQuery{
		{Original: query, Rewritten: cause, Purpose: "entity_a", Weight: 0.35},
		{Original: query, Rewritten: effect, Purpose: "entity_b", Weight: 0.35},
		{Original: query, Rewritten: cause + " " + effect + " cause mechanism relationship", Purpose: "criteria", Weight: 0.3},
	}
	return d
}

// extractCausalEntities pulls cause and effect from a causal question.
func extractCausalEntities(lower string) (string, string) {
	for _, sep := range []string{
		" cause ", " causes ", " caused ",
		" lead to ", " leads to ", " led to ",
		" result in ", " results in ", " resulted in ",
		" affect ", " affects ", " affected ",
	} {
		idx := strings.Index(lower, sep)
		if idx >= 0 {
			a := stripCausalPrefix(strings.TrimSpace(lower[:idx]))
			b := strings.TrimRight(strings.TrimSpace(lower[idx+len(sep):]), "?.!,;:")
			if a != "" && b != "" {
				return a, b
			}
		}
	}
	return "", ""
}

// stripCausalPrefix removes "why did", "what causes", etc.
func stripCausalPrefix(s string) string {
	for _, p := range []string{
		"why did ", "why does ", "why do ",
		"why is ", "why are ", "why was ", "why were ",
		"what causes ", "what cause ", "what caused ",
		"what leads to ", "what led to ",
		"how did ", "how does ", "how do ",
	} {
		if strings.HasPrefix(s, p) {
			return strings.TrimSpace(s[len(p):])
		}
	}
	return s
}

// decomposeTemporal splits temporal questions into period-based sub-queries.
func (qr *QueryRewriter) decomposeTemporal(query, lower string) *QueryDecomposition {
	d := &QueryDecomposition{
		Original:  query,
		QueryType: "temporal",
		IsComplex: true,
	}

	topic := stripTemporalPrefix(lower)
	topic = strings.TrimRight(topic, "?.!,;:")
	topic = strings.TrimSpace(topic)
	if topic == "" {
		topic = qr.RewriteForRetrieval(query)
	}

	d.SubQueries = []RewrittenQuery{
		{Original: query, Rewritten: topic, Purpose: "definition", Weight: 0.3},
		{Original: query, Rewritten: topic + " history origins early", Purpose: "entity_a", Weight: 0.25},
		{Original: query, Rewritten: topic + " recent current modern", Purpose: "entity_b", Weight: 0.25},
		{Original: query, Rewritten: topic + " change evolution development timeline", Purpose: "criteria", Weight: 0.2},
	}
	return d
}

// stripTemporalPrefix removes temporal question prefixes.
func stripTemporalPrefix(s string) string {
	for _, p := range []string{
		"how has ", "how have ",
		"history of ", "the history of ",
		"evolution of ", "the evolution of ",
		"timeline of ", "the timeline of ",
		"when did ", "since when ",
		"progression of ", "the progression of ",
	} {
		if strings.HasPrefix(s, p) {
			return strings.TrimSpace(s[len(p):])
		}
	}
	// Strip trailing "over time", "changed", etc.
	for _, suffix := range []string{
		" over time", " changed over time", " changed",
		" evolved", " developed",
	} {
		if strings.HasSuffix(s, suffix) {
			return strings.TrimSpace(s[:len(s)-len(suffix)])
		}
	}
	return s
}

// decomposeMultiEntity splits "what is X and how does it relate to Y".
func (qr *QueryRewriter) decomposeMultiEntity(query, lower string) *QueryDecomposition {
	d := &QueryDecomposition{
		Original:  query,
		QueryType: "multi_entity",
		IsComplex: true,
	}

	// Try to split on "and how/what/why".
	for _, conj := range []string{
		" and how ", " and what ", " and why ",
	} {
		idx := strings.Index(lower, conj)
		if idx < 0 {
			continue
		}
		partA := strings.TrimSpace(lower[:idx])
		partB := strings.TrimSpace(lower[idx+len(conj)-4:]) // keep "how..."
		partA = stripQuestionPrefix(partA)
		partA = strings.TrimRight(partA, "?.!,;:")

		d.SubQueries = []RewrittenQuery{
			{Original: query, Rewritten: partA, Purpose: "entity_a", Weight: 0.4},
			{Original: query, Rewritten: qr.RewriteForRetrieval(partB), Purpose: "entity_b", Weight: 0.35},
			{Original: query, Rewritten: partA + " " + qr.RewriteForRetrieval(partB) + " relationship", Purpose: "criteria", Weight: 0.25},
		}
		return d
	}

	// Fallback: treat as single query.
	d.SubQueries = []RewrittenQuery{
		{Original: query, Rewritten: qr.RewriteForRetrieval(query), Purpose: "definition", Weight: 1.0},
	}
	return d
}

// stripQuestionPrefix removes leading question words.
func stripQuestionPrefix(s string) string {
	for _, p := range []string{
		"what is ", "what are ", "what was ", "what were ",
		"who is ", "who are ", "who was ",
		"how does ", "how do ", "how is ", "how are ",
		"why does ", "why do ", "why is ",
		"where is ", "where are ",
		"tell me about ", "explain ", "describe ",
	} {
		if strings.HasPrefix(s, p) {
			return strings.TrimSpace(s[len(p):])
		}
	}
	return s
}

// RewriteForRetrieval produces an optimized retrieval query from a
// natural language question. It strips question words, filler words,
// and expands common abbreviations.
func (qr *QueryRewriter) RewriteForRetrieval(query string) string {
	lower := strings.ToLower(strings.TrimSpace(query))
	if lower == "" {
		return ""
	}

	// Remove trailing punctuation.
	lower = strings.TrimRight(lower, "?.!,;:")
	lower = strings.TrimSpace(lower)

	// Strip question prefixes.
	lower = stripQuestionPrefix(lower)

	// Remove filler words.
	for _, filler := range []string{
		"actually", "basically", "really", "just",
		"simply", "literally", "honestly",
		"please", "kindly", "could you", "can you",
		"i want to know", "i need to know",
		"i was wondering", "i am curious about",
	} {
		lower = strings.ReplaceAll(lower, filler, " ")
	}

	// Expand abbreviations.
	abbreviations := map[string]string{
		"ai":   "artificial intelligence",
		"ml":   "machine learning",
		"nlp":  "natural language processing",
		"dl":   "deep learning",
		"cv":   "computer vision",
		"db":   "database",
		"api":  "application programming interface",
		"os":   "operating system",
		"iot":  "internet of things",
		"ui":   "user interface",
		"ux":   "user experience",
		"cpu":  "central processing unit",
		"gpu":  "graphics processing unit",
		"ram":  "random access memory",
		"sql":  "structured query language",
		"html": "hypertext markup language",
		"css":  "cascading style sheets",
	}

	words := strings.Fields(lower)
	expanded := make([]string, 0, len(words))
	for _, w := range words {
		if exp, ok := abbreviations[w]; ok {
			expanded = append(expanded, exp)
		} else if len(w) > 1 && !isStopWord(w) {
			expanded = append(expanded, w)
		}
	}

	result := strings.Join(expanded, " ")
	result = strings.TrimSpace(result)
	if result == "" {
		// Final fallback: return the original (minus punctuation).
		return strings.TrimRight(strings.TrimSpace(query), "?.!,;:")
	}
	return result
}
