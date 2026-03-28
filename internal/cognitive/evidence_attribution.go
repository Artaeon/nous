package cognitive

import (
	"math"
	"regexp"
	"strings"
)

// Attribution links a claim to its supporting evidence.
type Attribution struct {
	Claim      string
	Evidence   []EvidenceLink
	Supported  bool    // at least one evidence link above threshold
	Confidence float64 // strength of attribution (max relevance)
}

// EvidenceLink connects a claim to a specific source.
type EvidenceLink struct {
	SourceText string
	Source     string  // origin (knowledge base, graph, etc.)
	Relevance  float64 // how well it supports the claim (0-1)
	Tier       RetrieverTier
}

// Contradiction describes a conflict between two claims.
type Contradiction struct {
	ClaimA     string
	ClaimB     string
	Type       string  // "direct_negation", "incompatible_values", "temporal_conflict"
	Confidence float64
}

// EvidenceAttributor manages claim-evidence linking and contradiction
// detection.
type EvidenceAttributor struct {
	retriever *TwoTierRetriever
}

// NewEvidenceAttributor creates an attributor backed by a two-tier retriever.
func NewEvidenceAttributor(retriever *TwoTierRetriever) *EvidenceAttributor {
	return &EvidenceAttributor{retriever: retriever}
}

// AttributeClaims finds evidence for each claim in a ContentPlan.
func (ea *EvidenceAttributor) AttributeClaims(plan *ContentPlan) []Attribution {
	if plan == nil || len(plan.Claims) == 0 {
		return nil
	}

	attrs := make([]Attribution, 0, len(plan.Claims))
	for _, claim := range plan.Claims {
		results := ea.retriever.Retrieve(claim.Text, 5)
		attr := ea.buildAttribution(claim.Text, results)
		attrs = append(attrs, attr)
	}
	return attrs
}

// buildAttribution creates an Attribution for a single claim.
func (ea *EvidenceAttributor) buildAttribution(claim string, results []RetrievalResult) Attribution {
	const relevanceThreshold = 0.3

	attr := Attribution{Claim: claim}
	claimTokens := tokenizeForRetrieval(claim)

	for _, r := range results {
		rel := textSimilarity(claimTokens, tokenizeForRetrieval(r.Text))
		if rel >= relevanceThreshold {
			attr.Evidence = append(attr.Evidence, EvidenceLink{
				SourceText: r.Text,
				Source:     r.Source,
				Relevance:  rel,
				Tier:       r.Tier,
			})
			if rel > attr.Confidence {
				attr.Confidence = rel
			}
		}
	}

	attr.Supported = len(attr.Evidence) > 0
	return attr
}

// CheckContradictions finds contradictions between a list of claims.
func (ea *EvidenceAttributor) CheckContradictions(claims []string) []Contradiction {
	if len(claims) < 2 {
		return nil
	}

	var contradictions []Contradiction

	for i := 0; i < len(claims); i++ {
		for j := i + 1; j < len(claims); j++ {
			if c := detectContradiction(claims[i], claims[j]); c != nil {
				contradictions = append(contradictions, *c)
			}
		}
	}
	return contradictions
}

// ValidateResponse checks that every claim in the response has attribution
// and looks for internal contradictions.
func (ea *EvidenceAttributor) ValidateResponse(response string, sources []RetrievalResult) *ValidationResult {
	sentences := splitSentences(response)
	if len(sentences) == 0 {
		return &ValidationResult{SupportRate: 1.0}
	}

	vr := &ValidationResult{}

	// Attribute each sentence.
	for _, sent := range sentences {
		sentTokens := tokenizeForRetrieval(sent)
		if len(sentTokens) == 0 {
			continue
		}

		attr := Attribution{Claim: sent}
		for _, src := range sources {
			rel := textSimilarity(sentTokens, tokenizeForRetrieval(src.Text))
			if rel >= 0.3 {
				attr.Evidence = append(attr.Evidence, EvidenceLink{
					SourceText: src.Text,
					Source:     src.Source,
					Relevance:  rel,
					Tier:       src.Tier,
				})
				if rel > attr.Confidence {
					attr.Confidence = rel
				}
			}
		}
		attr.Supported = len(attr.Evidence) > 0
		vr.Attributions = append(vr.Attributions, attr)

		if !attr.Supported {
			vr.UnsupportedClaims = append(vr.UnsupportedClaims, sent)
		}
	}

	// Contradiction detection across all sentences.
	vr.Contradictions = ea.CheckContradictions(sentences)

	// Compute rates.
	totalClaims := len(vr.Attributions)
	if totalClaims > 0 {
		supported := 0
		for _, a := range vr.Attributions {
			if a.Supported {
				supported++
			}
		}
		vr.SupportRate = float64(supported) / float64(totalClaims)
	}

	totalPairs := len(sentences) * (len(sentences) - 1) / 2
	if totalPairs > 0 {
		vr.ContradictionRate = float64(len(vr.Contradictions)) / float64(totalPairs)
	}

	return vr
}

// ValidationResult summarises the quality of a response.
type ValidationResult struct {
	Attributions      []Attribution
	Contradictions    []Contradiction
	UnsupportedClaims []string
	SupportRate       float64 // fraction of claims with evidence
	ContradictionRate float64 // fraction of claim pairs that contradict
}

// ---------- Contradiction detection ----------

// negationWords used to detect direct negation.
var negationWords = map[string]bool{
	"not": true, "no": true, "never": true, "none": true,
	"neither": true, "nor": true, "nothing": true,
	"nowhere": true, "cannot": true, "isn't": true,
	"aren't": true, "wasn't": true, "weren't": true,
	"don't": true, "doesn't": true, "didn't": true,
	"won't": true, "wouldn't": true, "couldn't": true,
	"shouldn't": true, "hasn't": true, "haven't": true,
}

// numericValueRe matches patterns like "X is 5" or "X = 10".
var numericValueRe = regexp.MustCompile(`(?i)(\w[\w\s]*?)\s+(?:is|are|was|were|=|equals?)\s+(\d+(?:\.\d+)?)`)

// temporalOrderRe matches "X before Y" or "X after Y".
var temporalOrderRe = regexp.MustCompile(`(?i)([\w][\w\s]*?)\s+(?:\w+\s+)?(before|after|precedes?|follows?)\s+([\w][\w\s]*\w)[.\s]*$`)

func detectContradiction(a, b string) *Contradiction {
	// 1. Direct negation: one sentence is the negated form of the other.
	if c := checkDirectNegation(a, b); c != nil {
		return c
	}

	// 2. Incompatible numeric values: "X is 5" vs "X is 10".
	if c := checkIncompatibleValues(a, b); c != nil {
		return c
	}

	// 3. Temporal conflicts: "X before Y" vs "Y before X".
	if c := checkTemporalConflict(a, b); c != nil {
		return c
	}

	return nil
}

func checkDirectNegation(a, b string) *Contradiction {
	aWords := strings.Fields(strings.ToLower(a))
	bWords := strings.Fields(strings.ToLower(b))

	aNeg := countNegations(aWords)
	bNeg := countNegations(bWords)

	// If one has negation and the other doesn't, check if the content
	// words overlap substantially.
	if (aNeg%2 == 0) == (bNeg%2 == 0) {
		return nil // same polarity
	}

	// Compare content words (non-stop, non-negation).
	aContent := evidenceContentWords(aWords)
	bContent := evidenceContentWords(bWords)

	overlap := jaccardSimilarity(aContent, bContent)
	if overlap >= 0.5 {
		return &Contradiction{
			ClaimA:     a,
			ClaimB:     b,
			Type:       "direct_negation",
			Confidence: overlap,
		}
	}
	return nil
}

func countNegations(words []string) int {
	count := 0
	for _, w := range words {
		if negationWords[w] {
			count++
		}
	}
	return count
}

func evidenceContentWords(words []string) []string {
	out := make([]string, 0, len(words))
	for _, w := range words {
		if !isStopWord(w) && !negationWords[w] && len(w) > 1 {
			out = append(out, w)
		}
	}
	return out
}

func checkIncompatibleValues(a, b string) *Contradiction {
	aMatch := numericValueRe.FindStringSubmatch(strings.ToLower(a))
	bMatch := numericValueRe.FindStringSubmatch(strings.ToLower(b))

	if len(aMatch) < 3 || len(bMatch) < 3 {
		return nil
	}

	// Same subject, different values.
	aSubj := strings.TrimSpace(aMatch[1])
	bSubj := strings.TrimSpace(bMatch[1])
	aVal := aMatch[2]
	bVal := bMatch[2]

	// Check if subjects overlap.
	subjSim := jaccardSimilarity(
		strings.Fields(aSubj),
		strings.Fields(bSubj),
	)
	if subjSim >= 0.5 && aVal != bVal {
		return &Contradiction{
			ClaimA:     a,
			ClaimB:     b,
			Type:       "incompatible_values",
			Confidence: subjSim,
		}
	}
	return nil
}

func checkTemporalConflict(a, b string) *Contradiction {
	aMatch := temporalOrderRe.FindStringSubmatch(strings.ToLower(a))
	bMatch := temporalOrderRe.FindStringSubmatch(strings.ToLower(b))

	if len(aMatch) < 4 || len(bMatch) < 4 {
		return nil
	}

	aSubj := strings.TrimSpace(aMatch[1])
	aRel := normalizeTemporalRel(aMatch[2])
	aObj := strings.TrimSpace(aMatch[3])

	bSubj := strings.TrimSpace(bMatch[1])
	bRel := normalizeTemporalRel(bMatch[2])
	bObj := strings.TrimSpace(bMatch[3])

	// Conflict: "X before Y" and "Y before X" (or equivalently "X after Y" and "Y after X").
	if aRel != bRel {
		// Check if the entities are swapped.
		if stringSimilar(aSubj, bSubj) && stringSimilar(aObj, bObj) {
			return &Contradiction{
				ClaimA:     a,
				ClaimB:     b,
				Type:       "temporal_conflict",
				Confidence: 0.8,
			}
		}
		// Or reversed.
		if stringSimilar(aSubj, bObj) && stringSimilar(aObj, bSubj) {
			return &Contradiction{
				ClaimA:     a,
				ClaimB:     b,
				Type:       "temporal_conflict",
				Confidence: 0.9,
			}
		}
	} else {
		// Same relation but entities are swapped: "X before Y" vs "Y before X".
		if stringSimilar(aSubj, bObj) && stringSimilar(aObj, bSubj) {
			return &Contradiction{
				ClaimA:     a,
				ClaimB:     b,
				Type:       "temporal_conflict",
				Confidence: 0.9,
			}
		}
	}
	return nil
}

func normalizeTemporalRel(rel string) string {
	rel = strings.ToLower(rel)
	switch {
	case strings.HasPrefix(rel, "before"), strings.HasPrefix(rel, "precede"):
		return "before"
	case strings.HasPrefix(rel, "after"), strings.HasPrefix(rel, "follow"):
		return "after"
	}
	return rel
}

func stringSimilar(a, b string) bool {
	return strings.TrimSpace(a) == strings.TrimSpace(b)
}

// ---------- Text similarity ----------

// textSimilarity computes a term-overlap similarity between two tokenized
// texts, combining Jaccard and a weighted overlap coefficient.
func textSimilarity(tokensA, tokensB []string) float64 {
	if len(tokensA) == 0 || len(tokensB) == 0 {
		return 0
	}

	setA := make(map[string]bool, len(tokensA))
	for _, t := range tokensA {
		setA[t] = true
	}
	setB := make(map[string]bool, len(tokensB))
	for _, t := range tokensB {
		setB[t] = true
	}

	intersection := 0
	for t := range setA {
		if setB[t] {
			intersection++
		}
	}

	if intersection == 0 {
		return 0
	}

	// Jaccard similarity.
	union := len(setA) + len(setB) - intersection
	jaccard := float64(intersection) / float64(union)

	// Overlap coefficient (intersection / min set size).
	minSize := len(setA)
	if len(setB) < minSize {
		minSize = len(setB)
	}
	overlap := float64(intersection) / float64(minSize)

	// Blend: weighted average favoring overlap for shorter texts.
	return 0.4*jaccard + 0.6*overlap
}

// jaccardSimilarity computes the Jaccard index of two word lists.
func jaccardSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	setA := make(map[string]bool, len(a))
	for _, w := range a {
		setA[w] = true
	}
	inter := 0
	setB := make(map[string]bool, len(b))
	for _, w := range b {
		setB[w] = true
		if setA[w] {
			inter++
		}
	}
	union := len(setA) + len(setB) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// ---------- Helpers ----------

// absFloat returns the absolute value of a float64.
func absFloat(x float64) float64 {
	return math.Abs(x)
}
