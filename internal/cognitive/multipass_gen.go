package cognitive

import (
	"fmt"
	"strings"
	"time"
)

// GenerationPass represents one stage in multi-pass generation.
type GenerationPass int

const (
	PassPlan   GenerationPass = iota // build content plan
	PassDraft                        // generate initial draft
	PassVerify                       // check against plan + quality
	PassRefine                       // fix issues found in verification
)

func passName(p GenerationPass) string {
	switch p {
	case PassPlan:
		return "plan"
	case PassDraft:
		return "draft"
	case PassVerify:
		return "verify"
	case PassRefine:
		return "refine"
	default:
		return "unknown"
	}
}

// PassResult captures the output of each generation pass.
type PassResult struct {
	Pass     GenerationPass
	Text     string
	Plan     *ContentPlan
	Score    NLGScore
	Issues   []QualityIssue
	Duration time.Duration
}

// QualityIssue identifies a specific problem found during verification.
type QualityIssue struct {
	Type        string // "missing_claim", "contradiction", "filler", "repetition", "off_topic", "low_specificity"
	Description string
	Severity    string // "critical", "major", "minor"
	Location    int    // approximate sentence index
}

// MultiPassConfig controls the multi-pass pipeline.
type MultiPassConfig struct {
	MaxRefineIterations int     // max refinement passes (default 2)
	MinQualityScore     float64 // minimum total NLG score to accept (0.65)
	RequirePlan         bool    // always require content plan (true for explain/compare)
	EnableVerification  bool    // run verification pass (default true)
}

// DefaultMultiPassConfig returns a sensible default configuration.
func DefaultMultiPassConfig() *MultiPassConfig {
	return &MultiPassConfig{
		MaxRefineIterations: 2,
		MinQualityScore:     0.65,
		RequirePlan:         true,
		EnableVerification:  true,
	}
}

// MultiPassResult holds the output of the complete generation pipeline.
type MultiPassResult struct {
	FinalText   string
	Passes      []PassResult
	FinalScore  NLGScore
	TotalPasses int
	Accepted    bool // true if quality threshold met
}

// MultiPassGenerate runs the full plan -> draft -> verify -> refine pipeline.
func (te *ThinkingEngine) MultiPassGenerate(query string, task ThinkTask, params *TaskParams, config *MultiPassConfig) *MultiPassResult {
	if config == nil {
		config = DefaultMultiPassConfig()
	}
	if params == nil {
		params = &TaskParams{}
	}

	result := &MultiPassResult{}
	fd := NewFillerDetector()

	// ---- Pass 1: Plan ----
	planStart := time.Now()
	plan := te.BuildContentPlan(query, task, params)

	// For explain/compare tasks, enforce plan structure
	var enforced *EnforcedPlan
	if config.RequirePlan || taskRequiresPlan(task) {
		template := GetPlanTemplate(task)
		enforced = EnforceContentPlan(plan, template, params.Topic)
		// Update plan claims from enforced plan if it added any
		if len(enforced.Claims) > len(plan.Claims) {
			plan.Claims = make([]PlanClaim, len(enforced.Claims))
			for i, ec := range enforced.Claims {
				plan.Claims[i] = ec.PlanClaim
			}
		}
	}

	planResult := PassResult{
		Pass:     PassPlan,
		Plan:     plan,
		Duration: time.Since(planStart),
	}
	result.Passes = append(result.Passes, planResult)

	// ---- Pass 2: Draft ----
	draftStart := time.Now()
	draftText := te.generateDraft(query, task, plan, params)

	// Apply filler removal on draft
	if cleaned, changed := fd.EnforcePolicy(draftText, true); changed {
		draftText = cleaned
	}

	draftScore := te.ScoreCandidate(plan, draftText)
	draftResult := PassResult{
		Pass:     PassDraft,
		Text:     draftText,
		Plan:     plan,
		Score:    draftScore,
		Duration: time.Since(draftStart),
	}
	result.Passes = append(result.Passes, draftResult)

	currentText := draftText
	currentScore := draftScore

	// ---- Pass 3: Verify ----
	if config.EnableVerification {
		verifyStart := time.Now()
		issues := verifyDraft(currentText, plan, fd)
		verifyResult := PassResult{
			Pass:     PassVerify,
			Text:     currentText,
			Plan:     plan,
			Score:    currentScore,
			Issues:   issues,
			Duration: time.Since(verifyStart),
		}
		result.Passes = append(result.Passes, verifyResult)

		// ---- Pass 4: Refine ----
		if len(issues) > 0 {
			for iter := 0; iter < config.MaxRefineIterations; iter++ {
				refineStart := time.Now()
				refined := refineDraft(currentText, plan, issues, enforced, fd)

				// Apply filler removal again after refinement
				if cleaned, changed := fd.EnforcePolicy(refined, true); changed {
					refined = cleaned
				}

				refinedScore := te.ScoreCandidate(plan, refined)
				refineIssues := verifyDraft(refined, plan, fd)

				refineResult := PassResult{
					Pass:     PassRefine,
					Text:     refined,
					Plan:     plan,
					Score:    refinedScore,
					Issues:   refineIssues,
					Duration: time.Since(refineStart),
				}
				result.Passes = append(result.Passes, refineResult)

				// Only accept refinement if it improves score
				if refinedScore.Total >= currentScore.Total {
					currentText = refined
					currentScore = refinedScore
				}

				issues = refineIssues

				// Stop refining if no critical/major issues remain
				hasCritical := false
				for _, issue := range refineIssues {
					if issue.Severity == "critical" || issue.Severity == "major" {
						hasCritical = true
						break
					}
				}
				if !hasCritical {
					break
				}
			}
		}
	}

	result.FinalText = currentText
	result.FinalScore = currentScore
	result.TotalPasses = len(result.Passes)
	result.Accepted = currentScore.Total >= config.MinQualityScore

	return result
}

// taskRequiresPlan returns true for tasks that benefit from structured planning.
func taskRequiresPlan(task ThinkTask) bool {
	switch task {
	case TaskTeach, TaskCompare, TaskAnalyze, TaskDebate, TaskPlan:
		return true
	default:
		return false
	}
}

// generateDraft creates the initial draft using existing candidate generation.
func (te *ThinkingEngine) generateDraft(query string, task ThinkTask, plan *ContentPlan, params *TaskParams) string {
	// Build multiple candidates and pick the best
	var frame *Frame
	candidates := te.BuildPlanCandidates(query, task, plan, frame, "")

	if len(candidates) == 0 {
		// Fallback: generate directly from plan
		return realizePlanDirect(plan)
	}

	// Use the deterministic reranker for selection
	reranker := NewDeterministicReranker()
	rankResult := reranker.Rerank(candidates, plan, query)
	if rankResult.BestIndex >= 0 && rankResult.BestIndex < len(candidates) {
		return candidates[rankResult.BestIndex]
	}

	// Fallback to existing selection
	return te.SelectBestCandidate(plan, candidates)
}

// realizePlanDirect generates text directly from a content plan without candidates.
func realizePlanDirect(plan *ContentPlan) string {
	if plan == nil {
		return ""
	}

	var parts []string
	if plan.Thesis != "" {
		parts = append(parts, strings.TrimSpace(plan.Thesis))
	}

	for _, c := range plan.Claims {
		if c.Text != "" {
			parts = append(parts, strings.TrimSpace(c.Text))
		}
	}

	if plan.Counterpoint != "" {
		parts = append(parts, strings.TrimSpace(plan.Counterpoint))
	}

	return strings.Join(parts, " ")
}

// verifyDraft checks a draft against a plan and identifies quality issues.
func verifyDraft(text string, plan *ContentPlan, fd *FillerDetector) []QualityIssue {
	var issues []QualityIssue

	if strings.TrimSpace(text) == "" {
		issues = append(issues, QualityIssue{
			Type:        "missing_claim",
			Description: "Response is empty",
			Severity:    "critical",
			Location:    0,
		})
		return issues
	}

	sentences := splitSentences(text)
	lower := strings.ToLower(text)

	// Check 1: Missing claims from the plan
	if plan != nil {
		for _, claim := range plan.Claims {
			if claim.Text == "" {
				continue
			}
			claimLower := strings.ToLower(claim.Text)
			ks := strings.ToLower(keySpan(claim.Text))

			if !strings.Contains(lower, claimLower) && !strings.Contains(lower, ks) {
				// Also check content word overlap
				claimWords := rerankerContentWords(claim.Text)
				matchCount := 0
				for _, w := range claimWords {
					if strings.Contains(lower, w) {
						matchCount++
					}
				}
				threshold := 0.6
				if len(claimWords) > 0 && float64(matchCount)/float64(len(claimWords)) < threshold {
					issues = append(issues, QualityIssue{
						Type:        "missing_claim",
						Description: fmt.Sprintf("Plan claim not covered: %s", claim.Text),
						Severity:    "major",
						Location:    -1,
					})
				}
			}
		}
	}

	// Check 2: Contradictions between sentences
	for i := 0; i < len(sentences); i++ {
		si := strings.ToLower(strings.TrimSpace(sentences[i]))
		for j := i + 1; j < len(sentences); j++ {
			sj := strings.ToLower(strings.TrimSpace(sentences[j]))
			if detectRerankerContradiction(si, sj) {
				issues = append(issues, QualityIssue{
					Type:        "contradiction",
					Description: fmt.Sprintf("Sentences %d and %d may contradict each other", i+1, j+1),
					Severity:    "critical",
					Location:    i,
				})
			}
		}
	}

	// Check 3: Filler detection
	fillerInstances := fd.DetectFiller(text)
	for _, fi := range fillerInstances {
		severity := "minor"
		if fi.Severity == "must_remove" {
			severity = "major"
		}
		issues = append(issues, QualityIssue{
			Type:        "filler",
			Description: fmt.Sprintf("Filler detected (%s): %s", fi.Type, fi.Text),
			Severity:    severity,
			Location:    findSentenceIndex(sentences, fi.Start),
		})
	}

	// Check 4: Repetition between nearby sentences
	for i := 0; i < len(sentences); i++ {
		end := i + 4
		if end > len(sentences) {
			end = len(sentences)
		}
		for j := i + 1; j < end; j++ {
			si := strings.TrimSpace(sentences[i])
			sj := strings.TrimSpace(sentences[j])
			if si == "" || sj == "" {
				continue
			}
			if sentenceOverlap(si, sj) > 0.5 {
				issues = append(issues, QualityIssue{
					Type:        "repetition",
					Description: fmt.Sprintf("Sentences %d and %d are highly similar", i+1, j+1),
					Severity:    "major",
					Location:    j,
				})
			}
		}
	}

	// Check 5: Off-topic — response should mention the plan topic
	if plan != nil && plan.Topic != "" {
		topicLower := strings.ToLower(plan.Topic)
		if !strings.Contains(lower, topicLower) {
			issues = append(issues, QualityIssue{
				Type:        "off_topic",
				Description: fmt.Sprintf("Response does not mention the topic: %s", plan.Topic),
				Severity:    "major",
				Location:    0,
			})
		}
	}

	// Check 6: Low specificity — check for too many vague terms
	vagueTerms := []string{"things", "stuff", "very", "really", "basically", "a lot", "kind of", "sort of"}
	vagueCount := 0
	for _, vt := range vagueTerms {
		vagueCount += strings.Count(lower, vt)
	}
	wordCount := len(strings.Fields(text))
	if wordCount > 0 && float64(vagueCount)/float64(wordCount) > 0.05 {
		issues = append(issues, QualityIssue{
			Type:        "low_specificity",
			Description: fmt.Sprintf("High vague-word density: %d vague terms in %d words", vagueCount, wordCount),
			Severity:    "minor",
			Location:    -1,
		})
	}

	return issues
}

// refineDraft applies targeted fixes for each identified issue.
func refineDraft(text string, plan *ContentPlan, issues []QualityIssue, enforced *EnforcedPlan, fd *FillerDetector) string {
	if len(issues) == 0 {
		return text
	}

	sentences := splitSentences(text)
	if len(sentences) == 0 {
		return text
	}

	// Track which sentences to remove
	removeSet := make(map[int]bool)
	// Track claims to insert
	var insertClaims []string

	for _, issue := range issues {
		switch issue.Type {
		case "missing_claim":
			// Extract the missing claim text from the description
			claim := extractMissingClaim(issue.Description)
			if claim != "" {
				insertClaims = append(insertClaims, claim)
			}

		case "contradiction":
			// Remove the later contradictory sentence
			if issue.Location >= 0 && issue.Location+1 < len(sentences) {
				removeSet[issue.Location+1] = true
			}

		case "filler":
			// Filler is handled by the filler detector
			// The filler removal pass will clean this up

		case "repetition":
			// Remove the later repetitive sentence
			if issue.Location >= 0 && issue.Location < len(sentences) {
				removeSet[issue.Location] = true
			}

		case "off_topic":
			// Add a topic-anchoring sentence at the beginning
			if plan != nil && plan.Topic != "" {
				anchor := fmt.Sprintf("Regarding %s:", plan.Topic)
				if len(sentences) > 0 {
					// Prepend topic anchor
					sentences = append([]string{anchor + " "}, sentences...)
				}
			}

		case "low_specificity":
			// Replace vague words with more specific alternatives where possible
			// This is a best-effort replacement
		}
	}

	// Build refined text: remove flagged sentences
	var refined []string
	for i, sent := range sentences {
		if removeSet[i] {
			continue
		}
		s := strings.TrimSpace(sent)
		if s != "" {
			refined = append(refined, s)
		}
	}

	// Insert missing claims at appropriate positions
	for _, claim := range insertClaims {
		claim = strings.TrimSpace(claim)
		if claim == "" {
			continue
		}
		if !hasSentenceEnding(claim) {
			claim += "."
		}
		// Insert after the first sentence (after thesis)
		if len(refined) > 1 {
			// Insert at position 1 (after thesis)
			newRefined := make([]string, 0, len(refined)+1)
			newRefined = append(newRefined, refined[0])
			newRefined = append(newRefined, claim)
			newRefined = append(newRefined, refined[1:]...)
			refined = newRefined
		} else {
			refined = append(refined, claim)
		}
	}

	// Add uncertainty and recap from enforced plan if missing
	if enforced != nil {
		resultText := strings.Join(refined, " ")
		resultLower := strings.ToLower(resultText)

		if enforced.Uncertainty != "" && !strings.Contains(resultLower, "uncertain") && !strings.Contains(resultLower, "unclear") {
			refined = append(refined, enforced.Uncertainty)
		}
		if enforced.Recap != "" && !strings.Contains(resultLower, "in summary") && !strings.Contains(resultLower, "overall") {
			refined = append(refined, enforced.Recap)
		}
	}

	result := strings.Join(refined, " ")

	// Final filler pass
	if cleaned, _ := fd.EnforcePolicy(result, true); cleaned != "" {
		result = cleaned
	}

	return result
}

// extractMissingClaim extracts the claim text from a missing_claim issue description.
func extractMissingClaim(desc string) string {
	prefix := "Plan claim not covered: "
	if idx := strings.Index(desc, prefix); idx >= 0 {
		return strings.TrimSpace(desc[idx+len(prefix):])
	}
	return ""
}

// findSentenceIndex returns the sentence index that contains the given character offset.
func findSentenceIndex(sentences []string, charOffset int) int {
	pos := 0
	for i, sent := range sentences {
		pos += len(sent)
		if charOffset < pos {
			return i
		}
	}
	return len(sentences) - 1
}

// sentenceOverlap computes word-level Jaccard similarity between two sentences.
func sentenceOverlap(a, b string) float64 {
	aWords := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(a)) {
		clean := strings.Trim(w, ".,;:!?()")
		if len(clean) >= 3 {
			aWords[clean] = true
		}
	}

	bWords := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(b)) {
		clean := strings.Trim(w, ".,;:!?()")
		if len(clean) >= 3 {
			bWords[clean] = true
		}
	}

	if len(aWords) == 0 || len(bWords) == 0 {
		return 0
	}

	intersection := 0
	for w := range aWords {
		if bWords[w] {
			intersection++
		}
	}

	union := len(aWords)
	for w := range bWords {
		if !aWords[w] {
			union++
		}
	}

	if union == 0 {
		return 0
	}

	return float64(intersection) / float64(union)
}
