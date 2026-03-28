package training

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// FailureRecord represents a logged failure from production.
type FailureRecord struct {
	Input        string            `json:"input"`
	Response     string            `json:"response"`
	FailureType  string            `json:"failure_type"`  // "hallucination", "off_topic", "low_quality", "crash", "timeout", "filler_response", "wrong_intent"
	Severity     string            `json:"severity"`      // "critical", "major", "minor"
	UserFeedback string            `json:"user_feedback"` // if user corrected/complained
	DetectedBy   string            `json:"detected_by"`   // "quality_gate", "user_correction", "auto_check", "retry_detector"
	Timestamp    time.Time         `json:"timestamp"`
	Metadata     map[string]string `json:"metadata"`
}

// FailureStore manages failure records.
type FailureStore struct {
	dir     string
	records []FailureRecord
	mu      sync.RWMutex
}

// NewFailureStore creates a store backed by the given directory.
func NewFailureStore(dir string) *FailureStore {
	return &FailureStore{
		dir: dir,
	}
}

// RecordFailure appends a failure record and persists to disk.
func (fs *FailureStore) RecordFailure(record *FailureRecord) error {
	if record == nil {
		return fmt.Errorf("failure: nil record")
	}
	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now()
	}
	if record.Metadata == nil {
		record.Metadata = make(map[string]string)
	}

	fs.mu.Lock()
	fs.records = append(fs.records, *record)
	data, err := json.MarshalIndent(fs.records, "", "  ")
	fs.mu.Unlock()
	if err != nil {
		return fmt.Errorf("failure: marshal: %w", err)
	}

	if fs.dir == "" {
		return nil
	}
	return safefile.WriteAtomic(filepath.Join(fs.dir, "failure_records.json"), data, 0644)
}

// LoadRecords reads all records from disk into memory.
func (fs *FailureStore) LoadRecords() ([]FailureRecord, error) {
	if fs.dir == "" {
		fs.mu.RLock()
		out := make([]FailureRecord, len(fs.records))
		copy(out, fs.records)
		fs.mu.RUnlock()
		return out, nil
	}

	data, err := os.ReadFile(filepath.Join(fs.dir, "failure_records.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failure: read: %w", err)
	}

	var records []FailureRecord
	if err := json.Unmarshal(data, &records); err != nil {
		return nil, fmt.Errorf("failure: unmarshal: %w", err)
	}

	fs.mu.Lock()
	fs.records = records
	fs.mu.Unlock()

	out := make([]FailureRecord, len(records))
	copy(out, records)
	return out, nil
}

// FailureAnalysis summarizes patterns in failures.
type FailureAnalysis struct {
	TotalFailures    int
	ByType           map[string]int
	BySeverity       map[string]int
	TopPatterns      []FailurePattern
	RegressionRisks  []string
	RecommendedFixes []string
}

// FailurePattern represents a recurring pattern in failures.
type FailurePattern struct {
	Pattern     string   // keyword pattern
	FailureType string
	Count       int
	Examples    []string // up to 5 example inputs
}

// Analyze finds patterns in failures.
func (fs *FailureStore) Analyze() *FailureAnalysis {
	fs.mu.RLock()
	records := make([]FailureRecord, len(fs.records))
	copy(records, fs.records)
	fs.mu.RUnlock()

	analysis := &FailureAnalysis{
		TotalFailures: len(records),
		ByType:        make(map[string]int),
		BySeverity:    make(map[string]int),
	}

	if len(records) == 0 {
		return analysis
	}

	// Count by type and severity
	for _, r := range records {
		analysis.ByType[r.FailureType]++
		analysis.BySeverity[r.Severity]++
	}

	// Extract patterns by finding common words/phrases in failing inputs per type
	patterns := findFailurePatterns(records)
	analysis.TopPatterns = patterns

	// Identify regression risks: failure types that are increasing
	analysis.RegressionRisks = identifyRegressionRisks(records)

	// Recommend fixes based on failure type distribution
	analysis.RecommendedFixes = recommendFixes(analysis.ByType)

	return analysis
}

// findFailurePatterns extracts common word patterns from failures grouped by type.
func findFailurePatterns(records []FailureRecord) []FailurePattern {
	// Group records by failure type
	byType := make(map[string][]FailureRecord)
	for _, r := range records {
		byType[r.FailureType] = append(byType[r.FailureType], r)
	}

	var patterns []FailurePattern

	for fType, recs := range byType {
		// Count word frequencies across failing inputs for this type
		wordFreq := make(map[string]int)
		for _, r := range recs {
			words := strings.Fields(strings.ToLower(r.Input))
			seen := make(map[string]bool) // count each word once per record
			for _, w := range words {
				w = strings.Trim(w, ".,!?;:'\"()[]{}") // strip punctuation
				if len(w) > 3 && !isStopWord(w) && !seen[w] {
					wordFreq[w]++
					seen[w] = true
				}
			}
		}

		// Find words that appear in >30% of failures of this type (min 2)
		threshold := int(math.Max(2, float64(len(recs))*0.3))
		for word, count := range wordFreq {
			if count >= threshold {
				// Collect example inputs containing this word
				var examples []string
				for _, r := range recs {
					if strings.Contains(strings.ToLower(r.Input), word) {
						examples = append(examples, r.Input)
						if len(examples) >= 5 {
							break
						}
					}
				}

				patterns = append(patterns, FailurePattern{
					Pattern:     word,
					FailureType: fType,
					Count:       count,
					Examples:    examples,
				})
			}
		}
	}

	// Sort by count descending
	sort.Slice(patterns, func(i, j int) bool {
		return patterns[i].Count > patterns[j].Count
	})

	// Cap at 20 patterns
	if len(patterns) > 20 {
		patterns = patterns[:20]
	}

	return patterns
}

// isStopWord returns true for common English stop words.
func isStopWord(w string) bool {
	stops := map[string]bool{
		"that": true, "this": true, "with": true, "from": true,
		"have": true, "been": true, "were": true, "they": true,
		"their": true, "there": true, "about": true, "would": true,
		"could": true, "should": true, "which": true, "where": true,
		"when": true, "what": true, "will": true, "does": true,
		"your": true, "into": true, "also": true, "than": true,
		"then": true, "some": true, "more": true, "very": true,
		"just": true, "like": true, "each": true, "make": true,
		"only": true, "even": true, "most": true, "much": true,
	}
	return stops[w]
}

// identifyRegressionRisks finds failure types whose rate is increasing.
func identifyRegressionRisks(records []FailureRecord) []string {
	if len(records) < 4 {
		return nil
	}

	// Split records into halves by time (assuming roughly chronological order)
	mid := len(records) / 2
	firstHalf := records[:mid]
	secondHalf := records[mid:]

	firstCounts := make(map[string]int)
	secondCounts := make(map[string]int)

	for _, r := range firstHalf {
		firstCounts[r.FailureType]++
	}
	for _, r := range secondHalf {
		secondCounts[r.FailureType]++
	}

	var risks []string
	firstN := float64(len(firstHalf))
	secondN := float64(len(secondHalf))

	for fType, secondCount := range secondCounts {
		firstCount := firstCounts[fType]
		firstRate := float64(firstCount) / firstN
		secondRate := float64(secondCount) / secondN

		// Flag if rate increased by >50% and there are at least 2 occurrences
		if secondRate > firstRate*1.5 && secondCount >= 2 {
			risks = append(risks, fType)
		}
	}

	sort.Strings(risks)
	return risks
}

// recommendFixes suggests fixes based on the failure type distribution.
func recommendFixes(byType map[string]int) []string {
	var fixes []string

	type typeCount struct {
		fType string
		count int
	}
	var sorted []typeCount
	for t, c := range byType {
		sorted = append(sorted, typeCount{t, c})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].count > sorted[j].count
	})

	for _, tc := range sorted {
		switch tc.fType {
		case "hallucination":
			fixes = append(fixes, fmt.Sprintf("Add grounding verification: %d hallucination failures detected. Strengthen fact-checking pipeline and add source attribution.", tc.count))
		case "off_topic":
			fixes = append(fixes, fmt.Sprintf("Improve intent classification: %d off-topic failures. Retrain NLU with more diverse examples and add topic-relevance scoring.", tc.count))
		case "low_quality":
			fixes = append(fixes, fmt.Sprintf("Raise quality thresholds: %d low-quality responses. Increase minimum NLG score and add iterative refinement pass.", tc.count))
		case "crash":
			fixes = append(fixes, fmt.Sprintf("Fix stability issues: %d crashes detected. Add nil checks, bounds validation, and recovery handlers.", tc.count))
		case "timeout":
			fixes = append(fixes, fmt.Sprintf("Optimize latency: %d timeouts. Profile slow paths, add caching, and consider fast-path routing.", tc.count))
		case "filler_response":
			fixes = append(fixes, fmt.Sprintf("Eliminate filler responses: %d filler detections. Strengthen no-filler policy and add response substance checks.", tc.count))
		case "wrong_intent":
			fixes = append(fixes, fmt.Sprintf("Retrain intent classifier: %d wrong-intent failures. Collect more training data for confused intent pairs.", tc.count))
		default:
			fixes = append(fixes, fmt.Sprintf("Investigate %q failures: %d occurrences.", tc.fType, tc.count))
		}
	}

	return fixes
}

// RegressionCase is a targeted test case mined from failures.
type RegressionCase struct {
	Input          string   `json:"input"`
	FailureType    string   `json:"failure_type"`
	MustNotContain []string `json:"must_not_contain"` // patterns that triggered the failure
	MustContain    []string `json:"must_contain"`     // patterns that indicate correct behavior
	Priority       int      `json:"priority"`         // higher = more important
}

// MineRegressionSuite extracts targeted test cases from failures.
func (fs *FailureStore) MineRegressionSuite() []RegressionCase {
	fs.mu.RLock()
	records := make([]FailureRecord, len(fs.records))
	copy(records, fs.records)
	fs.mu.RUnlock()

	var cases []RegressionCase

	// Deduplicate by input to avoid redundant test cases
	seen := make(map[string]bool)

	for _, r := range records {
		inputKey := strings.ToLower(strings.TrimSpace(r.Input))
		if seen[inputKey] {
			continue
		}
		seen[inputKey] = true

		rc := RegressionCase{
			Input:       r.Input,
			FailureType: r.FailureType,
		}

		// Set priority based on severity
		switch r.Severity {
		case "critical":
			rc.Priority = 3
		case "major":
			rc.Priority = 2
		case "minor":
			rc.Priority = 1
		default:
			rc.Priority = 1
		}

		// Extract must-not-contain patterns from the failed response
		rc.MustNotContain = extractBadPatterns(r.Response, r.FailureType)

		// Extract must-contain hints from user feedback or metadata
		rc.MustContain = extractGoodHints(r)

		cases = append(cases, rc)
	}

	// Sort by priority descending
	sort.Slice(cases, func(i, j int) bool {
		return cases[i].Priority > cases[j].Priority
	})

	return cases
}

// extractBadPatterns identifies patterns in the response that indicate failure.
func extractBadPatterns(response, failureType string) []string {
	var patterns []string
	respLower := strings.ToLower(response)

	switch failureType {
	case "hallucination":
		// Look for fabricated-looking claims
		if strings.Contains(respLower, "according to") {
			patterns = append(patterns, "unsourced claims")
		}
		if strings.Contains(respLower, "studies show") || strings.Contains(respLower, "research shows") {
			patterns = append(patterns, "fabricated citations")
		}
	case "filler_response":
		fillers := []string{
			"i'm not sure", "i don't know", "that's a great question",
			"interesting question", "let me think", "i'll do my best",
		}
		for _, f := range fillers {
			if strings.Contains(respLower, f) {
				patterns = append(patterns, f)
			}
		}
	case "off_topic":
		// The response itself is the bad pattern since it missed the topic
		if len(response) > 50 {
			patterns = append(patterns, "off-topic content")
		}
	case "low_quality":
		if len(strings.Fields(response)) < 10 {
			patterns = append(patterns, "insufficient detail")
		}
	}

	return patterns
}

// extractGoodHints derives what a correct response should contain.
func extractGoodHints(r FailureRecord) []string {
	var hints []string

	// Extract key terms from the input as expected topics
	inputWords := strings.Fields(strings.ToLower(r.Input))
	for _, w := range inputWords {
		w = strings.Trim(w, ".,!?;:'\"()[]{}") // strip punctuation
		if len(w) > 4 && !isStopWord(w) {
			hints = append(hints, w)
		}
	}

	// Cap hints at 5
	if len(hints) > 5 {
		hints = hints[:5]
	}

	// If user provided feedback, include keywords from it
	if r.UserFeedback != "" {
		fbWords := strings.Fields(strings.ToLower(r.UserFeedback))
		for _, w := range fbWords {
			w = strings.Trim(w, ".,!?;:'\"()[]{}") // strip punctuation
			if len(w) > 4 && !isStopWord(w) {
				hints = append(hints, w)
				if len(hints) >= 8 {
					break
				}
			}
		}
	}

	return hints
}

// TrainingTarget is a failure converted into a targeted training example.
type TrainingTarget struct {
	Input       string   `json:"input"`
	FailureType string   `json:"failure_type"`
	BadResponse string   `json:"bad_response"`
	GoodHints   []string `json:"good_hints"`
}

// MineTrainingData converts failures into targeted training examples.
func (fs *FailureStore) MineTrainingData() []TrainingTarget {
	fs.mu.RLock()
	records := make([]FailureRecord, len(fs.records))
	copy(records, fs.records)
	fs.mu.RUnlock()

	var targets []TrainingTarget

	// Deduplicate by input
	seen := make(map[string]bool)

	for _, r := range records {
		inputKey := strings.ToLower(strings.TrimSpace(r.Input))
		if seen[inputKey] {
			continue
		}
		seen[inputKey] = true

		target := TrainingTarget{
			Input:       r.Input,
			FailureType: r.FailureType,
			BadResponse: r.Response,
		}

		// Derive good hints from the input and failure type
		target.GoodHints = deriveGoodHints(r)

		targets = append(targets, target)
	}

	return targets
}

// deriveGoodHints generates hints for what a correct response should contain.
func deriveGoodHints(r FailureRecord) []string {
	var hints []string

	// Type-specific hints
	switch r.FailureType {
	case "hallucination":
		hints = append(hints, "verify claims against known facts")
		hints = append(hints, "cite specific sources")
	case "off_topic":
		hints = append(hints, "address the user's actual question")
		// Extract key nouns from input
		for _, w := range extractContentWords(r.Input) {
			hints = append(hints, "discuss "+w)
		}
	case "low_quality":
		hints = append(hints, "provide detailed explanation")
		hints = append(hints, "include specific examples")
	case "filler_response":
		hints = append(hints, "give a substantive answer")
		hints = append(hints, "avoid hedging language")
	case "wrong_intent":
		hints = append(hints, "correctly identify user intent")
	case "crash":
		hints = append(hints, "handle edge cases gracefully")
	case "timeout":
		hints = append(hints, "provide timely response")
	}

	// Add content words from user feedback
	if r.UserFeedback != "" {
		for _, w := range extractContentWords(r.UserFeedback) {
			hints = append(hints, w)
		}
	}

	// Cap at 8 hints
	if len(hints) > 8 {
		hints = hints[:8]
	}

	return hints
}

// extractContentWords returns meaningful words from text (>4 chars, non-stop).
func extractContentWords(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	var content []string
	seen := make(map[string]bool)
	for _, w := range words {
		w = strings.Trim(w, ".,!?;:'\"()[]{}") // strip punctuation
		if len(w) > 4 && !isStopWord(w) && !seen[w] {
			content = append(content, w)
			seen[w] = true
		}
	}
	if len(content) > 5 {
		content = content[:5]
	}
	return content
}

// FailureDetector automatically detects failures in real-time responses.
type FailureDetector struct {
	fillerPatterns    []*regexp.Regexp
	hallPatterns      []*regexp.Regexp
	lowQualitySignals []string
}

// NewFailureDetector creates a detector with default failure patterns.
func NewFailureDetector() *FailureDetector {
	fd := &FailureDetector{}

	// Filler response patterns: hedging, non-answers, stalling
	fillerStrs := []string{
		`(?i)i'?m not sure`,
		`(?i)i don'?t know`,
		`(?i)that'?s a great question`,
		`(?i)that'?s an interesting question`,
		`(?i)let me think about`,
		`(?i)i'?ll do my best`,
		`(?i)i can'?t really say`,
		`(?i)it'?s hard to say`,
		`(?i)^(hmm|well|so|uh|um)[,.]`,
		`(?i)i'?m afraid i`,
		`(?i)unfortunately,? i (can'?t|cannot|don'?t|am not able)`,
		`(?i)as an ai`,
	}
	for _, p := range fillerStrs {
		if re, err := regexp.Compile(p); err == nil {
			fd.fillerPatterns = append(fd.fillerPatterns, re)
		}
	}

	// Hallucination patterns: fabricated references, false certainty
	hallStrs := []string{
		`(?i)according to a (\d{4} )?study`,
		`(?i)research (from|at|by) [A-Z]\w+ (University|Institute|Lab)`,
		`(?i)published in (the )?[A-Z]\w+( \w+)* (Journal|Review|Proceedings)`,
		`(?i)dr\.? [A-Z]\w+ (found|discovered|showed|demonstrated)`,
		`(?i)\d+(\.\d+)?% of (people|studies|experts|scientists)`,
		`(?i)it is (well-)?known that`,
		`(?i)scientists have (proven|confirmed|established)`,
	}
	for _, p := range hallStrs {
		if re, err := regexp.Compile(p); err == nil {
			fd.hallPatterns = append(fd.hallPatterns, re)
		}
	}

	// Low quality signals
	fd.lowQualitySignals = []string{
		"error", "sorry", "apologize", "mistake",
	}

	return fd
}

// DetectedFailure describes a single detected failure in a response.
type DetectedFailure struct {
	Type       string
	Severity   string
	Evidence   string
	Confidence float64
}

// Detect checks a response for failure patterns.
func (fd *FailureDetector) Detect(input, response string, latencyMs int64) []DetectedFailure {
	var failures []DetectedFailure

	respLower := strings.ToLower(response)
	respWords := strings.Fields(response)

	// Check for timeout
	if latencyMs > 30000 { // 30 seconds
		failures = append(failures, DetectedFailure{
			Type:       "timeout",
			Severity:   "major",
			Evidence:   fmt.Sprintf("response took %dms", latencyMs),
			Confidence: 1.0,
		})
	}

	// Check for empty/very short responses
	if len(respWords) == 0 {
		failures = append(failures, DetectedFailure{
			Type:       "crash",
			Severity:   "critical",
			Evidence:   "empty response",
			Confidence: 1.0,
		})
		return failures
	}

	// Check for filler responses
	fillerHits := 0
	var fillerEvidence []string
	for _, re := range fd.fillerPatterns {
		if match := re.FindString(response); match != "" {
			fillerHits++
			fillerEvidence = append(fillerEvidence, match)
		}
	}
	if fillerHits > 0 {
		confidence := math.Min(float64(fillerHits)*0.3, 0.95)
		severity := "minor"
		if fillerHits >= 2 {
			severity = "major"
			confidence = math.Min(confidence+0.2, 0.95)
		}
		failures = append(failures, DetectedFailure{
			Type:       "filler_response",
			Severity:   severity,
			Evidence:   strings.Join(fillerEvidence, "; "),
			Confidence: confidence,
		})
	}

	// Check for hallucination patterns
	hallHits := 0
	var hallEvidence []string
	for _, re := range fd.hallPatterns {
		if match := re.FindString(response); match != "" {
			hallHits++
			hallEvidence = append(hallEvidence, match)
		}
	}
	if hallHits > 0 {
		confidence := math.Min(float64(hallHits)*0.35, 0.9)
		severity := "major"
		if hallHits >= 3 {
			severity = "critical"
		}
		failures = append(failures, DetectedFailure{
			Type:       "hallucination",
			Severity:   severity,
			Evidence:   strings.Join(hallEvidence, "; "),
			Confidence: confidence,
		})
	}

	// Check for off-topic response (low overlap with input)
	if len(input) > 0 {
		inputWords := strings.Fields(strings.ToLower(input))
		respSet := make(map[string]bool)
		for _, w := range strings.Fields(respLower) {
			w = strings.Trim(w, ".,!?;:'\"()[]{}") // strip punctuation
			if len(w) > 3 {
				respSet[w] = true
			}
		}
		contentWordsInInput := 0
		covered := 0
		for _, w := range inputWords {
			w = strings.Trim(w, ".,!?;:'\"()[]{}") // strip punctuation
			if len(w) > 3 && !isStopWord(w) {
				contentWordsInInput++
				if respSet[w] {
					covered++
				}
			}
		}
		if contentWordsInInput > 2 && covered == 0 {
			failures = append(failures, DetectedFailure{
				Type:       "off_topic",
				Severity:   "major",
				Evidence:   fmt.Sprintf("0/%d content words from input found in response", contentWordsInInput),
				Confidence: 0.6,
			})
		}
	}

	// Check for low quality: too short, repetitive, or error-heavy
	if len(respWords) < 5 && len(respWords) > 0 {
		failures = append(failures, DetectedFailure{
			Type:       "low_quality",
			Severity:   "minor",
			Evidence:   fmt.Sprintf("response only %d words", len(respWords)),
			Confidence: 0.5,
		})
	}

	// Check for excessive repetition
	if len(respWords) > 10 {
		wordCounts := make(map[string]int)
		for _, w := range strings.Fields(respLower) {
			wordCounts[w]++
		}
		maxRepeat := 0
		var repeatedWord string
		for w, c := range wordCounts {
			if c > maxRepeat && len(w) > 3 {
				maxRepeat = c
				repeatedWord = w
			}
		}
		repeatRatio := float64(maxRepeat) / float64(len(respWords))
		if repeatRatio > 0.15 && maxRepeat >= 4 {
			failures = append(failures, DetectedFailure{
				Type:       "low_quality",
				Severity:   "minor",
				Evidence:   fmt.Sprintf("word %q repeated %d times (%.0f%% of response)", repeatedWord, maxRepeat, repeatRatio*100),
				Confidence: math.Min(repeatRatio*2, 0.8),
			})
		}
	}

	// Check for low quality signals
	lowQualityHits := 0
	for _, signal := range fd.lowQualitySignals {
		if strings.Contains(respLower, signal) {
			lowQualityHits++
		}
	}
	if lowQualityHits >= 3 {
		failures = append(failures, DetectedFailure{
			Type:       "low_quality",
			Severity:   "minor",
			Evidence:   fmt.Sprintf("%d low-quality signals detected", lowQualityHits),
			Confidence: math.Min(float64(lowQualityHits)*0.2, 0.7),
		})
	}

	return failures
}
