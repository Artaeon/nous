package cognitive

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestAssess_KnownDomain(t *testing.T) {
	sm := NewSelfModel()

	// Build up a strong track record in science_explain.
	for i := 0; i < 20; i++ {
		sm.RecordOutcome("science_explain", "explain quantum entanglement", 0.85, true, "knowledge")
	}

	a := sm.Assess("how does photosynthesis work?", "science_explain")
	if a == nil {
		t.Fatal("assessment should not be nil")
	}
	if !a.CanHandle {
		t.Error("should report CanHandle for high-success domain")
	}
	if a.Confidence < 0.7 {
		t.Errorf("confidence = %.2f, want >= 0.7", a.Confidence)
	}
	if a.Disclaimer != "" {
		t.Errorf("high confidence domain should have no disclaimer, got %q", a.Disclaimer)
	}
	if a.BestApproach != "knowledge_lookup" {
		t.Errorf("approach = %q, want knowledge_lookup", a.BestApproach)
	}
}

func TestAssess_UnknownDomain(t *testing.T) {
	sm := NewSelfModel()

	a := sm.Assess("what career should I pursue?", "career_coaching")
	if a == nil {
		t.Fatal("assessment should not be nil")
	}
	if a.Confidence != 0.5 {
		t.Errorf("unknown domain confidence = %.2f, want 0.5", a.Confidence)
	}
	if a.Disclaimer == "" {
		t.Error("unknown domain should include a disclaimer")
	}
	if len(a.Limitations) == 0 {
		t.Error("unknown domain should list limitations")
	}
}

func TestAssess_WeakDomain(t *testing.T) {
	sm := NewSelfModel()

	// Build a poor track record.
	for i := 0; i < 10; i++ {
		sm.RecordOutcome("interpersonal", "help with a conflict", 0.2, false, "reasoning")
	}

	a := sm.Assess("how do I deal with a difficult coworker?", "interpersonal")
	if a.CanHandle {
		t.Error("weak domain should report CanHandle=false")
	}
	if a.Confidence > 0.3 {
		t.Errorf("weak domain confidence = %.2f, want <= 0.3", a.Confidence)
	}
	if a.Disclaimer == "" {
		t.Error("weak domain should have a disclaimer")
	}
}

func TestRecordOutcome(t *testing.T) {
	sm := NewSelfModel()

	sm.RecordOutcome("factual_qa", "what is the speed of light?", 0.9, true, "knowledge")
	sm.RecordOutcome("factual_qa", "who invented the telephone?", 0.8, true, "knowledge")
	sm.RecordOutcome("factual_qa", "when was the moon landing?", 0.3, false, "knowledge")

	sm.mu.RLock()
	profile := sm.capabilities["factual_qa"]
	sm.mu.RUnlock()

	if profile == nil {
		t.Fatal("factual_qa profile should exist")
	}
	if profile.Successes != 2 {
		t.Errorf("successes = %d, want 2", profile.Successes)
	}
	if profile.Failures != 1 {
		t.Errorf("failures = %d, want 1", profile.Failures)
	}
	if profile.AvgQuality < 0.3 || profile.AvgQuality > 1.0 {
		t.Errorf("avg quality = %.2f, want in (0.3, 1.0)", profile.AvgQuality)
	}
	if len(sm.interactions) != 3 {
		t.Errorf("interactions = %d, want 3", len(sm.interactions))
	}
}

func TestRecordOutcome_TrendDetection(t *testing.T) {
	sm := NewSelfModel()

	// First batch: low quality.
	for i := 0; i < 5; i++ {
		sm.RecordOutcome("comparison", "compare X vs Y", 0.3, false, "reasoning")
	}
	// Second batch: high quality (improving).
	for i := 0; i < 5; i++ {
		sm.RecordOutcome("comparison", "compare A vs B", 0.9, true, "synthesis")
	}

	sm.mu.RLock()
	trend := sm.capabilities["comparison"].Trend
	sm.mu.RUnlock()

	if trend != "improving" {
		t.Errorf("trend = %q, want improving", trend)
	}

	// Now reverse: add bad outcomes to make it declining.
	sm2 := NewSelfModel()
	for i := 0; i < 5; i++ {
		sm2.RecordOutcome("creative", "write a poem", 0.9, true, "generative")
	}
	for i := 0; i < 5; i++ {
		sm2.RecordOutcome("creative", "write a story", 0.2, false, "generative")
	}

	sm2.mu.RLock()
	trend2 := sm2.capabilities["creative"].Trend
	sm2.mu.RUnlock()

	if trend2 != "declining" {
		t.Errorf("trend = %q, want declining", trend2)
	}
}

func TestClassifyDomain(t *testing.T) {
	sm := NewSelfModel()

	tests := []struct {
		query string
		want  string
	}{
		{"explain how quantum physics works", "science_explain"},
		{"tell me about the French Revolution", "history_explain"},
		{"what is the meaning of life?", "philosophy_explain"},
		{"help me with my resume for a new job", "career_coaching"},
		{"compare Python vs Go for web development", "comparison"},
		{"write a poem about the ocean", "creative"},
		{"how do I debug a segfault in my code?", "technical"},
		{"who are you and what can you do?", "meta"},
		{"should I choose option A or option B?", "decision_support"},
		{"help me plan my project timeline", "planning"},
	}

	for _, tt := range tests {
		got := sm.ClassifyDomain(tt.query)
		if got != tt.want {
			t.Errorf("ClassifyDomain(%q) = %q, want %q", tt.query, got, tt.want)
		}
	}
}

func TestGenerateReport(t *testing.T) {
	sm := NewSelfModel()

	// No interactions — should return empty report.
	r := sm.GenerateReport()
	if r.TotalInteractions != 0 {
		t.Errorf("empty report interactions = %d, want 0", r.TotalInteractions)
	}

	// Add mixed data across several domains.
	for i := 0; i < 10; i++ {
		sm.RecordOutcome("science_explain", "explain gravity", 0.9, true, "knowledge")
	}
	for i := 0; i < 8; i++ {
		sm.RecordOutcome("technical", "fix a bug", 0.8, true, "reasoning")
	}
	for i := 0; i < 6; i++ {
		sm.RecordOutcome("career_coaching", "resume tips", 0.3, false, "reasoning")
	}

	r = sm.GenerateReport()
	if r.TotalInteractions != 24 {
		t.Errorf("interactions = %d, want 24", r.TotalInteractions)
	}
	if r.OverallQuality < 0.5 {
		t.Errorf("overall quality = %.2f, want > 0.5", r.OverallQuality)
	}
	if len(r.StrongestDomains) == 0 {
		t.Fatal("should have strongest domains")
	}
	if r.StrongestDomains[0].Domain != "science_explain" {
		t.Errorf("strongest = %q, want science_explain", r.StrongestDomains[0].Domain)
	}
	if len(r.WeakestDomains) == 0 {
		t.Fatal("should have weakest domains")
	}
	if r.WeakestDomains[0].Domain != "career_coaching" {
		t.Errorf("weakest = %q, want career_coaching", r.WeakestDomains[0].Domain)
	}
}

func TestHonestDisclaimer(t *testing.T) {
	sm := NewSelfModel()

	// Unknown domain — should still produce a meaningful disclaimer.
	d := sm.HonestDisclaimer("career_coaching")
	if d == "" {
		t.Error("disclaimer should not be empty even without data")
	}
	if d == "I have limited experience with this type of question, so my answer may not be as strong as in other areas." {
		// This is the truly-unknown fallback, career_coaching should have a
		// more specific template.
		if d != domainDisclaimers["career_coaching"] {
			t.Errorf("expected domain-specific disclaimer for career_coaching")
		}
	}

	// Build data to trigger augmented disclaimer.
	for i := 0; i < 5; i++ {
		sm.RecordOutcome("career_coaching", "salary negotiation", 0.2, false, "reasoning")
	}

	d = sm.HonestDisclaimer("career_coaching")
	if d == "" {
		t.Error("disclaimer should not be empty")
	}
	// Should mention it's a weaker area.
	if len(d) < 30 {
		t.Errorf("disclaimer seems too short: %q", d)
	}

	// Completely unknown domain.
	d = sm.HonestDisclaimer("underwater_basket_weaving")
	if d == "" {
		t.Error("unknown domain should still produce a disclaimer")
	}
}

func TestSuggestApproach(t *testing.T) {
	sm := NewSelfModel()

	tests := []struct {
		domain     string
		confidence float64
		want       string
	}{
		{"science_explain", 0.9, "knowledge_lookup"},
		{"science_explain", 0.7, "knowledge_lookup"},
		{"factual_qa", 0.6, "synthesis"},
		{"career_coaching", 0.35, "socratic"},
		{"decision_support", 0.3, "socratic"},
		{"technical", 0.35, "reasoning"},
		{"interpersonal", 0.1, "honest_limit"},
		{"science_explain", 0.1, "honest_limit"},
	}

	for _, tt := range tests {
		got := sm.SuggestApproach(tt.domain, tt.confidence)
		if got != tt.want {
			t.Errorf("SuggestApproach(%q, %.2f) = %q, want %q",
				tt.domain, tt.confidence, got, tt.want)
		}
	}
}

func TestSelfModelSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "self_model.json")

	sm := NewSelfModel()
	sm.RecordOutcome("science_explain", "explain gravity", 0.9, true, "knowledge")
	sm.RecordOutcome("science_explain", "explain entropy", 0.85, true, "knowledge")
	sm.RecordOutcome("career_coaching", "resume tips", 0.3, false, "reasoning")

	if err := sm.Save(path); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	// Verify file exists.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("saved file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	// Load into a fresh model.
	sm2 := NewSelfModel()
	if err := sm2.Load(path); err != nil {
		t.Fatalf("Load error: %v", err)
	}

	// Verify state was preserved.
	sm2.mu.RLock()
	defer sm2.mu.RUnlock()

	if len(sm2.interactions) != 3 {
		t.Errorf("loaded interactions = %d, want 3", len(sm2.interactions))
	}
	sci := sm2.capabilities["science_explain"]
	if sci == nil {
		t.Fatal("science_explain capability should exist after load")
	}
	if sci.Successes != 2 {
		t.Errorf("loaded successes = %d, want 2", sci.Successes)
	}
	career := sm2.capabilities["career_coaching"]
	if career == nil {
		t.Fatal("career_coaching capability should exist after load")
	}
	if career.Failures != 1 {
		t.Errorf("loaded failures = %d, want 1", career.Failures)
	}
}

func TestSelfModelThreadSafe(t *testing.T) {
	sm := NewSelfModel()

	var wg sync.WaitGroup
	// Concurrent writes.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			domain := "science_explain"
			if n%3 == 0 {
				domain = "career_coaching"
			}
			sm.RecordOutcome(domain, "test query", 0.5, n%2 == 0, "test")
		}(i)
	}

	// Concurrent reads.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sm.Assess("test", "science_explain")
			sm.GenerateReport()
			sm.ClassifyDomain("explain quantum physics")
			sm.HonestDisclaimer("science_explain")
			sm.SuggestApproach("career_coaching", 0.5)
		}()
	}

	wg.Wait()

	// Just verify we didn't panic and state is consistent.
	sm.mu.RLock()
	total := len(sm.interactions)
	sm.mu.RUnlock()

	if total != 50 {
		t.Errorf("after concurrent ops, interactions = %d, want 50", total)
	}
}

func TestRecordOutcome_MaxRecords(t *testing.T) {
	sm := NewSelfModel()
	sm.maxRecords = 10

	for i := 0; i < 20; i++ {
		sm.RecordOutcome("factual_qa", "test", 0.5, true, "test")
	}

	sm.mu.RLock()
	count := len(sm.interactions)
	sm.mu.RUnlock()

	if count != 10 {
		t.Errorf("interactions = %d, want 10 (max)", count)
	}
}

func TestClassifyDomain_Defaults(t *testing.T) {
	sm := NewSelfModel()

	// A query with no matching keywords should default to factual_qa.
	got := sm.ClassifyDomain("xyzzy plugh")
	if got != "factual_qa" {
		t.Errorf("default classification = %q, want factual_qa", got)
	}
}

func TestAssess_ModerateDomain(t *testing.T) {
	sm := NewSelfModel()

	// Build a mediocre track record (50% success).
	for i := 0; i < 5; i++ {
		sm.RecordOutcome("planning", "plan my project", 0.6, true, "reasoning")
	}
	for i := 0; i < 5; i++ {
		sm.RecordOutcome("planning", "schedule a roadmap", 0.4, false, "reasoning")
	}

	a := sm.Assess("help me plan a vacation", "planning")
	if a == nil {
		t.Fatal("assessment should not be nil")
	}
	if a.Disclaimer == "" {
		t.Error("moderate domain should include a disclaimer")
	}
	if a.Confidence < 0.3 || a.Confidence > 0.7 {
		t.Errorf("moderate confidence = %.2f, want in (0.3, 0.7)", a.Confidence)
	}
}
