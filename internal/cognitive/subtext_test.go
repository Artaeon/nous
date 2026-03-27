package cognitive

import (
	"math"
	"testing"
	"time"
)

// -----------------------------------------------------------------------
// Hedging detection
// -----------------------------------------------------------------------

func TestDetectHedging(t *testing.T) {
	se := NewSubtextEngine(nil)

	tests := []struct {
		name    string
		input   string
		wantHit bool
		minW    float64
	}{
		{"single hedge", "Maybe I should try a different approach", true, 0.2},
		{"multi-word hedge", "I think maybe this is wrong", true, 0.2},
		{"double hedge", "I guess maybe we could sort of try it", true, 0.5},
		{"no hedge", "Fix the database connection", false, 0},
		{"hedge phrase: not sure if", "I'm not sure if this is right", true, 0.2},
		{"empty input", "", false, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sig, ok := se.detectHedging(tt.input)
			if ok != tt.wantHit {
				t.Errorf("detectHedging(%q): got hit=%v, want %v", tt.input, ok, tt.wantHit)
			}
			if ok && sig.Weight < tt.minW {
				t.Errorf("detectHedging(%q): weight=%.2f, want >= %.2f", tt.input, sig.Weight, tt.minW)
			}
			if ok && sig.Type != "hedging" {
				t.Errorf("detectHedging(%q): type=%q, want 'hedging'", tt.input, sig.Type)
			}
		})
	}
}

// -----------------------------------------------------------------------
// Urgency detection
// -----------------------------------------------------------------------

func TestDetectUrgency(t *testing.T) {
	se := NewSubtextEngine(nil)

	tests := []struct {
		name string
		input string
		minU  float64
		maxU  float64
	}{
		{"calm question", "what time is it", 0.0, 0.2},
		{"exclamation", "help me!", 0.2, 0.7},
		{"ALL CAPS", "THIS IS BROKEN AND I NEED HELP", 0.3, 1.0},
		{"repeated punctuation", "why isn't this working??!!", 0.3, 1.0},
		{"urgency word", "I need this fixed immediately", 0.2, 0.8},
		{"combined signals", "HELP!!! This is URGENT!!!", 0.6, 1.0},
		{"common acronym ignored", "The AI and API are fine", 0.0, 0.2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := se.detectUrgency(tt.input)
			if u < tt.minU || u > tt.maxU {
				t.Errorf("detectUrgency(%q): got %.2f, want [%.2f, %.2f]", tt.input, u, tt.minU, tt.maxU)
			}
		})
	}
}

// -----------------------------------------------------------------------
// Emotional state detection
// -----------------------------------------------------------------------

func TestDetectEmotionalState(t *testing.T) {
	se := NewSubtextEngine(nil)

	tests := []struct {
		name        string
		input       string
		wantValence string // "positive", "negative", "neutral"
		wantDom     string // expected dominant (empty = any)
	}{
		{"positive", "I got promoted! This is amazing and wonderful!", "positive", "excited"},
		{"negative", "This is terrible, I hate this stupid thing", "negative", ""},
		{"neutral", "Can you show me the weather", "neutral", "neutral"},
		{"angry explicit", "I am so angry about this", "negative", "angry"},
		{"sad explicit", "I feel really sad and depressed", "negative", "sad"},
		{"anxious explicit", "I'm very anxious and worried about tomorrow", "negative", "anxious"},
		{"grateful", "I'm so grateful for your help, thanks!", "positive", "grateful"},
		{"curious", "I'm curious about how this works", "neutral", "curious"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			es := se.detectEmotionalState(tt.input)

			switch tt.wantValence {
			case "positive":
				if es.Valence <= 0 {
					t.Errorf("valence=%.2f, want positive", es.Valence)
				}
			case "negative":
				if es.Valence >= 0 {
					t.Errorf("valence=%.2f, want negative", es.Valence)
				}
			case "neutral":
				if math.Abs(es.Valence) > 0.3 {
					t.Errorf("valence=%.2f, want near-neutral", es.Valence)
				}
			}

			if tt.wantDom != "" && es.Dominant != tt.wantDom {
				t.Errorf("dominant=%q, want %q (valence=%.2f, arousal=%.2f)", es.Dominant, tt.wantDom, es.Valence, es.Arousal)
			}
		})
	}
}

// -----------------------------------------------------------------------
// Temporal context
// -----------------------------------------------------------------------

func TestClassifyTimeOfDay(t *testing.T) {
	tests := []struct {
		hour int
		want string
	}{
		{2, "late_night"},
		{5, "late_night"},
		{7, "early_morning"},
		{10, "morning"},
		{14, "afternoon"},
		{19, "evening"},
		{23, "late_night"},
	}

	for _, tt := range tests {
		got := classifyTimeOfDay(tt.hour)
		if got != tt.want {
			t.Errorf("classifyTimeOfDay(%d)=%q, want %q", tt.hour, got, tt.want)
		}
	}
}

func TestTemporalContextWeekend(t *testing.T) {
	se := NewSubtextEngine(nil)

	// Saturday at noon
	se.now = func() time.Time {
		return time.Date(2026, 3, 28, 12, 0, 0, 0, time.UTC) // Saturday
	}
	tc := se.matchTemporalPatterns("hello", nil)
	if !tc.IsWeekend {
		t.Error("expected IsWeekend=true for Saturday")
	}
	if tc.TimeOfDay != "afternoon" {
		t.Errorf("TimeOfDay=%q, want 'afternoon'", tc.TimeOfDay)
	}

	// Monday at 8am
	se.now = func() time.Time {
		return time.Date(2026, 3, 23, 8, 0, 0, 0, time.UTC) // Monday
	}
	tc = se.matchTemporalPatterns("hello", nil)
	if tc.IsWeekend {
		t.Error("expected IsWeekend=false for Monday")
	}
	if tc.TimeOfDay != "early_morning" {
		t.Errorf("TimeOfDay=%q, want 'early_morning'", tc.TimeOfDay)
	}
}

// -----------------------------------------------------------------------
// Implied need inference
// -----------------------------------------------------------------------

func TestInferImpliedNeed(t *testing.T) {
	se := NewSubtextEngine(nil)

	tests := []struct {
		name     string
		intent   string
		emotion  EmotionalState
		signals  []BehavioralSignal
		temporal TemporalContext
		want     string
	}{
		{
			name:    "celebration: excited positive",
			intent:  "share",
			emotion: EmotionalState{Valence: 0.7, Arousal: 0.6, Dominant: "excited"},
			want:    NeedCelebration,
		},
		{
			name:    "venting: frustration no question",
			intent:  "complain",
			emotion: EmotionalState{Valence: -0.5, Arousal: 0.6, Dominant: "frustrated"},
			signals: []BehavioralSignal{
				{Type: "venting", Evidence: "this stupid", Weight: 0.7},
			},
			want: NeedVenting,
		},
		{
			name:    "validation: seeking + hedging",
			intent:  "opinion",
			emotion: EmotionalState{Valence: 0.0, Arousal: 0.2, Dominant: "neutral"},
			signals: []BehavioralSignal{
				{Type: "seeking_validation", Evidence: "what do you think", Weight: 0.6},
				{Type: "hedging", Evidence: "maybe", Weight: 0.3},
			},
			want: NeedValidation,
		},
		{
			name:     "reassurance: repetition + late night",
			intent:   "status",
			emotion:  EmotionalState{Valence: -0.1, Arousal: 0.3, Dominant: "neutral"},
			signals:  []BehavioralSignal{{Type: "repetition", Evidence: "project", Weight: 0.5}},
			temporal: TemporalContext{TimeOfDay: "late_night"},
			want:     NeedReassurance,
		},
		{
			name:    "practical help: direct request",
			intent:  "help fix the database",
			emotion: EmotionalState{Valence: 0.0, Arousal: 0.1, Dominant: "neutral"},
			want:    NeedPracticalHelp,
		},
		{
			name:    "information: factual question",
			intent:  "what is Go",
			emotion: EmotionalState{Valence: 0.0, Arousal: 0.0, Dominant: "neutral"},
			want:    NeedInformation,
		},
		{
			name:    "connection: greeting",
			intent:  "greeting",
			emotion: EmotionalState{Valence: 0.1, Arousal: 0.1, Dominant: "neutral"},
			want:    NeedConnection,
		},
		{
			name:    "guidance: hedging without validation",
			intent:  "trying something",
			emotion: EmotionalState{Valence: 0.0, Arousal: 0.1, Dominant: "neutral"},
			signals: []BehavioralSignal{{Type: "hedging", Evidence: "maybe, i guess", Weight: 0.6}},
			want:    NeedGuidance,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := se.inferImpliedNeed(tt.intent, tt.emotion, tt.signals, tt.temporal)
			if got != tt.want {
				t.Errorf("inferImpliedNeed: got %q, want %q", got, tt.want)
			}
		})
	}
}

// -----------------------------------------------------------------------
// Full Analyze integration
// -----------------------------------------------------------------------

func TestAnalyzeIntegration(t *testing.T) {
	se := NewSubtextEngine(nil)
	se.now = func() time.Time {
		return time.Date(2026, 3, 26, 23, 30, 0, 0, time.UTC) // late night
	}

	// Scenario: frustrated user venting at night
	nlu := &NLUResult{Intent: "complain", Raw: "this stupid bug keeps breaking everything"}
	result := se.Analyze("this stupid bug keeps breaking everything", nlu, nil)

	if result.ImpliedNeed != NeedVenting {
		t.Errorf("implied need=%q, want %q", result.ImpliedNeed, NeedVenting)
	}
	if result.EmotionalState.Valence >= 0 {
		t.Errorf("valence=%.2f, want negative", result.EmotionalState.Valence)
	}
	if result.TemporalContext.TimeOfDay != "late_night" {
		t.Errorf("time_of_day=%q, want 'late_night'", result.TemporalContext.TimeOfDay)
	}
	if result.Confidence <= 0 {
		t.Error("confidence should be > 0")
	}

	// Scenario: excited celebration
	nlu2 := &NLUResult{Intent: "share", Raw: "I got promoted! This is amazing!"}
	result2 := se.Analyze("I got promoted! This is amazing!", nlu2, nil)

	if result2.ImpliedNeed != NeedCelebration {
		t.Errorf("implied need=%q, want %q", result2.ImpliedNeed, NeedCelebration)
	}
	if result2.EmotionalState.Valence <= 0 {
		t.Errorf("valence=%.2f, want positive", result2.EmotionalState.Valence)
	}

	// Scenario: hedging validation-seeker
	nlu3 := &NLUResult{Intent: "opinion", Raw: "what do you think, maybe I should quit my job?"}
	result3 := se.Analyze("what do you think, maybe I should quit my job?", nlu3, nil)

	if result3.ImpliedNeed != NeedValidation {
		t.Errorf("implied need=%q, want %q", result3.ImpliedNeed, NeedValidation)
	}
}

// -----------------------------------------------------------------------
// Brevity detection
// -----------------------------------------------------------------------

func TestDetectBrevity(t *testing.T) {
	se := NewSubtextEngine(nil)

	// History with decreasing message lengths
	history := []ConvTurn{
		{Input: "I have been working on this project for a while now and I want to discuss some ideas"},
		{Input: "The architecture needs to handle concurrent requests efficiently"},
		{Input: "What about using channels for the worker pool design"},
		{Input: "yeah channels"},
		{Input: "ok"},
	}

	sig, ok := se.detectBrevity("fine", history)
	if !ok {
		t.Error("expected brevity to be detected with shrinking messages")
	}
	if ok && sig.Type != "brevity" {
		t.Errorf("type=%q, want 'brevity'", sig.Type)
	}

	// No brevity with consistent length
	consistentHistory := []ConvTurn{
		{Input: "tell me about the weather today please"},
		{Input: "what about the temperature forecast"},
		{Input: "and will there be rain tomorrow"},
	}
	_, ok = se.detectBrevity("how about the weekend forecast", consistentHistory)
	if ok {
		t.Error("expected no brevity detection with consistent message lengths")
	}
}

// -----------------------------------------------------------------------
// Venting detection
// -----------------------------------------------------------------------

func TestDetectVenting(t *testing.T) {
	se := NewSubtextEngine(nil)

	tests := []struct {
		name    string
		input   string
		wantHit bool
	}{
		{"venting no question", "this stupid thing is so annoying", true},
		{"venting with question", "this stupid thing, how do I fix it?", true},
		{"not venting", "can you help me with this function", false},
		{"ugh venting", "ugh I can't believe this happened again", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sig, ok := se.detectVenting(tt.input)
			if ok != tt.wantHit {
				t.Errorf("detectVenting(%q): got hit=%v, want %v", tt.input, ok, tt.wantHit)
			}
			if ok {
				if sig.Type != "venting" {
					t.Errorf("type=%q, want 'venting'", sig.Type)
				}
				// No question mark = higher weight
				if tt.name == "venting no question" && sig.Weight < 0.5 {
					t.Errorf("weight=%.2f, expected higher for no-question venting", sig.Weight)
				}
			}
		})
	}
}

// -----------------------------------------------------------------------
// Validation seeking detection
// -----------------------------------------------------------------------

func TestDetectValidationSeeking(t *testing.T) {
	se := NewSubtextEngine(nil)

	tests := []struct {
		input   string
		wantHit bool
	}{
		{"what do you think about my approach?", true},
		{"do you think I should take the job?", true},
		{"does that make sense?", true},
		{"build me a REST API", false},
	}

	for _, tt := range tests {
		sig, ok := se.detectValidationSeeking(tt.input)
		if ok != tt.wantHit {
			t.Errorf("detectValidationSeeking(%q): got %v, want %v", tt.input, ok, tt.wantHit)
		}
		if ok && sig.Type != "seeking_validation" {
			t.Errorf("type=%q, want 'seeking_validation'", sig.Type)
		}
	}
}

// -----------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------

func TestSubtextTokenize(t *testing.T) {
	got := subtextTokenize("Hello, world! This is a TEST.")
	want := []string{"hello", "world", "this", "is", "a", "test"}
	if len(got) != len(want) {
		t.Fatalf("tokenize: got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tokenize[%d]=%q, want %q", i, got[i], want[i])
		}
	}
}

func TestSubtextSplitSentences(t *testing.T) {
	got := subtextSplitSentences("Hello world. How are you? Great!")
	if len(got) != 3 {
		t.Fatalf("splitSentences: got %d sentences, want 3: %v", len(got), got)
	}
}
