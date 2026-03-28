package cognitive

import (
	"math"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// User Preference Model — learns how a user prefers responses to be
// shaped, without any explicit configuration.
//
// Signals come from two sources:
//   1. Explicit: user says "keep it simple", "be more technical"
//   2. Behavioral: query length, clarification rate, follow-up patterns
//
// The model influences generation parameters (verbosity, tone, examples,
// formatting) so that Nous adapts to each user over time.
// -----------------------------------------------------------------------

// UserPreference captures a learned preference with confidence.
type UserPreference struct {
	Key        string    // preference key
	Value      string    // preference value
	Confidence float64   // how confident we are (0-1)
	Source     string    // "explicit" (user said it), "inferred" (from behavior)
	LearnedAt  time.Time
	SeenCount  int       // how many times this preference was reinforced
}

// PreferenceModel tracks user preferences across dimensions.
type PreferenceModel struct {
	Verbosity      float64 // 0=terse, 0.5=balanced, 1.0=verbose
	TechnicalDepth float64 // 0=simple, 0.5=moderate, 1.0=expert
	RiskTolerance  float64 // 0=conservative, 0.5=moderate, 1.0=bold
	FormattingPref string  // "prose", "bullets", "structured", "minimal"
	TonePref       Tone    // preferred tone
	ExamplePref    float64 // 0=no examples, 1.0=lots of examples

	// Per-preference tracking
	Preferences map[string]*UserPreference

	// Behavioral signals
	AverageQueryLength float64
	ClarificationRate  float64 // how often user asks for clarification
	CorrectionRate     float64 // how often user corrects responses
	FollowUpRate       float64 // how often user asks follow-ups
	TurnsSampled       int     // number of turns observed

	// Internal accumulators
	totalQueryLen       float64
	clarificationCount  int
	correctionCount     int
	followUpCount       int
	queryLengthSamples  []int
	technicalWordCount  int
	simpleRequestCount  int
	exampleRequestCount int
	bulletRequestCount  int
	prosePref           int
}

// NewPreferenceModel creates a preference model with balanced defaults.
func NewPreferenceModel() *PreferenceModel {
	return &PreferenceModel{
		Verbosity:      0.5,
		TechnicalDepth: 0.5,
		RiskTolerance:  0.5,
		FormattingPref: "prose",
		TonePref:       ToneNeutral,
		ExamplePref:    0.5,
		Preferences:    make(map[string]*UserPreference),
	}
}

// ObserveTurn updates preferences based on a conversation turn.
func (pm *PreferenceModel) ObserveTurn(input, response string, wasFollowUp, wasClarification, wasCorrection bool) {
	pm.TurnsSampled++

	// Track query length
	queryLen := len(strings.Fields(input))
	pm.totalQueryLen += float64(queryLen)
	pm.AverageQueryLength = pm.totalQueryLen / float64(pm.TurnsSampled)
	pm.queryLengthSamples = append(pm.queryLengthSamples, queryLen)
	if len(pm.queryLengthSamples) > 100 {
		pm.queryLengthSamples = pm.queryLengthSamples[1:]
	}

	// Track satisfaction signals
	if wasClarification {
		pm.clarificationCount++
	}
	if wasCorrection {
		pm.correctionCount++
	}
	if wasFollowUp {
		pm.followUpCount++
	}

	// Update rates
	if pm.TurnsSampled > 0 {
		pm.ClarificationRate = float64(pm.clarificationCount) / float64(pm.TurnsSampled)
		pm.CorrectionRate = float64(pm.correctionCount) / float64(pm.TurnsSampled)
		pm.FollowUpRate = float64(pm.followUpCount) / float64(pm.TurnsSampled)
	}

	// Detect technical language in input
	lower := strings.ToLower(input)
	pm.detectTechnicalSignals(lower)
	pm.detectFormatSignals(lower)
	pm.detectVerbositySignals(lower)
	pm.detectExplicitPreferences(lower)

	// Re-infer after enough samples
	if pm.TurnsSampled >= 3 {
		pm.InferPreferences()
	}
}

// InferPreferences derives preference scores from behavioral signals.
func (pm *PreferenceModel) InferPreferences() {
	if pm.TurnsSampled == 0 {
		return
	}

	// Verbosity inference from query length and satisfaction signals
	// Short queries (< 5 words avg) → user prefers terse
	// Long queries (> 15 words avg) → user is verbose, probably wants detail
	if pm.AverageQueryLength < 5 {
		pm.Verbosity = blend(pm.Verbosity, 0.25, 0.3)
	} else if pm.AverageQueryLength > 15 {
		pm.Verbosity = blend(pm.Verbosity, 0.8, 0.3)
	} else {
		pm.Verbosity = blend(pm.Verbosity, 0.5, 0.1)
	}

	// High clarification rate → system should be MORE verbose/explicit
	if pm.ClarificationRate > 0.2 {
		pm.Verbosity = blend(pm.Verbosity, 0.8, 0.4)
	}

	// High correction rate → system should be more careful/hedging
	if pm.CorrectionRate > 0.15 {
		pm.RiskTolerance = blend(pm.RiskTolerance, 0.2, 0.4)
	}

	// High follow-up rate → user is engaged, wants depth
	if pm.FollowUpRate > 0.3 {
		pm.Verbosity = blend(pm.Verbosity, 0.7, 0.2)
		pm.TechnicalDepth = blend(pm.TechnicalDepth, 0.7, 0.2)
	}

	// Technical depth from language analysis
	if pm.TurnsSampled > 0 {
		techRatio := float64(pm.technicalWordCount) / float64(pm.TurnsSampled)
		if techRatio > 2.0 {
			pm.TechnicalDepth = blend(pm.TechnicalDepth, 0.9, 0.3)
		} else if techRatio > 0.5 {
			pm.TechnicalDepth = blend(pm.TechnicalDepth, 0.7, 0.2)
		}

		simpleRatio := float64(pm.simpleRequestCount) / float64(pm.TurnsSampled)
		if simpleRatio > 0.2 {
			pm.TechnicalDepth = blend(pm.TechnicalDepth, 0.2, 0.4)
		}
	}

	// Example preference from explicit signals
	if pm.exampleRequestCount > 0 {
		pm.ExamplePref = blend(pm.ExamplePref, 0.9, 0.4)
	}

	// Format preference from signals
	if pm.bulletRequestCount > pm.prosePref {
		pm.FormattingPref = "bullets"
	} else if pm.prosePref > pm.bulletRequestCount {
		pm.FormattingPref = "prose"
	}

	// Clamp all values to [0, 1]
	pm.Verbosity = clamp01(pm.Verbosity)
	pm.TechnicalDepth = clamp01(pm.TechnicalDepth)
	pm.RiskTolerance = clamp01(pm.RiskTolerance)
	pm.ExamplePref = clamp01(pm.ExamplePref)
}

// SetExplicit records an explicitly stated preference.
func (pm *PreferenceModel) SetExplicit(key, value string) {
	key = strings.ToLower(strings.TrimSpace(key))
	value = strings.TrimSpace(value)

	pref := &UserPreference{
		Key:        key,
		Value:      value,
		Confidence: 1.0,
		Source:     "explicit",
		LearnedAt:  time.Now(),
		SeenCount:  1,
	}

	if existing, ok := pm.Preferences[key]; ok {
		existing.Value = value
		existing.SeenCount++
		existing.Confidence = 1.0
		existing.Source = "explicit"
		existing.LearnedAt = time.Now()
	} else {
		pm.Preferences[key] = pref
	}

	// Apply well-known explicit preferences immediately
	switch key {
	case "verbosity":
		switch strings.ToLower(value) {
		case "terse", "short", "brief", "concise":
			pm.Verbosity = 0.15
		case "verbose", "detailed", "long":
			pm.Verbosity = 0.85
		case "balanced", "moderate", "normal":
			pm.Verbosity = 0.5
		}
	case "technical_depth", "depth":
		switch strings.ToLower(value) {
		case "simple", "basic", "beginner":
			pm.TechnicalDepth = 0.15
		case "expert", "advanced", "technical":
			pm.TechnicalDepth = 0.9
		case "moderate", "intermediate":
			pm.TechnicalDepth = 0.5
		}
	case "tone":
		switch strings.ToLower(value) {
		case "casual":
			pm.TonePref = ToneCasual
		case "warm", "friendly":
			pm.TonePref = ToneWarm
		case "direct":
			pm.TonePref = ToneDirect
		case "neutral", "formal":
			pm.TonePref = ToneNeutral
		}
	case "format":
		switch strings.ToLower(value) {
		case "bullets", "bullet points", "list":
			pm.FormattingPref = "bullets"
		case "prose", "paragraph", "flowing":
			pm.FormattingPref = "prose"
		case "structured":
			pm.FormattingPref = "structured"
		case "minimal":
			pm.FormattingPref = "minimal"
		}
	case "examples":
		switch strings.ToLower(value) {
		case "yes", "lots", "many", "more":
			pm.ExamplePref = 0.9
		case "no", "none", "few":
			pm.ExamplePref = 0.1
		case "some", "moderate":
			pm.ExamplePref = 0.5
		}
	}
}

// ApplyToParams modifies TaskParams based on learned preferences.
func (pm *PreferenceModel) ApplyToParams(params *TaskParams) *TaskParams {
	if params == nil {
		return params
	}

	// Apply tone if not already specified
	if params.Tone == ToneNeutral && pm.TonePref != ToneNeutral {
		params.Tone = pm.TonePref
	}

	// Audience based on technical depth
	if params.Audience == "" {
		if pm.TechnicalDepth < 0.3 {
			params.Audience = "beginners"
		} else if pm.TechnicalDepth > 0.7 {
			params.Audience = "experts"
		}
	}

	return params
}

// GenerationHints returns hints for the generation pipeline based on preferences.
func (pm *PreferenceModel) GenerationHints() *GenerationHints {
	hints := &GenerationHints{}

	// Target length from verbosity
	if pm.Verbosity < 0.3 {
		hints.TargetLength = "short"
	} else if pm.Verbosity > 0.7 {
		hints.TargetLength = "long"
	} else {
		hints.TargetLength = "medium"
	}

	// Examples
	hints.UseExamples = pm.ExamplePref > 0.5

	// Bullet points
	hints.UseBulletPoints = pm.FormattingPref == "bullets" ||
		pm.FormattingPref == "structured"

	// Technical level
	if pm.TechnicalDepth < 0.3 {
		hints.TechnicalLevel = "simple"
	} else if pm.TechnicalDepth > 0.7 {
		hints.TechnicalLevel = "advanced"
	} else {
		hints.TechnicalLevel = "intermediate"
	}

	// Tone
	hints.Tone = pm.TonePref

	// Recap for verbose users or users who clarify often
	hints.IncludeRecap = pm.Verbosity > 0.7 || pm.ClarificationRate > 0.2

	// Warnings for risk-averse users
	hints.IncludeWarnings = pm.RiskTolerance < 0.3

	return hints
}

// GenerationHints provides generation pipeline hints based on user preferences.
type GenerationHints struct {
	TargetLength    string // "short" (<100 words), "medium" (100-300), "long" (300+)
	UseExamples     bool
	UseBulletPoints bool
	TechnicalLevel  string // "simple", "intermediate", "advanced"
	Tone            Tone
	IncludeRecap    bool
	IncludeWarnings bool // for risk-averse users
}

// -----------------------------------------------------------------------
// Internal detection methods
// -----------------------------------------------------------------------

// detectTechnicalSignals counts technical language in the input.
func (pm *PreferenceModel) detectTechnicalSignals(lower string) {
	technicalTerms := []string{
		"api", "algorithm", "function", "variable", "parameter",
		"protocol", "implementation", "architecture", "framework",
		"mutex", "goroutine", "concurrency", "asynchronous",
		"latency", "throughput", "complexity", "polymorphism",
		"inheritance", "recursion", "hash", "binary", "heap",
		"stack", "queue", "graph", "tree", "node",
		"microservice", "container", "kubernetes", "docker",
		"regex", "regexp", "sql", "database", "schema",
		"runtime", "compiler", "interpreter", "bytecode",
		"tcp", "http", "grpc", "rest", "websocket",
		"encryption", "authentication", "authorization",
		"neural", "gradient", "backpropagation", "embedding",
	}
	for _, term := range technicalTerms {
		if strings.Contains(lower, term) {
			pm.technicalWordCount++
		}
	}
}

// detectFormatSignals detects formatting preferences from input.
func (pm *PreferenceModel) detectFormatSignals(lower string) {
	bulletSignals := []string{
		"bullet point", "bullet points", "list", "bulleted",
		"numbered list", "step by step",
	}
	proseSignals := []string{
		"paragraph", "flowing", "essay", "narrative",
		"in prose", "explain in full",
	}
	for _, s := range bulletSignals {
		if strings.Contains(lower, s) {
			pm.bulletRequestCount++
			return
		}
	}
	for _, s := range proseSignals {
		if strings.Contains(lower, s) {
			pm.prosePref++
			return
		}
	}
}

// detectVerbositySignals detects explicit verbosity cues.
func (pm *PreferenceModel) detectVerbositySignals(lower string) {
	terseSignals := []string{
		"keep it short", "briefly", "in brief", "tldr",
		"tl;dr", "quick answer", "just tell me",
		"short answer", "one sentence",
	}
	verboseSignals := []string{
		"be specific", "in detail", "detailed explanation",
		"thorough", "comprehensive", "elaborate",
		"tell me everything", "deep dive",
	}
	simpleSignals := []string{
		"keep it simple", "explain simply", "in simple terms",
		"explain like i'm five", "eli5", "for a beginner",
		"in layman's terms", "plain english",
	}
	exampleSignals := []string{
		"give me an example", "for example", "with examples",
		"show me an example", "such as", "like what",
	}

	for _, s := range terseSignals {
		if strings.Contains(lower, s) {
			pm.Verbosity = blend(pm.Verbosity, 0.15, 0.5)
		}
	}
	for _, s := range verboseSignals {
		if strings.Contains(lower, s) {
			pm.Verbosity = blend(pm.Verbosity, 0.85, 0.5)
		}
	}
	for _, s := range simpleSignals {
		if strings.Contains(lower, s) {
			pm.simpleRequestCount++
		}
	}
	for _, s := range exampleSignals {
		if strings.Contains(lower, s) {
			pm.exampleRequestCount++
		}
	}
}

// detectExplicitPreferences detects when the user explicitly states preferences.
func (pm *PreferenceModel) detectExplicitPreferences(lower string) {
	// "pros and cons" → risk-aware user
	if strings.Contains(lower, "pros and cons") ||
		strings.Contains(lower, "advantages and disadvantages") ||
		strings.Contains(lower, "risks") {
		pm.RiskTolerance = blend(pm.RiskTolerance, 0.3, 0.3)
	}

	// Casual tone signals
	if strings.Contains(lower, "chill") || strings.Contains(lower, "casually") ||
		strings.Contains(lower, "no need to be formal") {
		pm.TonePref = ToneCasual
	}

	// Direct tone signals
	if strings.Contains(lower, "get to the point") ||
		strings.Contains(lower, "straight answer") ||
		strings.Contains(lower, "just the facts") {
		pm.TonePref = ToneDirect
	}
}

// -----------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------

// blend smoothly mixes a current value toward a target with a given weight.
func blend(current, target, weight float64) float64 {
	return current*(1.0-weight) + target*weight
}

// clamp01 clamps a float to [0, 1].
func clamp01(v float64) float64 {
	return math.Max(0.0, math.Min(1.0, v))
}
