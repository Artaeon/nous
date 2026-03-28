package cognitive

import "math"

// -----------------------------------------------------------------------
// Confidence calibration and abstention logic.
//
// The calibration pipeline takes raw confidence scores from pattern and
// neural NLU, combines agreement/disagreement signals, slot fill rates,
// and query characteristics to produce a single calibrated confidence.
//
// Abstention prevents the system from generating low-quality responses:
// instead of guessing, it returns a structured uncertainty result that
// the response layer can format as a clarifying question.
// -----------------------------------------------------------------------

// CalibrationConfig holds thresholds for confidence-based decisions.
type CalibrationConfig struct {
	AbstainThreshold    float64 // below this, refuse to answer (default 0.25)
	LowConfThreshold    float64 // below this, add hedging (default 0.50)
	HighConfThreshold   float64 // above this, answer directly (default 0.80)
	AgreementBonus      float64 // bonus when neural+pattern agree (default 0.15)
	DisagreementPenalty float64 // penalty when they disagree (default 0.20)
}

// DefaultCalibration returns a CalibrationConfig with production defaults.
func DefaultCalibration() *CalibrationConfig {
	return &CalibrationConfig{
		AbstainThreshold:    0.25,
		LowConfThreshold:    0.50,
		HighConfThreshold:   0.80,
		AgreementBonus:      0.15,
		DisagreementPenalty: 0.20,
	}
}

// ConfidenceSignals captures all signals that inform confidence calibration.
type ConfidenceSignals struct {
	PatternConf   float64 // pattern NLU confidence
	NeuralConf    float64 // neural NLU confidence
	PatternIntent string  // what pattern NLU thinks
	NeuralIntent  string  // what neural NLU thinks
	QueryLength   int     // longer queries = more info
	HasEntities   bool    // extracted entities boost confidence
	SlotsFilled   int     // more filled slots = clearer intent
	SlotsExpected int     // how many slots this intent needs
	IsAmbiguous   bool    // multiple strong intent candidates
}

// AbstentionResult describes why the system should or should not abstain.
type AbstentionResult struct {
	ShouldAbstain bool
	Reason        string  // "low_confidence", "ambiguous_intent", "missing_critical_slots"
	Confidence    float64 // calibrated confidence
	Suggestion    string  // what to say to the user if abstaining
}

// CalibrateConfidence adjusts raw confidence using multiple signals.
// It combines pattern and neural agreement, slot fill rate, query length,
// and ambiguity to produce a calibrated confidence in [0, 1].
func CalibrateConfidence(raw float64, signals ConfidenceSignals) float64 {
	conf := raw

	// 1. Agreement/disagreement between pattern and neural classifiers
	if signals.PatternIntent != "" && signals.NeuralIntent != "" {
		if signals.PatternIntent == signals.NeuralIntent {
			// Both classifiers agree: boost confidence
			conf += DefaultCalibration().AgreementBonus
		} else {
			// Disagreement: penalize
			conf -= DefaultCalibration().DisagreementPenalty
		}
	}

	// 2. Slot fill rate: if we expected slots and got them, boost
	if signals.SlotsExpected > 0 {
		fillRate := float64(signals.SlotsFilled) / float64(signals.SlotsExpected)
		// Boost up to +0.10 for fully filled slots
		conf += fillRate * 0.10
	}

	// 3. Entity presence: having entities means clearer intent
	if signals.HasEntities {
		conf += 0.05
	}

	// 4. Query length heuristic: very short queries (1-2 words) are often
	// ambiguous; longer queries (5+ words) carry more signal.
	switch {
	case signals.QueryLength <= 2:
		conf -= 0.10 // very short = ambiguous
	case signals.QueryLength >= 5:
		conf += 0.05 // longer = more context
	}

	// 5. Ambiguity penalty: if multiple intents scored similarly
	if signals.IsAmbiguous {
		conf -= 0.15
	}

	// 6. Combine pattern and neural if both present and no strong leader
	if signals.PatternConf > 0 && signals.NeuralConf > 0 {
		// Weighted average: favor the higher-confidence one
		avg := (signals.PatternConf + signals.NeuralConf) / 2.0
		maxConf := math.Max(signals.PatternConf, signals.NeuralConf)
		// Blend: 70% max, 30% average
		blended := maxConf*0.7 + avg*0.3
		// Only use blended if it would improve over raw
		if blended > conf {
			conf = blended
		}
	}

	// Clamp to [0, 1]
	if conf < 0 {
		conf = 0
	}
	if conf > 1 {
		conf = 1
	}

	return conf
}

// CalibrateConfidenceWithConfig is like CalibrateConfidence but uses a
// custom configuration instead of defaults.
func CalibrateConfidenceWithConfig(raw float64, signals ConfidenceSignals, config *CalibrationConfig) float64 {
	conf := raw

	// 1. Agreement/disagreement
	if signals.PatternIntent != "" && signals.NeuralIntent != "" {
		if signals.PatternIntent == signals.NeuralIntent {
			conf += config.AgreementBonus
		} else {
			conf -= config.DisagreementPenalty
		}
	}

	// 2. Slot fill rate
	if signals.SlotsExpected > 0 {
		fillRate := float64(signals.SlotsFilled) / float64(signals.SlotsExpected)
		conf += fillRate * 0.10
	}

	// 3. Entity presence
	if signals.HasEntities {
		conf += 0.05
	}

	// 4. Query length
	switch {
	case signals.QueryLength <= 2:
		conf -= 0.10
	case signals.QueryLength >= 5:
		conf += 0.05
	}

	// 5. Ambiguity
	if signals.IsAmbiguous {
		conf -= 0.15
	}

	// 6. Blend pattern + neural
	if signals.PatternConf > 0 && signals.NeuralConf > 0 {
		avg := (signals.PatternConf + signals.NeuralConf) / 2.0
		maxConf := math.Max(signals.PatternConf, signals.NeuralConf)
		blended := maxConf*0.7 + avg*0.3
		if blended > conf {
			conf = blended
		}
	}

	if conf < 0 {
		conf = 0
	}
	if conf > 1 {
		conf = 1
	}

	return conf
}

// ShouldAbstain determines whether the system should refuse to answer
// based on calibrated confidence and supporting signals.
//
// Abstention reasons:
//   - low_confidence: calibrated score below threshold
//   - ambiguous_intent: multiple plausible intents with similar scores
//   - missing_critical_slots: intent recognized but key slots unfilled
func ShouldAbstain(conf float64, signals ConfidenceSignals, config *CalibrationConfig) AbstentionResult {
	if config == nil {
		config = DefaultCalibration()
	}

	// Check for missing critical slots first -- even high confidence
	// should not proceed if the query is clearly incomplete.
	if signals.SlotsExpected > 0 && signals.SlotsFilled == 0 {
		// Zero slots filled out of expected: likely missing critical info
		if conf < config.LowConfThreshold {
			return AbstentionResult{
				ShouldAbstain: true,
				Reason:        "missing_critical_slots",
				Confidence:    conf,
				Suggestion:    "I think I understand what you're asking about, but could you provide more details?",
			}
		}
	}

	// Ambiguous intent: pattern and neural disagree AND both have
	// moderate confidence -- the system genuinely doesn't know.
	if signals.IsAmbiguous {
		if conf < config.LowConfThreshold {
			return AbstentionResult{
				ShouldAbstain: true,
				Reason:        "ambiguous_intent",
				Confidence:    conf,
				Suggestion:    "I'm not sure I understand. Could you rephrase that or give me more context?",
			}
		}
	}

	// Low confidence: raw calibrated score is below abstention threshold
	if conf < config.AbstainThreshold {
		return AbstentionResult{
			ShouldAbstain: true,
			Reason:        "low_confidence",
			Confidence:    conf,
			Suggestion:    "I'm not confident I understand what you mean. Could you say that differently?",
		}
	}

	// No abstention
	return AbstentionResult{
		ShouldAbstain: false,
		Reason:        "",
		Confidence:    conf,
		Suggestion:    "",
	}
}
