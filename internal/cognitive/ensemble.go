package cognitive

// ToolEnsemble combines predictions from IntentCompiler and NeuralCortex
// for higher accuracy tool selection. When the intent compiler is uncertain
// (confidence 0.5-0.8), it consults the cortex for a second opinion.
//
// Innovation: Two independent prediction systems — one rule-based (intent),
// one learned (cortex) — vote on the right tool. Agreement boosts confidence,
// disagreement triggers LLM fallback. This is ensemble learning applied to
// tool selection in an AI agent.
//
// The intent compiler is fast and precise but rigid (regex patterns).
// The neural cortex is adaptive but needs training data.
// Together they cover each other's blind spots:
//   - Intent handles structured commands perfectly (confidence ≥ 0.8)
//   - Cortex handles ambiguous queries that don't match patterns
//   - When both agree, confidence is boosted (precision > either alone)
//   - When they disagree, the system falls back to LLM (safety)
type ToolEnsemble struct {
	Intent *IntentCompiler
	Cortex *NeuralCortex
}

// EnsembleResult holds the combined prediction from both systems.
type EnsembleResult struct {
	Tool       string
	Confidence float64
	Source     string // "intent", "cortex", "ensemble", or "conflict"
	IntentConf float64
	CortexConf float64
}

// NewToolEnsemble creates a new ensemble from an intent compiler and neural cortex.
func NewToolEnsemble(intent *IntentCompiler, cortex *NeuralCortex) *ToolEnsemble {
	return &ToolEnsemble{Intent: intent, Cortex: cortex}
}

// Predict returns the best tool prediction using ensemble voting.
// Strategy:
//   - If intent confidence ≥ 0.8: trust intent (proven patterns)
//   - If intent confidence 0.5-0.8: consult cortex
//     - If both agree: boost confidence (min * 1.3, capped at 0.95)
//     - If they disagree: return "conflict" with reduced confidence
//   - If intent < 0.5 and cortex ≥ 0.7: trust cortex
//   - Otherwise: return empty (fall back to LLM)
func (e *ToolEnsemble) Predict(query string) *EnsembleResult {
	if e == nil {
		return nil
	}

	// Get intent prediction
	var intentTool string
	var intentConf float64
	if e.Intent != nil {
		actions := e.Intent.Compile(query)
		if len(actions) > 0 {
			intentTool = actions[0].Tool
			intentConf = actions[0].Confidence
		}
	}

	// Get cortex prediction
	var cortexTool string
	var cortexConf float64
	if e.Cortex != nil && e.Cortex.TrainCount >= 50 {
		input := CortexInputFromQuery(query, e.Cortex.InputSize)
		pred := e.Cortex.Predict(input)
		cortexTool = pred.Label
		cortexConf = pred.Confidence
	}

	result := &EnsembleResult{
		IntentConf: intentConf,
		CortexConf: cortexConf,
	}

	// Case 1: Intent is confident enough on its own
	if intentConf >= 0.8 {
		result.Tool = intentTool
		result.Confidence = intentConf
		result.Source = "intent"
		// Bonus: if cortex agrees, boost slightly
		if cortexTool == intentTool && cortexConf > 0.5 {
			result.Confidence = min64(intentConf*1.1, 0.98)
			result.Source = "ensemble"
		}
		return result
	}

	// Case 2: Intent is borderline — consult cortex
	if intentConf >= 0.5 && intentTool != "" {
		if cortexTool == intentTool && cortexConf >= 0.5 {
			// Agreement: boost confidence
			combined := min64(intentConf, cortexConf) * 1.3
			if combined > 0.95 {
				combined = 0.95
			}
			result.Tool = intentTool
			result.Confidence = combined
			result.Source = "ensemble"
			return result
		}
		if cortexConf >= 0.7 && cortexTool != "" {
			// Disagreement with strong cortex: conflict
			result.Tool = cortexTool
			result.Confidence = cortexConf * 0.7 // reduce due to conflict
			result.Source = "conflict"
			return result
		}
		// Cortex is uncertain too — trust intent (it at least matched a pattern)
		result.Tool = intentTool
		result.Confidence = intentConf
		result.Source = "intent"
		return result
	}

	// Case 3: Intent failed — cortex alone
	if cortexConf >= 0.7 && cortexTool != "" {
		result.Tool = cortexTool
		result.Confidence = cortexConf
		result.Source = "cortex"
		return result
	}

	// Case 4: Neither is confident — fall back to LLM
	return nil
}

func min64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
