package cognitive

import "math"

// -----------------------------------------------------------------------
// Intent classification evaluation framework.
//
// Provides confusion matrix, per-intent precision/recall/F1, and macro/
// weighted F1 scores for evaluating intent classifiers. The evaluation
// set can be auto-generated from the NLU's word lists to ensure coverage
// of all intents.
// -----------------------------------------------------------------------

// ConfusionMatrix tracks predicted-vs-actual counts for multi-class
// classification evaluation.
type ConfusionMatrix struct {
	Labels   []string
	Matrix   [][]int // [predicted][actual]
	Total    int
	labelIdx map[string]int
}

// IntentEvalResult holds the full evaluation output.
type IntentEvalResult struct {
	Matrix     ConfusionMatrix
	Accuracy   float64
	PerIntent  map[string]IntentMetrics
	MacroF1    float64
	WeightedF1 float64
}

// IntentMetrics holds precision, recall, and F1 for one intent.
type IntentMetrics struct {
	Precision float64
	Recall    float64
	F1        float64
	Support   int // number of true examples
}

// IntentEvalExample is one labeled test example.
type IntentEvalExample struct {
	Input    string
	Expected string // expected intent
}

// NewConfusionMatrix creates a confusion matrix for the given labels.
func NewConfusionMatrix(labels []string) *ConfusionMatrix {
	n := len(labels)
	matrix := make([][]int, n)
	for i := range matrix {
		matrix[i] = make([]int, n)
	}
	idx := make(map[string]int, n)
	for i, l := range labels {
		idx[l] = i
	}
	return &ConfusionMatrix{
		Labels:   labels,
		Matrix:   matrix,
		labelIdx: idx,
	}
}

// Record records one prediction. If a label is unknown, it is added
// dynamically (expanding the matrix).
func (cm *ConfusionMatrix) Record(predicted, actual string) {
	// Ensure both labels exist
	cm.ensureLabel(predicted)
	cm.ensureLabel(actual)

	pi := cm.labelIdx[predicted]
	ai := cm.labelIdx[actual]
	cm.Matrix[pi][ai]++
	cm.Total++
}

// ensureLabel adds a label if not already present, expanding the matrix.
func (cm *ConfusionMatrix) ensureLabel(label string) {
	if _, ok := cm.labelIdx[label]; ok {
		return
	}

	idx := len(cm.Labels)
	cm.Labels = append(cm.Labels, label)
	cm.labelIdx[label] = idx

	// Expand existing rows
	for i := range cm.Matrix {
		cm.Matrix[i] = append(cm.Matrix[i], 0)
	}
	// Add new row
	newRow := make([]int, len(cm.Labels))
	cm.Matrix = append(cm.Matrix, newRow)
}

// Accuracy returns the overall accuracy (correct / total).
func (cm *ConfusionMatrix) Accuracy() float64 {
	if cm.Total == 0 {
		return 0
	}
	correct := 0
	for i := range cm.Labels {
		correct += cm.Matrix[i][i]
	}
	return float64(correct) / float64(cm.Total)
}

// PerIntentMetrics computes precision, recall, and F1 for each label.
func (cm *ConfusionMatrix) PerIntentMetrics() map[string]IntentMetrics {
	metrics := make(map[string]IntentMetrics, len(cm.Labels))

	for i, label := range cm.Labels {
		tp := cm.Matrix[i][i]

		// FP: other labels predicted as this label (sum of row i, minus diagonal)
		fp := 0
		for j := range cm.Labels {
			if j != i {
				fp += cm.Matrix[i][j]
			}
		}

		// FN: this label predicted as other labels (sum of column i, minus diagonal)
		fn := 0
		for j := range cm.Labels {
			if j != i {
				fn += cm.Matrix[j][i]
			}
		}

		// Support: total actual examples of this label (column sum)
		support := 0
		for j := range cm.Labels {
			support += cm.Matrix[j][i]
		}

		var precision, recall, f1 float64
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}
		if tp+fn > 0 {
			recall = float64(tp) / float64(tp+fn)
		}
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}

		metrics[label] = IntentMetrics{
			Precision: precision,
			Recall:    recall,
			F1:        f1,
			Support:   support,
		}
	}

	return metrics
}

// MacroF1 returns the unweighted average F1 across all intents.
func (cm *ConfusionMatrix) MacroF1() float64 {
	metrics := cm.PerIntentMetrics()
	if len(metrics) == 0 {
		return 0
	}

	sumF1 := 0.0
	count := 0
	for _, m := range metrics {
		if m.Support > 0 {
			sumF1 += m.F1
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sumF1 / float64(count)
}

// WeightedF1 returns the support-weighted average F1 across all intents.
func (cm *ConfusionMatrix) WeightedF1() float64 {
	metrics := cm.PerIntentMetrics()
	if len(metrics) == 0 {
		return 0
	}

	sumF1 := 0.0
	totalSupport := 0
	for _, m := range metrics {
		sumF1 += m.F1 * float64(m.Support)
		totalSupport += m.Support
	}
	if totalSupport == 0 {
		return 0
	}
	return sumF1 / float64(totalSupport)
}

// EvaluateIntentClassifier runs a classifier function against labeled
// examples and returns full evaluation metrics.
func EvaluateIntentClassifier(classify func(string) string, examples []IntentEvalExample) *IntentEvalResult {
	// Collect all unique labels
	labelSet := make(map[string]bool)
	for _, ex := range examples {
		labelSet[ex.Expected] = true
	}
	labels := make([]string, 0, len(labelSet))
	for l := range labelSet {
		labels = append(labels, l)
	}
	// Sort labels for deterministic output
	sortStrings(labels)

	cm := NewConfusionMatrix(labels)

	for _, ex := range examples {
		predicted := classify(ex.Input)
		cm.Record(predicted, ex.Expected)
	}

	perIntent := cm.PerIntentMetrics()

	return &IntentEvalResult{
		Matrix:     *cm,
		Accuracy:   cm.Accuracy(),
		PerIntent:  perIntent,
		MacroF1:    cm.MacroF1(),
		WeightedF1: cm.WeightedF1(),
	}
}

// GenerateIntentEvalSet produces labeled examples for evaluation.
// These are hand-curated examples designed to test boundary cases
// and ensure balanced intent coverage.
func GenerateIntentEvalSet() []IntentEvalExample {
	return []IntentEvalExample{
		// Greetings
		{Input: "hello there", Expected: "greeting"},
		{Input: "hey how are you", Expected: "greeting"},
		{Input: "hi nous", Expected: "greeting"},
		{Input: "howdy", Expected: "greeting"},

		// Farewells
		{Input: "goodbye", Expected: "farewell"},
		{Input: "see you later", Expected: "farewell"},
		{Input: "bye bye", Expected: "farewell"},
		{Input: "gotta go", Expected: "farewell"},

		// Meta
		{Input: "who are you", Expected: "meta"},
		{Input: "what can you do", Expected: "meta"},
		{Input: "are you an AI", Expected: "meta"},
		{Input: "what is your name", Expected: "meta"},

		// Weather (navigation)
		{Input: "what's the weather like", Expected: "weather"},
		{Input: "is it going to rain today", Expected: "weather"},
		{Input: "temperature outside", Expected: "weather"},
		{Input: "weather forecast for tomorrow", Expected: "weather"},

		// Calculator (navigation)
		{Input: "calculate 25 times 4", Expected: "calculate"},
		{Input: "what is 15% of 200", Expected: "calculate"},
		{Input: "compute the square root of 144", Expected: "calculate"},
		{Input: "solve 100 divided by 7", Expected: "calculate"},

		// Timer (navigation)
		{Input: "set a timer for 5 minutes", Expected: "timer"},
		{Input: "start a pomodoro timer", Expected: "timer"},
		{Input: "how much time is left", Expected: "timer"},

		// Explain (query)
		{Input: "explain how photosynthesis works", Expected: "explain"},
		{Input: "what is quantum physics", Expected: "explain"},
		{Input: "tell me about the roman empire", Expected: "explain"},
		{Input: "how does gravity work", Expected: "explain"},

		// Question (query)
		{Input: "is the earth flat", Expected: "question"},
		{Input: "how far is the moon", Expected: "question"},
		{Input: "why do we dream", Expected: "question"},

		// Compare (query)
		{Input: "python vs golang", Expected: "compare"},
		{Input: "what's the difference between RAM and ROM", Expected: "compare"},
		{Input: "compare electric cars to gas cars", Expected: "compare"},

		// Creative (task)
		{Input: "write me a poem about the ocean", Expected: "creative"},
		{Input: "tell me a joke", Expected: "creative"},
		{Input: "compose a haiku about autumn", Expected: "creative"},

		// Recommendation (task)
		{Input: "suggest a good book to read", Expected: "recommendation"},
		{Input: "recommend a movie for tonight", Expected: "recommendation"},

		// Transform (task)
		{Input: "summarize this article", Expected: "transform"},
		{Input: "rewrite this more formally", Expected: "transform"},

		// Remember (task)
		{Input: "remember my name is Raphael", Expected: "remember"},
		{Input: "my favorite color is blue", Expected: "remember"},

		// Recall (query)
		{Input: "what's my name", Expected: "recall"},
		{Input: "what do you know about me", Expected: "recall"},

		// Conversation
		{Input: "I had a great day today", Expected: "conversation"},
		{Input: "I'm going to the gym later", Expected: "conversation"},

		// Affirmation
		{Input: "thanks", Expected: "affirmation"},
		{Input: "perfect", Expected: "affirmation"},
		{Input: "ok", Expected: "affirmation"},

		// Dictionary (navigation)
		{Input: "define serendipity", Expected: "dict"},
		{Input: "synonyms for happy", Expected: "dict"},

		// Translate (navigation)
		{Input: "translate hello to french", Expected: "translate"},
		{Input: "how do you say thank you in japanese", Expected: "translate"},

		// Reminder (navigation)
		{Input: "remind me to call mom tomorrow", Expected: "reminder"},
		{Input: "set a reminder for 5pm", Expected: "reminder"},

		// Todo (navigation)
		{Input: "add buy milk to my todo list", Expected: "todo"},
		{Input: "show my tasks", Expected: "todo"},

		// Notes (navigation)
		{Input: "save a note about the meeting", Expected: "note"},
		{Input: "show my notes", Expected: "note"},

		// Convert (navigation)
		{Input: "convert 100 miles to km", Expected: "convert"},
		{Input: "32 fahrenheit in celsius", Expected: "convert"},
	}
}

// sortStrings sorts a string slice in ascending order (simple insertion sort
// to avoid importing sort package and keeping zero dependencies within cognitive).
func sortStrings(ss []string) {
	for i := 1; i < len(ss); i++ {
		key := ss[i]
		j := i - 1
		for j >= 0 && ss[j] > key {
			ss[j+1] = ss[j]
			j--
		}
		ss[j+1] = key
	}
}

// absFloat returns the absolute value of a float64.
// We use this to avoid depending on math.Abs in trivial cases,
// though we do import math for NaN checks elsewhere.
var _ = math.IsNaN // ensure math import is used
