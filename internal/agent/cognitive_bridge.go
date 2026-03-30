package agent

import (
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/cognitive"
)

// CognitiveBridge connects the autonomous agent to Nous's cognitive systems.
// It gives the agent the ability to THINK — summarize, reason, generate
// documents, and compose natural language — not just run tools.
type CognitiveBridge struct {
	Summarizer *cognitive.Summarizer
	Thinker    *cognitive.ThinkingEngine
	DeepReason *cognitive.DeepReasoner
	DocGen     *cognitive.DocumentGenerator
	Composer   *cognitive.Composer
	NLG        *cognitive.NLGEngine
	Graph      *cognitive.CognitiveGraph
}

// NewCognitiveBridge creates a bridge from an ActionRouter's cognitive systems.
// Returns nil if the ActionRouter has no cognitive systems wired.
func NewCognitiveBridge(ar *cognitive.ActionRouter) *CognitiveBridge {
	if ar == nil {
		return nil
	}
	return &CognitiveBridge{
		Summarizer: ar.Summarizer,
		Thinker:    ar.Thinker,
		DeepReason: ar.DeepReason,
		DocGen:     ar.DocGen,
		Composer:   ar.Composer,
		NLG:        ar.NLG,
		Graph:      ar.CogGraph,
	}
}

// Summarize condenses text to the given number of sentences.
// Falls back to simple truncation if the summarizer is not available.
func (cb *CognitiveBridge) Summarize(text string, maxSentences int) string {
	if cb == nil || cb.Summarizer == nil {
		return truncateBridge(text, 500)
	}
	return cb.Summarizer.Summarize(text, maxSentences)
}

// SummarizeToLength condenses text to approximately maxWords words.
func (cb *CognitiveBridge) SummarizeToLength(text string, maxWords int) string {
	if cb == nil || cb.Summarizer == nil {
		return truncateBridge(text, maxWords*6) // rough chars estimate
	}
	return cb.Summarizer.SummarizeToLength(text, maxWords)
}

// Think uses the ThinkingEngine to generate a thoughtful response to a query.
// Returns empty string if the engine is not available or can't handle the query.
func (cb *CognitiveBridge) Think(query string) string {
	if cb == nil || cb.Thinker == nil {
		return ""
	}
	result := cb.Thinker.Think(query, nil)
	if result == nil {
		return ""
	}
	return result.Text
}

// Reason performs deep multi-step reasoning on a question.
// Returns the answer and reasoning trace, or empty strings if unavailable.
func (cb *CognitiveBridge) Reason(question string) (answer string, trace string) {
	if cb == nil || cb.DeepReason == nil {
		return "", ""
	}
	result := cb.DeepReason.Reason(question)
	if result == nil {
		return "", ""
	}
	return result.FinalAnswer, result.Trace
}

// GenerateDocument creates a structured multi-section document about a topic.
// style: "overview", "report", "essay", or "guide".
// Returns the formatted document text, or empty string if unavailable.
func (cb *CognitiveBridge) GenerateDocument(topic, style string) string {
	if cb == nil || cb.DocGen == nil {
		return ""
	}
	doc := cb.DocGen.Generate(topic, style)
	if doc == nil || len(doc.Sections) == 0 {
		return ""
	}
	return formatDocument(doc)
}

// Compose generates a natural language response for a query.
// Returns the composed text, or empty string if unavailable.
func (cb *CognitiveBridge) Compose(query string) string {
	if cb == nil || cb.Composer == nil {
		return ""
	}
	resp := cb.Composer.Compose(query, cognitive.RespFactual, nil)
	if resp == nil || resp.Text == "" {
		return ""
	}
	return resp.Text
}

// SynthesizeResults takes raw tool outputs and produces a coherent summary.
// This is the key function that turns the agent from a checklist runner
// into a thinking worker: it reads multiple raw results and writes a
// synthesis that a human would actually want to read.
func (cb *CognitiveBridge) SynthesizeResults(goal string, results map[string]string) string {
	if cb == nil {
		return concatenateResults(results)
	}

	// 1. Concatenate all results into raw material
	var raw strings.Builder
	for id, result := range results {
		fmt.Fprintf(&raw, "--- %s ---\n%s\n\n", id, result)
	}
	rawText := raw.String()

	// 2. Summarize the raw material
	summary := cb.Summarize(rawText, 10)

	// 3. Try to think about it in the context of the goal
	thought := cb.Think("Based on this information, " + goal + ": " + summary)
	if thought != "" {
		return thought
	}

	// 4. Try composing a response
	composed := cb.Compose("Summarize findings about: " + goal)
	if composed != "" {
		return composed
	}

	// 5. Fallback to the summary
	if summary != "" {
		return summary
	}

	return concatenateResults(results)
}

// WriteReport generates a full document from accumulated results.
// This is used at the end of a research or analysis goal to produce
// the final deliverable.
func (cb *CognitiveBridge) WriteReport(topic, style string, results map[string]string) string {
	// Try document generator first
	doc := cb.GenerateDocument(topic, style)
	if doc != "" {
		return doc
	}

	// Fallback: synthesize from results
	return cb.SynthesizeResults("write a "+style+" about "+topic, results)
}

// formatDocument converts a GeneratedDocument into readable markdown text.
func formatDocument(doc *cognitive.GeneratedDocument) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s\n\n", doc.Title)
	for _, sec := range doc.Sections {
		fmt.Fprintf(&b, "## %s\n\n%s\n\n", sec.Heading, sec.Content)
	}
	return b.String()
}

// concatenateResults joins all results into a simple text block.
func concatenateResults(results map[string]string) string {
	var parts []string
	for _, v := range results {
		if v != "" {
			parts = append(parts, v)
		}
	}
	return strings.Join(parts, "\n\n")
}

// truncateBridge shortens text to maxLen characters.
func truncateBridge(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
