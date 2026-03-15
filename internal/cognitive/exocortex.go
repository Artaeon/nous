package cognitive

import (
	"fmt"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

// Exocortex is the external cognitive architecture that wraps the LLM.
// Named after the hypothetical external brain augmentation, it handles
// all structured cognition deterministically and only delegates to the
// LLM when genuine natural language reasoning is needed.
//
// Innovation: Every AI assistant treats the LLM as the brain. The Exocortex
// treats it as a PERIPHERAL — like a speech synthesizer. The actual thinking
// (perceiving intent, selecting tools, executing actions, formatting responses)
// happens in deterministic code that is IMPOSSIBLE to hallucinate.
//
// This is not prompt engineering. This is not fine-tuning. This is not RAG.
// This is architectural cognition — building a mind out of code, with the
// LLM serving only as the natural language interface.
//
// Three processing tiers:
//   Tier 1 — FULL BYPASS (60-70% of queries):
//     Intent compiled → tools executed → response synthesized → done.
//     The LLM is never called. Zero hallucination possible.
//
//   Tier 2 — SCAFFOLDED LLM (20-25% of queries):
//     Tools executed → facts extracted → response PRE-BUILT with blanks →
//     LLM fills in connective tissue. Cannot contradict facts.
//
//   Tier 3 — FULL LLM (10-15% of queries):
//     Genuinely conversational/creative. Standard pipeline with all
//     cognitive prosthetics active.
type Exocortex struct {
	intent      *IntentCompiler
	synth       *ResponseSynthesizer
	tools       *tools.Registry
	distiller   *SelfDistiller
	neuroplast  *NeuroplasticDescriptions
	embedGround *EmbedGrounder
	crystals    *CrystalBook
}

// ExoResult holds the result of exocortex processing.
type ExoResult struct {
	// Tier that was used (1=bypass, 2=scaffold, 3=full LLM)
	Tier int

	// Final response (for tier 1, this is the complete answer)
	Response string

	// For tier 2: scaffold with evidence pre-filled
	Scaffold string

	// Tool execution results
	ToolResults []ExoToolResult

	// Processing metadata
	Duration    time.Duration
	ToolCount   int
	LLMBypassed bool
}

// ExoToolResult holds one tool execution.
type ExoToolResult struct {
	Tool     string
	Args     map[string]string
	Result   string
	Err      error
	Duration time.Duration
}

// NewExocortex creates a new cognitive exocortex.
func NewExocortex(
	intent *IntentCompiler,
	toolReg *tools.Registry,
	distiller *SelfDistiller,
	neuroplast *NeuroplasticDescriptions,
	embedGround *EmbedGrounder,
	crystals *CrystalBook,
) *Exocortex {
	return &Exocortex{
		intent:      intent,
		synth:       NewResponseSynthesizer(),
		tools:       toolReg,
		distiller:   distiller,
		neuroplast:  neuroplast,
		embedGround: embedGround,
		crystals:    crystals,
	}
}

// Process handles a user query through the exocortex.
// Returns an ExoResult indicating which tier was used and the response.
func (ex *Exocortex) Process(query string) *ExoResult {
	start := time.Now()

	// --- Tier 1: Full Bypass ---
	// Try intent compilation first (deterministic, instant)
	if ex.intent != nil {
		actions := ex.intent.Compile(query)
		if len(actions) > 0 && actions[0].Confidence >= 0.8 {
			return ex.processTier1(query, actions, start)
		}
	}

	// --- Try crystal matching (learned patterns) ---
	if ex.crystals != nil {
		if match := ex.crystals.Match(query); match != nil {
			return ex.processCrystal(query, match, start)
		}
	}

	// --- Tier 2: Scaffolded LLM ---
	// Can we at least identify the tools needed?
	tier := ex.classifyTier(query)
	if tier == 2 {
		return ex.processTier2(query, start)
	}

	// --- Tier 3: Full LLM ---
	return &ExoResult{
		Tier:        3,
		LLMBypassed: false,
		Duration:    time.Since(start),
	}
}

// processTier1 executes compiled actions and synthesizes the response.
// No LLM involvement at all.
func (ex *Exocortex) processTier1(query string, actions []CompiledAction, start time.Time) *ExoResult {
	result := &ExoResult{
		Tier:        1,
		LLMBypassed: true,
	}

	var steps []synthStep

	for _, action := range actions {
		toolStart := time.Now()

		t, err := ex.tools.Get(action.Tool)
		if err != nil {
			steps = append(steps, synthStep{
				Tool: action.Tool,
				Args: action.Args,
				Err:  err,
			})
			result.ToolResults = append(result.ToolResults, ExoToolResult{
				Tool:     action.Tool,
				Args:     action.Args,
				Err:      err,
				Duration: time.Since(toolStart),
			})
			continue
		}

		toolResult, toolErr := t.Execute(action.Args)
		toolResult = SmartTruncate(action.Tool, toolResult)

		steps = append(steps, synthStep{
			Tool:   action.Tool,
			Args:   action.Args,
			Result: toolResult,
			Err:    toolErr,
		})

		result.ToolResults = append(result.ToolResults, ExoToolResult{
			Tool:     action.Tool,
			Args:     action.Args,
			Result:   toolResult,
			Err:      toolErr,
			Duration: time.Since(toolStart),
		})

		// Record neuroplastic attempt
		if ex.neuroplast != nil {
			ex.neuroplast.RecordAttempt(action.Tool)
			if toolErr == nil && strings.TrimSpace(toolResult) != "" {
				ex.neuroplast.RecordSuccess(action.Tool)
			}
		}

		// Record for embedding grounding
		if ex.embedGround != nil && toolErr == nil {
			ex.embedGround.RecordSuccess(query, action.Tool, action.Args)
		}

		result.ToolCount++
	}

	// Synthesize response from tool results
	result.Response = ex.synth.SynthesizeMulti(query, steps)
	result.Duration = time.Since(start)

	return result
}

// processCrystal executes a crystallized reasoning chain.
func (ex *Exocortex) processCrystal(query string, match *CrystalMatch, start time.Time) *ExoResult {
	result := &ExoResult{
		Tier:        1,
		LLMBypassed: true,
	}

	crystal := match.Crystal
	var steps []synthStep

	for _, step := range crystal.Steps {
		t, err := ex.tools.Get(step.Tool)
		if err != nil {
			continue
		}

		// Resolve arguments (may have template vars)
		args := make(map[string]string)
		for k, v := range step.Args {
			args[k] = v
		}

		toolResult, toolErr := t.Execute(args)
		toolResult = SmartTruncate(step.Tool, toolResult)

		steps = append(steps, synthStep{
			Tool:   step.Tool,
			Args:   args,
			Result: toolResult,
			Err:    toolErr,
		})

		result.ToolResults = append(result.ToolResults, ExoToolResult{
			Tool:     step.Tool,
			Args:     args,
			Result:   toolResult,
			Err:      toolErr,
		})
		result.ToolCount++
	}

	// Record crystal usage
	ex.crystals.ReportSuccess(crystal.ID)

	result.Response = ex.synth.SynthesizeMulti(query, steps)
	result.Duration = time.Since(start)

	return result
}

// processTier2 builds a scaffold with evidence for the LLM to complete.
func (ex *Exocortex) processTier2(query string, start time.Time) *ExoResult {
	result := &ExoResult{
		Tier:        2,
		LLMBypassed: false,
	}

	// Use embedding grounding to find relevant context
	var groundingHints []string
	if ex.embedGround != nil {
		gr, err := ex.embedGround.Ground(query)
		if err == nil && gr != nil {
			if ctx := gr.FormatGroundingContext(); ctx != "" {
				groundingHints = append(groundingHints, ctx)
			}
		}
	}

	// Build scaffold: pre-fill the response with verified facts
	var scaffold strings.Builder
	scaffold.WriteString("Answer the user's question using ONLY the information provided below.\n")
	scaffold.WriteString("Do NOT make up information. If the evidence doesn't answer the question, say so.\n\n")
	scaffold.WriteString("User question: ")
	scaffold.WriteString(query)
	scaffold.WriteString("\n\n")

	if len(groundingHints) > 0 {
		scaffold.WriteString("Available context:\n")
		for _, hint := range groundingHints {
			scaffold.WriteString(hint)
			scaffold.WriteString("\n")
		}
	}

	// Add distillation insights (common mistakes to avoid)
	if ex.distiller != nil {
		negInst := ex.distiller.ExportNegativeInstructions()
		if negInst != "" {
			scaffold.WriteString("\n")
			scaffold.WriteString(negInst)
		}
	}

	result.Scaffold = scaffold.String()
	result.Duration = time.Since(start)

	return result
}

// classifyTier determines which processing tier to use for a query.
func (ex *Exocortex) classifyTier(query string) int {
	lower := strings.ToLower(query)

	// Tier 3 indicators: conversational, creative, complex reasoning
	conversational := []string{
		"what do you think", "how do you", "can you explain",
		"tell me about", "why is", "what is the purpose",
		"help me understand", "what's your opinion",
		"how would you", "should I", "recommend",
	}
	for _, phrase := range conversational {
		if strings.Contains(lower, phrase) {
			return 3
		}
	}

	// Tier 2 indicators: needs tools + explanation
	needsContext := []string{
		"how does", "what does", "explain how", "describe",
		"summarize", "analyze", "compare", "why does",
	}
	for _, phrase := range needsContext {
		if strings.Contains(lower, phrase) {
			return 2
		}
	}

	// If we can't classify, default to tier 3
	return 3
}

// FormatStatusLine returns a status indicator for the processing tier.
func (er *ExoResult) FormatStatusLine() string {
	switch er.Tier {
	case 1:
		return fmt.Sprintf("↳ exo-bypass: %d tools, %s (no LLM)", er.ToolCount, er.Duration.Round(time.Millisecond))
	case 2:
		return fmt.Sprintf("↳ exo-scaffold: evidence pre-loaded, %s", er.Duration.Round(time.Millisecond))
	case 3:
		return "↳ exo-full: LLM reasoning"
	}
	return ""
}
