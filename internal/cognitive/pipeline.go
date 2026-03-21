package cognitive

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

// StepResult holds the compressed outcome of one reasoning step.
type StepResult struct {
	StepNum   int
	ToolName  string
	Summary   string // one-line compressed summary of the tool result
	RawResult string // the actual result (kept for the current step only)
}

// Pipeline manages fresh-context reasoning across multiple steps.
// Instead of accumulating messages that fill the context window,
// each step gets a fresh LLM call with only:
// 1. A compact system prompt
// 2. The original user question
// 3. One-line summaries of all previous steps
// 4. The current tool result (if any)
type Pipeline struct {
	steps     []StepResult
	userQuery string
	distiller *ollama.Client // optional: fast model for thought distillation
}

// NewPipeline creates a pipeline for managing fresh-context reasoning.
func NewPipeline(query string) *Pipeline {
	return &Pipeline{
		userQuery: query,
	}
}

// SetDistiller configures an LLM client (typically the fast model) for
// thought distillation. When set, step summaries are produced by the LLM
// instead of rule-based compression, yielding much richer context for the
// reasoning model.
func (p *Pipeline) SetDistiller(llm *ollama.Client) {
	p.distiller = llm
}

// simpleTools are tools whose results can be summarized with rules, no LLM needed.
var simpleTools = map[string]bool{
	"read": true, "ls": true, "glob": true, "grep": true, "tree": true, "sysinfo": true,
}

// AddStep compresses and stores the result of a tool execution.
// For simple tools (read, ls, glob, grep), uses fast rule-based compression.
// For complex tools, tries LLM distillation with rule-based fallback.
func (p *Pipeline) AddStep(toolName, rawResult string) {
	summary := ""

	// Simple tools always use rule-based compression — no LLM call needed.
	// This eliminates ~3s of overhead per simple tool result.
	if simpleTools[toolName] {
		summary = CompressStep(toolName, rawResult)
	}

	// Try thought distillation for complex tools only
	if summary == "" && p.distiller != nil && len(rawResult) > 80 {
		if distilled := distillStep(p.distiller, p.userQuery, toolName, rawResult); distilled != "" {
			summary = distilled
		}
	}

	// Fall back to rule-based compression
	if summary == "" {
		summary = CompressStep(toolName, rawResult)
	}

	step := StepResult{
		StepNum:   len(p.steps) + 1,
		ToolName:  toolName,
		Summary:   summary,
		RawResult: rawResult,
	}
	// Clear RawResult from all previous steps to save memory
	for i := range p.steps {
		p.steps[i].RawResult = ""
	}
	p.steps = append(p.steps, step)
}

// distillStep uses the fast model to produce a semantically rich summary
// of a tool result. Returns empty string on failure (caller falls back to
// rule-based compression). Timeout: 3 seconds to avoid blocking.
func distillStep(llm *ollama.Client, userQuery, toolName, rawResult string) string {
	if llm == nil {
		return ""
	}

	// Truncate very large results before sending to distiller
	result := rawResult
	if len(result) > 1500 {
		result = result[:1500] + "\n... (truncated)"
	}

	prompt := fmt.Sprintf(`Tool "%s" returned this result. Summarize what it means for the task in one sentence (max 40 words). Focus on key content and findings, not metadata.

Task: %s

Result:
%s

Summary:`, toolName, userQuery, result)

	// Use a channel + goroutine for timeout control
	type resp struct {
		text string
		err  error
	}
	ch := make(chan resp, 1)
	go func() {
		r, err := llm.Chat([]ollama.Message{
			{Role: "user", Content: prompt},
		}, &ollama.ModelOptions{
			Temperature: 0.1,
			NumPredict:  60,
		})
		if err != nil {
			ch <- resp{err: err}
			return
		}
		ch <- resp{text: r.Message.Content}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return ""
		}
		// Clean up: take first line, trim whitespace
		summary := strings.TrimSpace(r.text)
		if idx := strings.IndexByte(summary, '\n'); idx > 0 {
			summary = strings.TrimSpace(summary[:idx])
		}
		// Sanity: if distillation returned garbage or too long, reject
		if len(summary) < 5 || len(summary) > 200 {
			return ""
		}
		return fmt.Sprintf("[%s] %s", toolName, summary)
	case <-time.After(3 * time.Second):
		return "" // timeout — fall back to rule-based
	}
}

// CompressStep applies rule-based compression to produce a one-line summary.
// This is purely rule-based — no LLM call — to keep it fast and deterministic.
func CompressStep(toolName, rawResult string) string {
	// Handle errors first
	if strings.HasPrefix(rawResult, "Error:") || strings.HasPrefix(rawResult, "error:") {
		first := firstLine(rawResult)
		return "Error: " + first
	}

	lines := strings.Split(rawResult, "\n")

	switch toolName {
	case "read":
		// "Read FILE_PATH: FIRST_LINE... (N lines)"
		firstMeaningful := firstNonEmptyLine(lines)
		if len(firstMeaningful) > 60 {
			firstMeaningful = firstMeaningful[:60] + "..."
		}
		path := extractPath(rawResult)
		if path != "" {
			return fmt.Sprintf("Read %s: %s (%d lines)", path, firstMeaningful, len(lines))
		}
		return fmt.Sprintf("Read file: %s (%d lines)", firstMeaningful, len(lines))

	case "ls", "tree":
		// "Listed DIR: N entries including [first 3 names]..."
		entries := nonEmptyLines(lines)
		names := firstN(entries, 3)
		dir := extractDir(rawResult)
		if dir == "" {
			dir = "directory"
		}
		return fmt.Sprintf("Listed %s: %d entries including %s", dir, len(entries), strings.Join(names, ", "))

	case "grep":
		// "Searched PATTERN: N matches in [files]"
		matches := nonEmptyLines(lines)
		files := extractUniqueFiles(matches)
		fileStr := strings.Join(firstN(files, 3), ", ")
		if len(files) > 3 {
			fileStr += fmt.Sprintf(" (+%d more)", len(files)-3)
		}
		return fmt.Sprintf("Searched: %d matches in %s", len(matches), fileStr)

	case "glob":
		// "Found N files matching PATTERN"
		entries := nonEmptyLines(lines)
		return fmt.Sprintf("Found %d files matching pattern", len(entries))

	case "git":
		// "Git COMMAND: FIRST_LINE_OF_OUTPUT"
		first := firstNonEmptyLine(lines)
		if len(first) > 80 {
			first = first[:80] + "..."
		}
		return fmt.Sprintf("Git: %s", first)

	case "write", "edit", "patch", "find_replace", "replace_all":
		// "Modified FILE_PATH"
		path := extractPath(rawResult)
		if path != "" {
			return fmt.Sprintf("Modified %s", path)
		}
		return "Modified file"

	case "sysinfo":
		// "System: OS ARCH CPUs"
		first := firstNonEmptyLine(lines)
		if len(first) > 80 {
			first = first[:80] + "..."
		}
		return fmt.Sprintf("System: %s", first)

	default:
		// Default: first 80 chars of result
		trimmed := strings.TrimSpace(rawResult)
		if len(trimmed) <= 80 {
			return trimmed
		}
		return trimmed[:80] + "..."
	}
}

// BuildContext returns the accumulated step summaries as a formatted block.
// Returns empty string if no steps have been recorded.
func (p *Pipeline) BuildContext() string {
	if len(p.steps) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("[Previous steps]\n")
	for _, s := range p.steps {
		sb.WriteString(fmt.Sprintf("%d. %s\n", s.StepNum, s.Summary))
	}
	return strings.TrimRight(sb.String(), "\n")
}

// StepCount returns the number of steps completed so far.
func (p *Pipeline) StepCount() int {
	return len(p.steps)
}

// LastResult returns the raw result of the most recent step.
// Returns empty string if no steps exist.
func (p *Pipeline) LastResult() string {
	if len(p.steps) == 0 {
		return ""
	}
	return p.steps[len(p.steps)-1].RawResult
}

// --- helper functions ---

func firstLine(s string) string {
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		return strings.TrimSpace(s[:idx])
	}
	return strings.TrimSpace(s)
}

func firstNonEmptyLine(lines []string) string {
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func nonEmptyLines(lines []string) []string {
	var result []string
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			result = append(result, strings.TrimSpace(l))
		}
	}
	return result
}

func firstN(items []string, n int) []string {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

// extractPath tries to find a file path in a tool result.
// Many tool results start with the path or contain "path:" references.
func extractPath(result string) string {
	lines := strings.Split(result, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Look for common path patterns
		if strings.Contains(trimmed, "/") || strings.Contains(trimmed, ".go") ||
			strings.Contains(trimmed, ".py") || strings.Contains(trimmed, ".js") ||
			strings.Contains(trimmed, ".ts") || strings.Contains(trimmed, ".md") {
			// Extract just the path-like portion
			words := strings.Fields(trimmed)
			for _, w := range words {
				if strings.Contains(w, "/") || strings.Contains(w, ".") {
					// Clean up common prefixes/suffixes
					w = strings.TrimRight(w, ":")
					if filepath.Ext(w) != "" || strings.Contains(w, "/") {
						return w
					}
				}
			}
		}
	}
	return ""
}

// extractDir tries to find a directory reference in a tool result.
func extractDir(result string) string {
	path := extractPath(result)
	if path != "" {
		if strings.HasSuffix(path, "/") {
			return path
		}
		dir := filepath.Dir(path)
		if dir != "." {
			return dir
		}
	}
	return ""
}

// extractUniqueFiles pulls file paths from grep-style output (path:line:content).
func extractUniqueFiles(lines []string) []string {
	seen := make(map[string]bool)
	var files []string
	for _, line := range lines {
		// grep output format: "file:line:content" or "file:content"
		if idx := strings.IndexByte(line, ':'); idx > 0 {
			candidate := line[:idx]
			if !seen[candidate] && (strings.Contains(candidate, "/") ||
				strings.Contains(candidate, ".")) {
				seen[candidate] = true
				files = append(files, candidate)
			}
		}
	}
	if len(files) == 0 {
		return []string{"(inline)"}
	}
	return files
}

// -----------------------------------------------------------------------
// Reasoning Pipeline — orchestrates all cognitive engines for a single
// query. Runs graph lookup, inference, multi-hop reasoning, causal
// analysis, analogy, and thinking in sequence. Each step is nil-safe.
// -----------------------------------------------------------------------

// ReasoningPipeline ties all cognitive engines together for query processing.
type ReasoningPipeline struct {
	Graph     *CognitiveGraph
	Inference *InferenceEngine
	Reasoner  *ReasoningEngine
	Thinker   *ThinkingEngine
	Causal    *GraphCausalReasoner
	Composer  *Composer
	Semantic  *SemanticEngine
	Analogy   *AnalogyEngine
}

// PipelineResult holds the combined output from all cognitive engines.
type PipelineResult struct {
	DirectFacts    []string // from graph query
	InferredFacts  []string // from inference engine at query time
	ReasoningTrace string   // from reasoning chain
	CausalTrace    string   // from causal reasoning
	AnalogyTrace   string   // from analogy engine
	ThinkingResult string   // from thinking engine
	Confidence     float64
	Sources        []string // which engines contributed
}

// NewReasoningPipeline creates a pipeline. All parameters are optional (nil-safe).
func NewReasoningPipeline(
	graph *CognitiveGraph,
	inference *InferenceEngine,
	reasoner *ReasoningEngine,
	thinker *ThinkingEngine,
	causal *GraphCausalReasoner,
	composer *Composer,
	semantic *SemanticEngine,
	analogy *AnalogyEngine,
) *ReasoningPipeline {
	return &ReasoningPipeline{
		Graph:     graph,
		Inference: inference,
		Reasoner:  reasoner,
		Thinker:   thinker,
		Causal:    causal,
		Composer:  composer,
		Semantic:  semantic,
		Analogy:   analogy,
	}
}

// Process runs the full reasoning pipeline for a query.
func (rp *ReasoningPipeline) Process(query string) *PipelineResult {
	result := &PipelineResult{}
	qtype := rp.classifyQuestion(query)

	// Step 1: Graph query — extract direct facts and activated node IDs.
	var activatedIDs []string
	if rp.Graph != nil {
		ga := rp.Graph.Query(query)
		if ga != nil {
			result.DirectFacts = ga.DirectFacts
			result.InferredFacts = ga.InferredFacts
			result.Confidence = ga.Confidence
			result.Sources = append(result.Sources, "graph")

			// Collect activated node IDs for targeted inference.
			active := rp.Graph.MostActive(10)
			for _, n := range active {
				activatedIDs = append(activatedIDs, n.ID)
			}
		}
	}

	// Early exit: if we have strong graph facts for a simple factual question,
	// skip expensive engines unless more depth is needed.
	earlyExit := qtype == "factual" && len(result.DirectFacts) >= 3 && result.Confidence >= 0.8

	// Step 2: Targeted inference on activated nodes.
	if !earlyExit && rp.Inference != nil && len(activatedIDs) > 0 {
		inferences := rp.Inference.InferAt(activatedIDs)
		for _, inf := range inferences {
			fact := fmt.Sprintf("%s %s %s", inf.Subject, inf.Relation, inf.Object)
			result.InferredFacts = append(result.InferredFacts, fact)
		}
		if len(inferences) > 0 {
			result.Sources = append(result.Sources, "inference")
		}
	}

	// Step 3: Multi-hop reasoning.
	if !earlyExit && rp.Reasoner != nil {
		chain := rp.Reasoner.Reason(query)
		if chain != nil && chain.Answer != "" {
			result.ReasoningTrace = chain.Trace
			if chain.Confidence > result.Confidence {
				result.Confidence = chain.Confidence
			}
			result.Sources = append(result.Sources, "reasoning")

			// If we had no graph facts, promote the reasoning answer.
			if len(result.DirectFacts) == 0 {
				result.DirectFacts = append(result.DirectFacts, chain.Answer)
			}
		}
	}

	// Step 4: Causal reasoning for why/what-if questions.
	if rp.Causal != nil && (qtype == "causal" || qtype == "creative") {
		result.CausalTrace = rp.runCausalAnalysis(query)
		if result.CausalTrace != "" {
			result.Sources = append(result.Sources, "causal")
		}
	}

	// Step 5: Analogy for similarity/comparison questions.
	if rp.Analogy != nil && qtype == "comparative" {
		result.AnalogyTrace = rp.runAnalogyAnalysis(query)
		if result.AnalogyTrace != "" {
			result.Sources = append(result.Sources, "analogy")
		}
	}

	// Step 6: Thinking engine for complex tasks.
	if !earlyExit && rp.Thinker != nil {
		if qtype == "analytical" || qtype == "comparative" || qtype == "creative" {
			thought := rp.Thinker.Think(query, nil)
			if thought != nil && thought.Text != "" {
				result.ThinkingResult = thought.Text
				result.Sources = append(result.Sources, "thinking")
			}
		}
	}

	return result
}

// ComposeResponse turns a PipelineResult into a coherent natural language response.
func (rp *ReasoningPipeline) ComposeResponse(query string, result *PipelineResult) string {
	if result == nil {
		return ""
	}

	var parts []string

	// Priority 1: ThinkingResult if substantial.
	if len(result.ThinkingResult) > 50 {
		parts = append(parts, result.ThinkingResult)
	}

	// Priority 2: Reasoning trace answer.
	if len(parts) == 0 && result.ReasoningTrace != "" {
		parts = append(parts, result.ReasoningTrace)
	}

	// Priority 3: Direct + inferred facts — use prose Composer when available.
	if len(parts) == 0 && (len(result.DirectFacts) > 0 || len(result.InferredFacts) > 0) {
		// Try prose Composer first for natural language output.
		if rp.Composer != nil && rp.Composer.Graph != nil {
			resp := rp.Composer.Compose(query, RespFactual, nil)
			if resp != nil && resp.Text != "" {
				parts = append(parts, resp.Text)
			}
		}

		// Fallback: structured fact sentences via graph ComposeAnswer.
		if len(parts) == 0 && rp.Graph != nil {
			ga := &GraphAnswer{
				DirectFacts:   result.DirectFacts,
				InferredFacts: result.InferredFacts,
				Confidence:    result.Confidence,
			}
			composed := rp.Graph.ComposeAnswer(query, ga)
			if composed != "" {
				parts = append(parts, composed)
			}
		}

		// Last resort: join facts directly.
		if len(parts) == 0 {
			all := make([]string, 0, len(result.DirectFacts)+len(result.InferredFacts))
			all = append(all, result.DirectFacts...)
			all = append(all, result.InferredFacts...)
			for i := range all {
				all[i] = ensurePeriod(all[i])
			}
			parts = append(parts, strings.Join(all, " "))
		}
	}

	// Append causal trace as supplementary paragraph.
	if result.CausalTrace != "" {
		parts = append(parts, result.CausalTrace)
	}

	// Append analogy trace as supplementary paragraph.
	if result.AnalogyTrace != "" {
		parts = append(parts, result.AnalogyTrace)
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// -----------------------------------------------------------------------
// Question classification
// -----------------------------------------------------------------------

// classifyQuestion categorizes a query to decide which engines to prioritize.
func (rp *ReasoningPipeline) classifyQuestion(query string) string {
	lower := strings.ToLower(query)

	// Causal — why / what-if / cause / effect
	causalWords := []string{"why ", "cause", "effect", "what if", "what would happen", "implications"}
	for _, w := range causalWords {
		if strings.Contains(lower, w) {
			return "causal"
		}
	}

	// Comparative — compare / contrast / difference / similar / like / analogy
	compWords := []string{"compare", "contrast", "difference", "similar", " like ", "analogy", "versus", " vs "}
	for _, w := range compWords {
		if strings.Contains(lower, w) {
			return "comparative"
		}
	}

	// Creative — imagine / suppose / brainstorm
	creativeWords := []string{"imagine", "suppose", "brainstorm", "pretend", "what would"}
	for _, w := range creativeWords {
		if strings.Contains(lower, w) {
			return "creative"
		}
	}

	// Analytical — how does / explain / teach / analyze
	analyticalWords := []string{"how does", "how do", "explain", "teach", "analyze", "analyse", "break down"}
	for _, w := range analyticalWords {
		if strings.Contains(lower, w) {
			return "analytical"
		}
	}

	// Default: factual
	return "factual"
}

// -----------------------------------------------------------------------
// Engine-specific runners
// -----------------------------------------------------------------------

// runCausalAnalysis runs causal reasoning appropriate to the query.
func (rp *ReasoningPipeline) runCausalAnalysis(query string) string {
	lower := strings.ToLower(query)

	// "what if" / "what would happen" → WhatIf
	if strings.Contains(lower, "what if") || strings.Contains(lower, "what would happen") {
		hypothesis := query
		for _, prefix := range []string{"what would happen if ", "what if "} {
			idx := strings.Index(lower, prefix)
			if idx >= 0 {
				hypothesis = strings.TrimSpace(query[idx+len(prefix):])
				break
			}
		}
		hypothesis = strings.TrimRight(hypothesis, "?!.")
		result := rp.Causal.WhatIf(hypothesis)
		if result != nil {
			return result.Trace
		}
		return ""
	}

	// "without" → WhatIfRemoved
	if strings.Contains(lower, "without") {
		entity := query
		idx := strings.Index(lower, "without ")
		if idx >= 0 {
			entity = strings.TrimSpace(query[idx+len("without "):])
			entity = strings.TrimRight(entity, "?!.")
		}
		result := rp.Causal.WhatIfRemoved(entity)
		if result != nil {
			return result.Trace
		}
		return ""
	}

	// "why" → forward trace from the subject
	for _, prefix := range []string{"why does ", "why do ", "why is ", "why are ", "why did ", "why "} {
		if strings.HasPrefix(lower, prefix) {
			subject := strings.TrimSpace(query[len(prefix):])
			subject = strings.TrimRight(subject, "?!.")
			result := rp.Causal.WhatIf(subject)
			if result != nil {
				return result.Trace
			}
			break
		}
	}

	return ""
}

// runAnalogyAnalysis runs analogy reasoning appropriate to the query.
func (rp *ReasoningPipeline) runAnalogyAnalysis(query string) string {
	lower := strings.ToLower(query)

	// Try to extract two terms for comparison.
	separators := []string{" is like ", " similar to ", " vs ", " versus ", " compared to "}
	for _, sep := range separators {
		idx := strings.Index(lower, sep)
		if idx > 0 {
			a := strings.TrimSpace(query[:idx])
			b := strings.TrimSpace(query[idx+len(sep):])
			b = strings.TrimRight(b, "?!.")
			// Strip leading question words from a.
			for _, prefix := range []string{"how is ", "is ", "what makes "} {
				if strings.HasPrefix(strings.ToLower(a), prefix) {
					a = strings.TrimSpace(a[len(prefix):])
					break
				}
			}
			result := rp.Analogy.FindAnalogy(a, b, "")
			if result != nil && result.Explanation != "" {
				return result.Explanation
			}
		}
	}

	// Fallback: apply principles to the primary entity.
	entity := extractPrimaryEntity(lower)
	if entity != "" {
		explanation := rp.Analogy.ApplyPrinciples(entity, query)
		if explanation != "" {
			return explanation
		}
	}

	return ""
}
