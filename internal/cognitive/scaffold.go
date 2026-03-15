package cognitive

import (
	"fmt"
	"strings"
)

// NeuralScaffold pre-builds a response structure with verified facts,
// leaving only natural language "connective tissue" for the LLM to fill.
// This is like giving a student a fill-in-the-blank worksheet instead
// of an essay prompt — the facts are locked in, the model just makes
// them sound natural.
//
// Innovation: Standard prompting says "here are results, summarize them."
// The model then ignores 80% of the facts and hallucinates the rest.
// Neural Scaffolding inverts this: the response is PRE-BUILT from facts,
// and the model can only ADD words between them, never contradict them.
//
// Example:
//   Standard:   "Summarize grep results" → "No results found" (hallucination)
//   Scaffolded: "The search found [3] matches in [file.go]. The key finding
//               is [line 42: func NewReasoner]. {MODEL: explain significance}"
//
// The facts in [brackets] are injected deterministically. The model only
// generates the {MODEL: ...} parts.
type NeuralScaffold struct{}

// NewNeuralScaffold creates a new neural scaffold builder.
func NewNeuralScaffold() *NeuralScaffold {
	return &NeuralScaffold{}
}

// ScaffoldedPrompt is a prompt with facts pre-filled and gaps for the LLM.
type ScaffoldedPrompt struct {
	SystemPrompt string // minimal system prompt
	UserMessage  string // query + evidence
	ResponseSeed string // pre-filled response start that the LLM continues
}

// BuildFromToolResult creates a scaffolded prompt for a single tool result.
func (ns *NeuralScaffold) BuildFromToolResult(query, tool string, args map[string]string, result string) *ScaffoldedPrompt {
	// Minimal system prompt — only what's needed
	sys := "You are a helpful assistant. The user asked a question and a tool has already been executed. " +
		"Your response MUST begin with the exact text provided in the response seed. " +
		"Continue from where the seed ends. Be concise and factual."

	// Build evidence block
	evidence := fmt.Sprintf("[Tool: %s]\n[Result]\n%s\n[/Result]", tool, truncateEvidence(result, 1500))

	user := fmt.Sprintf("Question: %s\n\n%s", query, evidence)

	// Build response seed — the beginning of the answer that's factually locked
	seed := ns.buildSeed(tool, args, result)

	return &ScaffoldedPrompt{
		SystemPrompt: sys,
		UserMessage:  user,
		ResponseSeed: seed,
	}
}

// BuildFromMultipleResults creates a scaffold from multiple tool results.
func (ns *NeuralScaffold) BuildFromMultipleResults(query string, steps []synthStep) *ScaffoldedPrompt {
	if len(steps) == 1 {
		return ns.BuildFromToolResult(query, steps[0].Tool, steps[0].Args, steps[0].Result)
	}

	sys := "You are a helpful assistant. Multiple tools were executed to answer the user's question. " +
		"Your response MUST begin with the exact text provided in the response seed. " +
		"Continue from where the seed ends. Synthesize the results concisely."

	var evidence strings.Builder
	for i, step := range steps {
		if step.Err != nil {
			continue
		}
		evidence.WriteString(fmt.Sprintf("[Tool %d: %s]\n", i+1, step.Tool))
		evidence.WriteString(truncateEvidence(step.Result, 800))
		evidence.WriteString("\n\n")
	}

	user := fmt.Sprintf("Question: %s\n\n%s", query, evidence.String())

	// Build combined seed
	var seed strings.Builder
	for _, step := range steps {
		if step.Err != nil {
			continue
		}
		partSeed := ns.buildSeed(step.Tool, step.Args, step.Result)
		if partSeed != "" {
			seed.WriteString(partSeed)
			seed.WriteString(" ")
		}
	}

	return &ScaffoldedPrompt{
		SystemPrompt: sys,
		UserMessage:  user,
		ResponseSeed: strings.TrimSpace(seed.String()),
	}
}

// buildSeed creates the factual beginning of a response that the LLM continues.
func (ns *NeuralScaffold) buildSeed(tool string, args map[string]string, result string) string {
	result = strings.TrimSpace(result)
	if result == "" {
		return ns.buildEmptySeed(tool, args)
	}

	switch tool {
	case "grep":
		return ns.grepSeed(args, result)
	case "read":
		return ns.readSeed(args, result)
	case "ls":
		return ns.lsSeed(args, result)
	case "tree":
		return ns.treeSeed(args, result)
	case "glob":
		return ns.globSeed(args, result)
	case "git":
		return ns.gitSeed(args, result)
	default:
		return ""
	}
}

func (ns *NeuralScaffold) grepSeed(args map[string]string, result string) string {
	pattern := args["pattern"]
	lines := strings.Split(result, "\n")
	matchCount := countNonEmpty(lines)

	if matchCount == 0 {
		return fmt.Sprintf("I searched for `%s` but found no matches.", pattern)
	}

	// Extract first match file and line
	firstMatch := strings.TrimSpace(lines[0])
	parts := strings.SplitN(firstMatch, ":", 3)
	if len(parts) >= 2 {
		file := parts[0]
		return fmt.Sprintf("I found %d matches for `%s`. The first match is in `%s`:", matchCount, pattern, file)
	}

	return fmt.Sprintf("I found %d matches for `%s`:", matchCount, pattern)
}

func (ns *NeuralScaffold) readSeed(args map[string]string, result string) string {
	path := args["path"]
	if path == "" {
		path = args["file"]
	}
	lines := strings.Split(result, "\n")
	return fmt.Sprintf("Here is the content of `%s` (%d lines):", path, len(lines))
}

func (ns *NeuralScaffold) lsSeed(args map[string]string, result string) string {
	dir := args["path"]
	if dir == "" {
		dir = "the current directory"
	}
	lines := strings.Split(result, "\n")
	entryCount := countNonEmpty(lines)
	return fmt.Sprintf("The directory `%s` contains %d entries:", dir, entryCount)
}

func (ns *NeuralScaffold) treeSeed(args map[string]string, result string) string {
	dir := args["path"]
	if dir == "" {
		dir = "the project"
	}
	return fmt.Sprintf("Here is the structure of `%s`:", dir)
}

func (ns *NeuralScaffold) globSeed(args map[string]string, result string) string {
	pattern := args["pattern"]
	lines := strings.Split(result, "\n")
	fileCount := countNonEmpty(lines)

	if fileCount == 0 {
		return fmt.Sprintf("No files match the pattern `%s`.", pattern)
	}
	return fmt.Sprintf("Found %d files matching `%s`:", fileCount, pattern)
}

func (ns *NeuralScaffold) gitSeed(args map[string]string, result string) string {
	cmd := args["command"]
	if strings.HasPrefix(cmd, "status") {
		if strings.Contains(result, "nothing to commit") {
			return "The working tree is clean — nothing to commit."
		}
		return "Here is the current git status:"
	}
	if strings.HasPrefix(cmd, "log") {
		return "Here are the recent commits:"
	}
	if strings.HasPrefix(cmd, "diff") {
		return "Here are the current changes:"
	}
	return fmt.Sprintf("Here is the output of `git %s`:", cmd)
}

func (ns *NeuralScaffold) buildEmptySeed(tool string, args map[string]string) string {
	switch tool {
	case "grep":
		return fmt.Sprintf("I searched for `%s` but found no matches.", args["pattern"])
	case "glob":
		return fmt.Sprintf("No files match the pattern `%s`.", args["pattern"])
	case "ls":
		dir := args["path"]
		if dir == "" {
			dir = "the current directory"
		}
		return fmt.Sprintf("The directory `%s` appears to be empty.", dir)
	default:
		return fmt.Sprintf("The %s tool returned no output.", tool)
	}
}

// --- Validation ---

// ValidateResponse checks if an LLM response contradicts the scaffold.
// Returns the original response if valid, or the scaffold-based response if not.
func (ns *NeuralScaffold) ValidateResponse(response, seed string, tool string, result string) string {
	response = strings.TrimSpace(response)

	// Check for common hallucination patterns
	hallucinations := []string{
		"no results found",
		"no matches found",
		"i couldn't find",
		"i was unable to find",
		"nothing was found",
		"the search returned no",
		"no occurrences",
		"does not contain",
		"not found in",
	}

	lower := strings.ToLower(response)
	hasResults := strings.TrimSpace(result) != "" && len(strings.Split(result, "\n")) > 1

	if hasResults {
		for _, h := range hallucinations {
			if strings.Contains(lower, h) {
				// LLM hallucinated "no results" despite having results
				// Return the seed as the response instead
				return seed
			}
		}
	}

	return response
}

// --- Helpers ---

func countNonEmpty(lines []string) int {
	count := 0
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			count++
		}
	}
	return count
}

func truncateEvidence(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... (truncated)"
}
