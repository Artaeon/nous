package agent

import (
	"fmt"
	"os"
	"strings"
)

// PhaseEvaluation is the outcome of evaluating a completed phase.
type PhaseEvaluation struct {
	QualitySufficient    bool     // is the output good enough to proceed?
	NeedsMoreData        []string // topics that need additional research
	SuggestedAdjustments []string // changes to remaining phases
	SkipNextPhase        bool     // if this phase already covered the next one
	ContentWords         int      // word count of actual content produced
	Reason               string   // human-readable explanation
}

// evaluatePhase examines the results of a completed phase and decides
// whether the agent should proceed, retry, inject new tasks, or skip ahead.
func (a *Agent) evaluatePhase(phaseIdx int) *PhaseEvaluation {
	a.State.mu.RLock()
	if a.State.Plan == nil || phaseIdx >= len(a.State.Plan.Phases) {
		a.State.mu.RUnlock()
		return &PhaseEvaluation{QualitySufficient: true, Reason: "no plan or invalid phase"}
	}
	phase := a.State.Plan.Phases[phaseIdx]
	results := make(map[string]string, len(a.State.Results))
	for k, v := range a.State.Results {
		results[k] = v
	}
	goal := a.State.CurrentGoal
	a.State.mu.RUnlock()

	// Gather all content produced by this phase's tasks.
	// For write tasks, read the actual file content instead of the
	// "wrote N bytes to path" result string.
	var phaseContent strings.Builder
	completedTasks := 0
	failedTasks := 0
	for _, task := range phase.Tasks {
		switch task.Status {
		case TaskCompleted:
			completedTasks++
			result := task.Result
			if stored, ok := results[task.ID]; ok && stored != "" {
				result = stored
			}
			// If result is "wrote to <path>", read the actual file
			if strings.HasPrefix(result, "wrote ") && strings.Contains(result, " to ") {
				if path := extractWritePath(result); path != "" {
					if data, err := os.ReadFile(path); err == nil {
						phaseContent.Write(data)
						phaseContent.WriteString("\n")
						continue
					}
				}
			}
			if result != "" {
				phaseContent.WriteString(result)
				phaseContent.WriteString("\n")
			}
		case TaskFailed:
			failedTasks++
		}
	}

	content := phaseContent.String()
	contentWords := countContentWords(content)

	eval := &PhaseEvaluation{
		ContentWords: contentWords,
	}

	// All tasks failed — definitely not sufficient
	if completedTasks == 0 && failedTasks > 0 {
		eval.QualitySufficient = false
		eval.Reason = fmt.Sprintf("all %d tasks failed", failedTasks)
		eval.NeedsMoreData = suggestAlternativeQueries(goal, phase)
		return eval
	}

	// Check content substance: < 100 words of real content is too thin
	if contentWords < 100 {
		eval.QualitySufficient = false
		eval.Reason = fmt.Sprintf("only %d content words (minimum 100)", contentWords)
		eval.NeedsMoreData = suggestAlternativeQueries(goal, phase)
		return eval
	}

	// Use the brain to evaluate if we have enough data
	if a.Brain != nil && contentWords < 300 {
		// Borderline — check if summarization produces anything meaningful
		summary := a.Brain.Summarize(content, 3)
		summaryWords := len(strings.Fields(summary))
		if summaryWords < 20 {
			eval.QualitySufficient = false
			eval.Reason = fmt.Sprintf("content summarizes to only %d words", summaryWords)
			eval.NeedsMoreData = suggestAlternativeQueries(goal, phase)
			return eval
		}
	}

	// Check if this phase's output already covers the next phase's work.
	// E.g., if the research phase already produced a structured analysis,
	// the analysis phase can be skipped.
	a.State.mu.RLock()
	nextPhaseIdx := phaseIdx + 1
	if nextPhaseIdx < len(a.State.Plan.Phases) {
		nextPhase := a.State.Plan.Phases[nextPhaseIdx]
		if phaseAlreadyCoversNext(content, nextPhase) {
			eval.SkipNextPhase = true
			eval.SuggestedAdjustments = append(eval.SuggestedAdjustments,
				fmt.Sprintf("skip phase %q — already covered by %q output", nextPhase.Name, phase.Name))
		}
	}
	a.State.mu.RUnlock()

	eval.QualitySufficient = true
	eval.Reason = fmt.Sprintf("%d content words from %d tasks", contentWords, completedTasks)
	return eval
}

// suggestAlternativeQueries generates new search queries when a phase
// produced insufficient results. Uses different angles on the topic.
func suggestAlternativeQueries(goal string, phase Phase) []string {
	topic := extractTopic(goal)
	if topic == "" {
		topic = goal
	}

	// Look at what queries were already tried
	tried := make(map[string]bool)
	for _, task := range phase.Tasks {
		for _, step := range task.ToolChain {
			if step.Tool == "websearch" {
				if q, ok := step.Args["query"]; ok {
					tried[strings.ToLower(q)] = true
				}
			}
		}
	}

	// Generate alternative angles
	alternatives := []string{
		topic + " explained simply",
		topic + " introduction guide",
		topic + " latest developments",
		topic + " key facts",
		"what is " + topic,
		topic + " wikipedia",
	}

	var suggestions []string
	for _, alt := range alternatives {
		if !tried[strings.ToLower(alt)] {
			suggestions = append(suggestions, alt)
		}
		if len(suggestions) >= 3 {
			break
		}
	}
	return suggestions
}

// phaseAlreadyCoversNext checks if the current phase's output already
// contains the substance that the next phase would produce.
func phaseAlreadyCoversNext(content string, nextPhase Phase) bool {
	lower := strings.ToLower(content)
	nextLower := strings.ToLower(nextPhase.Name)

	// If the content already has structured sections matching the next phase
	if strings.Contains(nextLower, "analysis") || strings.Contains(nextLower, "synthesize") {
		// Check if content already has analytical structure
		analysisMarkers := []string{"findings", "analysis", "conclusion", "therefore", "suggests that", "indicates"}
		matches := 0
		for _, m := range analysisMarkers {
			if strings.Contains(lower, m) {
				matches++
			}
		}
		if matches >= 3 {
			return true
		}
	}

	if strings.Contains(nextLower, "report") || strings.Contains(nextLower, "document") {
		// Check if content is already structured as a report
		if strings.Contains(lower, "## ") || strings.Contains(lower, "# ") {
			sectionCount := strings.Count(lower, "## ")
			if sectionCount >= 3 {
				return true
			}
		}
	}

	return false
}

// injectSearchTasks creates new websearch tasks for the suggested queries
// and prepends them to the given phase.
func injectSearchTasks(phase *Phase, queries []string, baseID string) {
	var newTasks []Task
	for i, query := range queries {
		newTasks = append(newTasks, Task{
			ID:          fmt.Sprintf("%s-extra-%d", baseID, i+1),
			Description: fmt.Sprintf("Additional search: %s", query),
			ToolChain: []ToolStep{
				{
					Tool:      "websearch",
					Args:      map[string]string{"query": query},
					DependsOn: -1,
					OutputKey: fmt.Sprintf("extra_%d", i+1),
				},
			},
		})
	}
	// Prepend new tasks before existing ones
	phase.Tasks = append(newTasks, phase.Tasks...)
}

// extractWritePath pulls the file path from a "wrote N bytes to /path" string.
func extractWritePath(result string) string {
	// Format: "wrote N bytes to /path/to/file" or "wrote to /path/to/file"
	if idx := strings.Index(result, " to /"); idx >= 0 {
		return strings.TrimSpace(result[idx+4:])
	}
	if idx := strings.Index(result, " to "); idx >= 0 {
		path := strings.TrimSpace(result[idx+4:])
		if strings.Contains(path, "/") {
			return path
		}
	}
	return ""
}

// countContentWords counts non-trivial words in text, skipping URLs,
// tool output markers, markdown headers, and short fragments.
func countContentWords(text string) int {
	count := 0
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		// Skip empty lines, URLs, tool output markers
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "http") || strings.HasPrefix(line, "wrote ") {
			continue
		}
		if strings.HasPrefix(line, "---") || strings.HasPrefix(line, "===") {
			continue
		}
		// Skip markdown headers (## Section) — they're structure, not content
		if strings.HasPrefix(line, "#") {
			continue
		}
		// Skip "wrote N bytes to" patterns
		if strings.Contains(line, "bytes to ") {
			continue
		}
		words := strings.Fields(line)
		for _, w := range words {
			if len(w) <= 1 {
				continue
			}
			count++
		}
	}
	return count
}
