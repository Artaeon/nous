package agent

import (
	"fmt"
	"strings"
	"time"
)

// Planner decomposes high-level goals into executable plans with phases,
// tasks, and tool chains. Pure code — no LLM calls.
type Planner struct {
	toolNames []string // available tool names for chain building
}

// NewPlanner creates a goal planner that knows about available tools.
func NewPlanner(toolNames []string) *Planner {
	return &Planner{toolNames: toolNames}
}

// DecomposeGoal breaks a high-level goal into an executable plan.
func (p *Planner) DecomposeGoal(goal string) (*Plan, error) {
	if strings.TrimSpace(goal) == "" {
		return nil, fmt.Errorf("empty goal")
	}

	goalType := classifyGoal(goal)
	phases := p.buildPhases(goal, goalType)

	if len(phases) == 0 {
		// Fallback: single-phase generic plan
		phases = p.genericPhases(goal)
	}

	return &Plan{
		Goal:        goal,
		Phases:      phases,
		CreatedAt:   time.Now(),
		EstDuration: estimateDuration(phases),
	}, nil
}

// goalType classifies what kind of goal the user described.
type goalType int

const (
	goalResearch goalType = iota
	goalWriting
	goalAnalysis
	goalPlanning
	goalBuilding
	goalMonitoring
	goalGeneric
)

// classifyGoal determines the goal type from natural language.
func classifyGoal(goal string) goalType {
	lower := strings.ToLower(goal)

	researchSignals := []string{
		"research", "investigate", "find out", "look into", "explore",
		"survey", "study", "learn about", "gather information", "compare",
		"market for", "competitors", "landscape",
	}
	for _, s := range researchSignals {
		if strings.Contains(lower, s) {
			return goalResearch
		}
	}

	writingSignals := []string{
		"write", "create a document", "draft", "compose", "author",
		"blog post", "article", "report", "essay", "business plan",
		"proposal", "documentation",
	}
	for _, s := range writingSignals {
		if strings.Contains(lower, s) {
			return goalWriting
		}
	}

	analysisSignals := []string{
		"analyze", "analyse", "evaluate", "assess", "review",
		"audit", "benchmark", "measure", "diagnose", "swot",
	}
	for _, s := range analysisSignals {
		if strings.Contains(lower, s) {
			return goalAnalysis
		}
	}

	planningSignals := []string{
		"plan", "strategy", "roadmap", "schedule", "organize",
		"outline", "design", "architect", "brainstorm",
	}
	for _, s := range planningSignals {
		if strings.Contains(lower, s) {
			return goalPlanning
		}
	}

	buildingSignals := []string{
		"build", "implement", "code", "program", "develop",
		"set up", "configure", "install", "deploy", "automate",
	}
	for _, s := range buildingSignals {
		if strings.Contains(lower, s) {
			return goalBuilding
		}
	}

	monitorSignals := []string{
		"monitor", "watch", "track", "check", "alert",
		"notify", "keep an eye", "follow",
	}
	for _, s := range monitorSignals {
		if strings.Contains(lower, s) {
			return goalMonitoring
		}
	}

	return goalGeneric
}

// buildPhases generates phases appropriate for the goal type.
func (p *Planner) buildPhases(goal string, gt goalType) []Phase {
	switch gt {
	case goalResearch:
		return p.researchPhases(goal)
	case goalWriting:
		return p.writingPhases(goal)
	case goalAnalysis:
		return p.analysisPhases(goal)
	case goalPlanning:
		return p.planningPhases(goal)
	case goalBuilding:
		return p.buildingPhases(goal)
	case goalMonitoring:
		return p.monitoringPhases(goal)
	default:
		return p.genericPhases(goal)
	}
}

func (p *Planner) researchPhases(goal string) []Phase {
	topic := extractTopic(goal)
	return []Phase{
		{
			Name:        "Information Gathering",
			Description: "Search for information on " + topic,
			Tasks: []Task{
				{
					ID:          "research-1",
					Description: "Web search for " + topic,
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " overview 2026"}, DependsOn: -1, OutputKey: "search_results"},
					},
				},
				{
					ID:          "research-2",
					Description: "Search for competitors and alternatives",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " competitors comparison"}, DependsOn: -1, OutputKey: "competitors"},
					},
				},
				{
					ID:          "research-3",
					Description: "Search for market data and trends",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " market size trends"}, DependsOn: -1, OutputKey: "market_data"},
					},
				},
			},
		},
		{
			Name:        "Analysis",
			Description: "Synthesize and analyze gathered information",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "analyze-1",
					Description: "Synthesize research findings",
					ToolChain: []ToolStep{
						{Tool: "_synthesize", Args: map[string]string{"goal": "analyze key findings about " + topic}, DependsOn: -1, OutputKey: "synthesis"},
					},
				},
				{
					ID:          "analyze-2",
					Description: "Save analysis notes",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/research_notes.md", "content": "${synthesis}"}, DependsOn: -1, OutputKey: "notes_file"},
					},
				},
			},
		},
		{
			Name:        "Report",
			Description: "Compile research report",
			DependsOn:   []int{1},
			Tasks: []Task{
				{
					ID:          "report-1",
					Description: "Generate research report document",
					ToolChain: []ToolStep{
						{Tool: "_generate_doc", Args: map[string]string{"topic": topic, "style": "report"}, DependsOn: -1, OutputKey: "report_content"},
					},
				},
				{
					ID:          "report-2",
					Description: "Save research report",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/research_report.md", "content": "${report_content}"}, DependsOn: -1, OutputKey: "report_file"},
					},
				},
			},
		},
	}
}

func (p *Planner) writingPhases(goal string) []Phase {
	topic := extractTopic(goal)
	return []Phase{
		{
			Name:        "Research",
			Description: "Gather background material for " + topic,
			Tasks: []Task{
				{
					ID:          "write-research-1",
					Description: "Search for background information",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic}, DependsOn: -1, OutputKey: "background"},
					},
				},
			},
		},
		{
			Name:        "Outline",
			Description: "Create document outline",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "write-outline-1",
					Description: "Think about document structure",
					ToolChain: []ToolStep{
						{Tool: "_think", Args: map[string]string{"query": "Create an outline for a document about " + topic}, DependsOn: -1, OutputKey: "outline_text"},
					},
				},
				{
					ID:          "write-outline-2",
					Description: "Save outline",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/outline.md", "content": "${outline_text}"}, DependsOn: -1, OutputKey: "outline_file"},
					},
				},
				{
					ID:          "write-outline-3",
					Description: "Review outline — may need human input on scope",
					NeedsHuman:  true,
					HumanPrompt: "I've created an outline. Would you like to adjust the scope or sections?",
				},
			},
		},
		{
			Name:        "Draft",
			Description: "Write each section of the document",
			DependsOn:   []int{1},
			Tasks: []Task{
				{
					ID:          "write-draft-1",
					Description: "Generate document draft",
					ToolChain: []ToolStep{
						{Tool: "_generate_doc", Args: map[string]string{"topic": topic, "style": "overview"}, DependsOn: -1, OutputKey: "draft_text"},
					},
				},
				{
					ID:          "write-draft-2",
					Description: "Save draft",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/draft.md", "content": "${draft_text}"}, DependsOn: -1, OutputKey: "draft_file"},
					},
				},
			},
		},
		{
			Name:        "Review & Finalize",
			Description: "Review, summarize, and produce final document",
			DependsOn:   []int{2},
			Tasks: []Task{
				{
					ID:          "write-review-1",
					Description: "Read and review the draft",
					ToolChain: []ToolStep{
						{Tool: "read", Args: map[string]string{"path": "agent_workspace/draft.md"}, DependsOn: -1, OutputKey: "draft_content"},
					},
				},
				{
					ID:          "write-final-1",
					Description: "Summarize and finalize the document",
					ToolChain: []ToolStep{
						{Tool: "_summarize", Args: map[string]string{"text": "${draft_content}"}, DependsOn: 0, OutputKey: "summary"},
					},
				},
				{
					ID:          "write-final-2",
					Description: "Save final document",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/final.md", "content": "${draft_content}"}, DependsOn: -1, OutputKey: "final_file"},
					},
				},
			},
		},
	}
}

func (p *Planner) analysisPhases(goal string) []Phase {
	topic := extractTopic(goal)
	return []Phase{
		{
			Name:        "Data Gathering",
			Description: "Collect data for analysis of " + topic,
			Tasks: []Task{
				{
					ID:          "data-1",
					Description: "Search for relevant data",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " data statistics"}, DependsOn: -1, OutputKey: "raw_data"},
					},
				},
				{
					ID:          "data-2",
					Description: "Search for benchmarks and comparisons",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " benchmark comparison"}, DependsOn: -1, OutputKey: "benchmarks"},
					},
				},
			},
		},
		{
			Name:        "Analysis",
			Description: "Analyze the collected data",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "analysis-1",
					Description: "Synthesize and analyze findings",
					ToolChain: []ToolStep{
						{Tool: "_synthesize", Args: map[string]string{"goal": "analyze " + topic}, DependsOn: -1, OutputKey: "analysis_text"},
					},
				},
				{
					ID:          "analysis-2",
					Description: "Save analysis",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/analysis.md", "content": "${analysis_text}"}, DependsOn: -1, OutputKey: "analysis_file"},
					},
				},
			},
		},
		{
			Name:        "Presentation",
			Description: "Present findings",
			DependsOn:   []int{1},
			Tasks: []Task{
				{
					ID:          "present-1",
					Description: "Generate findings report",
					ToolChain: []ToolStep{
						{Tool: "_generate_doc", Args: map[string]string{"topic": topic, "style": "report"}, DependsOn: -1, OutputKey: "findings_text"},
					},
				},
				{
					ID:          "present-2",
					Description: "Save findings report",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/findings.md", "content": "${findings_text}"}, DependsOn: -1, OutputKey: "findings_file"},
					},
				},
			},
		},
	}
}

func (p *Planner) planningPhases(goal string) []Phase {
	topic := extractTopic(goal)
	return []Phase{
		{
			Name:        "Brainstorm",
			Description: "Generate ideas for " + topic,
			Tasks: []Task{
				{
					ID:          "brainstorm-1",
					Description: "Research existing approaches",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " best practices"}, DependsOn: -1, OutputKey: "approaches"},
					},
				},
			},
		},
		{
			Name:        "Structure",
			Description: "Organize ideas into a plan",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "structure-1",
					Description: "Create structured plan",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/plan.md"}, DependsOn: -1, OutputKey: "plan_file"},
					},
				},
				{
					ID:          "structure-2",
					Description: "Review plan with human",
					NeedsHuman:  true,
					HumanPrompt: "I've created a plan outline. Would you like to adjust priorities or add constraints?",
				},
			},
		},
		{
			Name:        "Estimate & Document",
			Description: "Add estimates and finalize the plan",
			DependsOn:   []int{1},
			Tasks: []Task{
				{
					ID:          "estimate-1",
					Description: "Write final plan with estimates",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/final_plan.md"}, DependsOn: -1, OutputKey: "final_plan"},
					},
				},
			},
		},
	}
}

func (p *Planner) buildingPhases(goal string) []Phase {
	topic := extractTopic(goal)
	return []Phase{
		{
			Name:        "Design",
			Description: "Design approach for " + topic,
			Tasks: []Task{
				{
					ID:          "design-1",
					Description: "Research existing solutions",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " tutorial guide"}, DependsOn: -1, OutputKey: "research"},
					},
				},
				{
					ID:          "design-2",
					Description: "Write design document",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/design.md"}, DependsOn: -1, OutputKey: "design_file"},
					},
				},
			},
		},
		{
			Name:        "Implement",
			Description: "Build the solution",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "implement-1",
					Description: "Implementation — may need human guidance",
					NeedsHuman:  true,
					HumanPrompt: "I have a design ready. Should I proceed with implementation? Any constraints?",
				},
			},
		},
		{
			Name:        "Test & Verify",
			Description: "Test the implementation",
			DependsOn:   []int{1},
			Tasks: []Task{
				{
					ID:          "test-1",
					Description: "Verify the implementation",
					ToolChain: []ToolStep{
						{Tool: "read", Args: map[string]string{"path": "agent_workspace/design.md"}, DependsOn: -1, OutputKey: "design_review"},
					},
				},
				{
					ID:          "test-2",
					Description: "Write results summary",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/results.md"}, DependsOn: -1, OutputKey: "results_file"},
					},
				},
			},
		},
	}
}

func (p *Planner) monitoringPhases(goal string) []Phase {
	topic := extractTopic(goal)
	return []Phase{
		{
			Name:        "Setup",
			Description: "Configure monitoring for " + topic,
			Tasks: []Task{
				{
					ID:          "monitor-setup-1",
					Description: "Determine what to monitor",
					NeedsHuman:  true,
					HumanPrompt: "What specific metrics or conditions should I monitor? What thresholds should trigger an alert?",
				},
			},
		},
		{
			Name:        "Monitor",
			Description: "Run monitoring checks",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "monitor-1",
					Description: "Check " + topic,
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": topic + " current status"}, DependsOn: -1, OutputKey: "status"},
					},
				},
				{
					ID:          "monitor-2",
					Description: "Record findings",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/monitor_log.md"}, DependsOn: -1, OutputKey: "log_file"},
					},
				},
			},
		},
	}
}

func (p *Planner) genericPhases(goal string) []Phase {
	return []Phase{
		{
			Name:        "Research",
			Description: "Gather information for: " + goal,
			Tasks: []Task{
				{
					ID:          "generic-1",
					Description: "Search for relevant information",
					ToolChain: []ToolStep{
						{Tool: "web_search", Args: map[string]string{"query": goal}, DependsOn: -1, OutputKey: "search_results"},
					},
				},
			},
		},
		{
			Name:        "Execute",
			Description: "Work on the goal",
			DependsOn:   []int{0},
			Tasks: []Task{
				{
					ID:          "generic-2",
					Description: "Process results and take action",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/output.md"}, DependsOn: -1, OutputKey: "output_file"},
					},
				},
			},
		},
		{
			Name:        "Report",
			Description: "Summarize results",
			DependsOn:   []int{1},
			Tasks: []Task{
				{
					ID:          "generic-3",
					Description: "Write final summary",
					ToolChain: []ToolStep{
						{Tool: "write", Args: map[string]string{"path": "agent_workspace/summary.md"}, DependsOn: -1, OutputKey: "summary_file"},
					},
				},
			},
		},
	}
}

// extractTopic pulls the subject matter from a goal string.
func extractTopic(goal string) string {
	lower := strings.ToLower(goal)

	// Strip common goal verbs
	prefixes := []string{
		"research the market for ",
		"research ", "investigate ", "find out about ",
		"look into ", "explore ", "study ",
		"write a ", "write an ", "write ",
		"create a ", "create an ", "create ",
		"draft a ", "draft an ", "draft ",
		"analyze ", "analyse ", "evaluate ", "assess ",
		"build a ", "build an ", "build ",
		"implement a ", "implement an ", "implement ",
		"plan a ", "plan an ", "plan for ", "plan ",
		"monitor ", "track ", "watch ",
		"set up ", "configure ", "deploy ",
	}

	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(goal[len(p):])
		}
	}

	// Strip trailing goal markers
	suffixes := []string{
		" and create a business plan",
		" and write a report",
		" and summarize",
	}
	result := goal
	for _, s := range suffixes {
		if idx := strings.Index(strings.ToLower(result), s); idx > 0 {
			result = result[:idx]
		}
	}

	return strings.TrimSpace(result)
}

// estimateDuration roughly estimates how long a plan will take.
func estimateDuration(phases []Phase) time.Duration {
	var total time.Duration
	for _, phase := range phases {
		for _, task := range phase.Tasks {
			if task.NeedsHuman {
				total += 5 * time.Minute // assume 5 min for human
			} else {
				total += time.Duration(len(task.ToolChain)+1) * 30 * time.Second
			}
		}
	}
	return total
}
