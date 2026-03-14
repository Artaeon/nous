package hands

// BuiltinHands returns the default set of autonomous hands.
func BuiltinHands() []Hand {
	return []Hand{
		{
			Name:        "researcher",
			Description: "Takes a topic and uses fetch, grep, and read tools to gather information, then produces a summary report.",
			Schedule:    "",   // manual trigger only by default
			Enabled:     false,
			Config: HandConfig{
				MaxSteps: 8,
				Timeout:  300,
				Tools:    []string{"fetch", "read", "grep", "glob", "write"},
			},
			Prompt: "Research the assigned topic thoroughly. Use the fetch tool to retrieve relevant web content, grep and read to analyze local files for context, then write a comprehensive summary report to ~/.nous/hands/researcher/reports/latest.md. Include key findings, sources, and actionable insights.",
		},
		{
			Name:        "collector",
			Description: "Monitors URLs for changes, detects diffs, and stores change history.",
			Schedule:    "@hourly",
			Enabled:     false,
			Config: HandConfig{
				MaxSteps: 6,
				Timeout:  120,
				Tools:    []string{"fetch", "read", "write"},
			},
			Prompt: "Check the monitored URLs for content changes. Fetch each URL, compare with the previously stored version in ~/.nous/hands/collector/, and if changes are detected, save the new version and append a change summary to ~/.nous/hands/collector/changes.log with a timestamp.",
		},
		{
			Name:        "digest",
			Description: "Produces a daily digest summarizing recent activity, tasks, and reminders.",
			Schedule:    "@daily",
			Enabled:     false,
			Config: HandConfig{
				MaxSteps: 4,
				Timeout:  90,
				Tools:    []string{"read", "write"},
			},
			Prompt: "Create a daily digest summarizing today's activity. Read recent episodic memory and any pending tasks or reminders. Write a concise digest to ~/.nous/hands/digest/YYYY-MM-DD.md covering: completed tasks, key interactions, upcoming deadlines, and suggested priorities for tomorrow.",
		},
		{
			Name:        "guardian",
			Description: "Watches for filesystem changes and runs security checks on modified files.",
			Schedule:    "*/30 * * * *", // every 30 minutes
			Enabled:     false,
			Config: HandConfig{
				MaxSteps:         6,
				Timeout:          120,
				Tools:            []string{"read", "grep", "glob", "write"},
				RequiresApproval: true,
			},
			Prompt: "Scan recently modified files for potential security issues. Use glob to find files changed in the last 30 minutes, read their contents, and grep for common security patterns (hardcoded secrets, unsafe permissions, SQL injection vectors, etc). Write findings to ~/.nous/hands/guardian/report.md. Flag any files that need immediate attention.",
		},
		{
			Name:        "coder",
			Description: "Scans the codebase for TODO comments and proposes fixes or implementations.",
			Schedule:    "@daily",
			Enabled:     false,
			Config: HandConfig{
				MaxSteps:         8,
				Timeout:          180,
				Tools:            []string{"grep", "read", "glob", "write"},
				RequiresApproval: true,
			},
			Prompt: "Scan the codebase for TODO, FIXME, and HACK comments using grep. For each one found, read the surrounding code context, then write proposed fixes or implementations to ~/.nous/hands/coder/proposals/. Each proposal should include the file path, the TODO text, and a concrete code suggestion. Prioritize by impact and feasibility.",
		},
		{
			Name:        "predictor",
			Description: "Analyzes trends and makes calibrated predictions with confidence scores and reasoning chains.",
			Schedule:    "@daily",
			Enabled:     false,
			Config: HandConfig{
				MaxSteps: 6,
				Timeout:  240,
				Tools:    []string{"fetch", "read", "write", "grep"},
			},
			Prompt: "Read recent episodic memory, fetch relevant news and signals, and analyze local project data to make predictions about project progress, deadlines, and potential blockers. Write predictions to ~/.nous/hands/predictor/predictions.md with confidence percentages and explicit reasoning chains for each prediction. Track past prediction accuracy by comparing previous predictions against actual outcomes and include a calibration summary.",
		},
		{
			Name:        "monitor",
			Description: "Continuously monitors system health, resource usage, and service availability.",
			Schedule:    "*/15 * * * *",
			Enabled:     false,
			Config: HandConfig{
				MaxSteps: 6,
				Timeout:  120,
				Tools:    []string{"sysinfo", "shell", "fetch", "read", "write"},
			},
			Prompt: "Check system health using the sysinfo tool, monitor disk usage, and verify that key services are running via the shell tool. Fetch any configured health check URLs. Compare current metrics against previous readings stored in ~/.nous/hands/monitor/metrics.json. Alert if any metric crosses thresholds (>80% CPU, >90% disk, service down). Write current status to ~/.nous/hands/monitor/status.md with timestamps and trend indicators.",
		},
		{
			Name:        "planner",
			Description: "Creates and maintains a prioritized daily action plan based on tasks, reminders, and project context.",
			Schedule:    "0 8 * * *",
			Enabled:     false,
			Config: HandConfig{
				MaxSteps: 6,
				Timeout:  180,
				Tools:    []string{"read", "grep", "glob", "write"},
			},
			Prompt: "Review pending tasks, reminders, recent activity from episodic memory, and the project's current state. Create a prioritized daily plan with time estimates, considering dependencies between tasks. Write the plan to ~/.nous/hands/planner/plan-YYYY-MM-DD.md. Include a morning briefing summary and suggested focus blocks for deep work.",
		},
	}
}
