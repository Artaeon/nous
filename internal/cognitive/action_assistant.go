package cognitive

import (
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------
// Personal assistant action handlers.
// Each handler delegates to the corresponding tool in the tools registry.
// -----------------------------------------------------------------------

// handleWeather fetches weather for a location.
func (ar *ActionRouter) handleWeather(nlu *NLUResult) *ActionResult {
	location := nlu.Entities["location"]
	if location == "" {
		location = nlu.Entities["topic"]
	}
	if location == "" {
		// Try to extract location from raw input
		location = extractWeatherLocation(nlu.Raw)
	}
	if location == "" {
		location = "auto" // will use IP geolocation
	}

	if ar.Tools == nil {
		return &ActionResult{Data: "weather tool unavailable", Source: "weather", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("weather")
	if err != nil {
		return &ActionResult{Data: "weather tool not found", Source: "weather", NeedsLLM: true}
	}
	result, err := tool.Execute(map[string]string{"location": location})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("weather error: %v", err), Source: "weather", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "weather"}
}

// extractWeatherLocation pulls a city/location from weather queries.
func extractWeatherLocation(raw string) string {
	lower := strings.ToLower(raw)
	// "weather in London" / "weather for Paris" / "forecast for NYC"
	for _, prep := range []string{" in ", " for ", " at "} {
		if idx := strings.Index(lower, prep); idx >= 0 {
			loc := strings.TrimSpace(raw[idx+len(prep):])
			loc = strings.TrimRight(loc, "?!.")
			if loc != "" {
				return loc
			}
		}
	}
	return ""
}

// handleConvert performs unit or currency conversion.
func (ar *ActionRouter) handleConvert(nlu *NLUResult) *ActionResult {
	expr := nlu.Entities["expr"]
	if expr == "" {
		expr = nlu.Raw
	}

	// Strip common verb prefixes so the parser sees "5 miles to km" not "convert 5 miles to km"
	lower := strings.ToLower(expr)
	for _, prefix := range []string{"convert ", "how much is ", "how many ", "what is ", "what's "} {
		if strings.HasPrefix(lower, prefix) {
			expr = expr[len(prefix):]
			break
		}
	}

	if ar.Tools == nil {
		return &ActionResult{Data: "conversion tools unavailable", Source: "convert", NeedsLLM: true}
	}

	// Try unit conversion first
	tool, err := ar.Tools.Get("convert")
	if err == nil {
		result, err := tool.Execute(map[string]string{"input": expr})
		if err == nil && result != "" {
			return &ActionResult{DirectResponse: result, Source: "convert"}
		}
	}

	// Try currency conversion
	tool, err = ar.Tools.Get("currency")
	if err == nil {
		result, err := tool.Execute(map[string]string{"input": expr})
		if err == nil && result != "" {
			return &ActionResult{DirectResponse: result, Source: "currency"}
		}
	}

	return &ActionResult{Data: expr, Source: "convert", NeedsLLM: true}
}

// handleReminder creates or lists reminders.
func (ar *ActionRouter) handleReminder(nlu *NLUResult) *ActionResult {
	lower := strings.ToLower(nlu.Raw)

	// List reminders
	if strings.Contains(lower, "list") || strings.Contains(lower, "show") || strings.Contains(lower, "my reminder") {
		if ar.Reminders == nil {
			return &ActionResult{DirectResponse: "No reminders set.", Source: "reminder"}
		}
		list := ar.Reminders.ListReminders()
		if len(list) == 0 {
			return &ActionResult{DirectResponse: "No active reminders.", Source: "reminder"}
		}
		var lines []string
		for _, r := range list {
			lines = append(lines, fmt.Sprintf("- [%d] %s (fires at %s)", r.ID, r.Message, r.FireAt.Format("15:04")))
		}
		return &ActionResult{DirectResponse: strings.Join(lines, "\n"), Source: "reminder"}
	}

	// Create reminder: extract duration and message
	if ar.Reminders == nil {
		return &ActionResult{DirectResponse: "Reminder system not initialized.", Source: "reminder"}
	}

	// Parse "remind me in 30 minutes to check the oven"
	// or "set a timer for 5 minutes"
	dur, err := ar.Reminders.ParseDuration(nlu.Raw)
	if err != nil {
		return &ActionResult{Data: nlu.Raw, Source: "reminder", NeedsLLM: true}
	}

	msg := extractReminderMessage(nlu.Raw)
	if msg == "" {
		msg = "Timer done!"
	}

	r := ar.Reminders.AddReminder(msg, dur)
	return &ActionResult{
		DirectResponse: fmt.Sprintf("Reminder set: \"%s\" in %s (fires at %s)", r.Message, dur.String(), r.FireAt.Format("15:04")),
		Source:         "reminder",
	}
}

// extractReminderMessage pulls the reminder text from the raw input.
func extractReminderMessage(raw string) string {
	lower := strings.ToLower(raw)
	// "remind me in 30 minutes to check the oven" → "check the oven"
	for _, marker := range []string{" to ", " that ", " about "} {
		// Find the marker after the duration part
		idx := strings.LastIndex(lower, marker)
		if idx > 0 {
			msg := strings.TrimSpace(raw[idx+len(marker):])
			msg = strings.TrimRight(msg, "?!.")
			if msg != "" {
				return msg
			}
		}
	}
	return ""
}

// handleSysInfo returns system information.
func (ar *ActionRouter) handleSysInfo(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "sysinfo tool unavailable", Source: "sysinfo", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("sysinfo")
	if err != nil {
		return &ActionResult{Data: "sysinfo tool not found", Source: "sysinfo", NeedsLLM: true}
	}

	query := nlu.Entities["topic"]
	if query == "" {
		query = nlu.Raw
	}
	result, err := tool.Execute(map[string]string{"query": query})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("sysinfo error: %v", err), Source: "sysinfo", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "sysinfo"}
}

// handleClipboard reads from or writes to the system clipboard.
func (ar *ActionRouter) handleClipboard(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "clipboard tool unavailable", Source: "clipboard", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("clipboard")
	if err != nil {
		return &ActionResult{Data: "clipboard tool not found", Source: "clipboard", NeedsLLM: true}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{"action": "read"}

	if strings.Contains(lower, "copy") || strings.Contains(lower, "write") {
		args["action"] = "write"
		if content := nlu.Entities["quoted"]; content != "" {
			args["content"] = content
		}
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("clipboard error: %v", err), Source: "clipboard", NeedsLLM: true}
	}

	if args["action"] == "read" {
		return &ActionResult{DirectResponse: fmt.Sprintf("Clipboard contents:\n%s", result), Source: "clipboard"}
	}
	return &ActionResult{DirectResponse: result, Source: "clipboard"}
}

// handleNotes manages markdown notes.
func (ar *ActionRouter) handleNotes(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "notes tool unavailable", Source: "notes", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("notes")
	if err != nil {
		return &ActionResult{Data: "notes tool not found", Source: "notes", NeedsLLM: true}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{}

	switch {
	case strings.Contains(lower, "list") || strings.Contains(lower, "show") || lower == "my notes" || lower == "notes":
		args["action"] = "list"
	case strings.Contains(lower, "search") || strings.Contains(lower, "find"):
		args["action"] = "search"
		args["query"] = nlu.Entities["topic"]
	case strings.Contains(lower, "delete") || strings.Contains(lower, "remove"):
		args["action"] = "delete"
		args["title"] = nlu.Entities["topic"]
	case strings.Contains(lower, "save") || strings.Contains(lower, "create") || strings.Contains(lower, "new note") || strings.Contains(lower, "note about") || strings.Contains(lower, "note:"):
		args["action"] = "save"
		title := nlu.Entities["topic"]
		if title == "" {
			title = nlu.Entities["quoted"]
		}
		args["title"] = title
		content := nlu.Entities["quoted"]
		if content == "" {
			content = nlu.Raw
		}
		args["content"] = content
	default:
		args["action"] = "get"
		args["title"] = nlu.Entities["topic"]
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("notes error: %v", err), Source: "notes", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "notes"}
}

// handleTodos manages the todo list.
func (ar *ActionRouter) handleTodos(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "todos tool unavailable", Source: "todos", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("todos")
	if err != nil {
		return &ActionResult{Data: "todos tool not found", Source: "todos", NeedsLLM: true}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{}

	switch {
	case strings.Contains(lower, "done") || strings.Contains(lower, "complete") || strings.Contains(lower, "finish"):
		args["action"] = "complete"
		args["text"] = nlu.Entities["topic"]
	case strings.Contains(lower, "delete") || strings.Contains(lower, "remove"):
		args["action"] = "delete"
		args["text"] = nlu.Entities["topic"]
	case strings.Contains(lower, "list") || strings.Contains(lower, "show") || strings.Contains(lower, "my task") || strings.Contains(lower, "my todo"):
		args["action"] = "list"
	case strings.Contains(lower, "add") || strings.Contains(lower, "new") || strings.Contains(lower, "create"):
		args["action"] = "add"
		text := extractTodoText(nlu.Raw)
		if text == "" {
			text = nlu.Entities["quoted"]
		}
		if text == "" {
			text = nlu.Entities["topic"]
		}
		args["text"] = text
	default:
		args["action"] = "list"
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("todos error: %v", err), Source: "todos", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "todos"}
}

// extractTodoText pulls the task text from add commands.
func extractTodoText(raw string) string {
	lower := strings.ToLower(raw)
	for _, prefix := range []string{
		"add task ", "add todo ", "add a task ", "add a todo ",
		"new task ", "new todo ", "create task ", "create todo ",
		"add task: ", "add todo: ", "todo: ", "task: ",
	} {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			text := strings.TrimSpace(raw[idx+len(prefix):])
			return strings.TrimRight(text, "?!.")
		}
	}
	return ""
}

// handleFindFiles searches the local filesystem.
func (ar *ActionRouter) handleFindFiles(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "file finder unavailable", Source: "filefinder", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("filefinder")
	if err != nil {
		return &ActionResult{Data: "file finder not found", Source: "filefinder", NeedsLLM: true}
	}

	result, err := tool.Execute(map[string]string{"query": nlu.Raw})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("file finder error: %v", err), Source: "filefinder", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "filefinder"}
}

// handleSummarizeURL fetches and summarizes a URL.
func (ar *ActionRouter) handleSummarizeURL(nlu *NLUResult) *ActionResult {
	url := nlu.Entities["url"]
	if url == "" {
		return &ActionResult{Data: "no URL provided to summarize", Source: "summarize", NeedsLLM: true}
	}

	if ar.Tools == nil {
		return &ActionResult{Data: "summarize tool unavailable", Source: "summarize", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("summarize")
	if err != nil {
		return &ActionResult{Data: "summarize tool not found", Source: "summarize", NeedsLLM: true}
	}

	result, err := tool.Execute(map[string]string{"url": url})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("summarize error: %v", err), Source: "summarize", NeedsLLM: true}
	}

	// Content extracted, needs LLM to actually summarize
	return &ActionResult{
		Data:     fmt.Sprintf("[URL Content from %s]\n%s", url, result),
		Source:   "summarize",
		NeedsLLM: true,
	}
}

// handleNews fetches RSS/news feeds.
func (ar *ActionRouter) handleNews(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "news tool unavailable", Source: "news", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("rss")
	if err != nil {
		return &ActionResult{Data: "news tool not found", Source: "news", NeedsLLM: true}
	}

	// Detect which feed
	feed := "tech" // default
	lower := strings.ToLower(nlu.Raw)
	for _, name := range []string{"world", "science", "linux", "golang", "go", "tech", "technology"} {
		if strings.Contains(lower, name) {
			feed = name
			break
		}
	}

	result, err := tool.Execute(map[string]string{"feed": feed, "count": "5"})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("news error: %v", err), Source: "news", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "news"}
}

// handleRunCode executes code in a sandbox.
func (ar *ActionRouter) handleRunCode(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "code runner unavailable", Source: "coderunner", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("coderunner")
	if err != nil {
		return &ActionResult{Data: "code runner not found", Source: "coderunner", NeedsLLM: true}
	}

	code := nlu.Entities["quoted"]
	if code == "" {
		code = extractCodeBlock(nlu.Raw)
	}
	lang := nlu.Entities["language"]
	if code == "" {
		// Try "run python: code" or "run bash: code" patterns
		code, lang = extractCodeAfterColon(nlu.Raw)
	}
	if code == "" {
		return &ActionResult{Data: nlu.Raw, Source: "coderunner", NeedsLLM: true}
	}
	result, err := tool.Execute(map[string]string{"code": code, "language": lang})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("code runner error: %v", err), Source: "coderunner", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "coderunner"}
}

// extractCodeBlock pulls code from markdown-style code blocks or inline backticks.
func extractCodeBlock(raw string) string {
	// Check for ```code``` blocks
	if idx := strings.Index(raw, "```"); idx >= 0 {
		rest := raw[idx+3:]
		// Skip optional language tag on first line
		if nl := strings.Index(rest, "\n"); nl >= 0 {
			rest = rest[nl+1:]
		}
		if end := strings.Index(rest, "```"); end >= 0 {
			return strings.TrimSpace(rest[:end])
		}
		return strings.TrimSpace(rest)
	}
	// Check for `code` inline
	if idx := strings.Index(raw, "`"); idx >= 0 {
		rest := raw[idx+1:]
		if end := strings.Index(rest, "`"); end >= 0 {
			return strings.TrimSpace(rest[:end])
		}
	}
	return ""
}

// extractCodeAfterColon extracts code from patterns like "run python: code here".
func extractCodeAfterColon(raw string) (code, lang string) {
	lower := strings.ToLower(raw)
	langs := []string{"python", "bash", "javascript", "node", "sh"}
	for _, l := range langs {
		if strings.Contains(lower, l) {
			lang = l
			if lang == "node" || lang == "sh" {
				if lang == "node" {
					lang = "javascript"
				} else {
					lang = "bash"
				}
			}
			break
		}
	}
	// Extract code after colon
	if idx := strings.Index(raw, ":"); idx >= 0 {
		code = strings.TrimSpace(raw[idx+1:])
	}
	return
}

// handleCalendar reads calendar events.
func (ar *ActionRouter) handleCalendar(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "calendar tool unavailable", Source: "calendar", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("calendar")
	if err != nil {
		return &ActionResult{Data: "calendar tool not found", Source: "calendar", NeedsLLM: true}
	}

	days := "7"
	lower := strings.ToLower(nlu.Raw)
	if strings.Contains(lower, "today") {
		days = "1"
	} else if strings.Contains(lower, "tomorrow") {
		days = "2"
	} else if strings.Contains(lower, "this week") {
		days = "7"
	}

	result, err := tool.Execute(map[string]string{"days": days})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("calendar error: %v", err), Source: "calendar", NeedsLLM: true}
	}
	if result == "" {
		return &ActionResult{DirectResponse: "No upcoming events.", Source: "calendar"}
	}
	return &ActionResult{DirectResponse: result, Source: "calendar"}
}

// handleCheckEmail checks for new emails.
func (ar *ActionRouter) handleCheckEmail(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "email tool unavailable", Source: "email", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("email")
	if err != nil {
		return &ActionResult{Data: "email tool not found", Source: "email", NeedsLLM: true}
	}

	result, err := tool.Execute(map[string]string{"count": "5"})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("email error: %v", err), Source: "email", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "email"}
}

// handleScreenshot captures a screenshot.
func (ar *ActionRouter) handleScreenshot(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: "screenshot tool unavailable", Source: "screenshot", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("screenshot")
	if err != nil {
		return &ActionResult{Data: "screenshot tool not found", Source: "screenshot", NeedsLLM: true}
	}

	result, err := tool.Execute(map[string]string{})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("screenshot error: %v", err), Source: "screenshot", NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: "screenshot"}
}

// handleGenericTool is a reusable handler for tools that accept the raw query.
// It passes the full message as "query" and all entities as individual args.
func (ar *ActionRouter) handleGenericTool(nlu *NLUResult, toolName string) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{Data: toolName + " unavailable", Source: toolName, NeedsLLM: true}
	}
	tool, err := ar.Tools.Get(toolName)
	if err != nil {
		return &ActionResult{Data: toolName + " not found", Source: toolName, NeedsLLM: true}
	}

	args := make(map[string]string)
	args["query"] = nlu.Raw
	for k, v := range nlu.Entities {
		args[k] = v
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("%s error: %v", toolName, err), Source: toolName, NeedsLLM: true}
	}
	return &ActionResult{DirectResponse: result, Source: toolName}
}
