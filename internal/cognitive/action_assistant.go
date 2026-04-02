package cognitive

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

// normalizeMathFunc converts natural-language math function calls to proper syntax.
// "sqrt of 144" → "sqrt(144)", "abs of -5" → "abs(-5)"
var mathFuncNormRe = regexp.MustCompile(`(?i)^(sqrt|abs|sin|cos|tan|log|ln|ceil|floor|round)\s+(?:of\s+)?(-?[\d.]+)$`)

func normalizeMathFunc(expr string) string {
	if m := mathFuncNormRe.FindStringSubmatch(expr); len(m) == 3 {
		return strings.ToLower(m[1]) + "(" + m[2] + ")"
	}
	return expr
}

// convertMathWords replaces English math words with operator symbols.
// "15 times 23" → "15 * 23", "100 divided by 5" → "100 / 5"
func convertMathWords(expr string) string {
	r := regexp.MustCompile(`(?i)\s+times\s+`)
	expr = r.ReplaceAllString(expr, " * ")
	r = regexp.MustCompile(`(?i)\s+multiplied\s+by\s+`)
	expr = r.ReplaceAllString(expr, " * ")
	r = regexp.MustCompile(`(?i)\s+divided\s+by\s+`)
	expr = r.ReplaceAllString(expr, " / ")
	r = regexp.MustCompile(`(?i)\s+plus\s+`)
	expr = r.ReplaceAllString(expr, " + ")
	r = regexp.MustCompile(`(?i)\s+minus\s+`)
	expr = r.ReplaceAllString(expr, " - ")
	r = regexp.MustCompile(`(?i)\s+to\s+the\s+power\s+of\s+`)
	expr = r.ReplaceAllString(expr, " ^ ")
	r = regexp.MustCompile(`(?i)\s+mod\s+`)
	expr = r.ReplaceAllString(expr, " % ")
	return strings.TrimSpace(expr)
}

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
		return &ActionResult{DirectResponse: "weather tool unavailable", Source: "weather"}
	}
	tool, err := ar.Tools.Get("weather")
	if err != nil {
		return &ActionResult{DirectResponse: "weather tool not found", Source: "weather"}
	}
	result, err := tool.Execute(map[string]string{"location": location})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("weather error: %v", err), Source: "weather"}
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
		return &ActionResult{DirectResponse: "conversion tools unavailable", Source: "convert"}
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

	return &ActionResult{DirectResponse: expr, Source: "convert"}
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
		return &ActionResult{
			DirectResponse: "I couldn't figure out when to remind you. Try something like: \"remind me in 30 minutes to call mom\" or \"remind me tomorrow to buy groceries\".",
			Source:         "reminder",
		}
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
		return &ActionResult{DirectResponse: "sysinfo tool unavailable", Source: "sysinfo"}
	}
	tool, err := ar.Tools.Get("sysinfo")
	if err != nil {
		return &ActionResult{DirectResponse: "sysinfo tool not found", Source: "sysinfo"}
	}

	query := nlu.Entities["topic"]
	if query == "" {
		query = nlu.Raw
	}
	result, err := tool.Execute(map[string]string{"query": query})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("sysinfo error: %v", err), Source: "sysinfo"}
	}
	return &ActionResult{DirectResponse: result, Source: "sysinfo"}
}

// handleClipboard reads from or writes to the system clipboard.
func (ar *ActionRouter) handleClipboard(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "clipboard tool unavailable", Source: "clipboard"}
	}
	tool, err := ar.Tools.Get("clipboard")
	if err != nil {
		return &ActionResult{DirectResponse: "clipboard tool not found", Source: "clipboard"}
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
		return &ActionResult{DirectResponse: fmt.Sprintf("clipboard error: %v", err), Source: "clipboard"}
	}

	if args["action"] == "read" {
		return &ActionResult{DirectResponse: fmt.Sprintf("Clipboard contents:\n%s", result), Source: "clipboard"}
	}
	return &ActionResult{DirectResponse: result, Source: "clipboard"}
}

// handleNotes manages markdown notes.
func (ar *ActionRouter) handleNotes(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "notes tool unavailable", Source: "notes"}
	}
	tool, err := ar.Tools.Get("notes")
	if err != nil {
		return &ActionResult{DirectResponse: "notes tool not found", Source: "notes"}
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
		return &ActionResult{DirectResponse: fmt.Sprintf("notes error: %v", err), Source: "notes"}
	}
	return &ActionResult{DirectResponse: result, Source: "notes"}
}

// handleTodos manages the todo list.
func (ar *ActionRouter) handleTodos(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "todos tool unavailable", Source: "todos"}
	}
	tool, err := ar.Tools.Get("todos")
	if err != nil {
		return &ActionResult{DirectResponse: "todos tool not found", Source: "todos"}
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
		return &ActionResult{DirectResponse: fmt.Sprintf("todos error: %v", err), Source: "todos"}
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
		return &ActionResult{DirectResponse: "file finder unavailable", Source: "filefinder"}
	}
	tool, err := ar.Tools.Get("filefinder")
	if err != nil {
		return &ActionResult{DirectResponse: "file finder not found", Source: "filefinder"}
	}

	result, err := tool.Execute(map[string]string{"query": nlu.Raw})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("file finder error: %v", err), Source: "filefinder"}
	}
	return &ActionResult{DirectResponse: result, Source: "filefinder"}
}

// handleSummarizeURL fetches and summarizes a URL.
func (ar *ActionRouter) handleSummarizeURL(nlu *NLUResult) *ActionResult {
	url := nlu.Entities["url"]
	if url == "" {
		return &ActionResult{DirectResponse: "no URL provided to summarize", Source: "summarize"}
	}

	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "summarize tool unavailable", Source: "summarize"}
	}
	tool, err := ar.Tools.Get("summarize")
	if err != nil {
		return &ActionResult{DirectResponse: "summarize tool not found", Source: "summarize"}
	}

	result, err := tool.Execute(map[string]string{"url": url})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("summarize error: %v", err), Source: "summarize"}
	}

	// Store in working memory for raw retrieval
	if ar.WorkingMem != nil && result != "" {
		memContent := result
		if len(memContent) > 4000 {
			memContent = memContent[:4000]
		}
		ar.WorkingMem.Store("fetched:"+url, memContent, 0.9)
	}

	// Extract facts for follow-up questions
	topic := extractTopicFromURL(url)
	if ar.Tracker != nil && result != "" {
		ar.Tracker.IngestContent(result, url, topic)
	}

	// Use extractive summarization instead of LLM
	if ar.Tracker != nil {
		summary := ar.Tracker.TopicSummary()
		if summary != "" {
			return &ActionResult{
				DirectResponse: summary,
				Source:         "summarize",
			}
		}
	}

	// Fallback: return content directly
	return &ActionResult{
		DirectResponse: fmt.Sprintf("[URL Content from %s]\n%s", url, result),
		Source:         "summarize",
	}
}

// handleNews fetches RSS/news feeds.
func (ar *ActionRouter) handleNews(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "news tool unavailable", Source: "news"}
	}
	tool, err := ar.Tools.Get("rss")
	if err != nil {
		return &ActionResult{DirectResponse: "news tool not found", Source: "news"}
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
		return &ActionResult{DirectResponse: fmt.Sprintf("news error: %v", err), Source: "news"}
	}
	return &ActionResult{DirectResponse: result, Source: "news"}
}

// handleRunCode executes code in a sandbox.
func (ar *ActionRouter) handleRunCode(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "code runner unavailable", Source: "coderunner"}
	}
	tool, err := ar.Tools.Get("coderunner")
	if err != nil {
		return &ActionResult{DirectResponse: "code runner not found", Source: "coderunner"}
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
		return &ActionResult{DirectResponse: nlu.Raw, Source: "coderunner"}
	}
	result, err := tool.Execute(map[string]string{"code": code, "language": lang})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("code runner error: %v", err), Source: "coderunner"}
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
		return &ActionResult{DirectResponse: "calendar tool unavailable", Source: "calendar"}
	}
	tool, err := ar.Tools.Get("calendar")
	if err != nil {
		return &ActionResult{DirectResponse: "calendar tool not found", Source: "calendar"}
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
		return &ActionResult{DirectResponse: fmt.Sprintf("calendar error: %v", err), Source: "calendar"}
	}
	if result == "" {
		return &ActionResult{DirectResponse: "No upcoming events.", Source: "calendar"}
	}
	return &ActionResult{DirectResponse: result, Source: "calendar"}
}

// handleCheckEmail checks for new emails.
func (ar *ActionRouter) handleCheckEmail(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "email tool unavailable", Source: "email"}
	}
	tool, err := ar.Tools.Get("email")
	if err != nil {
		return &ActionResult{DirectResponse: "email tool not found", Source: "email"}
	}

	result, err := tool.Execute(map[string]string{"count": "5"})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("email error: %v", err), Source: "email"}
	}
	return &ActionResult{DirectResponse: result, Source: "email"}
}

// handleScreenshot captures a screenshot.
func (ar *ActionRouter) handleScreenshot(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "screenshot tool unavailable", Source: "screenshot"}
	}
	tool, err := ar.Tools.Get("screenshot")
	if err != nil {
		return &ActionResult{DirectResponse: "screenshot tool not found", Source: "screenshot"}
	}

	result, err := tool.Execute(map[string]string{})
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("screenshot error: %v", err), Source: "screenshot"}
	}
	return &ActionResult{DirectResponse: result, Source: "screenshot"}
}

// handleGenericTool is a reusable handler for tools that accept the raw query.
// It passes the full message as "query" and all entities as individual args.
func (ar *ActionRouter) handleGenericTool(nlu *NLUResult, toolName string) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: toolName + " unavailable", Source: toolName}
	}
	tool, err := ar.Tools.Get(toolName)
	if err != nil {
		return &ActionResult{DirectResponse: toolName + " not found", Source: toolName}
	}

	args := make(map[string]string)
	args["query"] = nlu.Raw
	for k, v := range nlu.Entities {
		args[k] = v
	}

	// Map NLU entities to tool-specific argument names.
	// The NLU extracts a "topic" but each tool has its own arg names.
	topic := nlu.Entities["topic"]
	switch toolName {
	case "dict":
		// dict expects "word" (required) and "action" (define/synonyms/antonyms)
		if topic != "" {
			args["word"] = topic
		}
		if args["action"] == "" {
			args["action"] = "define"
		}
	case "translate":
		// translate expects "text" (required) and "to" (required)
		ar.extractTranslateArgs(nlu.Raw, args)
	}

	timeout := genericToolTimeout(toolName)
	result, err := executeToolWithTimeout(tool.Execute, args, timeout)
	if err != nil {
		// Never surface raw tool errors to the user.
		return &ActionResult{
			DirectResponse: fmt.Sprintf("I tried to use %s for that, but I need more specific information. Could you rephrase what you'd like to do?", toolName),
			Source:         toolName,
		}
	}
	return &ActionResult{DirectResponse: result, Source: toolName}
}

func genericToolTimeout(toolName string) time.Duration {
	switch toolName {
	case "translate", "netcheck", "news", "weather", "email", "fetch", "summarize":
		return 4 * time.Second
	default:
		return 2 * time.Second
	}
}

func executeToolWithTimeout(execFn func(args map[string]string) (string, error), args map[string]string, timeout time.Duration) (string, error) {
	type toolResult struct {
		out string
		err error
	}
	ch := make(chan toolResult, 1)

	go func() {
		out, err := execFn(args)
		ch <- toolResult{out: out, err: err}
	}()

	select {
	case res := <-ch:
		return res.out, res.err
	case <-time.After(timeout):
		return "", fmt.Errorf("timed out after %s", timeout)
	}
}

// extractTranslateArgs parses "translate X to Y" or "X in Y" patterns.
func (ar *ActionRouter) extractTranslateArgs(raw string, args map[string]string) {
	lower := strings.ToLower(raw)

	// Pattern: "translate X to/into Y"
	for _, sep := range []string{" to ", " into "} {
		idx := strings.LastIndex(lower, sep)
		if idx > 0 {
			textPart := raw[:idx]
			langPart := strings.TrimSpace(raw[idx+len(sep):])

			// Strip "translate" prefix from text
			textLower := strings.ToLower(textPart)
			for _, prefix := range []string{"translate ", "can you translate ", "please translate "} {
				if strings.HasPrefix(textLower, prefix) {
					textPart = textPart[len(prefix):]
					break
				}
			}
			textPart = strings.TrimSpace(textPart)
			if textPart != "" && langPart != "" {
				args["text"] = textPart
				args["to"] = langPart
				return
			}
		}
	}

	// Pattern: "how do you say X in Y" / "X in spanish"
	for _, sep := range []string{" in "} {
		idx := strings.LastIndex(lower, sep)
		if idx > 0 {
			langPart := strings.TrimSpace(raw[idx+len(sep):])
			textPart := raw[:idx]
			// Strip common prefixes
			textLower := strings.ToLower(textPart)
			for _, prefix := range []string{"how do you say ", "how to say ", "say "} {
				if strings.HasPrefix(textLower, prefix) {
					textPart = textPart[len(prefix):]
					break
				}
			}
			textPart = strings.TrimSpace(textPart)
			if textPart != "" && langPart != "" {
				args["text"] = textPart
				args["to"] = langPart
				return
			}
		}
	}
}

// handleCalculate extracts a math expression from natural language and evaluates it.
func (ar *ActionRouter) handleCalculate(nlu *NLUResult) *ActionResult {
	// Extract the math expression from natural language
	expr := nlu.Entities["expression"]
	if expr == "" {
		expr = nlu.Entities["expr"]
	}
	if expr == "" {
		// Strip common prefixes to get the raw expression
		expr = nlu.Raw
		lower := strings.ToLower(expr)
		for _, prefix := range []string{
			"calculate ", "compute ", "evaluate ", "eval ", "solve ",
			"what is ", "what's ", "how much is ", "whats ",
		} {
			if strings.HasPrefix(lower, prefix) {
				expr = expr[len(prefix):]
				break
			}
		}
		expr = strings.TrimRight(expr, "?!. ")
	}

	// Normalize "sqrt of 144" → "sqrt(144)", "abs of -5" → "abs(-5)", etc.
	expr = normalizeMathFunc(expr)

	// Convert English operator words to symbols
	expr = convertMathWords(expr)

	// Try the calculator tool first.
	if ar.Tools != nil {
		if tool, err := ar.Tools.Get("calculator"); err == nil {
			result, err := tool.Execute(map[string]string{"expression": expr})
			if err == nil {
				return &ActionResult{DirectResponse: fmt.Sprintf("%s = %s", expr, result), Source: "calculator"}
			}
		}
	}

	// Fallback: use the built-in math evaluator directly.
	result, err := evaluateMath(expr)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("Could not calculate: %v", err), Source: "calculator"}
	}
	return &ActionResult{DirectResponse: fmt.Sprintf("%s = %s", expr, result), Source: "computed"}
}

// handlePassword generates passwords, passphrases, or PINs based on the request.
func (ar *ActionRouter) handlePassword(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "password tool unavailable", Source: "password"}
	}
	tool, err := ar.Tools.Get("password")
	if err != nil {
		return &ActionResult{DirectResponse: "password tool not found", Source: "password"}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{"type": "password"}

	if strings.Contains(lower, "passphrase") || strings.Contains(lower, "phrase") {
		args["type"] = "passphrase"
	} else if strings.Contains(lower, "pin") {
		args["type"] = "pin"
	}

	// Extract length from natural language ("16 characters", "6 digit", "5 words")
	if m := extractNumberBefore(lower, []string{"char", "long", "length", "digit", "word"}); m != "" {
		args["length"] = m
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("Could not generate: %v", err), Source: "password"}
	}

	label := "Password"
	if args["type"] == "passphrase" {
		label = "Passphrase"
	} else if args["type"] == "pin" {
		label = "PIN"
	}
	return &ActionResult{DirectResponse: fmt.Sprintf("%s: %s", label, result), Source: "password"}
}

// handleBookmark manages bookmark operations (save, list, search, delete).
func (ar *ActionRouter) handleBookmark(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "bookmarks unavailable", Source: "bookmarks"}
	}
	tool, err := ar.Tools.Get("bookmarks")
	if err != nil {
		return &ActionResult{DirectResponse: "bookmarks not found", Source: "bookmarks"}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{}

	switch {
	case strings.Contains(lower, "delete") || strings.Contains(lower, "remove"):
		args["action"] = "delete"
		if url := nlu.Entities["url"]; url != "" {
			args["url"] = url
		}
	case strings.Contains(lower, "search") || strings.Contains(lower, "find"):
		args["action"] = "search"
		args["query"] = nlu.Entities["topic"]
		if args["query"] == "" {
			args["query"] = nlu.Entities["quoted"]
		}
	case strings.Contains(lower, "save") || strings.Contains(lower, "bookmark this") ||
		strings.Contains(lower, "add bookmark") || nlu.Entities["url"] != "":
		args["action"] = "save"
		if url := nlu.Entities["url"]; url != "" {
			args["url"] = url
		}
		if title := nlu.Entities["quoted"]; title != "" {
			args["title"] = title
		} else if topic := nlu.Entities["topic"]; topic != "" {
			args["title"] = topic
		}
	default:
		args["action"] = "list"
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("Bookmark error: %v", err), Source: "bookmarks"}
	}
	return &ActionResult{DirectResponse: result, Source: "bookmarks"}
}

// handleJournal manages journal entries (write, today, list, search).
func (ar *ActionRouter) handleJournal(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "journal unavailable", Source: "journal"}
	}
	tool, err := ar.Tools.Get("journal")
	if err != nil {
		return &ActionResult{DirectResponse: "journal not found", Source: "journal"}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{}

	// Check if the input looks like a journal entry (starts with diary/journal prefix)
	hasEntryPrefix := false
	for _, prefix := range []string{
		"dear diary", "journal entry", "journal:", "write in journal",
		"diary ", "diary:", "add journal", "log journal", "new journal entry",
	} {
		if strings.HasPrefix(lower, prefix) {
			hasEntryPrefix = true
			break
		}
	}

	switch {
	case hasEntryPrefix:
		// Entry prefix detected — always write, even if "today" appears in the text
		args["action"] = "write"
	case strings.Contains(lower, "search") || strings.Contains(lower, "find"):
		args["action"] = "search"
		args["query"] = nlu.Entities["topic"]
		if args["query"] == "" {
			args["query"] = nlu.Entities["quoted"]
		}
	case strings.Contains(lower, "today") || strings.Contains(lower, "today's"):
		args["action"] = "today"
	case strings.Contains(lower, "this week") || strings.Contains(lower, "week"):
		args["action"] = "week"
	case strings.Contains(lower, "list") || strings.Contains(lower, "show") ||
		strings.Contains(lower, "entries") || strings.Contains(lower, "history"):
		args["action"] = "list"
	default:
		// Default: write a journal entry
		args["action"] = "write"
	}

	// For write action, extract the entry text
	if args["action"] == "write" {
		entry := extractJournalEntry(nlu.Raw)
		if entry == "" && nlu.Entities["quoted"] != "" {
			entry = nlu.Entities["quoted"]
		}
		if entry == "" {
			entry = nlu.Raw
		}
		args["entry"] = entry

		// Extract mood (1-5) if mentioned
		if m := extractNumberBefore(lower, []string{"mood", "feeling", "/5", "out of 5"}); m != "" {
			args["mood"] = m
		}
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("Journal error: %v", err), Source: "journal"}
	}

	// Record for causal analysis
	if ar.Causal != nil && args["action"] == "write" {
		tags := map[string]string{}
		if args["mood"] != "" {
			tags["mood"] = args["mood"]
		}
		ar.Causal.RecordEvent("journal", tags)
	}

	return &ActionResult{DirectResponse: result, Source: "journal"}
}

// handleHabit manages habit tracking (create, check, list, status, delete).
func (ar *ActionRouter) handleHabit(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "habits unavailable", Source: "habits"}
	}
	tool, err := ar.Tools.Get("habits")
	if err != nil {
		return &ActionResult{DirectResponse: "habits not found", Source: "habits"}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{}

	switch {
	case strings.Contains(lower, "delete") || strings.Contains(lower, "remove"):
		args["action"] = "delete"
		args["name"] = extractHabitName(nlu)
	case strings.Contains(lower, "check") || strings.Contains(lower, "done") ||
		strings.Contains(lower, "completed") || strings.Contains(lower, "did") ||
		strings.Contains(lower, "mark") || strings.Contains(lower, "log"):
		args["action"] = "check"
		args["name"] = extractHabitName(nlu)
	case strings.Contains(lower, "status") || strings.Contains(lower, "streak") ||
		strings.Contains(lower, "how am i doing") || strings.Contains(lower, "progress"):
		args["action"] = "status"
		args["name"] = extractHabitName(nlu)
	case strings.Contains(lower, "create") || strings.Contains(lower, "new habit") ||
		strings.Contains(lower, "add habit") || strings.Contains(lower, "start tracking") ||
		strings.Contains(lower, "track"):
		args["action"] = "create"
		args["name"] = extractHabitName(nlu)
		if strings.Contains(lower, "weekly") {
			args["frequency"] = "weekly"
		}
	default:
		args["action"] = "list"
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("Habit error: %v", err), Source: "habits"}
	}
	return &ActionResult{DirectResponse: result, Source: "habits"}
}

// handleExpense manages expense tracking (add, list, summary, delete).
func (ar *ActionRouter) handleExpense(nlu *NLUResult) *ActionResult {
	if ar.Tools == nil {
		return &ActionResult{DirectResponse: "expenses unavailable", Source: "expenses"}
	}
	tool, err := ar.Tools.Get("expenses")
	if err != nil {
		return &ActionResult{DirectResponse: "expenses not found", Source: "expenses"}
	}

	lower := strings.ToLower(nlu.Raw)
	args := map[string]string{}

	switch {
	case strings.Contains(lower, "summary") || strings.Contains(lower, "total") ||
		strings.Contains(lower, "how much") || strings.Contains(lower, "report"):
		args["action"] = "summary"
		if strings.Contains(lower, "today") {
			args["period"] = "today"
		} else if strings.Contains(lower, "this week") || strings.Contains(lower, "week") {
			args["period"] = "week"
		} else if strings.Contains(lower, "this month") || strings.Contains(lower, "month") {
			args["period"] = "month"
		}
	case strings.Contains(lower, "delete") || strings.Contains(lower, "remove") ||
		strings.Contains(lower, "undo"):
		args["action"] = "delete"
	case strings.Contains(lower, "list") || strings.Contains(lower, "show") ||
		strings.Contains(lower, "my expense") || strings.Contains(lower, "spending"):
		args["action"] = "list"
	default:
		// Default: add an expense — extract amount and description
		args["action"] = "add"
		amount, desc, cat := extractExpenseDetails(nlu.Raw)
		if amount != "" {
			args["amount"] = amount
		}
		if desc != "" {
			args["description"] = desc
		}
		if cat != "" {
			args["category"] = cat
		}
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{DirectResponse: fmt.Sprintf("Expense error: %v", err), Source: "expenses"}
	}

	// Record for causal analysis
	if ar.Causal != nil && args["action"] == "add" {
		tags := map[string]string{}
		if args["category"] != "" {
			tags["category"] = args["category"]
		}
		if args["amount"] != "" {
			tags["amount"] = args["amount"]
		}
		ar.Causal.RecordEvent("expense", tags)
	}

	return &ActionResult{DirectResponse: result, Source: "expenses"}
}

// -----------------------------------------------------------------------
// Entity extraction helpers for dedicated handlers.
// -----------------------------------------------------------------------

// extractNumberBefore finds a number that appears before any of the given suffix words.
// e.g. extractNumberBefore("generate 16 character password", ["char"]) → "16"
func extractNumberBefore(lower string, suffixes []string) string {
	numberRe := regexp.MustCompile(`(\d+)\s*(?:` + strings.Join(suffixes, "|") + `)`)
	if m := numberRe.FindStringSubmatch(lower); len(m) >= 2 {
		return m[1]
	}
	// Also check for standalone number
	standaloneRe := regexp.MustCompile(`\b(\d+)\b`)
	if m := standaloneRe.FindStringSubmatch(lower); len(m) >= 2 {
		return m[1]
	}
	return ""
}

// extractJournalEntry strips journal-related prefixes from the input to get the entry text.
func extractJournalEntry(raw string) string {
	lower := strings.ToLower(raw)
	for _, prefix := range []string{
		"dear diary ", "journal entry: ", "journal: ", "write in journal: ",
		"journal entry ", "write journal ", "write in journal ",
		"add journal ", "log journal ", "new journal entry ",
		"diary ", "diary: ", "dear diary, ",
	} {
		if strings.HasPrefix(lower, prefix) {
			return strings.TrimSpace(raw[len(prefix):])
		}
	}
	// Try "journal" + content after it
	for _, marker := range []string{"journal ", "diary "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			rest := strings.TrimSpace(raw[idx+len(marker):])
			if rest != "" {
				return rest
			}
		}
	}
	return ""
}

// extractHabitName extracts the habit name from natural language.
func extractHabitName(nlu *NLUResult) string {
	if topic := nlu.Entities["topic"]; topic != "" {
		return topic
	}
	if quoted := nlu.Entities["quoted"]; quoted != "" {
		return quoted
	}
	// Strip common prefixes
	lower := strings.ToLower(nlu.Raw)
	for _, prefix := range []string{
		"check habit ", "check off ", "mark ", "did ", "completed ",
		"create habit ", "new habit ", "add habit ", "start tracking ",
		"track ", "delete habit ", "remove habit ", "habit status ",
		"status of ", "streak for ", "progress on ",
	} {
		if strings.HasPrefix(lower, prefix) {
			return strings.TrimSpace(nlu.Raw[len(prefix):])
		}
	}
	return nlu.Entities["topic"]
}

// extractExpenseDetails parses amount, description, and category from expense input.
// Handles patterns like "spent 25 on groceries", "coffee 4.50", "$12.99 lunch", "bought shoes for 89.99"
func extractExpenseDetails(raw string) (amount, description, category string) {
	lower := strings.ToLower(raw)

	// Strip common prefixes
	for _, prefix := range []string{
		"add expense ", "log expense ", "spent ", "i spent ", "i paid ",
		"bought ", "i bought ", "expense ", "paid ",
	} {
		if strings.HasPrefix(lower, prefix) {
			raw = raw[len(prefix):]
			lower = strings.ToLower(raw)
			break
		}
	}

	// Pattern 1: "$25.50 on groceries" or "25.50 on groceries"
	re1 := regexp.MustCompile(`(?i)\$?([\d.]+)\s+(?:on|for|at)\s+(.+)`)
	if m := re1.FindStringSubmatch(raw); len(m) >= 3 {
		amount = m[1]
		description = strings.TrimSpace(m[2])
		category = guessExpenseCategory(description)
		return
	}

	// Pattern 2: "groceries for 25.50" or "shoes for $89.99" or "coffee $4.50"
	re2a := regexp.MustCompile(`(?i)^(.+?)\s+(?:for|at)\s+\$?([\d.]+)\s*$`)
	if m := re2a.FindStringSubmatch(raw); len(m) >= 3 {
		description = strings.TrimSpace(m[1])
		amount = m[2]
		category = guessExpenseCategory(description)
		return
	}

	// Pattern 2b: "groceries 25.50" or "coffee $4.50" (no preposition)
	re2 := regexp.MustCompile(`(?i)^(.+?)\s+\$?([\d.]+)\s*$`)
	if m := re2.FindStringSubmatch(raw); len(m) >= 3 {
		description = strings.TrimSpace(m[1])
		amount = m[2]
		category = guessExpenseCategory(description)
		return
	}

	// Pattern 3: just a number — bare amount
	re3 := regexp.MustCompile(`\$?([\d.]+)`)
	if m := re3.FindStringSubmatch(raw); len(m) >= 2 {
		amount = m[1]
		rest := strings.TrimSpace(re3.ReplaceAllString(raw, ""))
		rest = strings.Trim(rest, "$")
		if rest != "" {
			description = rest
		}
		category = guessExpenseCategory(description)
		return
	}

	return
}

// handleDailyBriefing generates a personalized daily briefing.
// Gathers data from multiple tools (weather, habits, todos, expenses, calendar)
// and composes a single overview — zero LLM calls.
func (ar *ActionRouter) handleDailyBriefing(nlu *NLUResult) *ActionResult {
	toolResults := make(map[string]string)

	// Gather data from available tools
	if ar.Tools != nil {
		type briefingTool struct {
			key      string
			toolName string
			args     map[string]string
		}
		sources := []briefingTool{
			{"weather", "weather", map[string]string{"location": "auto"}},
			{"habits", "habits", map[string]string{"action": "list"}},
			{"todos", "todos", map[string]string{"action": "list"}},
			{"expenses", "expenses", map[string]string{"action": "summary", "period": "today"}},
			{"calendar", "calendar", map[string]string{"days": "1"}},
		}
		for _, s := range sources {
			tool, err := ar.Tools.Get(s.toolName)
			if err != nil {
				continue
			}
			result, err := tool.Execute(s.args)
			if err == nil && result != "" {
				toolResults[s.key] = result
			}
		}
	}

	// Use PersonalResponseGenerator if available
	if ar.PersonalResp != nil {
		briefing := ar.PersonalResp.DailyBriefing(toolResults)
		return &ActionResult{DirectResponse: briefing, Source: "briefing"}
	}

	// Fallback: simple concatenation
	var parts []string
	if w, ok := toolResults["weather"]; ok {
		parts = append(parts, "Weather: "+w)
	}
	if h, ok := toolResults["habits"]; ok {
		parts = append(parts, "Habits:\n"+h)
	}
	if t, ok := toolResults["todos"]; ok {
		parts = append(parts, "Tasks:\n"+t)
	}
	if e, ok := toolResults["expenses"]; ok {
		parts = append(parts, "Spending: "+e)
	}
	if c, ok := toolResults["calendar"]; ok {
		parts = append(parts, "Schedule:\n"+c)
	}
	if len(parts) == 0 {
		return &ActionResult{DirectResponse: "Good morning! No data available yet — set up your tools to get a personalized briefing.", Source: "briefing"}
	}
	return &ActionResult{DirectResponse: strings.Join(parts, "\n\n"), Source: "briefing"}
}

// guessExpenseCategory maps common expense descriptions to categories.
func guessExpenseCategory(desc string) string {
	lower := strings.ToLower(desc)
	categories := map[string][]string{
		"food":          {"coffee", "lunch", "dinner", "breakfast", "groceries", "restaurant", "takeout", "pizza", "burger", "snack", "meal", "food", "eat"},
		"transport":     {"uber", "lyft", "taxi", "gas", "fuel", "parking", "bus", "train", "metro", "subway"},
		"entertainment": {"movie", "netflix", "spotify", "game", "concert", "ticket", "book", "magazine"},
		"shopping":      {"amazon", "clothes", "shoes", "shirt", "pants", "electronics", "bought"},
		"health":        {"gym", "pharmacy", "doctor", "medicine", "vitamins", "health"},
		"bills":         {"rent", "electric", "water", "internet", "phone", "insurance", "subscription"},
	}
	for cat, words := range categories {
		for _, w := range words {
			if strings.Contains(lower, w) {
				return cat
			}
		}
	}
	return "other"
}
