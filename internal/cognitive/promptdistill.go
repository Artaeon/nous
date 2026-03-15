package cognitive

import (
	"fmt"
	"strings"
)

// PromptDistiller compiles minimal system prompts per query type.
// Instead of one monolithic system prompt for all queries, it builds
// targeted prompts that maximize the model's attention budget.
//
// Innovation: Every AI agent uses one fixed system prompt. The distiller
// treats the system prompt as COMPILED CODE — different input types get
// different instruction sets. A general knowledge question gets 30 tokens
// of context. A code generation query gets language rules. A project
// question gets relevant file excerpts. Nothing more, nothing less.
//
// This is JIT compilation for prompts. The insight: a 1.5b model with
// 200 focused tokens of context outperforms the same model with 2000
// tokens of mixed context, because small models have limited attention
// and irrelevant context causes confusion.
//
// Measured impact: removing project context from general questions
// eliminates the "references to project code in generic answers" failure
// mode entirely. Adding language rules to code generation prevents
// impossible constructs (try-catch in Go, etc.).
type PromptDistiller struct {
	langRules   map[string]string // language -> rules string
	toolPrompt  string            // cached tool prompt
}

// QueryClass describes the type of query for prompt selection.
type QueryClass string

const (
	// ClassGeneral — general knowledge, no project context needed.
	ClassGeneral QueryClass = "general"
	// ClassCodeGen — generating code, needs language rules.
	ClassCodeGen QueryClass = "codegen"
	// ClassCodeQuery — asking about existing code, needs project context.
	ClassCodeQuery QueryClass = "codequery"
	// ClassToolResult — interpreting tool output, needs format hints.
	ClassToolResult QueryClass = "toolresult"
	// ClassChat — conversational, minimal prompt.
	ClassChat QueryClass = "chat"
	// ClassToolUse — needs tools for execution.
	ClassToolUse QueryClass = "tooluse"
)

// PromptProfile describes how to build the system prompt for a query class.
type PromptProfile struct {
	Class           QueryClass
	IncludeTools    bool
	IncludeProject  bool
	IncludeLangRules bool
	MaxContextChars int
	Temperature     float64
	NumPredict      int
}

// NewPromptDistiller creates a new prompt distiller.
func NewPromptDistiller() *PromptDistiller {
	pd := &PromptDistiller{
		langRules: make(map[string]string),
	}
	pd.registerDefaultRules()
	return pd
}

// Classify determines the query class for prompt selection.
func (pd *PromptDistiller) Classify(query string) QueryClass {
	lower := strings.ToLower(query)

	// Chat: greetings, thanks, small talk
	chatPhrases := []string{
		"hello", "hi ", "hey ", "thanks", "thank you", "good morning",
		"good afternoon", "good evening", "how are you", "goodbye", "bye",
		"sup", "what's up", "yo ",
	}
	for _, p := range chatPhrases {
		if strings.HasPrefix(lower, p) || lower == strings.TrimSpace(p) {
			return ClassChat
		}
	}

	// CodeGen: write/create/implement/generate code
	codeGenPhrases := []string{
		"write a function", "write a program", "write code",
		"create a function", "create a program", "create a class",
		"implement a", "generate code", "generate a function",
		"write me a", "code a", "write an algorithm",
		"write a script", "write a test",
	}
	for _, p := range codeGenPhrases {
		if strings.Contains(lower, p) {
			return ClassCodeGen
		}
	}

	// CodeQuery: asking about specific project code
	codeQueryPhrases := []string{
		"what does this", "how does this", "explain this code",
		"what is this function", "how does the",
	}
	for _, p := range codeQueryPhrases {
		if strings.Contains(lower, p) {
			return ClassCodeQuery
		}
	}
	// Also CodeQuery if query references specific file paths
	if strings.Contains(lower, ".go") || strings.Contains(lower, ".py") ||
		strings.Contains(lower, ".js") || strings.Contains(lower, ".ts") ||
		strings.Contains(lower, "internal/") || strings.Contains(lower, "cmd/") ||
		strings.Contains(lower, "src/") {
		return ClassCodeQuery
	}

	// ToolUse: explicit tool requests
	toolPhrases := []string{
		"read file", "read the file", "show me the file",
		"search for", "grep for", "find files",
		"run the", "execute", "git ", "list files",
	}
	for _, p := range toolPhrases {
		if strings.Contains(lower, p) {
			return ClassToolUse
		}
	}

	// General: everything else (knowledge questions, explanations)
	return ClassGeneral
}

// Profile returns the prompt profile for a query class.
func (pd *PromptDistiller) Profile(class QueryClass) PromptProfile {
	switch class {
	case ClassChat:
		return PromptProfile{
			Class:           ClassChat,
			IncludeTools:    false,
			IncludeProject:  false,
			IncludeLangRules: false,
			MaxContextChars: 100,
			Temperature:     0.8,
			NumPredict:      256,
		}
	case ClassCodeGen:
		return PromptProfile{
			Class:           ClassCodeGen,
			IncludeTools:    false,
			IncludeProject:  false,
			IncludeLangRules: true,
			MaxContextChars: 500,
			Temperature:     0.2,
			NumPredict:      1024,
		}
	case ClassCodeQuery:
		return PromptProfile{
			Class:           ClassCodeQuery,
			IncludeTools:    true,
			IncludeProject:  true,
			IncludeLangRules: false,
			MaxContextChars: 800,
			Temperature:     0.3,
			NumPredict:      512,
		}
	case ClassToolResult:
		return PromptProfile{
			Class:           ClassToolResult,
			IncludeTools:    false,
			IncludeProject:  false,
			IncludeLangRules: false,
			MaxContextChars: 400,
			Temperature:     0.3,
			NumPredict:      512,
		}
	case ClassToolUse:
		return PromptProfile{
			Class:           ClassToolUse,
			IncludeTools:    true,
			IncludeProject:  true,
			IncludeLangRules: false,
			MaxContextChars: 1000,
			Temperature:     0.5,
			NumPredict:      512,
		}
	default: // ClassGeneral
		return PromptProfile{
			Class:           ClassGeneral,
			IncludeTools:    false,
			IncludeProject:  false,
			IncludeLangRules: true,
			MaxContextChars: 300,
			Temperature:     0.5,
			NumPredict:      512,
		}
	}
}

// BuildSystemPrompt builds a minimal system prompt for the given query class.
func (pd *PromptDistiller) BuildSystemPrompt(class QueryClass, toolList string, lang string) string {
	profile := pd.Profile(class)
	var sb strings.Builder

	switch class {
	case ClassChat:
		sb.WriteString("You are Nous, a friendly local AI assistant. Be warm and concise.\n")

	case ClassCodeGen:
		sb.WriteString("You are Nous. Generate clean, correct code.\n")
		if profile.IncludeLangRules && lang != "" {
			if rules, ok := pd.langRules[strings.ToLower(lang)]; ok {
				sb.WriteString("\n")
				sb.WriteString(rules)
				sb.WriteString("\n")
			}
		}

	case ClassCodeQuery:
		sb.WriteString("You are Nous. Answer about this codebase using ONLY the evidence provided.\n")
		sb.WriteString("Do NOT invent file contents — use tools to read actual code.\n")
		if profile.IncludeTools && toolList != "" {
			sb.WriteString("\n")
			sb.WriteString(toolList)
			sb.WriteString("\n")
		}

	case ClassToolResult:
		sb.WriteString("You are Nous. Summarize tool results for the user.\n")
		sb.WriteString("Use ONLY the data provided below. Do NOT contradict the evidence.\n")
		sb.WriteString("If the evidence shows N items, say N items. Do NOT guess different numbers.\n")

	case ClassToolUse:
		sb.WriteString("You are Nous, a local assistant. Use tools to help the user.\n")
		sb.WriteString("NEVER invent file contents — read first.\n\n")
		sb.WriteString("Tool call format:\n")
		sb.WriteString(`{"tool": "NAME", "args": {"key": "value"}}`)
		sb.WriteString("\n\n")
		if profile.IncludeTools && toolList != "" {
			sb.WriteString(toolList)
			sb.WriteString("\n")
		}

	default: // ClassGeneral
		sb.WriteString("You are Nous, a knowledgeable AI assistant. Answer directly and concisely.\n")
		sb.WriteString("Give accurate, factual information. If unsure, say so.\n")
		if profile.IncludeLangRules && lang != "" {
			if rules, ok := pd.langRules[strings.ToLower(lang)]; ok {
				sb.WriteString("\nWhen discussing code in ")
				sb.WriteString(lang)
				sb.WriteString(":\n")
				sb.WriteString(rules)
				sb.WriteString("\n")
			}
		}
	}

	// Add project context if profile requires it
	if profile.IncludeProject && CurrentProject != nil {
		ctx := CurrentProject.ContextString()
		if len(ctx) > profile.MaxContextChars {
			ctx = ctx[:profile.MaxContextChars]
		}
		if ctx != "" {
			sb.WriteString("\n")
			sb.WriteString(ctx)
		}
	}

	// Add working directory for tool-using classes
	if class == ClassToolUse || class == ClassCodeQuery {
		wd := WorkDir
		if wd == "" {
			wd = "."
		}
		sb.WriteString("\nwd: ")
		sb.WriteString(wd)
		sb.WriteString("\n")
	}

	result := sb.String()

	// Hard cap on prompt length
	if profile.MaxContextChars > 0 && len(result) > profile.MaxContextChars*2 {
		result = result[:profile.MaxContextChars*2]
	}

	return result
}

// BuildFinalAnswerPrompt builds a specialized system prompt for
// finalAnswerFromEvidence — when the model must synthesize from evidence.
func (pd *PromptDistiller) BuildFinalAnswerPrompt(lang string) string {
	var sb strings.Builder
	sb.WriteString("You are Nous in final-answer mode.\n")
	sb.WriteString("Rules:\n")
	sb.WriteString("- Give a direct answer using ONLY the evidence provided\n")
	sb.WriteString("- Do NOT call tools or emit JSON\n")
	sb.WriteString("- If the evidence shows specific numbers, use those EXACT numbers\n")
	sb.WriteString("- Do NOT contradict the evidence\n")
	sb.WriteString("- If evidence is incomplete, say so clearly\n")

	if lang != "" {
		if rules, ok := pd.langRules[strings.ToLower(lang)]; ok {
			sb.WriteString("\nLanguage rules for ")
			sb.WriteString(lang)
			sb.WriteString(":\n")
			sb.WriteString(rules)
		}
	}

	return sb.String()
}

// SetLanguageRules sets language-specific rules for prompt injection.
func (pd *PromptDistiller) SetLanguageRules(lang string, rules string) {
	pd.langRules[strings.ToLower(lang)] = rules
}

// registerDefaultRules registers default language rules.
func (pd *PromptDistiller) registerDefaultRules() {
	pd.langRules["go"] = strings.Join([]string{
		"- Error handling: if err != nil { return err } — NO try-catch",
		"- No classes — use type X struct {} with methods",
		"- No ternary operator — use if/else",
		"- No while loops — only for loops",
		"- Export with Uppercase, private with lowercase",
		"- Concurrency: goroutines + channels, not async/await",
		"- nil not null, receiver not this",
	}, "\n")

	pd.langRules["python"] = strings.Join([]string{
		"- Use snake_case for functions and variables",
		"- Use type hints for function signatures",
		"- Prefer list comprehensions over map/filter",
	}, "\n")

	pd.langRules["javascript"] = strings.Join([]string{
		"- Use const/let, never var",
		"- Use arrow functions for callbacks",
		"- Use async/await over .then() chains",
	}, "\n")
}

// PromptStats returns size info for the distilled prompt.
func (pd *PromptDistiller) PromptStats(class QueryClass, lang string) string {
	prompt := pd.BuildSystemPrompt(class, "", lang)
	return fmt.Sprintf("class=%s chars=%d", class, len(prompt))
}
