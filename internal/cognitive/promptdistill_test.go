package cognitive

import (
	"strings"
	"testing"
)

// --- Prompt Distiller Tests ---

func TestPromptDistillerCreation(t *testing.T) {
	pd := NewPromptDistiller()
	if pd == nil {
		t.Fatal("NewPromptDistiller should not return nil")
	}
}

// --- Classification Tests ---

func TestClassifyChat(t *testing.T) {
	pd := NewPromptDistiller()

	chatQueries := []string{
		"hello",
		"hi there",
		"thanks",
		"thank you for your help",
		"good morning",
		"bye",
		"how are you",
	}

	for _, q := range chatQueries {
		class := pd.Classify(q)
		if class != ClassChat {
			t.Errorf("%q should be ClassChat, got %s", q, class)
		}
	}
}

func TestClassifyCodeGen(t *testing.T) {
	pd := NewPromptDistiller()

	codeGenQueries := []string{
		"write a function to reverse a string",
		"create a function that sorts numbers",
		"implement a binary search tree",
		"generate code for a web server",
		"write a test for the parser",
	}

	for _, q := range codeGenQueries {
		class := pd.Classify(q)
		if class != ClassCodeGen {
			t.Errorf("%q should be ClassCodeGen, got %s", q, class)
		}
	}
}

func TestClassifyCodeQuery(t *testing.T) {
	pd := NewPromptDistiller()

	codeQueryQueries := []string{
		"how does the reasoner.go work",
		"explain this code in main.go",
		"what does this function do in internal/cognitive",
		"show me src/index.ts",
	}

	for _, q := range codeQueryQueries {
		class := pd.Classify(q)
		if class != ClassCodeQuery {
			t.Errorf("%q should be ClassCodeQuery, got %s", q, class)
		}
	}
}

func TestClassifyToolUse(t *testing.T) {
	pd := NewPromptDistiller()

	toolQueries := []string{
		"read file config.yaml",
		"search for Pipeline",
		"grep for errors",
		"git status",
		"list files in the project",
	}

	for _, q := range toolQueries {
		class := pd.Classify(q)
		if class != ClassToolUse {
			t.Errorf("%q should be ClassToolUse, got %s", q, class)
		}
	}
}

func TestClassifyGeneral(t *testing.T) {
	pd := NewPromptDistiller()

	generalQueries := []string{
		"what is a goroutine",
		"explain polymorphism",
		"how does TCP work",
		"what are design patterns",
	}

	for _, q := range generalQueries {
		class := pd.Classify(q)
		if class != ClassGeneral {
			t.Errorf("%q should be ClassGeneral, got %s", q, class)
		}
	}
}

// --- Profile Tests ---

func TestProfileChat(t *testing.T) {
	pd := NewPromptDistiller()
	p := pd.Profile(ClassChat)

	if p.IncludeTools {
		t.Error("chat should not include tools")
	}
	if p.IncludeProject {
		t.Error("chat should not include project context")
	}
	if p.Temperature < 0.7 {
		t.Error("chat should have high temperature")
	}
}

func TestProfileCodeGen(t *testing.T) {
	pd := NewPromptDistiller()
	p := pd.Profile(ClassCodeGen)

	if p.IncludeTools {
		t.Error("codegen should not include tools")
	}
	if !p.IncludeLangRules {
		t.Error("codegen MUST include language rules")
	}
	if p.Temperature > 0.3 {
		t.Error("codegen should have low temperature")
	}
}

func TestProfileGeneral(t *testing.T) {
	pd := NewPromptDistiller()
	p := pd.Profile(ClassGeneral)

	if p.IncludeTools {
		t.Error("general should not include tools")
	}
	if p.IncludeProject {
		t.Error("general should NOT include project context")
	}
}

func TestProfileToolUse(t *testing.T) {
	pd := NewPromptDistiller()
	p := pd.Profile(ClassToolUse)

	if !p.IncludeTools {
		t.Error("tool use MUST include tools")
	}
}

// --- System Prompt Tests ---

func TestBuildPromptChat(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildSystemPrompt(ClassChat, "", "Go")

	if len(prompt) > 200 {
		t.Errorf("chat prompt should be minimal, got %d chars", len(prompt))
	}
	if strings.Contains(prompt, "Tool") || strings.Contains(prompt, "tool") {
		t.Error("chat prompt should not mention tools")
	}
}

func TestBuildPromptCodeGen(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildSystemPrompt(ClassCodeGen, "", "Go")

	if !strings.Contains(prompt, "try-catch") {
		t.Error("Go codegen prompt should warn about try-catch")
	}
	if !strings.Contains(prompt, "goroutine") {
		t.Error("Go codegen prompt should mention goroutines")
	}
	if strings.Contains(prompt, "tool") && strings.Contains(prompt, `"tool"`) {
		t.Error("codegen should not include tool call format")
	}
}

func TestBuildPromptGeneral(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildSystemPrompt(ClassGeneral, "", "Go")

	if strings.Contains(prompt, "wd:") {
		t.Error("general prompt should not include working directory")
	}
	if len(prompt) > 500 {
		t.Errorf("general prompt should be compact, got %d chars", len(prompt))
	}
}

func TestBuildPromptToolUse(t *testing.T) {
	pd := NewPromptDistiller()
	toolList := "grep(pattern): search files\nread(path): read file"
	prompt := pd.BuildSystemPrompt(ClassToolUse, toolList, "Go")

	if !strings.Contains(prompt, "grep") {
		t.Error("tool use prompt should include tool list")
	}
	if !strings.Contains(prompt, "wd:") {
		t.Error("tool use prompt should include working directory")
	}
}

func TestBuildPromptNoLangRules(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildSystemPrompt(ClassCodeGen, "", "")

	if strings.Contains(prompt, "try-catch") {
		t.Error("should not include Go rules when language is empty")
	}
}

func TestBuildPromptToolResult(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildSystemPrompt(ClassToolResult, "", "")

	if !strings.Contains(prompt, "ONLY") {
		t.Error("tool result prompt should emphasize using only provided data")
	}
	if !strings.Contains(prompt, "contradict") {
		t.Error("tool result prompt should warn against contradiction")
	}
}

// --- Final Answer Prompt Tests ---

func TestBuildFinalAnswerPrompt(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildFinalAnswerPrompt("Go")

	if !strings.Contains(prompt, "EXACT numbers") {
		t.Error("final answer prompt should emphasize exact numbers")
	}
	if !strings.Contains(prompt, "try-catch") {
		t.Error("final answer prompt should include Go language rules")
	}
}

func TestBuildFinalAnswerPromptNoLang(t *testing.T) {
	pd := NewPromptDistiller()
	prompt := pd.BuildFinalAnswerPrompt("")

	if strings.Contains(prompt, "try-catch") {
		t.Error("should not include language rules when no language specified")
	}
}

// --- Custom Language Rules ---

func TestSetCustomLanguageRules(t *testing.T) {
	pd := NewPromptDistiller()
	pd.SetLanguageRules("rust", "- No garbage collection\n- Ownership system")

	prompt := pd.BuildSystemPrompt(ClassCodeGen, "", "Rust")
	if !strings.Contains(prompt, "Ownership") {
		t.Error("should include custom Rust rules")
	}
}

// --- Prompt Size Comparison ---

func TestPromptSizeOrdering(t *testing.T) {
	pd := NewPromptDistiller()
	toolList := "grep(pattern): search\nread(path): read\nwrite(path,content): write\nls(path): list"

	chatPrompt := pd.BuildSystemPrompt(ClassChat, toolList, "Go")
	generalPrompt := pd.BuildSystemPrompt(ClassGeneral, toolList, "Go")
	codeGenPrompt := pd.BuildSystemPrompt(ClassCodeGen, toolList, "Go")
	toolUsePrompt := pd.BuildSystemPrompt(ClassToolUse, toolList, "Go")

	// Chat should be smallest
	if len(chatPrompt) > len(generalPrompt) {
		t.Errorf("chat (%d) should be smaller than general (%d)", len(chatPrompt), len(generalPrompt))
	}

	// ToolUse with full tool list should be the largest
	if len(toolUsePrompt) < len(chatPrompt) {
		t.Errorf("tooluse (%d) should be larger than chat (%d)", len(toolUsePrompt), len(chatPrompt))
	}
	_ = codeGenPrompt // size depends on language rules
}

// --- PromptStats ---

func TestPromptStats(t *testing.T) {
	pd := NewPromptDistiller()
	stats := pd.PromptStats(ClassChat, "Go")
	if !strings.Contains(stats, "chat") {
		t.Errorf("stats should mention class, got: %s", stats)
	}
	if !strings.Contains(stats, "chars=") {
		t.Errorf("stats should mention chars, got: %s", stats)
	}
}

// --- Benchmark ---

func BenchmarkPromptDistillerClassify(b *testing.B) {
	pd := NewPromptDistiller()
	queries := []string{
		"hello",
		"write a function to reverse a string",
		"what is a goroutine",
		"grep for Pipeline",
		"explain this code in main.go",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pd.Classify(queries[i%len(queries)])
	}
}

func BenchmarkPromptDistillerBuild(b *testing.B) {
	pd := NewPromptDistiller()
	toolList := "grep(pattern): search\nread(path): read"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pd.BuildSystemPrompt(ClassToolUse, toolList, "Go")
	}
}
