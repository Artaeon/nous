package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/tools"
)

func makeTestTools() []tools.Tool {
	names := []string{
		"read", "glob", "grep", "ls", "tree",
		"write", "edit", "patch", "find_replace", "replace_all", "mkdir",
		"shell", "run", "sysinfo", "clipboard",
		"git", "diff",
		"fetch",
	}
	var out []tools.Tool
	for _, n := range names {
		out = append(out, tools.Tool{Name: n, Description: "test " + n})
	}
	return out
}

func toolNames(tt []tools.Tool) map[string]bool {
	m := make(map[string]bool)
	for _, t := range tt {
		m[t.Name] = true
	}
	return m
}

func TestSelectToolsQuestion(t *testing.T) {
	all := makeTestTools()
	selected := SelectToolsForIntent("question", nil, "what files are here?", all)
	names := toolNames(selected)

	// Should include explore tools
	for _, name := range []string{"read", "glob", "grep", "ls", "tree"} {
		if !names[name] {
			t.Errorf("question intent should include %s", name)
		}
	}
	// Should NOT include modify tools
	if names["write"] {
		t.Error("question intent should not include write tool")
	}
}

func TestSelectToolsCommand(t *testing.T) {
	all := makeTestTools()
	selected := SelectToolsForIntent("command", nil, "create a new file", all)
	names := toolNames(selected)

	// Should include both explore and modify
	if !names["read"] {
		t.Error("command intent should include read")
	}
	if !names["write"] {
		t.Error("command intent should include write")
	}
}

func TestSelectToolsGitKeyword(t *testing.T) {
	all := makeTestTools()
	entities := map[string]string{"action": "commit"}
	selected := SelectToolsForIntent("command", entities, "commit this change", all)
	names := toolNames(selected)

	if !names["git"] {
		t.Error("git keyword in entities should include git tool")
	}
	if !names["diff"] {
		t.Error("git keyword should include diff tool")
	}
}

func TestSelectToolsRawInputKeywords(t *testing.T) {
	all := makeTestTools()
	// Even without entities, raw input scanning should catch "fetch"
	selected := SelectToolsForIntent("request", nil, "fetch the url http://example.com", all)
	names := toolNames(selected)

	if !names["fetch"] {
		t.Error("'fetch' in raw input should include fetch tool")
	}
}

func TestSelectToolsAlwaysIncludesBaseline(t *testing.T) {
	all := makeTestTools()
	// Even for git-only intent, read and ls should always be present
	selected := SelectToolsForIntent("unknown", nil, "show git log", all)
	names := toolNames(selected)

	if !names["read"] {
		t.Error("read should always be included as baseline")
	}
	if !names["ls"] {
		t.Error("ls should always be included as baseline")
	}
}

func TestExpandCategory(t *testing.T) {
	all := makeTestTools()
	current := []tools.Tool{{Name: "read"}, {Name: "ls"}}
	expanded := ExpandCategory(CategoryModify, all, current)

	names := toolNames(expanded)
	if !names["write"] || !names["edit"] {
		t.Error("expanding modify should add write and edit")
	}
	if names["read"] {
		t.Error("already-selected tools should not be re-added")
	}
}

func TestToolPromptForSubset(t *testing.T) {
	subset := []tools.Tool{
		{Name: "read", Description: "Read a file"},
		{Name: "ls", Description: "List directory"},
	}
	prompt := ToolPromptForSubset(subset)

	if !strings.Contains(prompt, "- read: Read a file") {
		t.Error("prompt should include read tool")
	}
	if !strings.Contains(prompt, "- ls: List directory") {
		t.Error("prompt should include ls tool")
	}
	if !strings.Contains(prompt, "request_tools") {
		t.Error("prompt should include request_tools meta-tool")
	}
}

func TestCategoryNames(t *testing.T) {
	all := makeTestTools()
	names := CategoryNames(CategoryGit, all)
	if len(names) != 2 {
		t.Errorf("git category should have 2 tools, got %d", len(names))
	}
}
