package cognitive

import (
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/tools"
)

// ToolCategory groups tools by purpose for progressive disclosure.
type ToolCategory string

const (
	CategoryExplore ToolCategory = "explore"
	CategoryModify  ToolCategory = "modify"
	CategorySystem  ToolCategory = "system"
	CategoryGit     ToolCategory = "git"
	CategoryWeb     ToolCategory = "web"
)

// AllCategories lists every tool category.
var AllCategories = []ToolCategory{
	CategoryExplore, CategoryModify, CategorySystem, CategoryGit, CategoryWeb,
}

// ToolCategoryMap maps each tool name to its category.
var ToolCategoryMap = map[string]ToolCategory{
	"read": CategoryExplore, "glob": CategoryExplore, "grep": CategoryExplore,
	"ls": CategoryExplore, "tree": CategoryExplore,
	"write": CategoryModify, "edit": CategoryModify, "patch": CategoryModify,
	"find_replace": CategoryModify, "replace_all": CategoryModify, "mkdir": CategoryModify,
	"shell": CategorySystem, "run": CategorySystem, "sysinfo": CategorySystem,
	"clipboard": CategorySystem,
	"git": CategoryGit, "diff": CategoryGit,
	"fetch": CategoryWeb,
}

// intentCategories maps perceived intents to default tool categories.
var intentCategories = map[string][]ToolCategory{
	"question":  {CategoryExplore},
	"unknown":   {CategoryExplore},
	"greeting":  {CategoryExplore},
	"command":   {CategoryExplore, CategoryModify},
	"request":   {CategoryExplore, CategoryModify},
	"statement": {CategoryExplore},
}

// entityKeywords maps entity keywords to additional categories.
var entityKeywords = map[string]ToolCategory{
	"git":       CategoryGit,
	"commit":    CategoryGit,
	"branch":    CategoryGit,
	"diff":      CategoryGit,
	"merge":     CategoryGit,
	"log":       CategoryGit,
	"push":      CategoryGit,
	"pull":      CategoryGit,
	"url":       CategoryWeb,
	"http":      CategoryWeb,
	"https":     CategoryWeb,
	"web":       CategoryWeb,
	"fetch":     CategoryWeb,
	"download":  CategoryWeb,
	"system":    CategorySystem,
	"os":        CategorySystem,
	"sysinfo":   CategorySystem,
	"clipboard": CategorySystem,
	"shell":     CategorySystem,
	"run":       CategorySystem,
	"execute":   CategorySystem,
	"write":     CategoryModify,
	"edit":      CategoryModify,
	"create":    CategoryModify,
	"modify":    CategoryModify,
	"change":    CategoryModify,
	"replace":   CategoryModify,
	"delete":    CategoryModify,
	"remove":    CategoryModify,
	"fix":       CategoryModify,
	"update":    CategoryModify,
	"add":       CategoryModify,
}

// SelectToolsForIntent picks relevant tools based on perceived intent and entities.
// Returns a subset of tools (typically 5-8) instead of all 18, saving ~500 tokens
// in the system prompt — critical for small model context windows.
func SelectToolsForIntent(intent string, entities map[string]string, rawInput string, allTools []tools.Tool) []tools.Tool {
	cats := make(map[ToolCategory]bool)

	// Start with intent-based defaults
	if defaults, ok := intentCategories[intent]; ok {
		for _, c := range defaults {
			cats[c] = true
		}
	} else {
		cats[CategoryExplore] = true
	}

	// Scan entities for keyword signals
	for k, v := range entities {
		kv := strings.ToLower(k + " " + v)
		for keyword, cat := range entityKeywords {
			if strings.Contains(kv, keyword) {
				cats[cat] = true
			}
		}
	}

	// Also scan the raw input for keywords (entities extraction is unreliable with small models)
	lower := strings.ToLower(rawInput)
	for keyword, cat := range entityKeywords {
		if strings.Contains(lower, keyword) {
			cats[cat] = true
		}
	}

	// Filter tools by selected categories
	var selected []tools.Tool
	seen := make(map[string]bool)

	for _, t := range allTools {
		cat, ok := ToolCategoryMap[t.Name]
		if !ok {
			continue
		}
		if cats[cat] && !seen[t.Name] {
			selected = append(selected, t)
			seen[t.Name] = true
		}
	}

	// Always ensure read and ls are available (baseline exploration)
	for _, name := range []string{"read", "ls"} {
		if !seen[name] {
			for _, t := range allTools {
				if t.Name == name {
					selected = append(selected, t)
					seen[name] = true
					break
				}
			}
		}
	}

	return selected
}

// ExpandCategory returns tools from a category that aren't already selected.
func ExpandCategory(cat ToolCategory, allTools []tools.Tool, current []tools.Tool) []tools.Tool {
	currentNames := make(map[string]bool)
	for _, t := range current {
		currentNames[t.Name] = true
	}

	var added []tools.Tool
	for _, t := range allTools {
		if ToolCategoryMap[t.Name] == cat && !currentNames[t.Name] {
			added = append(added, t)
		}
	}
	return added
}

// ToolPromptForSubset generates the tool list section of the system prompt
// for only the selected tools, plus the request_tools meta-tool.
func ToolPromptForSubset(selected []tools.Tool) string {
	var sb strings.Builder
	sb.WriteString("Available tools:\n")
	for _, t := range selected {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
	}
	sb.WriteString("- request_tools: Request more tools. Args: category (one of: explore, modify, system, git, web).\n")
	return sb.String()
}

// CategoryNames returns the names of tools added by ExpandCategory.
func CategoryNames(cat ToolCategory, allTools []tools.Tool) []string {
	var names []string
	for _, t := range allTools {
		if ToolCategoryMap[t.Name] == cat {
			names = append(names, t.Name)
		}
	}
	return names
}
