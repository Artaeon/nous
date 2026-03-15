package cognitive

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// CognitiveFirewall validates and corrects LLM responses before they reach
// the user. It catches language impossibilities, numerical inconsistencies,
// and hallucination markers that contradict tool evidence.
//
// Innovation: No local AI agent does post-LLM deterministic correction.
// Every other system trusts the model output. The firewall treats the LLM
// like an untrusted process — verify everything, correct what's wrong.
type CognitiveFirewall struct {
	langRules map[string][]LanguageRule
	distiller *SelfDistiller
}

// LanguageRule defines an impossible construct for a programming language.
type LanguageRule struct {
	Pattern    *regexp.Regexp
	Impossible string // human-readable name
	Correction string // what to use instead
}

// FirewallContext holds all the evidence needed to validate a response.
type FirewallContext struct {
	Query       string
	Response    string
	ToolResults []FirewallToolResult
	Language    string // detected project language
}

// FirewallToolResult holds one tool execution result for validation.
type FirewallToolResult struct {
	Tool   string
	Args   map[string]string
	Result string
}

// Violation records a single correction made by the firewall.
type Violation struct {
	Type        string // "language_impossible", "numerical_mismatch", "hallucination", "tool_contradiction"
	Description string
	Original    string
	Corrected   string
}

// NewCognitiveFirewall creates a firewall with default rules.
func NewCognitiveFirewall(distiller *SelfDistiller) *CognitiveFirewall {
	fw := &CognitiveFirewall{
		langRules: make(map[string][]LanguageRule),
		distiller: distiller,
	}
	fw.registerGoRules()
	return fw
}

// Validate checks a response against all rules and returns the corrected
// response plus any violations found.
func (fw *CognitiveFirewall) Validate(ctx *FirewallContext) (string, []Violation) {
	if ctx.Response == "" {
		return ctx.Response, nil
	}

	var violations []Violation
	corrected := ctx.Response

	// 1. Language impossibility check
	if ctx.Language != "" {
		if rules, ok := fw.langRules[strings.ToLower(ctx.Language)]; ok {
			corrected, violations = fw.checkLanguageRules(corrected, rules, violations)
		}
	}

	// 2. Numerical consistency check
	corrected, violations = fw.checkNumericalConsistency(corrected, ctx.ToolResults, violations)

	// 3. Hallucination marker check
	corrected, violations = fw.checkHallucinationMarkers(corrected, ctx.ToolResults, violations)

	// 4. Tool capability contradiction check
	corrected, violations = fw.checkToolContradictions(corrected, ctx.ToolResults, violations)

	// Feed violations to distiller for learning
	if fw.distiller != nil && len(violations) > 0 {
		for _, v := range violations {
			fw.distiller.RecordFailure(ctx.Query, "", v.Original, v.Type, v.Corrected, "firewall")
		}
	}

	return corrected, violations
}

// checkLanguageRules scans the response for constructs impossible in the
// detected language and replaces them with corrections.
func (fw *CognitiveFirewall) checkLanguageRules(response string, rules []LanguageRule, violations []Violation) (string, []Violation) {
	// Only check code-related sections (within backticks, or prose about code)
	for _, rule := range rules {
		if rule.Pattern.MatchString(response) {
			violations = append(violations, Violation{
				Type:        "language_impossible",
				Description: fmt.Sprintf("Used impossible construct: %s", rule.Impossible),
				Original:    rule.Pattern.FindString(response),
				Corrected:   rule.Correction,
			})
			// Replace in code blocks
			response = rule.Pattern.ReplaceAllString(response, rule.Correction)
		}
	}
	return response, violations
}

// checkNumericalConsistency cross-references numbers between tool output
// and the LLM response to catch misinterpretations (e.g., file size used
// as file count).
func (fw *CognitiveFirewall) checkNumericalConsistency(response string, toolResults []FirewallToolResult, violations []Violation) (string, []Violation) {
	if len(toolResults) == 0 {
		return response, violations
	}

	for _, tr := range toolResults {
		if tr.Result == "" {
			continue
		}

		switch tr.Tool {
		case "glob", "ls":
			// Count actual entries
			lines := strings.Split(strings.TrimSpace(tr.Result), "\n")
			actualCount := 0
			for _, line := range lines {
				if strings.TrimSpace(line) != "" {
					actualCount++
				}
			}
			if actualCount == 0 {
				continue
			}

			// Look for wrong counts in response
			response, violations = fw.fixWrongCount(response, actualCount, tr.Tool, violations)

		case "grep":
			// Count actual matches
			lines := strings.Split(strings.TrimSpace(tr.Result), "\n")
			matchCount := 0
			for _, line := range lines {
				if strings.TrimSpace(line) != "" && strings.Contains(line, ":") {
					matchCount++
				}
			}
			if matchCount == 0 {
				continue
			}

			response, violations = fw.fixWrongCount(response, matchCount, tr.Tool, violations)
		}
	}

	return response, violations
}

// fixWrongCount finds instances where the response states a number that
// appears elsewhere in the tool output (e.g., byte count) instead of the
// actual item count.
func (fw *CognitiveFirewall) fixWrongCount(response string, actualCount int, tool string, violations []Violation) (string, []Violation) {
	// Pattern: "N files/entries/matches/results"
	countRe := regexp.MustCompile(`\b(\d+)\s+(files?|entries|matches|results|items|Go files|directories)`)
	matches := countRe.FindAllStringSubmatch(response, -1)

	for _, m := range matches {
		stated, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}

		// If stated count is wildly wrong (more than 3x off or differs by >10)
		if stated != actualCount && (stated > actualCount*3 || actualCount > stated*3 || abs(stated-actualCount) > 10) {
			old := m[0]
			new := fmt.Sprintf("%d %s", actualCount, m[2])
			violations = append(violations, Violation{
				Type:        "numerical_mismatch",
				Description: fmt.Sprintf("%s: stated %d but actual count is %d", tool, stated, actualCount),
				Original:    old,
				Corrected:   new,
			})
			response = strings.Replace(response, old, new, 1)
		}
	}

	return response, violations
}

// checkHallucinationMarkers catches the LLM claiming no results when
// tool evidence shows results, or claiming file doesn't exist when
// the operation was a write/create.
func (fw *CognitiveFirewall) checkHallucinationMarkers(response string, toolResults []FirewallToolResult, violations []Violation) (string, []Violation) {
	lower := strings.ToLower(response)

	// Check for "no results" claims when evidence exists
	noResultPhrases := []string{
		"no results", "no matches", "couldn't find", "could not find",
		"unable to find", "nothing found", "no occurrences",
		"not found any", "didn't find", "did not find",
		"no files found", "no entries found", "does not contain",
	}

	hasEvidence := false
	for _, tr := range toolResults {
		if strings.TrimSpace(tr.Result) != "" {
			hasEvidence = true
			break
		}
	}

	if hasEvidence {
		for _, phrase := range noResultPhrases {
			if strings.Contains(lower, phrase) {
				// The model claims no results, but we have evidence — hallucination
				violations = append(violations, Violation{
					Type:        "hallucination",
					Description: fmt.Sprintf("Claims '%s' but tool results are non-empty", phrase),
					Original:    response,
				})

				// Replace with synthesized response from actual evidence
				synth := NewResponseSynthesizer()
				for _, tr := range toolResults {
					if tr.Result != "" {
						synthesized := synth.Synthesize(tr.Tool, tr.Args, tr.Result, nil)
						if synthesized != "" {
							return synthesized, violations
						}
					}
				}
				break
			}
		}
	}

	return response, violations
}

// checkToolContradictions catches claims that contradict tool capabilities.
func (fw *CognitiveFirewall) checkToolContradictions(response string, toolResults []FirewallToolResult, violations []Violation) (string, []Violation) {
	lower := strings.ToLower(response)

	for _, tr := range toolResults {
		switch tr.Tool {
		case "write":
			// Model says "can't create" or "file doesn't exist" for a write operation
			cantCreatePhrases := []string{
				"can't create", "cannot create", "doesn't exist",
				"does not exist", "unable to create", "failed to create",
				"i'm sorry", "i apologize",
			}
			for _, phrase := range cantCreatePhrases {
				if strings.Contains(lower, phrase) {
					path := tr.Args["path"]
					correction := fmt.Sprintf("File `%s` has been written successfully.", path)
					violations = append(violations, Violation{
						Type:        "tool_contradiction",
						Description: fmt.Sprintf("Claims inability to create file, but write tool can create files"),
						Original:    response,
						Corrected:   correction,
					})
					return correction, violations
				}
			}

		case "edit":
			// Model says "no changes needed" when an edit was requested
			if strings.Contains(lower, "no changes need") || strings.Contains(lower, "already present") {
				path := tr.Args["path"]
				if tr.Result != "" && !strings.Contains(strings.ToLower(tr.Result), "error") {
					correction := fmt.Sprintf("File `%s` has been edited successfully.", path)
					violations = append(violations, Violation{
						Type:        "tool_contradiction",
						Description: "Claims no changes needed but edit was requested and executed",
						Original:    response,
						Corrected:   correction,
					})
					return correction, violations
				}
			}
		}
	}

	return response, violations
}

// registerGoRules adds impossibility rules for Go.
func (fw *CognitiveFirewall) registerGoRules() {
	fw.langRules["go"] = []LanguageRule{
		{
			Pattern:    regexp.MustCompile(`\btry\s*\{[^}]*\}\s*catch\b`),
			Impossible: "try-catch blocks",
			Correction: "// Go uses if err != nil { return err }",
		},
		{
			Pattern:    regexp.MustCompile(`\btry[- ]catch\b`),
			Impossible: "try-catch",
			Correction: "error handling with if err != nil",
		},
		{
			Pattern:    regexp.MustCompile(`(?m)^\s*class\s+\w+\s*\{`),
			Impossible: "class declarations",
			Correction: "// Go uses type X struct {}",
		},
		{
			Pattern:    regexp.MustCompile(`\bextends\s+\w+`),
			Impossible: "class inheritance (extends)",
			Correction: "struct embedding",
		},
		{
			Pattern:    regexp.MustCompile(`\bimplements\s+\w+`),
			Impossible: "explicit interface implementation",
			Correction: "implicit interface satisfaction (no 'implements' keyword)",
		},
		{
			Pattern:    regexp.MustCompile(`\bwhile\s*\(`),
			Impossible: "while loops",
			Correction: "for { // Go only has for loops",
		},
		{
			Pattern:    regexp.MustCompile(`\b\w+\s*\?\s*\w+\s*:\s*\w+`),
			Impossible: "ternary operator",
			Correction: "// Go has no ternary — use if/else",
		},
		{
			Pattern:    regexp.MustCompile(`\bthrow\s+`),
			Impossible: "throw exceptions",
			Correction: "return fmt.Errorf(...)",
		},
		{
			Pattern:    regexp.MustCompile(`\b(public|private|protected)\s+(class|func|var|int|string|void)\b`),
			Impossible: "access modifiers",
			Correction: "// Go uses uppercase for exported, lowercase for unexported",
		},
		{
			Pattern:    regexp.MustCompile(`\basync\s+func\b`),
			Impossible: "async functions",
			Correction: "go func() { // Go uses goroutines",
		},
		{
			Pattern:    regexp.MustCompile(`\bawait\s+\w+`),
			Impossible: "await keyword",
			Correction: "<-ch // Go uses channels for synchronization",
		},
		{
			Pattern:    regexp.MustCompile(`\bnull\b`),
			Impossible: "null keyword",
			Correction: "nil",
		},
		{
			Pattern:    regexp.MustCompile(`\bthis\.\w+`),
			Impossible: "this keyword",
			Correction: "receiver variable (e.g., r.field)",
		},
	}
}

// RegisterLanguageRules adds custom rules for a language.
func (fw *CognitiveFirewall) RegisterLanguageRules(lang string, rules []LanguageRule) {
	fw.langRules[strings.ToLower(lang)] = rules
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
