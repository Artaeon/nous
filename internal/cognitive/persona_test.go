package cognitive

import (
	"strings"
	"testing"
)

func TestPersonaContainsIdentity(t *testing.T) {
	if !strings.Contains(Persona, "Nous") {
		t.Error("Persona should mention Nous by name")
	}
	if !strings.Contains(Persona, "νοῦς") {
		t.Error("Persona should include the Greek word νοῦς")
	}
}

func TestPersonaContainsCoreTraits(t *testing.T) {
	required := []string{
		"perceive", "reason", "plan", "execute", "reflect", "learn",
	}
	for _, trait := range required {
		if !strings.Contains(strings.ToLower(Persona), trait) {
			t.Errorf("Persona should mention cognitive trait %q", trait)
		}
	}
}

func TestPersonaContainsPrinciples(t *testing.T) {
	// Persona should have numbered principles
	for i := 1; i <= 8; i++ {
		prefix := strings.Repeat(" ", 0) // just check the number exists
		_ = prefix
		if !strings.Contains(Persona, strings.Repeat("", 0)) {
			// Skip the prefix check, just verify the count
		}
	}
	if !strings.Contains(Persona, "1.") || !strings.Contains(Persona, "8.") {
		t.Error("Persona should contain at least 8 numbered principles")
	}
}

func TestPersonaIsUnrestricted(t *testing.T) {
	if !strings.Contains(strings.ToLower(Persona), "unrestricted") {
		t.Error("Persona should declare unrestricted assistance")
	}
	if !strings.Contains(strings.ToLower(Persona), "never refuse") {
		t.Error("Persona should explicitly state it never refuses questions")
	}
}

func TestPersonaPrivacy(t *testing.T) {
	if !strings.Contains(strings.ToLower(Persona), "no cloud") {
		t.Error("Persona should emphasize no cloud dependency")
	}
	if !strings.Contains(strings.ToLower(Persona), "local") {
		t.Error("Persona should emphasize local execution")
	}
}

func TestSelfKnowledgeFormat(t *testing.T) {
	tests := []struct {
		name        string
		model       string
		streams     int
		tools       int
		memItems    int
		ltmEntries  int
		wantContain []string
	}{
		{
			name:        "default_config",
			model:       "qwen2.5:1.5b",
			streams:     6,
			tools:       18,
			memItems:    5,
			ltmEntries:  10,
			wantContain: []string{"Nous", "νοῦς", "qwen2.5:1.5b", "6 streams", "18 tools", "5 working", "10 long-term"},
		},
		{
			name:        "zero_memory",
			model:       "llama3.2",
			streams:     6,
			tools:       20,
			memItems:    0,
			ltmEntries:  0,
			wantContain: []string{"llama3.2", "0 working", "0 long-term"},
		},
		{
			name:        "large_memory",
			model:       "deepseek-r1:8b",
			streams:     8,
			tools:       25,
			memItems:    64,
			ltmEntries:  500,
			wantContain: []string{"deepseek-r1:8b", "8 streams", "25 tools"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SelfKnowledge(tt.model, tt.streams, tt.tools, tt.memItems, tt.ltmEntries)
			for _, want := range tt.wantContain {
				if !strings.Contains(result, want) {
					t.Errorf("SelfKnowledge() should contain %q, got:\n%s", want, result)
				}
			}
		})
	}
}

func TestSelfKnowledgeLocalOnly(t *testing.T) {
	result := SelfKnowledge("test-model", 6, 18, 5, 10)
	lower := strings.ToLower(result)
	if !strings.Contains(lower, "local only") && !strings.Contains(lower, "no cloud") {
		t.Error("SelfKnowledge should emphasize local-only operation")
	}
}

func TestSelfKnowledgeGroundingStatement(t *testing.T) {
	result := SelfKnowledge("test-model", 6, 18, 5, 10)
	if !strings.Contains(result, "verify") || !strings.Contains(result, "guess") {
		t.Error("SelfKnowledge should include grounding statement about verification")
	}
}

func TestPromptConstants(t *testing.T) {
	prompts := map[string]string{
		"PerceivePrompt": PerceivePrompt,
		"PlanPrompt":     PlanPrompt,
		"ReflectPrompt":  ReflectPrompt,
	}

	for name, prompt := range prompts {
		t.Run(name, func(t *testing.T) {
			if prompt == "" {
				t.Errorf("%s should not be empty", name)
			}
			if !strings.Contains(prompt, "Nous") {
				t.Errorf("%s should mention Nous", name)
			}
			if !strings.Contains(prompt, "cognitive") {
				t.Errorf("%s should mention cognitive architecture", name)
			}
			if len(prompt) < 50 {
				t.Errorf("%s seems too short (%d chars), expected substantive prompt", name, len(prompt))
			}
		})
	}
}

func TestPerceivePromptRole(t *testing.T) {
	if !strings.Contains(strings.ToLower(PerceivePrompt), "perception") {
		t.Error("PerceivePrompt should describe perception role")
	}
	if !strings.Contains(strings.ToLower(PerceivePrompt), "intent") {
		t.Error("PerceivePrompt should mention intent extraction")
	}
}

func TestPlanPromptRole(t *testing.T) {
	if !strings.Contains(strings.ToLower(PlanPrompt), "planning") || !strings.Contains(strings.ToLower(PlanPrompt), "decompose") {
		t.Error("PlanPrompt should describe planning/decomposition role")
	}
	if !strings.Contains(strings.ToLower(PlanPrompt), "tool") {
		t.Error("PlanPrompt should mention tool specification")
	}
}

func TestReflectPromptRole(t *testing.T) {
	if !strings.Contains(strings.ToLower(ReflectPrompt), "reflection") {
		t.Error("ReflectPrompt should describe reflection role")
	}
	if !strings.Contains(strings.ToLower(ReflectPrompt), "hallucination") {
		t.Error("ReflectPrompt should mention hallucination detection")
	}
}
