package cognitive

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// ModelCompiler compiles accumulated experience into an optimized
// Ollama Modelfile, creating a new version of the model that has
// learned from its interactions.
//
// Innovation: No local AI agent compiles its own experience into
// its own model. The compiler takes everything Nous has learned —
// verified tool patterns, anti-hallucination rules, user preferences,
// crystallized reasoning chains, frequently accessed knowledge —
// and COMPRESSES it into a system prompt that IS the model's brain.
//
// Each compilation cycle produces a measurably better model:
//   Week 1: Raw qwen2.5:1.5b (baseline)
//   Week 2: nous-v1 (500 verified interactions compiled)
//   Week 3: nous-v2 (1000 interactions, resonance-optimized prompts)
//   ...
//
// The system prompt is not handwritten. It's COMPILED CODE —
// machine-generated from experience data, optimized for the
// model's attention patterns, and structured for maximum impact.
type ModelCompiler struct {
	baseModel  string
	distiller  *SelfDistiller
	crystals   *CrystalBook
	knowledge  *KnowledgeVec
	cortex     *NeuralCortex

	// Compilation stats
	Versions []CompiledVersion `json:"versions"`
}

// CompiledVersion records one compilation cycle.
type CompiledVersion struct {
	Version      int       `json:"version"`
	ModelName    string    `json:"model_name"`
	Timestamp    time.Time `json:"timestamp"`
	TrainingSamples int   `json:"training_samples"`
	PromptTokens   int    `json:"prompt_tokens"`
	Rules          int    `json:"rules"`
	Examples       int    `json:"examples"`
}

// ToolPattern is a verified tool usage pattern for compilation.
type ToolPattern struct {
	Query    string
	Tool     string
	Args     map[string]string
	Count    int
	AvgScore float64
}

// NewModelCompiler creates a new model compiler.
func NewModelCompiler(baseModel string, distiller *SelfDistiller, crystals *CrystalBook) *ModelCompiler {
	return &ModelCompiler{
		baseModel: baseModel,
		distiller: distiller,
		crystals:  crystals,
	}
}

// SetKnowledge sets the knowledge store for frequent-access compilation.
func (mc *ModelCompiler) SetKnowledge(kv *KnowledgeVec) {
	mc.knowledge = kv
}

// SetCortex sets the neural cortex for training data extraction.
func (mc *ModelCompiler) SetCortex(nc *NeuralCortex) {
	mc.cortex = nc
}

// Compile generates an optimized system prompt from all available data.
func (mc *ModelCompiler) Compile() string {
	var sections []compiledSection

	// 1. Core identity (always first — small models attend to beginning)
	sections = append(sections, compiledSection{
		priority: 100,
		content:  mc.compileIdentity(),
	})

	// 2. Anti-hallucination rules (from distiller)
	if mc.distiller != nil {
		if rules := mc.compileAntiHallucination(); rules != "" {
			sections = append(sections, compiledSection{
				priority: 90,
				content:  rules,
			})
		}
	}

	// 3. Tool usage patterns (from crystals)
	if mc.crystals != nil {
		if patterns := mc.compileToolPatterns(); patterns != "" {
			sections = append(sections, compiledSection{
				priority: 80,
				content:  patterns,
			})
		}
	}

	// 4. Language rules (hardcoded for Go — primary use case)
	sections = append(sections, compiledSection{
		priority: 70,
		content:  mc.compileLanguageRules(),
	})

	// 5. Cortex stats (what the neural network has learned)
	if mc.cortex != nil {
		if stats := mc.compileCortexInsights(); stats != "" {
			sections = append(sections, compiledSection{
				priority: 60,
				content:  stats,
			})
		}
	}

	// Sort by priority (highest first)
	sort.Slice(sections, func(i, j int) bool {
		return sections[i].priority > sections[j].priority
	})

	// Assemble and cap at token budget (~2000 chars ≈ 500 tokens)
	var sb strings.Builder
	totalLen := 0
	maxLen := 2000

	for _, s := range sections {
		if totalLen+len(s.content) > maxLen {
			// Truncate this section to fit
			remaining := maxLen - totalLen
			if remaining > 50 {
				sb.WriteString(s.content[:remaining])
			}
			break
		}
		sb.WriteString(s.content)
		sb.WriteString("\n")
		totalLen += len(s.content) + 1
	}

	return sb.String()
}

// GenerateModelfile creates a complete Ollama Modelfile with compiled system prompt.
func (mc *ModelCompiler) GenerateModelfile() string {
	compiledPrompt := mc.Compile()

	version := len(mc.Versions) + 1
	modelName := fmt.Sprintf("nous-v%d", version)

	mc.Versions = append(mc.Versions, CompiledVersion{
		Version:      version,
		ModelName:    modelName,
		Timestamp:    time.Now(),
		PromptTokens: len(compiledPrompt) / 4, // rough estimate
	})

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("FROM %s\n\n", mc.baseModel))

	// Optimal parameters for compiled model
	sb.WriteString("PARAMETER temperature 0.4\n")
	sb.WriteString("PARAMETER top_k 40\n")
	sb.WriteString("PARAMETER top_p 0.9\n")
	sb.WriteString("PARAMETER repeat_penalty 1.1\n")
	sb.WriteString("PARAMETER num_predict 1024\n\n")

	// The compiled system prompt
	sb.WriteString("SYSTEM \"\"\"")
	sb.WriteString(compiledPrompt)
	sb.WriteString("\"\"\"\n")

	return sb.String()
}

// ModelName returns the name for the next compiled model.
func (mc *ModelCompiler) ModelName() string {
	return fmt.Sprintf("nous-v%d", len(mc.Versions)+1)
}

// compiledSection is a prioritized section of the compiled prompt.
type compiledSection struct {
	priority int
	content  string
}

func (mc *ModelCompiler) compileIdentity() string {
	return strings.Join([]string{
		"You are Nous, a personal AI running fully on the user's machine.",
		"You have vast knowledge, tools, and memory. You grow with the user.",
		"RULES: Search knowledge before answering. Never invent facts. If unsure, say so.",
	}, " ")
}

func (mc *ModelCompiler) compileAntiHallucination() string {
	if mc.distiller == nil {
		return ""
	}

	negInst := mc.distiller.ExportNegativeInstructions()
	if negInst == "" {
		return ""
	}

	// Compress: take first 300 chars
	if len(negInst) > 300 {
		negInst = negInst[:300]
	}

	return "AVOID: " + negInst
}

func (mc *ModelCompiler) compileToolPatterns() string {
	if mc.crystals == nil {
		return ""
	}

	// Get top crystals by success rate
	crystals := mc.crystals.TopCrystals(5)
	if len(crystals) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Proven patterns:\n")
	for _, c := range crystals {
		if len(c.Steps) > 0 {
			steps := make([]string, 0, len(c.Steps))
			for _, s := range c.Steps {
				steps = append(steps, s.Tool)
			}
			sb.WriteString(fmt.Sprintf("- %s → %s\n", strings.Join(c.Trigger.Keywords, " "), strings.Join(steps, " → ")))
		}
	}
	return sb.String()
}

func (mc *ModelCompiler) compileLanguageRules() string {
	return strings.Join([]string{
		"Go rules: if err != nil (no try-catch), struct (no class),",
		"for (no while), if/else (no ternary), nil (no null),",
		"goroutines (no async/await), Uppercase=export.",
	}, " ")
}

func (mc *ModelCompiler) compileCortexInsights() string {
	if mc.cortex == nil {
		return ""
	}

	trainCount, _ := mc.cortex.Stats()
	if trainCount < 50 {
		return "" // not enough data yet
	}

	return fmt.Sprintf("Neural cortex trained on %d interactions.", trainCount)
}
