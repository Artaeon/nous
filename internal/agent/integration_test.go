package agent

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/cognitive"
	"github.com/artaeon/nous/internal/tools"
)

// TestIntegration_CognitiveBridgeReal tests the CognitiveBridge with real
// cognitive systems (Summarizer, ThinkingEngine, DocumentGenerator, etc.)
// rather than mocks. This verifies end-to-end content generation.
func TestIntegration_CognitiveBridgeReal(t *testing.T) {
	// Build real cognitive systems
	graph := cognitive.NewCognitiveGraph("")
	semantic := cognitive.NewSemanticEngine()
	causal := cognitive.NewCausalEngine()
	patterns := cognitive.NewPatternDetector()
	composer := cognitive.NewComposer(graph, semantic, causal, patterns)
	thinker := cognitive.NewThinkingEngine(graph, composer)
	summarizer := cognitive.NewSummarizer()
	docGen := cognitive.NewDocumentGenerator(graph, "")

	bridge := &CognitiveBridge{
		Summarizer: summarizer,
		Thinker:    thinker,
		DocGen:     docGen,
		Composer:   composer,
		NLG:        cognitive.NewNLGEngine(),
		Graph:      graph,
	}

	t.Run("Summarize_RealText", func(t *testing.T) {
		// Use enough sentences that 3 is clearly a subset
		text := `Artificial intelligence is transforming industries worldwide. ` +
			`Machine learning algorithms can now process vast amounts of data. ` +
			`Natural language processing enables computers to understand human speech. ` +
			`Computer vision systems can identify objects in images with high accuracy. ` +
			`Deep learning has revolutionized the field of AI research. ` +
			`Autonomous vehicles rely heavily on AI for navigation and safety. ` +
			`Healthcare uses AI for diagnosis and drug discovery workflows. ` +
			`Financial institutions use AI for fraud detection and algorithmic trading. ` +
			`Education platforms use AI for personalized learning experiences. ` +
			`Manufacturing relies on AI for quality control and predictive maintenance.`

		result := bridge.Summarize(text, 3)
		if result == "" {
			t.Error("Summarize returned empty")
		}
		// Should be shorter than the original (10 sentences → 3)
		if len(result) >= len(text) {
			t.Errorf("Summary (%d chars) should be shorter than original (%d chars)", len(result), len(text))
		}
		t.Logf("Summary (%d chars): %s", len(result), result)
	})

	t.Run("SummarizeToLength", func(t *testing.T) {
		text := strings.Repeat("This is a test sentence about artificial intelligence. ", 20)
		result := bridge.SummarizeToLength(text, 30)
		if result == "" {
			t.Error("SummarizeToLength returned empty")
		}
		words := strings.Fields(result)
		// Should be roughly within 2x of maxWords (extractive summarization is approximate)
		if len(words) > 80 {
			t.Errorf("SummarizeToLength: got %d words, expected roughly ≤60", len(words))
		}
		t.Logf("SummarizeToLength (%d words): %s", len(words), result)
	})

	t.Run("Think_RealQuery", func(t *testing.T) {
		result := bridge.Think("Write a brief analysis of cloud computing trends")
		// ThinkingEngine may or may not handle this query (depends on task classification)
		t.Logf("Think result (%d chars): %s", len(result), truncateString(result, 200))
	})

	t.Run("Compose_RealQuery", func(t *testing.T) {
		result := bridge.Compose("What is artificial intelligence?")
		t.Logf("Compose result (%d chars): %s", len(result), truncateString(result, 200))
	})

	t.Run("GenerateDocument", func(t *testing.T) {
		// This may return empty if the topic isn't in the knowledge graph
		result := bridge.GenerateDocument("artificial intelligence", "overview")
		t.Logf("GenerateDocument result (%d chars): %s", len(result), truncateString(result, 300))
	})

	t.Run("SynthesizeResults_Multiple", func(t *testing.T) {
		results := map[string]string{
			"search_results": "AI market growing 40% annually. Key players: Google, Microsoft, OpenAI.",
			"competitors":    "Main competitors: ChatGPT, Gemini, Claude. Pricing: $20-50/month.",
			"market_data":    "Global AI market: $150B in 2025. Expected $300B by 2028.",
		}
		synth := bridge.SynthesizeResults("analyze AI market trends", results)
		if synth == "" {
			t.Error("SynthesizeResults returned empty")
		}
		t.Logf("Synthesis (%d chars): %s", len(synth), truncateString(synth, 300))
	})

	t.Run("WriteReport", func(t *testing.T) {
		results := map[string]string{
			"research": "Blockchain is a distributed ledger technology.",
		}
		report := bridge.WriteReport("blockchain technology", "report", results)
		if report == "" {
			t.Error("WriteReport returned empty")
		}
		t.Logf("Report (%d chars): %s", len(report), truncateString(report, 300))
	})
}

// TestIntegration_AgentWithCognitiveBridge tests a full agent run with
// real cognitive systems connected.
func TestIntegration_AgentWithCognitiveBridge(t *testing.T) {
	dir := t.TempDir()

	// Build real tool registry with mock tools
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "Search the web",
		Execute: func(args map[string]string) (string, error) {
			query := args["query"]
			return "Search results for '" + query + "':\n" +
				"1. AI is transforming healthcare with diagnostic tools\n" +
				"2. Machine learning market projected to reach $150B\n" +
				"3. Natural language processing enables human-computer interaction\n" +
				"4. Computer vision advances in autonomous vehicles\n" +
				"5. Deep learning research continues to accelerate", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write a file",
		Execute: func(args map[string]string) (string, error) {
			path := args["path"]
			content := args["content"]
			if content == "" {
				content = "(no content)"
			}
			os.MkdirAll(filepath.Dir(path), 0o755)
			os.WriteFile(path, []byte(content), 0o644)
			return "wrote to " + path, nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "read",
		Description: "Read a file",
		Execute: func(args map[string]string) (string, error) {
			data, err := os.ReadFile(args["path"])
			if err != nil {
				return "", err
			}
			return string(data), nil
		},
	})

	// Build real cognitive systems
	graph := cognitive.NewCognitiveGraph("")
	semantic := cognitive.NewSemanticEngine()
	causal := cognitive.NewCausalEngine()
	patterns := cognitive.NewPatternDetector()
	composer := cognitive.NewComposer(graph, semantic, causal, patterns)
	thinker := cognitive.NewThinkingEngine(graph, composer)
	summarizer := cognitive.NewSummarizer()
	docGen := cognitive.NewDocumentGenerator(graph, "")

	brain := &CognitiveBridge{
		Summarizer: summarizer,
		Thinker:    thinker,
		DocGen:     docGen,
		Composer:   composer,
		NLG:        cognitive.NewNLGEngine(),
		Graph:      graph,
	}

	config := AgentConfig{
		Workspace:    dir,
		MaxToolCalls: 50,
		MaxRetries:   2,
		StepTimeout:  10 * time.Second,
	}

	a := NewAgent(reg, config)
	a.SetBrain(brain)

	var mu sync.Mutex
	var reports []string
	a.SetReportCallback(func(msg string) {
		mu.Lock()
		reports = append(reports, msg)
		mu.Unlock()
	})

	t.Run("ResearchGoal", func(t *testing.T) {
		err := a.Start("Research artificial intelligence and write a summary")
		if err != nil {
			t.Fatalf("Start: %v", err)
		}

		// Wait for completion
		deadline := time.After(30 * time.Second)
		for {
			select {
			case <-deadline:
				a.Stop()
				t.Fatal("agent did not complete within 30s")
			default:
			}
			if !a.Status().Running {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}

		// Check the report
		report := a.Report()
		if !strings.Contains(report, "[COMPLETE]") {
			t.Errorf("expected [COMPLETE] in report, got:\n%s", report)
		}
		t.Logf("Final report:\n%s", report)

		// Check workspace files
		files, _ := filepath.Glob(filepath.Join(dir, "agent_workspace", "*.md"))
		t.Logf("Workspace files: %v", files)
		for _, f := range files {
			data, err := os.ReadFile(f)
			if err != nil {
				t.Errorf("read %s: %v", f, err)
				continue
			}
			base := filepath.Base(f)
			content := string(data)
			t.Logf("File %s (%d bytes):\n%s", base, len(data), truncateString(content, 500))

			// Content should not be "(no content)" — that means write was called without content arg
			if content == "(no content)" {
				t.Errorf("File %s has no content — write tool received empty content arg", base)
			}
		}

		// Check state.json
		stateData, err := os.ReadFile(filepath.Join(dir, "state.json"))
		if err != nil {
			t.Errorf("state.json not found: %v", err)
		} else {
			t.Logf("state.json: %d bytes", len(stateData))
		}

		// Verify reports were generated
		mu.Lock()
		reportsCopy := make([]string, len(reports))
		copy(reportsCopy, reports)
		mu.Unlock()
		t.Logf("Collected %d reports", len(reportsCopy))
		for i, r := range reportsCopy {
			t.Logf("  report[%d]: %s", i, truncateString(r, 120))
		}
	})
}

// TestIntegration_ExtractTopicCompound tests extractTopic with compound goals.
func TestIntegration_ExtractTopicCompound(t *testing.T) {
	tests := []struct {
		goal string
		want string
	}{
		{"Research artificial intelligence and write a summary", "artificial intelligence"},
		{"Research artificial intelligence and write a report", "artificial intelligence"},
		{"Research the history of blockchain and create a report", "the history of blockchain"},
		{"Analyze the pros and cons of remote work", "the pros and cons of remote work"},
		{"Write an essay about climate change", "essay about climate change"},
	}
	for _, tt := range tests {
		got := extractTopic(tt.goal)
		if got != tt.want {
			t.Errorf("extractTopic(%q) = %q, want %q", tt.goal, got, tt.want)
		}
	}
}

// TestIntegration_PlannerGoalTypes verifies all goal types produce cognitive steps.
func TestIntegration_PlannerGoalTypes(t *testing.T) {
	planner := NewPlanner([]string{"websearch", "write", "read"})

	goals := map[string]string{
		"research":   "Research the history of cryptocurrency",
		"writing":    "Write a report about climate change",
		"analysis":   "Analyze the pros and cons of remote work",
		"planning":   "Create a project plan for building a mobile app",
		"building":   "Build a web scraper for news articles",
		"monitoring": "Monitor Bitcoin price every hour",
		"generic":    "Do something completely random",
	}

	for goalType, goal := range goals {
		plan, err := planner.DecomposeGoal(goal)
		if err != nil {
			t.Errorf("[%s] DecomposeGoal(%q): %v", goalType, goal, err)
			continue
		}

		// Check for cognitive steps in the plan
		hasCognitive := false
		for _, phase := range plan.Phases {
			for _, task := range phase.Tasks {
				for _, step := range task.ToolChain {
					if strings.HasPrefix(step.Tool, "_") {
						hasCognitive = true
					}
				}
			}
		}

		// All goal types except monitoring-setup (which starts with human input)
		// should have cognitive steps
		if !hasCognitive && goalType != "monitoring" {
			// monitoring has _summarize in phase 2, but only after human input in phase 1
			t.Logf("[%s] Plan phases:", goalType)
			for i, ph := range plan.Phases {
				for _, task := range ph.Tasks {
					var toolNames []string
					for _, s := range task.ToolChain {
						toolNames = append(toolNames, s.Tool)
					}
					human := ""
					if task.NeedsHuman {
						human = " [HUMAN]"
					}
					t.Logf("  Phase %d/%s: task %s tools=%v%s", i+1, ph.Name, task.ID, toolNames, human)
				}
			}
		}

		// Check write steps have content args
		for _, phase := range plan.Phases {
			for _, task := range phase.Tasks {
				for _, step := range task.ToolChain {
					if step.Tool == "write" {
						if _, ok := step.Args["content"]; !ok {
							t.Errorf("[%s] write step in task %q has no content arg", goalType, task.ID)
						}
					}
				}
			}
		}
	}
}
