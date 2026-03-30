package agent

import (
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/tools"
)

func TestCountContentWords(t *testing.T) {
	tests := []struct {
		text string
		want int // approximate — just check range
		min  int
		max  int
	}{
		{"", 0, 0, 0},
		{"hello world", 2, 2, 2},
		{"wrote 480 bytes to file.md", 0, 0, 0}, // "wrote " prefix skipped
		{"http://example.com", 0, 0, 0},           // URL skipped
		{"--- separator ---", 0, 0, 0},             // separator skipped
		{"AI is transforming industries worldwide. Machine learning algorithms can process vast amounts of data.", 0, 10, 20},
	}

	for _, tt := range tests {
		got := countContentWords(tt.text)
		if tt.want > 0 && got != tt.want {
			t.Errorf("countContentWords(%q) = %d, want %d", tt.text, got, tt.want)
		}
		if got < tt.min || got > tt.max {
			t.Errorf("countContentWords(%q) = %d, want [%d, %d]", tt.text, got, tt.min, tt.max)
		}
	}
}

func TestSuggestAlternativeQueries(t *testing.T) {
	phase := Phase{
		Name: "Research",
		Tasks: []Task{
			{ToolChain: []ToolStep{{Tool: "websearch", Args: map[string]string{"query": "AI overview 2026"}}}},
		},
	}

	suggestions := suggestAlternativeQueries("Research artificial intelligence", phase)
	if len(suggestions) == 0 {
		t.Error("expected alternative queries")
	}

	// Should not suggest queries already tried
	for _, s := range suggestions {
		if strings.EqualFold(s, "AI overview 2026") {
			t.Errorf("suggested already-tried query: %s", s)
		}
	}

	t.Logf("Suggestions: %v", suggestions)
}

func TestPhaseAlreadyCoversNext(t *testing.T) {
	// Content with analysis markers should skip an "Analysis" phase
	content := "Our findings indicate that AI is growing. The analysis suggests that the market therefore will expand. The conclusion is clear."
	nextPhase := Phase{Name: "Analysis", Description: "Analyze the data"}
	if !phaseAlreadyCoversNext(content, nextPhase) {
		t.Error("content with analysis markers should cover Analysis phase")
	}

	// Content without analysis markers should NOT skip
	thinContent := "Some search results about AI."
	if phaseAlreadyCoversNext(thinContent, nextPhase) {
		t.Error("thin content should not cover Analysis phase")
	}

	// Content with markdown sections should skip Report phase
	reportContent := "## Introduction\nSome text.\n## Background\nMore text.\n## Analysis\nEven more.\n## Conclusion\nDone."
	reportPhase := Phase{Name: "Report", Description: "Compile the report"}
	if !phaseAlreadyCoversNext(reportContent, reportPhase) {
		t.Error("content with 3+ sections should cover Report phase")
	}
}

func TestInjectSearchTasks(t *testing.T) {
	phase := &Phase{
		Name: "Research",
		Tasks: []Task{
			{ID: "original-1", Description: "Original task"},
		},
	}

	injectSearchTasks(phase, []string{"AI explained simply", "AI key facts"}, "phase0")

	if len(phase.Tasks) != 3 {
		t.Fatalf("expected 3 tasks after injection, got %d", len(phase.Tasks))
	}

	// New tasks should be at the front
	if !strings.Contains(phase.Tasks[0].ID, "extra") {
		t.Errorf("first task should be injected, got ID=%s", phase.Tasks[0].ID)
	}
	if phase.Tasks[2].ID != "original-1" {
		t.Errorf("original task should be last, got ID=%s", phase.Tasks[2].ID)
	}

	// Injected tasks should be websearch
	for i := 0; i < 2; i++ {
		if len(phase.Tasks[i].ToolChain) == 0 || phase.Tasks[i].ToolChain[0].Tool != "websearch" {
			t.Errorf("injected task %d should use websearch", i)
		}
	}
}

func TestEvaluatePhase_Sufficient(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})

	// Set up state with a phase that produced good content
	longContent := strings.Repeat("Artificial intelligence is a broad field of computer science. ", 20)
	a.State.Reset("test goal")
	a.State.SetPlan(&Plan{
		Goal: "test goal",
		Phases: []Phase{
			{
				Name:   "Research",
				Status: PhaseCompleted,
				Tasks: []Task{
					{ID: "t1", Status: TaskCompleted, Result: longContent},
				},
			},
			{Name: "Analysis", DependsOn: []int{0}},
		},
	})
	a.State.RecordResult("t1", longContent)

	eval := a.evaluatePhase(0)
	if !eval.QualitySufficient {
		t.Errorf("expected sufficient quality, got: %s", eval.Reason)
	}
	if eval.ContentWords < 100 {
		t.Errorf("expected 100+ words, got %d", eval.ContentWords)
	}
}

func TestEvaluatePhase_Insufficient(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})

	// Phase with very thin content
	a.State.Reset("test goal")
	a.State.SetPlan(&Plan{
		Goal: "test goal",
		Phases: []Phase{
			{
				Name:   "Research",
				Status: PhaseCompleted,
				Tasks: []Task{
					{ID: "t1", Status: TaskCompleted, Result: "Short."},
				},
			},
		},
	})
	a.State.RecordResult("t1", "Short.")

	eval := a.evaluatePhase(0)
	if eval.QualitySufficient {
		t.Error("thin content should not be sufficient")
	}
	if len(eval.NeedsMoreData) == 0 {
		t.Error("should suggest more queries for thin results")
	}
	t.Logf("Evaluation: %s, suggestions: %v", eval.Reason, eval.NeedsMoreData)
}

func TestEvaluatePhase_AllFailed(t *testing.T) {
	reg := mockRegistry()
	a := NewAgent(reg, AgentConfig{Workspace: t.TempDir()})

	a.State.Reset("test goal")
	a.State.SetPlan(&Plan{
		Goal: "test goal",
		Phases: []Phase{
			{
				Name:   "Research",
				Status: PhaseCompleted,
				Tasks: []Task{
					{ID: "t1", Status: TaskFailed, Error: "network error"},
					{ID: "t2", Status: TaskFailed, Error: "timeout"},
				},
			},
		},
	})

	eval := a.evaluatePhase(0)
	if eval.QualitySufficient {
		t.Error("all-failed phase should not be sufficient")
	}
	if !strings.Contains(eval.Reason, "failed") {
		t.Errorf("reason should mention failure, got: %s", eval.Reason)
	}
}

// TestAdaptiveReplan_ThinResults tests that the agent retries a phase
// when web search returns thin results, then succeeds on retry.
func TestAdaptiveReplan_ThinResults(t *testing.T) {
	callCount := 0
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "Search",
		Execute: func(args map[string]string) (string, error) {
			callCount++
			query := args["query"]
			if callCount <= 3 {
				// First 3 calls return thin results (phase 1 original)
				return "1. " + query, nil
			}
			// Retry calls return substantial content
			return "Artificial intelligence is a broad field of computer science focused on creating " +
				"systems capable of performing tasks requiring human-like intelligence. The field was " +
				"formally established at the Dartmouth Conference in 1956. Machine learning, a subset " +
				"of AI, enables systems to learn patterns from data without explicit programming. " +
				"Deep learning uses neural networks with multiple layers to process complex data. " +
				"Natural language processing enables computers to understand and generate human language. " +
				"Computer vision allows machines to interpret visual information from the world.", nil
		},
	})
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write",
		Execute: func(args map[string]string) (string, error) {
			return "wrote to " + args["path"], nil
		},
	})

	config := AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
		MaxRetries:   1,
		StepTimeout:  5 * time.Second,
	}
	a := NewAgent(reg, config)

	var mu sync.Mutex
	var reports []string
	a.SetReportCallback(func(msg string) {
		mu.Lock()
		reports = append(reports, msg)
		mu.Unlock()
	})

	err := a.Start("Research artificial intelligence")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	deadline := time.After(15 * time.Second)
	for {
		select {
		case <-deadline:
			a.Stop()
			t.Fatal("agent did not complete within 15s")
		default:
		}
		if !a.Status().Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	mu.Lock()
	reportsCopy := make([]string, len(reports))
	copy(reportsCopy, reports)
	mu.Unlock()

	// Check that adaptive replanning happened
	adaptiveFound := false
	for _, r := range reportsCopy {
		if strings.Contains(r, "[ADAPTIVE]") {
			adaptiveFound = true
			t.Logf("Adaptive report: %s", r)
		}
	}
	if !adaptiveFound {
		t.Error("expected [ADAPTIVE] report — replanning should have triggered on thin results")
	}

	// Check that more than the original 3 search calls were made
	if callCount <= 3 {
		t.Errorf("expected retry searches, but only %d websearch calls were made", callCount)
	}

	t.Logf("Total websearch calls: %d", callCount)
	t.Logf("Reports collected: %d", len(reportsCopy))
	for i, r := range reportsCopy {
		t.Logf("  [%d] %s", i, truncateString(r, 100))
	}
}

// TestAdaptiveReplan_MaxOneRetry ensures phases only retry once.
func TestAdaptiveReplan_MaxOneRetry(t *testing.T) {
	reg := tools.NewRegistry()
	reg.Register(tools.Tool{
		Name:        "websearch",
		Description: "Search",
		Execute: func(args map[string]string) (string, error) {
			return "thin", nil // always returns thin results
		},
	})
	reg.Register(tools.Tool{
		Name:        "write",
		Description: "Write",
		Execute: func(args map[string]string) (string, error) {
			return "wrote", nil
		},
	})

	config := AgentConfig{
		Workspace:    t.TempDir(),
		MaxToolCalls: 50,
		MaxRetries:   1,
		StepTimeout:  5 * time.Second,
	}
	a := NewAgent(reg, config)

	var mu sync.Mutex
	retryCount := 0
	a.SetReportCallback(func(msg string) {
		mu.Lock()
		if strings.Contains(msg, "Retrying phase") {
			retryCount++
		}
		mu.Unlock()
	})

	a.Start("Research something obscure")
	deadline := time.After(15 * time.Second)
	for {
		select {
		case <-deadline:
			a.Stop()
			t.Fatal("timeout")
		default:
		}
		if !a.Status().Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	mu.Lock()
	rc := retryCount
	mu.Unlock()

	// Should retry at most once per phase.
	// With 3 phases, total retries can be up to 3 (one each).
	// But a single phase should never retry more than once.
	if rc > 3 {
		t.Errorf("expected max 3 retries (1 per phase), got %d", rc)
	}
	// Verify no phase retried more than once
	a.State.mu.RLock()
	for _, ph := range a.State.Plan.Phases {
		if ph.Retried > 1 {
			t.Errorf("phase %q retried %d times (max 1)", ph.Name, ph.Retried)
		}
	}
	a.State.mu.RUnlock()
	t.Logf("Retries: %d", rc)
}
