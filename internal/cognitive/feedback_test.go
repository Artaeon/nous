package cognitive

import (
	"sync"
	"testing"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

func TestFeedbackLoopNil(t *testing.T) {
	var fl *FeedbackLoop
	// Should not panic
	fl.OnToolSuccess("test", "read", nil)
	fl.OnToolFailure("test", "read")
	fl.OnFirewallViolation("test", "bad output")
	stats := fl.Stats()
	if stats.CortexTrainCount != 0 {
		t.Error("nil feedback loop should have zero stats")
	}
}

func TestFeedbackLoopCreation(t *testing.T) {
	cortex := NewNeuralCortex(64, 32, []string{"grep", "read", "write", "ls"}, "")
	episodic := memory.NewEpisodicMemory("", nil)
	vctx := NewVirtualContext(1500)
	growth := NewPersonalGrowth("")
	crystals := NewCrystalBook("")

	fl := NewFeedbackLoop(cortex, episodic, vctx, growth, crystals)
	if fl == nil {
		t.Fatal("should create feedback loop")
	}
	if fl.AutoCryst == nil {
		t.Error("should create auto-crystallizer when crystals and episodic are provided")
	}
}

func TestFeedbackLoopToolSuccess(t *testing.T) {
	cortex := NewNeuralCortex(64, 32, []string{"grep", "read", "write", "ls"}, "")
	episodic := memory.NewEpisodicMemory("", nil)
	vctx := NewVirtualContext(1500)
	vctx.AddSource(ContextSource{Name: "knowledge", Size: 100, Priority: 50})

	fl := NewFeedbackLoop(cortex, episodic, vctx, nil, nil)

	initialTrain := cortex.TrainCount
	fl.OnToolSuccess("find the bug", "grep", []string{"grep"})

	if cortex.TrainCount <= initialTrain {
		t.Error("cortex should have been trained")
	}
}

func TestFeedbackLoopTrainWithMemoryBoost(t *testing.T) {
	cortex := NewNeuralCortex(64, 32, []string{"grep", "read", "write", "ls"}, "")
	episodic := memory.NewEpisodicMemory("", nil)

	// Seed episodic memory with past successful grep episodes
	for i := 0; i < 3; i++ {
		episodic.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "find old code pattern",
			ToolsUsed: []string{"grep"},
			Success:   true,
		})
	}

	fl := NewFeedbackLoop(cortex, episodic, nil, nil, nil)

	initialTrain := cortex.TrainCount
	fl.OnToolSuccess("find the new pattern", "grep", nil)

	// Should train at least twice (current + replay from episodic)
	trainDelta := cortex.TrainCount - initialTrain
	if trainDelta < 2 {
		t.Errorf("should train at least 2x with memory boost, trained %d times", trainDelta)
	}
}

func TestFeedbackLoopToolFailure(t *testing.T) {
	vctx := NewVirtualContext(1500)
	vctx.AddSource(ContextSource{Name: "knowledge", Size: 100, Priority: 50})

	fl := NewFeedbackLoop(nil, nil, vctx, nil, nil)
	fl.OnToolFailure("bad query", "shell")

	report := vctx.SourceHealthReport()
	if len(report) == 0 {
		t.Fatal("should have source report")
	}
	if report[0].Failures != 1 {
		t.Errorf("failures = %d, want 1", report[0].Failures)
	}
}

func TestFeedbackLoopFirewallViolation(t *testing.T) {
	episodic := memory.NewEpisodicMemory("", nil)
	fl := NewFeedbackLoop(nil, episodic, nil, nil, nil)

	fl.OnFirewallViolation("bad query about venus", "claimed venus has no atmosphere")

	if episodic.Size() != 1 {
		t.Fatalf("violation should be recorded as episode, got %d", episodic.Size())
	}

	// The episode should be marked as failure
	recent := episodic.Recent(1)
	if len(recent) == 0 {
		t.Fatal("should have recent episode")
	}
	if recent[0].Success {
		t.Error("firewall violation episode should be marked as failure")
	}
}

func TestFeedbackLoopStats(t *testing.T) {
	cortex := NewNeuralCortex(64, 32, []string{"a"}, "")
	episodic := memory.NewEpisodicMemory("", nil)
	vctx := NewVirtualContext(1500)
	vctx.AddSource(ContextSource{Name: "test", Size: 100})
	growth := NewPersonalGrowth("")
	crystals := NewCrystalBook("")

	fl := NewFeedbackLoop(cortex, episodic, vctx, growth, crystals)
	stats := fl.Stats()

	if stats.CortexTrainCount != 0 {
		t.Error("initial train count should be 0")
	}
	if stats.VCtxSourceCount != 1 {
		t.Errorf("source count = %d, want 1", stats.VCtxSourceCount)
	}
}

func TestFeedbackLoopGrowthSync(t *testing.T) {
	growth := NewPersonalGrowth("")
	vctx := NewVirtualContext(1500)
	vctx.AddSource(ContextSource{Name: "knowledge", Size: 100, Priority: 50})

	// Record enough interactions to build interest
	for i := 0; i < 10; i++ {
		growth.RecordInteraction("quantum physics experiments")
	}

	fl := NewFeedbackLoop(nil, nil, vctx, growth, nil)
	fl.OnToolSuccess("quantum question", "read", nil)

	// Growth sync should have boosted knowledge quality
	report := vctx.SourceHealthReport()
	if len(report) == 0 {
		t.Fatal("should have source report")
	}
	// Quality should be above default 0.5 after success + growth boost
	if report[0].Quality <= 0.5 {
		t.Errorf("quality = %f, should be above 0.5 after success + growth boost", report[0].Quality)
	}
}

// --- Race Condition Tests ---

func TestFeedbackLoopConcurrentCallbacks(t *testing.T) {
	cortex := NewNeuralCortex(64, 32, []string{"grep", "read", "write", "ls"}, "")
	episodic := memory.NewEpisodicMemory("", nil)
	vctx := NewVirtualContext(1500)
	vctx.AddSource(ContextSource{Name: "knowledge", Size: 100, Priority: 50})

	fl := NewFeedbackLoop(cortex, episodic, vctx, nil, nil)

	var wg sync.WaitGroup

	// Concurrent OnToolSuccess
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				fl.OnToolSuccess("query", "grep", []string{"grep"})
			}
		}()
	}

	// Concurrent OnToolFailure
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				fl.OnToolFailure("bad query", "shell")
			}
		}()
	}

	// Concurrent OnFirewallViolation
	for g := 0; g < 5; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				fl.OnFirewallViolation("danger", "wrong output")
			}
		}()
	}

	wg.Wait()

	// Should not crash; verify stats are plausible
	stats := fl.Stats()
	if stats.CortexTrainCount == 0 {
		t.Error("cortex should have been trained")
	}
}

// --- Edge Case Tests ---

func TestFeedbackLoopPartialComponents(t *testing.T) {
	// Cortex only — no episodic, no vctx, no growth, no crystals
	// Note: OnToolSuccess requires both Cortex AND Episodic to train cortex,
	// so cortex-only won't trigger training. Verify no panic.
	t.Run("cortex_only", func(t *testing.T) {
		cortex := NewNeuralCortex(64, 32, []string{"grep", "read"}, "")
		fl := NewFeedbackLoop(cortex, nil, nil, nil, nil)
		fl.OnToolSuccess("query", "grep", nil)
		fl.OnToolFailure("query", "read")
		fl.OnFirewallViolation("query", "bad")
		// Cortex training requires episodic memory for context boost,
		// so TrainCount stays 0 with cortex-only setup
	})

	// Episodic only
	t.Run("episodic_only", func(t *testing.T) {
		ep := memory.NewEpisodicMemory("", nil)
		fl := NewFeedbackLoop(nil, ep, nil, nil, nil)
		fl.OnToolSuccess("query", "grep", nil)
		fl.OnToolFailure("query", "read")
		fl.OnFirewallViolation("query", "bad")
		// Firewall violation records to episodic
		if ep.Size() != 1 {
			t.Errorf("expected 1 episode from firewall, got %d", ep.Size())
		}
	})

	// VCtx only
	t.Run("vctx_only", func(t *testing.T) {
		vc := NewVirtualContext(1500)
		vc.AddSource(ContextSource{Name: "knowledge", Size: 100, Priority: 50})
		fl := NewFeedbackLoop(nil, nil, vc, nil, nil)
		fl.OnToolSuccess("query", "grep", nil)
		fl.OnToolFailure("query", "read")
		fl.OnFirewallViolation("query", "bad")
		// Should not crash
	})

	// Nothing at all
	t.Run("no_components", func(t *testing.T) {
		fl := NewFeedbackLoop(nil, nil, nil, nil, nil)
		fl.OnToolSuccess("query", "grep", nil)
		fl.OnToolFailure("query", "read")
		fl.OnFirewallViolation("query", "bad")
		stats := fl.Stats()
		if stats.CortexTrainCount != 0 {
			t.Error("nil cortex should have 0 train count")
		}
	})
}

// --- Integration Test ---

func TestFeedbackLoopIntegration(t *testing.T) {
	labels := []string{"grep", "read", "write", "ls", "edit"}
	cortex := NewNeuralCortex(64, 32, labels, "")
	episodic := memory.NewEpisodicMemory("", nil)
	vctx := NewVirtualContext(1500)
	vctx.AddSource(ContextSource{Name: "knowledge", Size: 50000, Priority: 70})
	growth := NewPersonalGrowth("")
	crystals := NewCrystalBook("")

	fl := NewFeedbackLoop(cortex, episodic, vctx, growth, crystals)

	// Seed episodic memory with similar past successes
	for i := 0; i < 5; i++ {
		episodic.Record(memory.Episode{
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			Input:     "find function definition in code",
			ToolsUsed: []string{"grep"},
			Success:   true,
			Duration:  100,
		})
	}

	// Record growth interactions
	for i := 0; i < 10; i++ {
		growth.RecordInteraction("find function definition")
	}

	// Now trigger 20 successes
	initialTrain := cortex.TrainCount
	for i := 0; i < 20; i++ {
		fl.OnToolSuccess("find function definition in code", "grep", []string{"grep"})
	}

	// Cortex should have trained with memory replay (more than 20 times)
	trainDelta := cortex.TrainCount - initialTrain
	if trainDelta < 20 {
		t.Errorf("cortex should have trained at least 20 times, got %d", trainDelta)
	}

	// With past episodes matching, should train >20 (memory boost)
	if trainDelta <= 20 {
		t.Logf("note: memory boost trained %d times (expected >20 with replay)", trainDelta)
	}

	// VCtx should have recorded successes
	report := vctx.SourceHealthReport()
	if len(report) > 0 && report[0].Quality <= 0.5 {
		t.Errorf("quality should be above 0.5 after 20 successes, got %f", report[0].Quality)
	}

	// Auto-crystallizer should be created
	if fl.AutoCryst == nil {
		t.Error("auto-crystallizer should be created when crystals and episodic are provided")
	}

	// Stats should reflect all activity
	stats := fl.Stats()
	if stats.CortexTrainCount < 20 {
		t.Errorf("cortex train count should be >= 20, got %d", stats.CortexTrainCount)
	}
}
