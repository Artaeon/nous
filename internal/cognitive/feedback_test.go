package cognitive

import (
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
