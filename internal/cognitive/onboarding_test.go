package cognitive

import (
	"strings"
	"testing"

	"github.com/artaeon/nous/internal/memory"
)

func TestOnboardingFirstRun(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	wm := memory.NewWorkingMemory(10)

	// Simulate user typing answers
	input := "Raphael\nsoftware engineer\nbuilding an AI assistant\ncosmology and Go\ncode partner\n"
	reader := strings.NewReader(input)

	ran := RunOnboarding(reader, ltm, wm)
	if !ran {
		t.Fatal("expected onboarding to run on first launch")
	}

	// Verify LTM storage
	tests := []struct {
		key  string
		want string
	}{
		{"user.name", "Raphael"},
		{"user.role", "software engineer"},
		{"user.current_work", "building an AI assistant"},
		{"user.interests", "cosmology and Go"},
		{"user.preferred_role", "code partner"},
	}

	for _, tt := range tests {
		val, ok := ltm.Retrieve(tt.key)
		if !ok {
			t.Errorf("LTM missing key %q", tt.key)
			continue
		}
		if val != tt.want {
			t.Errorf("LTM %s = %q, want %q", tt.key, val, tt.want)
		}
	}

	// Verify working memory
	wmVal, ok := wm.Retrieve("user.name")
	if !ok {
		t.Fatal("working memory missing user.name")
	}
	if wmVal != "Raphael" {
		t.Errorf("working mem user.name = %v, want Raphael", wmVal)
	}
}

func TestOnboardingSkipsWhenLTMHasData(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	ltm.Store("user.name", "Alice", "personal")

	reader := strings.NewReader("Bob\n")
	ran := RunOnboarding(reader, ltm, nil)
	if ran {
		t.Fatal("onboarding should not run when LTM already has data")
	}

	// Name should still be Alice, not overwritten
	val, _ := ltm.Retrieve("user.name")
	if val != "Alice" {
		t.Errorf("user.name = %q, want Alice (should not be overwritten)", val)
	}
}

func TestOnboardingSkipsOptional(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())

	// Only answer name and role, skip optional questions with empty lines
	input := "Raphael\nengineer\n\n\n\n"
	reader := strings.NewReader(input)

	ran := RunOnboarding(reader, ltm, nil)
	if !ran {
		t.Fatal("expected onboarding to run")
	}

	// Required answers stored
	val, ok := ltm.Retrieve("user.name")
	if !ok || val != "Raphael" {
		t.Errorf("user.name = %q, want Raphael", val)
	}

	// Optional answers should not exist
	_, ok = ltm.Retrieve("user.current_work")
	if ok {
		t.Error("user.current_work should not be set when skipped")
	}
}

func TestOnboardingNilLTM(t *testing.T) {
	reader := strings.NewReader("test\n")
	ran := RunOnboarding(reader, nil, nil)
	if ran {
		t.Fatal("onboarding should not run with nil LTM")
	}
}

func TestWelcomeBackReturningUser(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	ltm.Store("user.name", "Raphael", "personal")
	ltm.Store("user.current_work", "building Nous", "work")

	shown := WelcomeBack(ltm)
	if !shown {
		t.Fatal("expected welcome-back for returning user")
	}
}

func TestWelcomeBackEmptyLTM(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())

	shown := WelcomeBack(ltm)
	if shown {
		t.Fatal("should not show welcome-back with empty LTM")
	}
}

func TestWelcomeBackNoName(t *testing.T) {
	ltm := memory.NewLongTermMemory(t.TempDir())
	ltm.Store("user.role", "engineer", "personal")

	shown := WelcomeBack(ltm)
	if shown {
		t.Fatal("should not show welcome-back without user.name")
	}
}
