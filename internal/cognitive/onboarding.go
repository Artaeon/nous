package cognitive

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"github.com/artaeon/nous/internal/memory"
)

// OnboardingQuestion defines a single question in the first-run interview.
type OnboardingQuestion struct {
	Prompt   string // The question to display
	Key      string // LTM key to store the answer under
	Category string // LTM category
	Optional bool   // If true, user can skip with empty input
}

// DefaultOnboardingQuestions returns the standard first-run questions.
func DefaultOnboardingQuestions() []OnboardingQuestion {
	return []OnboardingQuestion{
		{
			Prompt:   "What's your name?",
			Key:      "user.name",
			Category: "personal",
		},
		{
			Prompt:   "What do you do? (e.g. software engineer, student, researcher)",
			Key:      "user.role",
			Category: "personal",
		},
		{
			Prompt:   "What are you working on right now?",
			Key:      "user.current_work",
			Category: "work",
			Optional: true,
		},
		{
			Prompt:   "What are your interests? (anything — hobbies, topics, technologies)",
			Key:      "user.interests",
			Category: "personal",
			Optional: true,
		},
		{
			Prompt:   "How should I help you? (e.g. coding assistant, research partner, brainstorm buddy)",
			Key:      "user.preferred_role",
			Category: "personal",
			Optional: true,
		},
	}
}

// RunOnboarding runs the first-time user interview and stores answers in LTM + working memory.
// Returns true if onboarding ran, false if skipped (not first run or non-interactive).
func RunOnboarding(r io.Reader, ltm *memory.LongTermMemory, wm *memory.WorkingMemory) bool {
	if ltm == nil || ltm.Size() > 0 {
		return false // Not first run
	}

	questions := DefaultOnboardingQuestions()
	scanner := bufio.NewScanner(r)

	// Welcome message
	fmt.Println()
	fmt.Print(Panel("Welcome", []string{
		"I'm " + Styled(ColorBold, "Nous") + " — your personal AI assistant, running fully on your machine.",
		"Let me learn a bit about you so I can be most helpful.",
		Styled(ColorDim, "Press Enter to skip any question."),
	}))
	fmt.Println()

	answered := 0
	for _, q := range questions {
		label := Styled(ColorCyan, "  ? ")
		if q.Optional {
			fmt.Printf("%s%s %s(optional)%s\n", label, q.Prompt, ColorDim, ColorReset)
		} else {
			fmt.Printf("%s%s\n", label, q.Prompt)
		}
		fmt.Print(Styled(ColorCyan, "  › "))

		if !scanner.Scan() {
			break
		}
		answer := strings.TrimSpace(scanner.Text())
		if answer == "" {
			continue
		}

		ltm.Store(q.Key, answer, q.Category)
		if wm != nil {
			wm.Store(q.Key, answer, 1.0)
		}
		answered++
	}

	if answered == 0 {
		return false
	}

	// Personalized confirmation
	name, hasName := ltm.Retrieve("user.name")
	fmt.Println()
	if hasName {
		fmt.Printf("  %s✓%s Got it, %s%s%s! I'll remember everything across sessions.\n",
			ColorGreen, ColorReset, ColorBold, name, ColorReset)
	} else {
		fmt.Printf("  %s✓%s Profile saved! I'll remember everything across sessions.\n",
			ColorGreen, ColorReset)
	}
	fmt.Println()

	return true
}

// WelcomeBack prints a personalized greeting for returning users.
// Returns true if a greeting was printed.
func WelcomeBack(ltm *memory.LongTermMemory) bool {
	if ltm == nil || ltm.Size() == 0 {
		return false
	}

	name, hasName := ltm.Retrieve("user.name")
	if !hasName {
		return false
	}

	// Build a context-aware greeting
	parts := []string{fmt.Sprintf("Welcome back, %s%s%s!", ColorBold, name, ColorReset)}

	if work, ok := ltm.Retrieve("user.current_work"); ok {
		parts = append(parts, Styled(ColorDim, "Last project: ")+work)
	}

	fmt.Println()
	fmt.Print(Panel("", parts))
	return true
}
