package cognitive

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupThinkingPipeline creates a fully wired thinking engine for tests.
func setupThinkingPipeline(t *testing.T) *ThinkingEngine {
	t.Helper()
	dir := t.TempDir()
	graph := NewCognitiveGraph(filepath.Join(dir, "graph.json"))
	semantic := NewSemanticEngine()
	causal := NewCausalEngine()
	patterns := NewPatternDetector()
	composer := NewComposer(graph, semantic, causal, patterns)

	// Load knowledge packages
	packDir := filepath.Join("..", "..", "packages")
	if _, err := os.Stat(packDir); err != nil {
		t.Skip("packages directory not found")
	}
	loader := NewPackageLoader(graph, composer.Generative, composer, packDir)
	loader.LoadAll()

	te := NewThinkingEngine(graph, composer)
	return te
}

func TestTaskClassification(t *testing.T) {
	te := &ThinkingEngine{}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  TASK CLASSIFICATION")
	fmt.Println(strings.Repeat("=", 70))

	tests := []struct {
		query    string
		expected ThinkTask
	}{
		// Compose
		{"Write me an email to my boss asking for vacation", TaskCompose},
		{"Draft a letter to the team about the new policy", TaskCompose},
		{"Compose a message to my professor", TaskCompose},
		{"Help me write an email about rescheduling", TaskCompose},

		// Brainstorm
		{"Brainstorm ideas for a birthday party", TaskBrainstorm},
		{"Give me ideas for a startup name", TaskBrainstorm},
		{"Help me think of creative solutions", TaskBrainstorm},
		{"Come up with names for my project", TaskBrainstorm},

		// Teach
		{"Teach me about recursion", TaskTeach},
		{"Explain machine learning to me", TaskTeach},
		{"Help me understand quantum physics", TaskTeach},
		{"Walk me through how databases work", TaskTeach},
		{"How does a compiler work?", TaskTeach},

		// Advise
		{"What should I do about my career?", TaskAdvise},
		{"I'm feeling overwhelmed with work", TaskAdvise},
		{"I need help with a decision", TaskAdvise},
		{"Any suggestions for managing stress?", TaskAdvise},

		// Compare
		{"Compare Python and Go", TaskCompare},
		{"What's the difference between TCP and UDP?", TaskCompare},
		{"Pros and cons of remote work", TaskCompare},
		{"Which is better, React or Vue?", TaskCompare},

		// Summarize
		{"Summarize the key principles of Stoicism", TaskSummarize},
		{"Give me the gist of machine learning", TaskSummarize},
		{"TL;DR on quantum mechanics", TaskSummarize},

		// Create
		{"Write a poem about the ocean", TaskCreate},
		{"Write a short story about adventure", TaskCreate},
		{"Create a haiku about mountains", TaskCreate},

		// Plan
		{"Help me plan a trip to Japan", TaskPlan},
		{"Create a plan for learning programming", TaskPlan},
		{"What's the best approach to building an app?", TaskPlan},

		// Debate
		{"Make the case for renewable energy", TaskDebate},
		{"Argue for open source software", TaskDebate},

		// Converse (default)
		{"Hello", TaskConverse},
		{"Tell me about Stoicism", TaskTeach},
		{"What is DNA?", TaskTeach},
		{"Give me an overview of operating systems", TaskTeach},
	}

	passed := 0
	for _, tt := range tests {
		got := te.classifyTask(tt.query)
		status := "PASS"
		if got != tt.expected {
			status = "FAIL"
			t.Errorf("classifyTask(%q) = %s, want %s", tt.query, taskName(got), taskName(tt.expected))
		} else {
			passed++
		}
		fmt.Printf("  [%s] %-55s → %s\n", status, tt.query, taskName(got))
	}
	fmt.Printf("\n  Classified: %d/%d\n", passed, len(tests))
}

func TestEmailComposition(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  EMAIL COMPOSITION")
	fmt.Println(strings.Repeat("=", 70))

	emails := []struct {
		query     string
		wantParts []string // substrings that should appear
	}{
		{
			"Write me an email to my boss asking for vacation",
			[]string{"Boss", "vacation"},
		},
		{
			"Draft an email to my professor about an extension",
			[]string{"Professor", "extension"},
		},
		{
			"Write a casual email to my friend about the weekend",
			[]string{"Friend", "weekend"},
		},
		{
			"Compose a formal email to the client about the project update",
			[]string{"Client", "project"},
		},
	}

	for _, tt := range emails {
		ctx := &ThinkContext{UserName: "Raphael"}
		result := te.Think(tt.query, ctx)
		if result == nil {
			t.Errorf("Think(%q) returned nil", tt.query)
			continue
		}

		fmt.Printf("\n  Q: %s\n", tt.query)
		fmt.Printf("  Frame: %s | Task: %s\n", result.Frame, taskName(result.Task))
		fmt.Printf("  ---\n")
		for _, line := range strings.Split(result.Text, "\n") {
			if line != "" {
				fmt.Printf("  | %s\n", line)
			} else {
				fmt.Printf("  |\n", )
			}
		}

		// Verify structure
		if result.Frame != "email" {
			t.Errorf("expected email frame, got %s", result.Frame)
		}
		if result.Task != TaskCompose {
			t.Errorf("expected TaskCompose, got %s", taskName(result.Task))
		}

		// Check email has greeting and sign-off
		text := result.Text
		hasGreeting := strings.Contains(text, "Dear") || strings.Contains(text, "Hi") ||
			strings.Contains(text, "Hey") || strings.Contains(text, "Hello") ||
			strings.Contains(text, "Good day") || strings.Contains(text, "Good morning")
		if !hasGreeting {
			t.Error("email should have a greeting")
		}

		hasSignoff := strings.Contains(text, "regards") || strings.Contains(text, "Regards") ||
			strings.Contains(text, "Sincerely") || strings.Contains(text, "Cheers") ||
			strings.Contains(text, "Thanks") || strings.Contains(text, "Best") ||
			strings.Contains(text, "Respectfully")
		if !hasSignoff {
			t.Error("email should have a sign-off")
		}

		// Check user name appears
		if strings.Contains(text, "Raphael") {
			fmt.Printf("  [OK] Personalized with user name\n")
		}
	}
}

func TestBrainstorming(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  BRAINSTORMING")
	fmt.Println(strings.Repeat("=", 70))

	topics := []string{
		"Brainstorm ideas for a tech startup",
		"Give me ideas for a birthday party",
		"Help me think of ways to learn programming",
		"Come up with names for a philosophy blog",
		"Ideas for improving my morning routine",
	}

	for _, query := range topics {
		result := te.Think(query, nil)
		if result == nil {
			t.Errorf("Think(%q) returned nil", query)
			continue
		}

		fmt.Printf("\n  Q: %s\n", query)
		fmt.Printf("  Frame: %s\n", result.Frame)

		// Count numbered ideas
		ideaCount := 0
		for _, line := range strings.Split(result.Text, "\n") {
			if len(line) > 2 && line[0] >= '1' && line[0] <= '9' && line[1] == '.' {
				ideaCount++
			}
			if line != "" {
				preview := line
				if len(preview) > 80 {
					preview = preview[:80] + "..."
				}
				fmt.Printf("  | %s\n", preview)
			}
		}
		fmt.Printf("  Ideas generated: %d\n", ideaCount)

		if ideaCount < 3 {
			t.Errorf("expected at least 3 ideas for %q, got %d", query, ideaCount)
		}
		if result.Frame != "brainstorm" {
			t.Errorf("expected brainstorm frame, got %s", result.Frame)
		}
	}
}

func TestExplanation(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  EXPLANATION / TEACHING")
	fmt.Println(strings.Repeat("=", 70))

	queries := []struct {
		query string
		frame string
	}{
		{"Teach me about Stoicism", "tutorial"},
		{"Explain quantum mechanics to me", "tutorial"},
		{"Help me understand DNA", "tutorial"},
		{"How does evolution work?", "tutorial"},
		{"Walk me through the basics of Python", "tutorial"},
	}

	for _, tt := range queries {
		result := te.Think(tt.query, nil)
		if result == nil {
			t.Errorf("Think(%q) returned nil", tt.query)
			continue
		}

		preview := result.Text
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		fmt.Printf("\n  Q: %s\n", tt.query)
		fmt.Printf("  Frame: %s | Task: %s\n", result.Frame, taskName(result.Task))
		fmt.Printf("  A: %s\n", preview)

		if len(result.Text) < 50 {
			t.Errorf("explanation for %q too short: %d chars", tt.query, len(result.Text))
		}
	}
}

func TestComparison(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  COMPARISON")
	fmt.Println(strings.Repeat("=", 70))

	comparisons := []struct {
		query string
		itemA string
		itemB string
	}{
		{"Compare Python and Go", "python", "go"},
		{"What's the difference between Stoicism and Epicureanism?", "stoicism", "epicureanism"},
		{"Pros and cons of city vs countryside living", "city", "countryside"},
	}

	for _, tt := range comparisons {
		result := te.Think(tt.query, nil)
		if result == nil {
			t.Errorf("Think(%q) returned nil", tt.query)
			continue
		}

		fmt.Printf("\n  Q: %s\n", tt.query)
		fmt.Printf("  Frame: %s\n", result.Frame)
		preview := result.Text
		if len(preview) > 300 {
			preview = preview[:300] + "..."
		}
		fmt.Printf("  A: %s\n", preview)

		if result.Frame != "comparison" {
			t.Errorf("expected comparison frame, got %s", result.Frame)
		}
		if len(result.Text) < 50 {
			t.Errorf("comparison too short: %d chars", len(result.Text))
		}
	}
}

func TestAdvice(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  ADVICE")
	fmt.Println(strings.Repeat("=", 70))

	queries := []string{
		"I'm feeling overwhelmed with work, what should I do?",
		"Any suggestions for managing stress?",
		"I need help deciding between two job offers",
		"What should I do about my procrastination?",
	}

	for _, query := range queries {
		result := te.Think(query, nil)
		if result == nil {
			t.Errorf("Think(%q) returned nil", query)
			continue
		}

		fmt.Printf("\n  Q: %s\n", query)
		fmt.Printf("  Frame: %s\n", result.Frame)
		preview := result.Text
		if len(preview) > 300 {
			preview = preview[:300] + "..."
		}
		fmt.Printf("  A: %s\n", preview)

		if result.Frame != "advice" {
			t.Errorf("expected advice frame, got %s", result.Frame)
		}

		// Should have numbered suggestions
		hasNumbers := strings.Contains(result.Text, "1.") && strings.Contains(result.Text, "2.")
		if !hasNumbers {
			t.Error("advice should contain numbered suggestions")
		}
	}
}

func TestCreativeWriting(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  CREATIVE WRITING")
	fmt.Println(strings.Repeat("=", 70))

	queries := []struct {
		query   string
		check   string // substring to look for
	}{
		{"Write a poem about the ocean", "ocean"},
		{"Create a haiku about mountains", "mountains"},
		{"Write a short story about discovery", "discovery"},
		{"Write a poem about silence", "silence"},
	}

	for _, tt := range queries {
		result := te.Think(tt.query, nil)
		if result == nil {
			t.Errorf("Think(%q) returned nil", tt.query)
			continue
		}

		fmt.Printf("\n  Q: %s\n", tt.query)
		fmt.Printf("  Frame: %s | Task: %s\n", result.Frame, taskName(result.Task))
		fmt.Printf("  ---\n")
		for _, line := range strings.Split(result.Text, "\n") {
			fmt.Printf("  | %s\n", line)
		}

		if result.Task != TaskCreate {
			t.Errorf("expected TaskCreate, got %s", taskName(result.Task))
		}
		if len(result.Text) < 20 {
			t.Errorf("creative text too short: %d chars", len(result.Text))
		}
	}
}

func TestPlanning(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  PLANNING")
	fmt.Println(strings.Repeat("=", 70))

	queries := []string{
		"Help me plan a trip to Japan",
		"Create a plan for learning to code",
		"What's the best approach to starting a business?",
	}

	for _, query := range queries {
		result := te.Think(query, nil)
		if result == nil {
			t.Errorf("Think(%q) returned nil", query)
			continue
		}

		fmt.Printf("\n  Q: %s\n", query)
		fmt.Printf("  Frame: %s\n", result.Frame)
		preview := result.Text
		if len(preview) > 300 {
			preview = preview[:300] + "..."
		}
		fmt.Printf("  A: %s\n", preview)

		if result.Frame != "plan" {
			t.Errorf("expected plan frame, got %s", result.Frame)
		}
		// Should have phases
		if !strings.Contains(result.Text, "Phase") {
			t.Error("plan should contain phases")
		}
	}
}

func TestAnalogy(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  ANALOGY ENGINE")
	fmt.Println(strings.Repeat("=", 70))

	topics := []string{"Python", "Stoicism", "DNA", "evolution", "Linux"}

	for _, topic := range topics {
		analogy := te.GenerateAnalogy(topic)
		if analogy != "" {
			fmt.Printf("  %s → %s\n", topic, analogy)
		} else {
			fmt.Printf("  %s → (no analogy found)\n", topic)
		}
	}
}

func TestConceptBlend(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  CONCEPT BLENDING")
	fmt.Println(strings.Repeat("=", 70))

	pairs := [][2]string{
		{"Python", "Stoicism"},
		{"DNA", "Linux"},
		{"evolution", "quantum mechanics"},
	}

	for _, pair := range pairs {
		blends := te.ConceptBlend(pair[0], pair[1])
		fmt.Printf("\n  %s + %s:\n", pair[0], pair[1])
		if len(blends) == 0 {
			fmt.Printf("    (no blends generated)\n")
		}
		for _, b := range blends {
			fmt.Printf("    → %s\n", b)
		}
	}
}

func TestThinkingVsLLM(t *testing.T) {
	te := setupThinkingPipeline(t)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("  THINKING ENGINE vs LLM COMPARISON")
	fmt.Println(strings.Repeat("=", 70))

	queries := []struct {
		query string
		desc  string
	}{
		{"Write me an email to my boss asking for a day off", "Email composition"},
		{"Brainstorm ideas for a tech startup", "Brainstorming"},
		{"Teach me about DNA", "Teaching/explaining"},
		{"Compare Python and Go", "Comparison"},
		{"I'm feeling stressed about exams, what should I do?", "Advice"},
		{"Write a poem about the stars", "Creative writing"},
		{"Help me plan learning Stoicism", "Planning"},
		{"Summarize the key principles of evolution", "Summary"},
		{"Make the case for open source software", "Argumentation"},
		{"What's the best approach to building a mobile app?", "Strategy"},
	}

	fmt.Printf("\n  %-50s %-12s %-8s %s\n", "QUERY", "FRAME", "WORDS", "QUALITY")
	fmt.Println("  " + strings.Repeat("-", 100))

	totalWords := 0
	allPassed := true
	for _, tt := range queries {
		result := te.Think(tt.query, &ThinkContext{UserName: "Raphael"})
		if result == nil {
			t.Errorf("Think(%q) returned nil", tt.query)
			allPassed = false
			continue
		}

		words := len(strings.Fields(result.Text))
		totalWords += words

		// Quality check: is the response useful?
		quality := "OK"
		if words < 10 {
			quality = "TOO SHORT"
			allPassed = false
		} else if words > 50 {
			quality = "GOOD"
		}

		queryPreview := tt.query
		if len(queryPreview) > 48 {
			queryPreview = queryPreview[:48] + ".."
		}
		fmt.Printf("  %-50s %-12s %-8d %s\n", queryPreview, result.Frame, words, quality)
	}

	fmt.Printf("\n  Total words generated: %d\n", totalWords)
	fmt.Printf("  Average words/response: %d\n", totalWords/len(queries))

	if !allPassed {
		t.Error("some thinking engine responses were too short")
	}

	// Now show a few full examples
	fmt.Println("\n" + strings.Repeat("-", 70))
	fmt.Println("  FULL EXAMPLES")
	fmt.Println(strings.Repeat("-", 70))

	examples := []string{
		"Write me an email to my professor asking for an extension on my paper",
		"Brainstorm ideas for improving my morning routine",
		"Write a poem about time",
	}

	for _, query := range examples {
		result := te.Think(query, &ThinkContext{UserName: "Raphael"})
		if result == nil {
			continue
		}
		fmt.Printf("\n  Q: %s\n", query)
		fmt.Printf("  Frame: %s | Task: %s | Words: %d\n",
			result.Frame, taskName(result.Task), len(strings.Fields(result.Text)))
		fmt.Printf("  ─────────────────────────────────\n")
		for _, line := range strings.Split(result.Text, "\n") {
			fmt.Printf("  %s\n", line)
		}
	}
}
