package cognitive

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// TestNeuralEvaluation is a comprehensive evaluation of the neural NLU system.
// All test inputs are NOVEL — not in the training data — to measure true generalization.
func TestNeuralEvaluation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping neural evaluation in short mode")
	}
	nlu := NewNLU()
	examples := GenerateTrainingData(nlu)
	augmented := AugmentExamples(examples)

	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)
	result := nc.Train(augmented, 80, 0.1)
	t.Logf("Training: %d examples, %d intents, %.1f%% accuracy",
		len(augmented), result.Intents, result.Accuracy*100)

	// Wire neural into NLU for full pipeline test
	nlu.Neural = &NeuralNLU{Classifier: nc}

	// ─── NOVEL TEST CASES ───
	// None of these exact strings appear in GenerateTrainingData.
	// They test generalization to unseen phrasings.
	tests := []struct {
		input string
		want  string
		cat   string // category for reporting
	}{
		// Greetings — novel phrasings
		{"wassup", "greeting", "greeting"},
		{"hey hey hey", "greeting", "greeting"},
		{"hiii", "greeting", "greeting"},
		{"good day sir", "greeting", "greeting"},
		{"heya", "greeting", "greeting"},
		{"hows it going mate", "greeting", "greeting"},
		{"aloha", "greeting", "greeting"},

		// Farewells — novel
		{"catch you later", "farewell", "farewell"},
		{"peace out", "farewell", "farewell"},
		{"gotta bounce", "farewell", "farewell"},
		{"im out", "farewell", "farewell"},
		{"have a good one", "farewell", "farewell"},
		{"adios", "farewell", "farewell"},

		// Dict — novel words and phrasings
		{"define obfuscate", "dict", "dict"},
		{"whats the definition of entropy", "dict", "dict"},
		{"meaning of soliloquy", "dict", "dict"},
		{"synonyms for melancholy", "dict", "dict"},
		{"what does verisimilitude mean", "dict", "dict"},
		{"define the word quintessential", "dict", "dict"},

		// Translate — novel languages and phrases
		{"translate goodbye to italian", "translate", "translate"},
		{"how do you say water in arabic", "translate", "translate"},
		{"what is cat in russian", "translate", "translate"},
		{"translate I love you to portuguese", "translate", "translate"},
		{"say cheese in mandarin", "translate", "translate"},

		// Meta — novel identity questions
		{"what kind of AI are you", "meta", "meta"},
		{"how old are you", "meta", "meta"},
		{"can you learn new things", "meta", "meta"},
		{"what language were you written in", "meta", "meta"},
		{"do you have a personality", "meta", "meta"},
		{"tell me about your capabilities", "meta", "meta"},

		// Reminders — novel
		{"remind me to water the plants at 6pm", "reminder", "reminder"},
		{"set a reminder to submit the report", "reminder", "reminder"},
		{"dont forget to feed the cat", "reminder", "reminder"},
		{"remind me about the dentist next tuesday", "reminder", "reminder"},

		// Remember/Recall — novel
		{"my dogs name is max", "remember", "memory"},
		{"remember that i prefer dark theme", "remember", "memory"},
		{"i graduated from MIT", "remember", "memory"},
		{"whats my dogs name", "recall", "memory"},
		{"do you know where i work", "recall", "memory"},
		{"what have i told you about myself", "recall", "memory"},

		// Creative — novel
		{"write me a sonnet about love", "creative", "creative"},
		{"compose something about the rain", "creative", "creative"},
		{"tell me a riddle", "creative", "creative"},
		{"make up a story about dragons", "creative", "creative"},
		{"write something beautiful", "creative", "creative"},
		{"give me something to think about", "creative", "creative"},
		{"philosophize about time", "creative", "creative"},

		// Explain — novel topics
		{"what is machine learning", "explain", "explain"},
		{"explain how batteries work", "explain", "explain"},
		{"tell me about the french revolution", "explain", "explain"},
		{"who was nikola tesla", "explain", "explain"},
		{"how does photosynthesis work", "explain", "explain"},
		{"what is stoicism all about", "explain", "explain"},

		// Weather — novel phrasings
		{"is it going to snow tonight", "weather", "weather"},
		{"whats the temperature outside", "weather", "weather"},
		{"will i need an umbrella tomorrow", "weather", "weather"},
		{"hows the weather looking for the weekend", "weather", "weather"},

		// Timer — novel
		{"start a 25 minute countdown", "timer", "timer"},
		{"set timer 10 min", "timer", "timer"},
		{"start a pomodoro session", "timer", "timer"},

		// Password — novel
		{"create a secure password for me", "password", "password"},
		{"i need a new strong password", "password", "password"},
		{"make me a random 20 character password", "password", "password"},

		// App — novel
		{"launch chrome browser", "app", "app"},
		{"fire up terminal", "app", "app"},
		{"close the music player", "app", "app"},
		{"what programs are running right now", "app", "app"},

		// Screenshot — novel
		{"capture my screen please", "screenshot", "screenshot"},
		{"grab a screenshot of this window", "screenshot", "screenshot"},

		// Daily briefing — novel
		{"morning report please", "daily_briefing", "briefing"},
		{"whats on my agenda today", "daily_briefing", "briefing"},
		{"start my day", "daily_briefing", "briefing"},
		{"give me the daily summary", "daily_briefing", "briefing"},

		// Affirmation — novel
		{"thats exactly what i needed", "affirmation", "affirmation"},
		{"perfect thanks a lot", "affirmation", "affirmation"},
		{"awesome work", "affirmation", "affirmation"},
		{"nah forget it", "affirmation", "affirmation"},

		// Convert — novel
		{"how many liters in a gallon", "convert", "convert"},
		{"50 celsius to fahrenheit", "convert", "convert"},
		{"convert 200 pounds to kg", "convert", "convert"},

		// Search — novel
		{"look up kubernetes best practices", "search", "search"},
		{"google how to make sourdough", "search", "search"},
		{"find articles about climate change", "search", "search"},

		// System info — novel
		{"show me my cpu usage", "sysinfo", "sysinfo"},
		{"what operating system am i running", "sysinfo", "sysinfo"},
		{"how much free memory do i have", "sysinfo", "sysinfo"},

		// File operations — novel
		{"show me whats in the downloads folder", "find_files", "files"},
		{"find all jpg files on my desktop", "find_files", "files"},
		{"where are my go source files", "find_files", "files"},

		// Notes — novel
		{"jot down a note about the project deadline", "note", "notes"},
		{"save a quick note", "note", "notes"},
		{"search my notes for meeting minutes", "note", "notes"},

		// Todo — novel
		{"add groceries to my task list", "todo", "todo"},
		{"whats left on my to do list", "todo", "todo"},
		{"mark the first item as complete", "todo", "todo"},

		// Calendar — novel
		{"do i have any meetings tomorrow morning", "calendar", "calendar"},
		{"show me my schedule for next week", "calendar", "calendar"},
		{"whats coming up on my calendar", "calendar", "calendar"},

		// Follow up — novel
		{"go deeper on that", "follow_up", "followup"},
		{"elaborate please", "follow_up", "followup"},
		{"can you explain that more", "follow_up", "followup"},

		// Journal — novel
		{"write in my diary about today", "journal", "journal"},
		{"what did i journal about last week", "journal", "journal"},

		// Habits — novel
		{"did i exercise today", "habit", "habit"},
		{"track my reading habit", "habit", "habit"},

		// Expense — novel
		{"i spent 30 bucks on gas", "expense", "expense"},
		{"log a 15 dollar purchase for lunch", "expense", "expense"},
		{"how much have i spent on food this month", "expense", "expense"},

		// Bookmark — novel
		{"save this url to my bookmarks", "bookmark", "bookmark"},
		{"show all my saved bookmarks", "bookmark", "bookmark"},

		// Network — novel
		{"is the internet working", "network", "network"},
		{"ping 192.168.1.1", "network", "network"},
		{"check if github.com is up", "network", "network"},

		// Hash — novel
		{"compute sha256 of this file", "hash", "hash"},
		{"base64 decode this string", "hash", "hash"},

		// Compare — novel
		{"rust versus c++", "compare", "compare"},
		{"whats the difference between tcp and udp", "compare", "compare"},

		// Recommendation — novel
		{"suggest a podcast about history", "recommendation", "recommend"},
		{"what movie should i watch tonight", "recommendation", "recommend"},
		{"any restaurant recommendations nearby", "recommendation", "recommend"},
	}

	// ─── RUN EVALUATION ───
	catCorrect := make(map[string]int)
	catTotal := make(map[string]int)
	totalCorrect := 0
	var misses []string

	for _, tt := range tests {
		intent, conf := nc.Classify(tt.input)
		catTotal[tt.cat]++
		if intent == tt.want {
			totalCorrect++
			catCorrect[tt.cat]++
		} else {
			misses = append(misses, fmt.Sprintf("  %-50s → %-16s (%.2f) want %-16s", tt.input, intent, conf, tt.want))
		}
	}

	// ─── REPORT ───
	totalAcc := float64(totalCorrect) / float64(len(tests)) * 100
	t.Logf("")
	t.Logf("═══════════════════════════════════════════════════════")
	t.Logf("  NEURAL NLU EVALUATION — NOVEL INPUTS ONLY")
	t.Logf("═══════════════════════════════════════════════════════")
	t.Logf("  Overall: %d/%d (%.1f%%)", totalCorrect, len(tests), totalAcc)
	t.Logf("")

	// Per-category breakdown
	cats := []string{
		"greeting", "farewell", "dict", "translate", "meta",
		"reminder", "memory", "creative", "explain", "weather",
		"timer", "password", "app", "screenshot", "briefing",
		"affirmation", "convert", "search", "sysinfo", "files",
		"notes", "todo", "calendar", "followup", "journal",
		"habit", "expense", "bookmark", "network", "hash",
		"compare", "recommend",
	}
	for _, cat := range cats {
		total := catTotal[cat]
		if total == 0 {
			continue
		}
		correct := catCorrect[cat]
		pct := float64(correct) / float64(total) * 100
		bar := strings.Repeat("█", int(pct/5))
		t.Logf("  %-14s %2d/%-2d %5.1f%% %s", cat, correct, total, pct, bar)
	}

	if len(misses) > 0 {
		t.Logf("")
		t.Logf("  MISSES (%d):", len(misses))
		for _, m := range misses {
			t.Logf("%s", m)
		}
	}

	t.Logf("═══════════════════════════════════════════════════════")

	if totalAcc < 70 {
		t.Errorf("overall accuracy too low: %.1f%% (want ≥70%%)", totalAcc)
	}
}

// TestNeuralVsPatternComparison compares neural and pattern NLU on the same inputs.
func TestNeuralVsPatternComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping neural comparison in short mode")
	}
	nlu := NewNLU()

	// Train neural
	examples := GenerateTrainingData(nlu)
	augmented := AugmentExamples(examples)
	nc := NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize)
	nc.Train(augmented, 80, 0.1)

	tests := []struct {
		input string
		want  string
	}{
		{"hello", "greeting"},
		{"yo whats up", "greeting"},
		{"hey man how are you doing", "greeting"},
		{"define serendipity", "dict"},
		{"what does ephemeral mean", "dict"},
		{"translate hello to french", "translate"},
		{"how do you say goodbye in spanish", "translate"},
		{"what is your name", "meta"},
		{"who made you", "meta"},
		{"do you have feelings", "meta"},
		{"good morning", "daily_briefing"},
		{"remind me to call mom tomorrow", "reminder"},
		{"remember my favorite color is blue", "remember"},
		{"what is my favorite color", "recall"},
		{"write me a poem about the ocean", "creative"},
		{"tell me something interesting", "creative"},
		{"I feel happy today", "greeting"},
		{"what is quantum physics", "explain"},
		{"what is the meaning of life", "creative"},
		{"set a timer for 5 minutes", "timer"},
		{"generate a password", "password"},
		{"open firefox", "app"},
		{"take a screenshot", "screenshot"},
		{"search for golang tutorials", "search"},
		{"suggest a good book", "recommendation"},
		{"bye", "farewell"},
		{"thanks", "affirmation"},
		{"later dude", "farewell"},
		{"convert 100 miles to km", "convert"},
		{"python vs golang", "compare"},
	}

	neuralCorrect := 0
	patternCorrect := 0
	var neuralTimes, patternTimes []time.Duration

	for _, tt := range tests {
		// Neural
		start := time.Now()
		neuralIntent, _ := nc.Classify(tt.input)
		neuralDur := time.Since(start)
		neuralTimes = append(neuralTimes, neuralDur)

		// Pattern (use a fresh NLU without neural)
		patternNLU := NewNLU()
		start = time.Now()
		patternResult := patternNLU.Understand(tt.input)
		patternDur := time.Since(start)
		patternTimes = append(patternTimes, patternDur)

		neuralOK := neuralIntent == tt.want
		patternOK := patternResult.Intent == tt.want

		if neuralOK {
			neuralCorrect++
		}
		if patternOK {
			patternCorrect++
		}

		status := "BOTH"
		if neuralOK && !patternOK {
			status = "NEURAL WINS"
		} else if !neuralOK && patternOK {
			status = "PATTERN WINS"
		} else if !neuralOK && !patternOK {
			status = "BOTH MISS"
		}

		if status != "BOTH" {
			t.Logf("%-45s neural=%-16s pattern=%-16s want=%-16s %s",
				tt.input, neuralIntent, patternResult.Intent, tt.want, status)
		}
	}

	// Compute average times
	var totalNeural, totalPattern time.Duration
	for i := range neuralTimes {
		totalNeural += neuralTimes[i]
		totalPattern += patternTimes[i]
	}
	avgNeural := totalNeural / time.Duration(len(neuralTimes))
	avgPattern := totalPattern / time.Duration(len(patternTimes))

	t.Logf("")
	t.Logf("═══════════════════════════════════════════════════════")
	t.Logf("  NEURAL vs PATTERN COMPARISON")
	t.Logf("═══════════════════════════════════════════════════════")
	t.Logf("  Neural accuracy:   %d/%d (%.1f%%)", neuralCorrect, len(tests), float64(neuralCorrect)/float64(len(tests))*100)
	t.Logf("  Pattern accuracy:  %d/%d (%.1f%%)", patternCorrect, len(tests), float64(patternCorrect)/float64(len(tests))*100)
	t.Logf("  Neural avg time:   %s", avgNeural)
	t.Logf("  Pattern avg time:  %s", avgPattern)
	t.Logf("  Speedup:           %.1fx", float64(avgPattern)/float64(avgNeural))
	t.Logf("═══════════════════════════════════════════════════════")
}

// TestFullPipelineE2E tests the entire NLU→Action pipeline end-to-end with neural.
func TestFullPipelineE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping full pipeline E2E in short mode")
	}
	nlu := NewNLU()
	if err := nlu.InitNeural(""); err != nil {
		t.Fatalf("InitNeural: %v", err)
	}

	tests := []struct {
		input      string
		wantIntent string
		wantAction string
	}{
		{"hello", "greeting", "respond"},
		{"define serendipity", "dict", "dict"},
		{"translate hello to french", "translate", "translate"},
		{"what is your name", "meta", "respond"},
		{"good morning", "daily_briefing", "daily_briefing"},
		{"remind me to call mom", "reminder", "reminder"},
		{"write me a poem", "creative", "creative"},
		{"what is quantum physics", "explain", "lookup_knowledge"},
		{"set a timer for 5 minutes", "timer", "timer"},
		{"generate a password", "password", "password"},
		{"bye", "farewell", "respond"},
		{"thanks", "affirmation", "respond"},
	}

	correct := 0
	for _, tt := range tests {
		result := nlu.Understand(tt.input)
		intentOK := result.Intent == tt.wantIntent
		actionOK := result.Action == tt.wantAction
		if intentOK && actionOK {
			correct++
		} else {
			t.Logf("%-40s intent=%-16s (want %-16s) action=%-16s (want %-16s)",
				tt.input, result.Intent, tt.wantIntent, result.Action, tt.wantAction)
		}
	}

	accuracy := float64(correct) / float64(len(tests)) * 100
	t.Logf("Full pipeline E2E: %d/%d (%.0f%%)", correct, len(tests), accuracy)

	if accuracy < 90 {
		t.Errorf("E2E accuracy too low: %.0f%%", accuracy)
	}
}
