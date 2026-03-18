package cognitive

import "testing"

// -----------------------------------------------------------------------
// Levenshtein distance
// -----------------------------------------------------------------------

func TestFuzzyLevenshteinDistance(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		// identical
		{"hello", "hello", 0},
		{"", "", 0},
		// one empty
		{"abc", "", 3},
		{"", "xyz", 3},
		// single edit
		{"cat", "bat", 1},    // substitution
		{"cat", "cats", 1},   // insertion
		{"cats", "cat", 1},   // deletion
		// two edits
		{"kitten", "sitting", 3},
		{"weather", "waether", 2}, // transposition = 2 edits in Levenshtein
		{"volume", "voume", 1},    // deletion
		{"translate", "trasnlate", 2},
		// completely different
		{"abc", "xyz", 3},
		// single character
		{"a", "b", 1},
		{"a", "a", 0},
	}
	for _, tt := range tests {
		got := levenshtein(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Fuzzy threshold
// -----------------------------------------------------------------------

func TestFuzzyThreshold(t *testing.T) {
	tests := []struct {
		wordLen int
		want    int
	}{
		{0, 0},
		{1, 0},
		{2, 1},
		{3, 1},
		{4, 1},
		{5, 2},
		{6, 2},
		{10, 2},
	}
	for _, tt := range tests {
		got := fuzzyThreshold(tt.wordLen)
		if got != tt.want {
			t.Errorf("fuzzyThreshold(%d) = %d, want %d", tt.wordLen, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Fuzzy word matching
// -----------------------------------------------------------------------

func TestFuzzyWordMatch(t *testing.T) {
	tests := []struct {
		candidate, target string
		want              bool
	}{
		// exact
		{"weather", "weather", true},
		{"volume", "volume", true},
		// typos within threshold
		{"waether", "weather", true},  // distance 2, len 7 >= 5
		{"weathr", "weather", true},   // distance 1
		{"voume", "volume", true},     // distance 1, len 6 >= 5
		{"volme", "volume", true},     // distance 1
		{"trasnlate", "translate", true}, // distance 2, len 9 >= 5
		{"brightnss", "brightness", true}, // distance 1
		// too many edits
		{"wther", "weather", true},    // distance 2, target len 7 >= 5, threshold 2
		{"vme", "volume", false},      // too short + too many edits
		// short words — now require both >= 4 chars for fuzzy
		{"mut", "mute", false},        // candidate len 3 < 4, no fuzzy
		{"muts", "mute", true},        // both len 4, distance 1
		{"zop", "zip", false},         // both too short for fuzzy
		// short words — threshold 0
		{"a", "b", false},             // len 1, threshold 0
		{"", "x", false},
		// same length but too far
		{"abcde", "xyzwv", false},     // distance 5
	}
	for _, tt := range tests {
		got := fuzzyWordMatch(tt.candidate, tt.target)
		if got != tt.want {
			t.Errorf("fuzzyWordMatch(%q, %q) = %v, want %v", tt.candidate, tt.target, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// fuzzyContains — matching within sentences
// -----------------------------------------------------------------------

func TestFuzzyContains(t *testing.T) {
	tests := []struct {
		input, phrase string
		want          bool
	}{
		// single word fuzzy
		{"check the waether today", "weather", true},
		{"set the voume to 50", "volume", true},
		{"trasnlate this to spanish", "translate", true},
		// exact match
		{"check the weather today", "weather", true},
		// multi-word phrase fuzzy
		{"whats the sreen brightnss", "screen brightness", true},
		// no match
		{"hello world", "weather", false},
		// empty inputs
		{"", "weather", false},
		{"hello", "", false},
		{"", "", false},
		// multi-word sliding window
		{"please turn up the music", "turn up", true},
	}
	for _, tt := range tests {
		got := fuzzyContains(tt.input, tt.phrase)
		if got != tt.want {
			t.Errorf("fuzzyContains(%q, %q) = %v, want %v", tt.input, tt.phrase, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Synonym expansion
// -----------------------------------------------------------------------

func TestFuzzySynonymExpansion(t *testing.T) {
	tests := []struct {
		input      string
		wantIntent string
		wantFound  bool
	}{
		// volume synonyms
		{"crank up the sound", "volume", true},
		{"increase the sound", "volume", true},
		{"raise the audio", "volume", true},
		// dictionary synonyms
		{"define photosynthesis", "dict", true},
		{"meaning of serendipity", "dict", true},
		// compute synonyms
		{"calculate 5 + 3", "compute", true},
		{"solve this equation", "compute", true},
		{"evaluate this expression", "compute", true},
		// journal synonyms
		{"dear diary today was great", "note", true},
		{"log entry for today", "note", true},
		// habit synonyms
		{"daily check", "todo", true},
		{"streak tracker", "todo", true},
		// expense synonyms
		{"i bought groceries", "note", true},
		{"spent 50 on dinner", "note", true},
		// no synonym match
		{"hello world", "", false},
		{"what time is it", "", false},
	}
	for _, tt := range tests {
		intent, _ := expandSynonyms(tt.input)
		found := intent != ""
		if found != tt.wantFound {
			t.Errorf("expandSynonyms(%q): found=%v, want found=%v", tt.input, found, tt.wantFound)
		}
		if found && intent != tt.wantIntent {
			t.Errorf("expandSynonyms(%q): intent=%q, want %q", tt.input, intent, tt.wantIntent)
		}
	}
}

// -----------------------------------------------------------------------
// End-to-end: NLU.Understand with fuzzy/synonym inputs
// -----------------------------------------------------------------------

func TestFuzzyNLUUnderstand(t *testing.T) {
	nlu := NewNLU()

	tests := []struct {
		input      string
		wantIntent string
		desc       string
	}{
		// Typo: "waether" → weather
		{"what's the waether like", "weather", "weather typo"},
		// Typo: "voume" → volume
		{"set the voume to 50", "volume", "volume typo"},
		// Typo: "trasnlate" → translate (synonym map)
		{"trasnlate hello to spanish", "translate", "translate typo via synonym"},
		// Synonym: "crank up" → volume
		{"crank up the speakers", "volume", "crank up synonym"},
		// Synonym: "define" → dict
		{"define serendipity", "dict", "define synonym"},
		// Synonym: "meaning of" → dict
		{"meaning of ephemeral", "dict", "meaning of synonym"},
		// "dear diary" → journal (exact word list match takes priority over synonym)
		{"dear diary today was amazing", "journal", "dear diary exact match"},
		// "bought" → expense (exact word list match takes priority over synonym)
		{"i bought a new laptop", "expense", "bought exact match"},
		// Exact match still works (not broken)
		{"what's the weather like", "weather", "exact weather still works"},
		{"set the volume to 80", "volume", "exact volume still works"},
		{"translate hello to french", "translate", "exact translate still works"},
		// Typo in brightness
		{"set screen brightnss to 50", "brightness", "brightness typo"},
		// Typo: "screnshot" → screenshot
		{"take a screnshot", "screenshot", "screenshot typo"},
	}

	for _, tt := range tests {
		result := nlu.Understand(tt.input)
		if result.Intent != tt.wantIntent {
			t.Errorf("[%s] Understand(%q).Intent = %q, want %q (confidence=%.2f)",
				tt.desc, tt.input, result.Intent, tt.wantIntent, result.Confidence)
		}
	}
}

// -----------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------

func TestFuzzyEdgeCases(t *testing.T) {
	// Very short words should not fuzzy-match wildly
	if fuzzyWordMatch("a", "z") {
		t.Error("single-char words should not fuzzy match different chars")
	}
	if fuzzyWordMatch("", "hello") {
		t.Error("empty string should not fuzzy match")
	}
	if fuzzyWordMatch("hello", "") {
		t.Error("should not fuzzy match empty target")
	}

	// All-typo input should not crash
	nlu := NewNLU()
	result := nlu.Understand("xzqwp bnmkl trfvg")
	if result == nil {
		t.Error("Understand should never return nil")
	}

	// Empty input
	result = nlu.Understand("")
	if result.Intent != "unknown" {
		t.Errorf("empty input intent = %q, want 'unknown'", result.Intent)
	}

	// Single character input
	result = nlu.Understand("x")
	if result == nil {
		t.Error("single char should not crash")
	}

	// Levenshtein with identical long strings
	dist := levenshtein("abcdefghijklmnop", "abcdefghijklmnop")
	if dist != 0 {
		t.Errorf("identical strings distance = %d, want 0", dist)
	}
}

// -----------------------------------------------------------------------
// Performance: ensure fuzzy matching doesn't blow up on longer input
// -----------------------------------------------------------------------

func TestFuzzyPerformanceBounded(t *testing.T) {
	nlu := NewNLU()
	// A moderately long input — should still complete quickly.
	longInput := "please help me with the waether forecast for tomorrow and also trasnlate something to french and set the voume louder"
	result := nlu.Understand(longInput)
	// Should match something (weather is the first fuzzy hit in the tool list order).
	if result.Intent == "unknown" {
		t.Error("long input with typos should still classify, got unknown")
	}
}
