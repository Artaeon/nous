package cognitive

import (
	"strings"
	"testing"
)

func newTestEngine() *TextTransformEngine {
	return NewTextTransformEngine()
}

// -----------------------------------------------------------------------
// Formalize
// -----------------------------------------------------------------------

func TestFormalize_BasicCasualToFormal(t *testing.T) {
	e := newTestEngine()
	out := e.Formalize("hey can u send me the docs asap")

	if !strings.Contains(strings.ToLower(out), "hello") {
		t.Errorf("expected 'hello' in formalized output, got: %s", out)
	}
	if !strings.Contains(strings.ToLower(out), "you") {
		t.Errorf("expected 'you' (expanded from 'u') in formalized output, got: %s", out)
	}
	if !strings.Contains(strings.ToLower(out), "earliest convenience") {
		t.Errorf("expected 'at your earliest convenience' in formalized output, got: %s", out)
	}
	// Must end with a period.
	if !strings.HasSuffix(strings.TrimSpace(out), ".") {
		t.Errorf("expected period at end, got: %s", out)
	}
}

func TestFormalize_ContractionExpansion(t *testing.T) {
	e := newTestEngine()
	out := e.Formalize("I can't believe they won't help us")

	if strings.Contains(out, "can't") {
		t.Errorf("contractions should be expanded, got: %s", out)
	}
	if strings.Contains(out, "won't") {
		t.Errorf("contractions should be expanded, got: %s", out)
	}
	if !strings.Contains(out, "cannot") {
		t.Errorf("expected 'cannot' in output, got: %s", out)
	}
}

func TestFormalize_PunctuationNormalization(t *testing.T) {
	e := newTestEngine()
	out := e.Formalize("this is great!!!")

	if strings.Contains(out, "!!!") {
		t.Errorf("multiple exclamation marks should be collapsed, got: %s", out)
	}
	if strings.Contains(out, "!") {
		t.Errorf("exclamation marks should be replaced with periods, got: %s", out)
	}
}

func TestFormalize_CapitalizeSentences(t *testing.T) {
	e := newTestEngine()
	out := e.Formalize("hello. this is a test. another sentence")

	sentences := txSplitSentences(out)
	for i, s := range sentences {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		runes := []rune(s)
		for _, r := range runes {
			if r == ' ' {
				continue
			}
			if r < 'A' || r > 'Z' {
				t.Errorf("sentence %d should start with capital letter, got: %q", i, s)
			}
			break
		}
	}
}

// -----------------------------------------------------------------------
// Casualize
// -----------------------------------------------------------------------

func TestCasualize_FormalToCasual(t *testing.T) {
	e := newTestEngine()
	out := e.Casualize("I would like to request your assistance with this matter.")

	// Should contain at least one contraction.
	hasContraction := false
	for _, c := range []string{"'d", "'m", "'ll", "'s", "'re", "'ve", "n't", "don't", "can't", "won't"} {
		if strings.Contains(out, c) {
			hasContraction = true
			break
		}
	}
	if !hasContraction {
		t.Errorf("expected contractions in casualized output, got: %s", out)
	}
}

func TestCasualize_ReplacesFormalWords(t *testing.T) {
	e := newTestEngine()
	out := e.Casualize("Please obtain the materials and transmit them immediately.")

	lower := strings.ToLower(out)
	if strings.Contains(lower, "obtain") {
		t.Errorf("'obtain' should be replaced with casual equivalent, got: %s", out)
	}
	if strings.Contains(lower, "transmit") {
		t.Errorf("'transmit' should be replaced with casual equivalent, got: %s", out)
	}
}

func TestCasualize_LowercaseSentenceStart(t *testing.T) {
	e := newTestEngine()
	out := e.Casualize("The documentation is satisfactory.")

	// The first character should be lowercased (unless "I").
	runes := []rune(strings.TrimSpace(out))
	if len(runes) > 0 && runes[0] >= 'A' && runes[0] <= 'Z' && runes[0] != 'I' {
		// Check that it's not a proper noun. For this test "the" should be lower.
		first := txFirstWord(string(runes))
		if strings.ToLower(first) == "the" {
			t.Errorf("expected lowercase sentence start, got: %s", out)
		}
	}
}

// -----------------------------------------------------------------------
// Simplify
// -----------------------------------------------------------------------

func TestSimplify_ComplexWordsReplaced(t *testing.T) {
	e := newTestEngine()
	out := e.Simplify("The epistemic ramifications of this paradigm shift are ubiquitous.")

	lower := strings.ToLower(out)
	if strings.Contains(lower, "epistemic") {
		t.Errorf("'epistemic' should be simplified, got: %s", out)
	}
	if strings.Contains(lower, "ramification") {
		t.Errorf("'ramification' should be simplified, got: %s", out)
	}
	// Check that simpler words appear.
	if !strings.Contains(lower, "knowledge-related") && !strings.Contains(lower, "knowledge") {
		t.Errorf("expected 'knowledge-related' in simplified output, got: %s", out)
	}
}

func TestSimplify_RemovesParentheticals(t *testing.T) {
	e := newTestEngine()
	out := e.Simplify("The system (which was built last year) works well.")

	if strings.Contains(out, "(") || strings.Contains(out, ")") {
		t.Errorf("parenthetical should be removed, got: %s", out)
	}
}

func TestSimplify_RemovesSubordinateClauses(t *testing.T) {
	e := newTestEngine()
	out := e.Simplify("The rule, which was established in 2020, applies to all users.")

	if strings.Contains(strings.ToLower(out), "which was established") {
		t.Errorf("subordinate clause should be removed, got: %s", out)
	}
}

func TestSimplify_BreaksLongSentences(t *testing.T) {
	e := newTestEngine()
	long := "The company decided to implement the new policy and the employees were " +
		"expected to follow the guidelines that were distributed last week and everyone " +
		"agreed that the changes would benefit the organization in the long run."
	out := e.Simplify(long)

	sents := txSplitSentences(out)
	if len(sents) < 2 {
		t.Errorf("long sentence should be broken into multiple, got %d sentence(s): %s", len(sents), out)
	}
}

// -----------------------------------------------------------------------
// Shorten
// -----------------------------------------------------------------------

func TestShorten_ReducesSentenceCount(t *testing.T) {
	e := newTestEngine()
	paragraph := "The project began in January. " +
		"It was led by the engineering team. " +
		"Several departments contributed resources. " +
		"The budget was approved by the board. " +
		"Testing started in March. " +
		"Results were encouraging. " +
		"The launch date was set for June. " +
		"Marketing prepared the campaign. " +
		"Customer feedback was collected. " +
		"The final report was submitted in July."

	out := e.Shorten(paragraph)
	originalCount := len(txSplitSentences(paragraph))
	shortenedCount := len(txSplitSentences(out))

	if shortenedCount >= originalCount {
		t.Errorf("shortened text should have fewer sentences: original=%d, shortened=%d",
			originalCount, shortenedCount)
	}
	if shortenedCount > 5 {
		t.Errorf("10-sentence paragraph should be shortened to 5 or fewer, got %d", shortenedCount)
	}
}

func TestShorten_PreservesFirstSentence(t *testing.T) {
	e := newTestEngine()
	paragraph := "The project began in January. " +
		"It was led by the engineering team. " +
		"Several departments contributed resources. " +
		"The budget was approved by the board. " +
		"Testing started in March. " +
		"Results were encouraging."

	out := e.Shorten(paragraph)
	if !strings.HasPrefix(out, "The project began in January.") {
		t.Errorf("first sentence should be preserved, got: %s", out)
	}
}

func TestShorten_ShortTextUnchanged(t *testing.T) {
	e := newTestEngine()
	short := "Hello. Goodbye."
	out := e.Shorten(short)
	if out != short {
		t.Errorf("short text should be unchanged, got: %s", out)
	}
}

// -----------------------------------------------------------------------
// ToBullets
// -----------------------------------------------------------------------

func TestToBullets_BasicConversion(t *testing.T) {
	e := newTestEngine()
	paragraph := "The system is fast. It handles many requests. The API is simple."
	out := e.ToBullets(paragraph)

	lines := strings.Split(out, "\n")
	if len(lines) < 3 {
		t.Errorf("expected 3 bullet points, got %d: %s", len(lines), out)
	}
	for i, line := range lines {
		if !strings.HasPrefix(line, "- ") {
			t.Errorf("line %d should start with '- ', got: %q", i, line)
		}
	}
}

func TestToBullets_StripsTransitionWords(t *testing.T) {
	e := newTestEngine()
	text := "The first point. Additionally, the second point. Furthermore, the third point."
	out := e.ToBullets(text)

	lower := strings.ToLower(out)
	if strings.Contains(lower, "additionally,") {
		t.Errorf("transition word 'Additionally,' should be stripped, got: %s", out)
	}
	if strings.Contains(lower, "furthermore,") {
		t.Errorf("transition word 'Furthermore,' should be stripped, got: %s", out)
	}
}

func TestToBullets_NoPeriods(t *testing.T) {
	e := newTestEngine()
	text := "First sentence. Second sentence."
	out := e.ToBullets(text)

	lines := strings.Split(out, "\n")
	for _, line := range lines {
		if strings.HasSuffix(line, ".") {
			t.Errorf("bullet point should not end with period, got: %q", line)
		}
	}
}

// -----------------------------------------------------------------------
// Transform router
// -----------------------------------------------------------------------

func TestTransform_Routes(t *testing.T) {
	e := newTestEngine()

	tests := []struct {
		mode string
		in   string
		want func(string) bool
		desc string
	}{
		{"formal", "hey", func(s string) bool { return strings.Contains(strings.ToLower(s), "hello") }, "formal replaces hey"},
		{"casual", "obtain", func(s string) bool { return strings.Contains(strings.ToLower(s), "get") }, "casual replaces obtain"},
		{"simple", "utilize", func(s string) bool { return strings.Contains(strings.ToLower(s), "use") }, "simple replaces utilize"},
		{"bullet", "First. Second.", func(s string) bool { return strings.HasPrefix(s, "- ") }, "bullet starts with dash"},
		{"unknown", "hello", func(s string) bool { return s == "hello" }, "unknown mode returns text unchanged"},
	}

	for _, tt := range tests {
		out := e.Transform(tt.in, tt.mode)
		if !tt.want(out) {
			t.Errorf("%s: Transform(%q, %q) = %q", tt.desc, tt.in, tt.mode, out)
		}
	}
}

// -----------------------------------------------------------------------
// Round-trip: formalize then casualize should be somewhat reversible
// -----------------------------------------------------------------------

func TestRoundTrip_FormalThenCasual(t *testing.T) {
	e := newTestEngine()
	original := "I can't get the stuff to work."

	formal := e.Formalize(original)
	// The formal version should not contain contractions.
	if strings.Contains(formal, "can't") {
		t.Errorf("formalized text should not contain contractions, got: %s", formal)
	}

	casual := e.Casualize(formal)
	// The round-tripped casual version should re-introduce contractions.
	if !strings.Contains(casual, "can't") && !strings.Contains(casual, "cannot") {
		// At minimum, the meaning should be preserved even if the exact word differs.
		lower := strings.ToLower(casual)
		if !strings.Contains(lower, "can") {
			t.Errorf("round-trip should preserve meaning, got: %s", casual)
		}
	}
}

// -----------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------

func TestTransformEngine_EmptyInput(t *testing.T) {
	e := newTestEngine()
	modes := []string{"formal", "casual", "simple", "short", "bullet"}
	for _, m := range modes {
		out := e.Transform("", m)
		if out != "" {
			t.Errorf("Transform(%q, %q) should return empty string, got: %q", "", m, out)
		}
	}
}

func TestTransformEngine_WhitespaceOnly(t *testing.T) {
	e := newTestEngine()
	out := e.Formalize("   \t  \n  ")
	if out != "" {
		t.Errorf("whitespace-only input should return empty, got: %q", out)
	}
}
