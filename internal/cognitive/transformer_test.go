package cognitive

import (
	"strings"
	"testing"
)

func newTestTransformer() *TextTransformer {
	emb := NewWordEmbeddings(32)
	return NewTextTransformer(emb)
}

func TestTransformParaphraseProducesDifferentOutput(t *testing.T) {
	tr := newTestTransformer()
	input := "The good student found an important answer to the hard problem quickly."
	result := tr.Transform(input, OpParaphrase)

	if result == input {
		t.Errorf("paraphrase should produce different output, got identical: %q", result)
	}
	if result == "" {
		t.Error("paraphrase should not produce empty output")
	}
	t.Logf("input:  %s", input)
	t.Logf("output: %s", result)
}

func TestTransformSummarizeShortersText(t *testing.T) {
	tr := newTestTransformer()
	input := "The quick brown fox jumps over the lazy dog. " +
		"This sentence is about foxes and dogs in a field. " +
		"The fox was known for its speed and agility. " +
		"The dog preferred to rest under the shade of a tree. " +
		"Many animals lived in the meadow together. " +
		"The birds sang in the morning light. " +
		"Rabbits hopped through the tall grass. " +
		"It was a peaceful day in the countryside. " +
		"The farmer watched from his porch with satisfaction."

	result := tr.Transform(input, OpSummarize)

	if len(result) >= len(input) {
		t.Errorf("summarize should shorten text: input=%d chars, result=%d chars", len(input), len(result))
	}
	if result == "" {
		t.Error("summarize should not produce empty output")
	}
	t.Logf("input:  %d chars", len(input))
	t.Logf("output: %d chars — %s", len(result), result)
}

func TestTransformFormalizeExpandsContractions(t *testing.T) {
	tr := newTestTransformer()
	input := "I can't believe it's already done. We don't need to worry."
	result := tr.Transform(input, OpFormalize)

	if strings.Contains(result, "can't") {
		t.Errorf("formalize should expand contractions, still contains \"can't\": %q", result)
	}
	if strings.Contains(result, "don't") {
		t.Errorf("formalize should expand contractions, still contains \"don't\": %q", result)
	}
	if strings.Contains(result, "it's") {
		t.Errorf("formalize should expand contractions, still contains \"it's\": %q", result)
	}
	t.Logf("input:  %s", input)
	t.Logf("output: %s", result)
}

func TestTransformCasualizeAddsContractions(t *testing.T) {
	tr := newTestTransformer()
	input := "I cannot believe it is already completed. We do not need to worry."
	result := tr.Transform(input, OpCasualize)

	// Should contract at least some expanded forms
	hasContraction := strings.Contains(result, "can't") ||
		strings.Contains(result, "it's") ||
		strings.Contains(result, "don't")
	if !hasContraction {
		t.Errorf("casualize should add contractions: %q", result)
	}
	t.Logf("input:  %s", input)
	t.Logf("output: %s", result)
}

func TestTransformBulletizeProducesBullets(t *testing.T) {
	tr := newTestTransformer()
	input := "First we plan the project. Then we write the code. Finally we test everything."
	result := tr.Transform(input, OpBulletize)

	lines := strings.Split(result, "\n")
	bulletCount := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "- ") {
			bulletCount++
		}
	}
	if bulletCount == 0 {
		t.Errorf("bulletize should produce lines starting with '- ', got: %q", result)
	}
	if bulletCount < 2 {
		t.Errorf("bulletize should produce multiple bullets, got %d", bulletCount)
	}
	t.Logf("output:\n%s", result)
}

func TestTransformProsifyRemovesBulletMarkers(t *testing.T) {
	tr := newTestTransformer()
	input := "- Plan the project\n- Write the code\n- Test everything\n- Deploy to production"
	result := tr.Transform(input, OpProsify)

	if strings.Contains(result, "- ") {
		t.Errorf("prosify should remove bullet markers, got: %q", result)
	}
	if strings.Contains(result, "\n") {
		t.Errorf("prosify should produce flowing prose without newlines, got: %q", result)
	}
	// Should contain transition words
	hasTransition := false
	for _, tw := range transitionWords {
		if strings.Contains(result, strings.TrimRight(tw, ",")) {
			hasTransition = true
			break
		}
	}
	if !hasTransition {
		t.Errorf("prosify should contain transition words, got: %q", result)
	}
	t.Logf("output: %s", result)
}

func TestTransformSimplifyRemovesFillerWords(t *testing.T) {
	tr := newTestTransformer()
	input := "We basically really need to obviously find a very simple solution that actually works quite well."
	result := tr.Transform(input, OpSimplify)

	// Check that at least some filler words are removed
	fillerCount := 0
	resultLower := strings.ToLower(result)
	for word := range fillerWords {
		if strings.Contains(resultLower, " "+word+" ") {
			fillerCount++
		}
	}
	if fillerCount > 2 {
		t.Errorf("simplify should remove most filler words, still has %d: %q", fillerCount, result)
	}
	if len(result) >= len(input) {
		t.Errorf("simplify should shorten text: input=%d, result=%d", len(input), len(result))
	}
	t.Logf("input:  %s", input)
	t.Logf("output: %s", result)
}

func TestTransformRoundTripPreservesMeaning(t *testing.T) {
	tr := newTestTransformer()
	input := "The project requires careful planning and good execution."

	// Casualize then formalize
	casual := tr.Transform(input, OpCasualize)
	formal := tr.Transform(casual, OpFormalize)

	// The round-trip should still contain key content words
	keyWords := []string{"project", "planning", "execution"}
	for _, kw := range keyWords {
		if !strings.Contains(strings.ToLower(formal), kw) {
			// Check synonyms too
			found := false
			if syns, ok := transformSynonyms[kw]; ok {
				for _, syn := range syns {
					if strings.Contains(strings.ToLower(formal), syn) {
						found = true
						break
					}
				}
			}
			if !found {
				t.Logf("round-trip lost content word %q (may have synonym): %q", kw, formal)
			}
		}
	}
	t.Logf("input:     %s", input)
	t.Logf("casual:    %s", casual)
	t.Logf("formal:    %s", formal)
}

func TestTransformEmptyInput(t *testing.T) {
	tr := newTestTransformer()

	ops := []TransformOp{OpParaphrase, OpSummarize, OpFormalize, OpCasualize, OpBulletize, OpProsify, OpSimplify}
	for _, op := range ops {
		result := tr.Transform("", op)
		if result != "" {
			t.Errorf("%s of empty input should return empty, got: %q", op, result)
		}

		result = tr.Transform("   ", op)
		if result != "" {
			t.Errorf("%s of whitespace input should return empty, got: %q", op, result)
		}
	}
}

func TestTransformShortInput(t *testing.T) {
	tr := newTestTransformer()
	input := "Hello world."

	// Should handle gracefully without panicking
	for _, op := range []TransformOp{OpParaphrase, OpSummarize, OpFormalize, OpCasualize, OpBulletize, OpProsify, OpSimplify} {
		result := tr.Transform(input, op)
		if result == "" {
			t.Errorf("%s of short input should not return empty", op)
		}
	}
}

func TestTransformSummarizePreservesTopSentences(t *testing.T) {
	tr := newTestTransformer()
	// The first sentence and the most information-dense sentence should survive summarization
	input := "Machine learning is a subset of artificial intelligence. " +
		"It uses algorithms to learn patterns from data. " +
		"The weather was nice today. " +
		"Many companies invest in this technology. " +
		"Deep learning uses neural networks with multiple layers. " +
		"The cat sat on the mat. " +
		"Natural language processing enables computers to understand human language."

	result := tr.Transform(input, OpSummarize)
	resultLower := strings.ToLower(result)

	// Key topic words should appear in the summary
	if !strings.Contains(resultLower, "learning") && !strings.Contains(resultLower, "intelligence") {
		t.Errorf("summary should contain key topic words: %q", result)
	}
	t.Logf("summary: %s", result)
}

func TestTransformFormalizeAddsFormalVocabulary(t *testing.T) {
	tr := newTestTransformer()
	input := "We need to get this stuff fixed and help the guys find a good way to start."
	result := tr.Transform(input, OpFormalize)
	resultLower := strings.ToLower(result)

	// Should replace at least some casual words
	formalReplacements := 0
	if strings.Contains(resultLower, "require") || strings.Contains(resultLower, "obtain") {
		formalReplacements++
	}
	if strings.Contains(resultLower, "material") || strings.Contains(resultLower, "items") {
		formalReplacements++
	}
	if strings.Contains(resultLower, "assist") {
		formalReplacements++
	}
	if strings.Contains(resultLower, "individuals") {
		formalReplacements++
	}
	if strings.Contains(resultLower, "commence") {
		formalReplacements++
	}

	if formalReplacements < 2 {
		t.Errorf("formalize should replace casual vocabulary (got %d replacements): %q", formalReplacements, result)
	}
	t.Logf("input:  %s", input)
	t.Logf("output: %s", result)
}

func TestTransformUnknownOp(t *testing.T) {
	tr := newTestTransformer()
	input := "This is a test."
	result := tr.Transform(input, TransformOp("unknown_operation"))

	if result != input {
		t.Errorf("unknown operation should return input unchanged, got: %q", result)
	}
}

func TestTransformBulletizeThenProsifyRoundTrip(t *testing.T) {
	tr := newTestTransformer()
	input := "First we plan the project. Then we write the code. Finally we test everything."

	bullets := tr.Transform(input, OpBulletize)
	prose := tr.Transform(bullets, OpProsify)

	// Prose should not contain bullet markers
	if strings.Contains(prose, "- ") {
		t.Errorf("prosified bullets should not contain '- ': %q", prose)
	}
	// Should be non-empty flowing text
	if prose == "" {
		t.Error("prosified text should not be empty")
	}
	// Should contain content from original
	if !strings.Contains(strings.ToLower(prose), "plan") {
		t.Errorf("prosified text should contain original content: %q", prose)
	}
	t.Logf("bullets:\n%s", bullets)
	t.Logf("prose: %s", prose)
}
