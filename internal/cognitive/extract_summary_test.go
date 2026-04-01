package cognitive

import (
	"strings"
	"testing"
)

const sampleArticle = `Artificial intelligence has transformed the technology industry over the past decade. ` +
	`Companies like Google, Microsoft, and OpenAI have invested billions of dollars in AI research. ` +
	`Machine learning models can now generate text, translate languages, and recognize images with remarkable accuracy. ` +
	`The field traces its origins to a 1956 workshop at Dartmouth College in New Hampshire. ` +
	`John McCarthy, Marvin Minsky, and Claude Shannon were among the founders. ` +
	`Early AI programs could play chess and prove mathematical theorems. ` +
	`Progress stalled during the so-called AI winters of the 1970s and 1980s. ` +
	`The modern resurgence began around 2012 with deep neural networks. ` +
	`Today, AI is used in healthcare for diagnostic imaging and drug discovery. ` +
	`Self-driving cars rely on AI perception and planning systems. ` +
	`Natural language processing powers virtual assistants like Siri and Alexa. ` +
	`Critics warn that AI could displace millions of jobs. ` +
	`Researchers are working on alignment to ensure AI systems remain safe and beneficial.`

func TestExtractSummary_CorrectCount(t *testing.T) {
	summary := ExtractSummary(sampleArticle, 3)
	sentences := splitSummarySentences(summary)
	if len(sentences) != 3 {
		t.Errorf("expected 3 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestExtractSummary_OriginalOrder(t *testing.T) {
	summary := ExtractSummary(sampleArticle, 4)
	sentences := splitSummarySentences(summary)

	prevIdx := -1
	for _, sent := range sentences {
		idx := strings.Index(sampleArticle, sent)
		if idx < 0 {
			t.Fatalf("summary sentence not found in original: %q", sent)
		}
		if idx <= prevIdx {
			t.Errorf("sentences not in original order: %q (idx %d) after previous idx %d", sent, idx, prevIdx)
		}
		prevIdx = idx
	}
}

func TestExtractSummary_FirstSentenceBonus(t *testing.T) {
	sentences := splitSummarySentences(sampleArticle)
	scored := scoreSummarySentences(sampleArticle, sentences)
	if len(scored) < 2 {
		t.Skip("not enough sentences")
	}

	firstScore := scored[0].score

	// The first sentence should rank among the top due to position bonus.
	// Count how many sentences outscore it.
	higherCount := 0
	for _, s := range scored[1:] {
		if s.score > firstScore {
			higherCount++
		}
	}
	// With the +50% bonus the first sentence should typically be selected
	// in a top-5 summary. Allow at most 4 to outscore it (out of 12+).
	if higherCount > 4 {
		t.Errorf("first sentence should rank highly due to position bonus, but %d sentences outscore it (score=%.3f)", higherCount, firstScore)
	}
}

func TestExtractBullets_Format(t *testing.T) {
	bullets := ExtractBullets(sampleArticle, 3)
	lines := strings.Split(bullets, "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 bullet lines, got %d", len(lines))
	}
	for i, line := range lines {
		if !strings.HasPrefix(line, "- ") {
			t.Errorf("line %d should start with '- ', got %q", i, line)
		}
	}
}

func TestExtractOneLiner_SingleSentence(t *testing.T) {
	result := ExtractOneLiner(sampleArticle)
	sentences := splitSummarySentences(result)
	if len(sentences) != 1 {
		t.Errorf("expected exactly 1 sentence, got %d: %q", len(sentences), result)
	}
}

func TestExtractSummary_EmptyInput(t *testing.T) {
	if got := ExtractSummary("", 5); got != "" {
		t.Errorf("expected empty string for empty input, got %q", got)
	}
	if got := ExtractBullets("", 3); got != "" {
		t.Errorf("expected empty string for empty input, got %q", got)
	}
	if got := ExtractOneLiner(""); got != "" {
		t.Errorf("expected empty string for empty input, got %q", got)
	}
}

func TestExtractSummary_SingleSentence(t *testing.T) {
	single := "This is the only sentence."
	got := ExtractSummary(single, 5)
	if got != single {
		t.Errorf("single sentence input should return itself, got %q", got)
	}
}

func TestExtractSummary_AbbreviationHandling(t *testing.T) {
	text := "Dr. Smith went to Washington. Mr. Jones stayed behind. The meeting was productive."
	sentences := splitSummarySentences(text)
	if len(sentences) != 3 {
		t.Errorf("expected 3 sentences (handling abbreviations), got %d: %v", len(sentences), sentences)
	}
}

func TestExtractSummary_HeadingBonus(t *testing.T) {
	text := "# Introduction\nAI is transforming the world at an unprecedented pace. " +
		"Many industries are affected by this transformation. " +
		"Small companies struggle to keep up with rapid changes. " +
		"## History\nThe field began in the 1950s with simple rule-based programs. " +
		"Early pioneers laid the groundwork for modern systems. " +
		"The journey has been long and winding over many decades."

	sentences := splitSummarySentences(text)
	scored := scoreSummarySentences(text, sentences)

	// Sentence after "# Introduction" should get heading bonus.
	for _, s := range scored {
		if strings.Contains(s.text, "transforming the world") {
			// This sentence is first AND after a heading, so it should score very high.
			if s.score < 0.1 {
				t.Errorf("sentence after heading should have bonus, score=%.3f", s.score)
			}
			break
		}
	}
}

func TestExtractSummary_FewerThanMax(t *testing.T) {
	text := "One sentence here. Another sentence there."
	got := ExtractSummary(text, 10)
	if got != "One sentence here. Another sentence there." {
		t.Errorf("when fewer sentences than max, should return all: got %q", got)
	}
}
