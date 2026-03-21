package cognitive

import (
	"strings"
	"testing"
)

func newTestCreativeEngine() *CreativeEngine {
	g := NewCognitiveGraph("")
	// Add some test knowledge
	g.EnsureNode("the ocean", NodeConcept)
	g.EnsureNode("body of water", NodeConcept)
	g.AddEdge("the ocean", "body of water", RelIsA, "test")
	g.EnsureNode("Earth", NodeEntity)
	g.AddEdge("the ocean", "Earth", RelPartOf, "test")

	g.EnsureNode("AI", NodeConcept)
	g.EnsureNode("technology", NodeConcept)
	g.AddEdge("AI", "technology", RelIsA, "test")
	g.EnsureNode("machine learning", NodeConcept)
	g.AddEdge("AI", "machine learning", RelHas, "test")

	g.EnsureNode("Go", NodeEntity)
	g.EnsureNode("programming language", NodeConcept)
	g.AddEdge("Go", "programming language", RelIsA, "test")
	g.EnsureNode("Google", NodeEntity)
	g.AddEdge("Go", "Google", RelCreatedBy, "test")

	sem := NewSemanticEngine()
	causal := NewCausalEngine()
	pat := NewPatternDetector()
	comp := NewComposer(g, sem, causal, pat)

	return NewCreativeEngine(g, comp)
}

func TestCreative_PoemOceanMultiline(t *testing.T) {
	ce := newTestCreativeEngine()
	poem := ce.WritePoem("the ocean", PoemFreeVerse)
	lines := strings.Split(poem, "\n")
	if len(lines) < 4 {
		t.Errorf("free verse poem should have at least 4 lines, got %d:\n%s", len(lines), poem)
	}
	if poem == "" {
		t.Fatal("poem should not be empty")
	}
	t.Logf("Ocean poem:\n%s", poem)
}

func TestCreative_PoemLonelinessEmotional(t *testing.T) {
	ce := newTestCreativeEngine()
	poem := ce.WritePoem("loneliness", PoemFreeVerse)
	lower := strings.ToLower(poem)
	// Should contain "loneliness" somewhere in the text
	if !strings.Contains(lower, "loneliness") {
		t.Errorf("poem about loneliness should mention the topic:\n%s", poem)
	}
	// Should have multiple lines
	lines := strings.Split(poem, "\n")
	if len(lines) < 4 {
		t.Errorf("expected at least 4 lines, got %d", len(lines))
	}
	t.Logf("Loneliness poem:\n%s", poem)
}

func TestCreative_HaikuThreeLines(t *testing.T) {
	ce := newTestCreativeEngine()
	haiku := ce.WritePoem("cherry blossoms", PoemHaiku)
	lines := strings.Split(haiku, "\n")
	if len(lines) != 3 {
		t.Errorf("haiku must have exactly 3 lines, got %d:\n%s", len(lines), haiku)
	}
	t.Logf("Haiku:\n%s", haiku)
}

func TestCreative_StoryBeginningMiddleEnd(t *testing.T) {
	ce := newTestCreativeEngine()
	story := ce.WriteStory("courage", 3)
	paragraphs := strings.Split(story, "\n\n")
	if len(paragraphs) < 3 {
		t.Errorf("story should have at least 3 paragraphs (setup/conflict/resolution), got %d:\n%s", len(paragraphs), story)
	}
	// First paragraph should have character setup indicators
	if len(paragraphs[0]) < 50 {
		t.Errorf("setup paragraph too short: %s", paragraphs[0])
	}
	t.Logf("Story:\n%s", story)
}

func TestCreative_StoryKnownTopicIncludesFacts(t *testing.T) {
	ce := newTestCreativeEngine()
	story := ce.WriteStory("Go", 3)
	lower := strings.ToLower(story)
	// Should include knowledge graph facts about Go
	hasFact := strings.Contains(lower, "programming language") || strings.Contains(lower, "google")
	if !hasFact {
		t.Errorf("story about known topic 'Go' should include facts from knowledge graph:\n%s", story)
	}
	t.Logf("Story about Go:\n%s", story)
}

func TestCreative_JokeQuestionAnswer(t *testing.T) {
	ce := newTestCreativeEngine()
	joke := ce.TellJoke("computers")
	if joke == "" {
		t.Fatal("joke should not be empty")
	}
	// Jokes should have some structure — question mark or observational setup
	hasStructure := strings.Contains(joke, "?") || strings.Contains(joke, "noticed") ||
		strings.Contains(joke, "joke") || strings.Contains(joke, "funny") ||
		strings.Contains(joke, "told") || strings.Contains(joke, "friend") ||
		strings.Contains(joke, "call") || strings.Contains(joke, "relationship")
	if !hasStructure {
		t.Errorf("joke should have recognizable structure (question, observation, etc.):\n%s", joke)
	}
	t.Logf("Joke:\n%s", joke)
}

func TestCreative_ReflectionAIMultiParagraph(t *testing.T) {
	ce := newTestCreativeEngine()
	reflection := ce.Reflect("AI", "what do you think about AI?")
	paragraphs := strings.Split(reflection, "\n\n")
	if len(paragraphs) < 3 {
		t.Errorf("reflection should have multiple paragraphs, got %d:\n%s", len(paragraphs), reflection)
	}
	lower := strings.ToLower(reflection)
	if !strings.Contains(lower, "ai") {
		t.Errorf("reflection about AI should mention AI:\n%s", reflection)
	}
	t.Logf("AI reflection:\n%s", reflection)
}

func TestCreative_ReflectionKnownTopicWeavesFacts(t *testing.T) {
	ce := newTestCreativeEngine()
	reflection := ce.Reflect("AI", "")
	lower := strings.ToLower(reflection)
	// Should include knowledge about AI from the graph
	hasFact := strings.Contains(lower, "technology") || strings.Contains(lower, "machine learning")
	if !hasFact {
		t.Errorf("reflection on known topic 'AI' should weave in graph knowledge:\n%s", reflection)
	}
	t.Logf("AI reflection with facts:\n%s", reflection)
}

func TestCreative_EmptyTopicHandling(t *testing.T) {
	ce := newTestCreativeEngine()
	// Empty topic should not panic, should pick a random topic
	poem := ce.Generate(CreativeRequest{Type: CreativePoem, Topic: ""})
	if poem == "" {
		t.Fatal("empty topic should still generate output")
	}
	story := ce.Generate(CreativeRequest{Type: CreativeStory, Topic: ""})
	if story == "" {
		t.Fatal("empty topic story should still generate output")
	}
	joke := ce.Generate(CreativeRequest{Type: CreativeJoke, Topic: ""})
	if joke == "" {
		t.Fatal("empty topic joke should still generate output")
	}
	t.Logf("Empty topic poem:\n%s", poem)
}

func TestCreative_DifferentPoemFormsDiffer(t *testing.T) {
	ce := newTestCreativeEngine()
	freeVerse := ce.WritePoem("stars", PoemFreeVerse)
	haiku := ce.WritePoem("stars", PoemHaiku)
	quatrain := ce.WritePoem("stars", PoemQuatrain)

	fvLines := len(strings.Split(freeVerse, "\n"))
	hkLines := len(strings.Split(haiku, "\n"))
	qtLines := len(strings.Split(quatrain, "\n"))

	// Haiku = 3 lines, quatrain = 4 lines, free verse = 5-9 lines
	if hkLines != 3 {
		t.Errorf("haiku should have 3 lines, got %d", hkLines)
	}
	if qtLines != 4 {
		t.Errorf("quatrain should have 4 lines, got %d", qtLines)
	}
	if fvLines < 5 {
		t.Errorf("free verse should have at least 5 lines, got %d", fvLines)
	}
	// All three should be different
	if freeVerse == haiku || haiku == quatrain || freeVerse == quatrain {
		t.Error("different poem forms should produce different outputs")
	}
	t.Logf("Free verse (%d lines):\n%s\n\nHaiku (%d lines):\n%s\n\nQuatrain (%d lines):\n%s",
		fvLines, freeVerse, hkLines, haiku, qtLines, quatrain)
}

func TestCreative_StoryLengthParameter(t *testing.T) {
	ce := newTestCreativeEngine()
	short := ce.WriteStory("adventure", 3)
	long := ce.WriteStory("adventure", 5)

	shortParas := len(strings.Split(short, "\n\n"))
	longParas := len(strings.Split(long, "\n\n"))

	if shortParas > longParas {
		t.Errorf("longer story should have more paragraphs: short=%d, long=%d", shortParas, longParas)
	}
	if longParas < 4 {
		t.Errorf("5-paragraph story should have at least 4 paragraphs, got %d", longParas)
	}
	t.Logf("Short story paragraphs: %d, Long story paragraphs: %d", shortParas, longParas)
}

func TestCreative_GenerateRouting(t *testing.T) {
	ce := newTestCreativeEngine()

	// Poem request
	poem := ce.Generate(CreativeRequest{Type: CreativePoem, Topic: "night", PoemForm: PoemHaiku})
	lines := strings.Split(poem, "\n")
	if len(lines) != 3 {
		t.Errorf("Generate with PoemHaiku should produce 3 lines, got %d", len(lines))
	}

	// Story request
	story := ce.Generate(CreativeRequest{Type: CreativeStory, Topic: "time", Length: 3})
	paragraphs := strings.Split(story, "\n\n")
	if len(paragraphs) < 3 {
		t.Errorf("Generate with story type should produce paragraphs, got %d", len(paragraphs))
	}

	// Joke request
	joke := ce.Generate(CreativeRequest{Type: CreativeJoke, Topic: "math"})
	if joke == "" {
		t.Error("Generate with joke type should produce output")
	}

	// Reflect request
	ref := ce.Generate(CreativeRequest{Type: CreativeReflect, Topic: "AI"})
	if len(strings.Split(ref, "\n\n")) < 3 {
		t.Error("Generate with reflect type should produce multi-paragraph output")
	}

	t.Log("All routing tests passed")
}

func TestCreative_QuatrainHasFourLines(t *testing.T) {
	ce := newTestCreativeEngine()
	for i := 0; i < 5; i++ {
		q := ce.WritePoem("love", PoemQuatrain)
		lines := strings.Split(q, "\n")
		if len(lines) != 4 {
			t.Errorf("quatrain iteration %d: expected 4 lines, got %d:\n%s", i, len(lines), q)
		}
	}
}

func TestCreative_TopicalJokeUsesKnowledge(t *testing.T) {
	ce := newTestCreativeEngine()
	// "Go" is in the knowledge graph
	joke := ce.TellJoke("Go")
	lower := strings.ToLower(joke)
	// Should mention the topic
	if !strings.Contains(lower, "go") {
		t.Errorf("topical joke should mention the topic 'Go':\n%s", joke)
	}
	t.Logf("Topical joke about Go:\n%s", joke)
}
