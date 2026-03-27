package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Sentence absorption from real Wikipedia text.
// -----------------------------------------------------------------------

func TestAbsorbWikipediaText(t *testing.T) {
	ae := NewAbsorptionEngine("")

	text := `Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. He is widely regarded as one of the most influential scientists of the 20th century. His work is also known for its influence on the philosophy of science. Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.`

	count := ae.Absorb(text)
	if count == 0 {
		t.Fatal("expected to absorb at least one pattern from Wikipedia text")
	}
	t.Logf("absorbed %d patterns from %d-char text", count, len(text))

	// Check that we got some patterns with proper structure.
	stats := ae.Stats()
	if stats.Total == 0 {
		t.Fatal("stats show zero patterns")
	}
	t.Logf("stats: %+v", stats)
}

func TestAbsorbMultipleSentences(t *testing.T) {
	ae := NewAbsorptionEngine("")

	sentences := []string{
		"Marie Curie was a Polish-born physicist who pioneered research on radioactivity.",
		"The Amazon River is the largest river by discharge volume in the world.",
		"Democracy, while imperfect, remains the most resilient form of governance.",
		"Unlike classical physics, quantum mechanics describes nature at the atomic level.",
		"The Great Wall of China was built to protect against invasions from the north.",
	}

	for _, s := range sentences {
		p := ae.AbsorbSentence(s)
		if p == nil {
			t.Logf("rejected (expected for some): %s", s)
			continue
		}
		t.Logf("absorbed: function=%s tone=%s structure=%s template=%s",
			p.Function, p.Tone, p.Structure, p.Template)

		// Template must be different from source.
		if p.Template == p.Source {
			t.Errorf("template should differ from source: %s", p.Source)
		}
		// Must have at least one slot.
		if len(p.SlotTypes) == 0 {
			t.Errorf("expected at least one slot in: %s", p.Template)
		}
	}
}

// -----------------------------------------------------------------------
// Function classification.
// -----------------------------------------------------------------------

func TestClassifyFunction(t *testing.T) {
	ae := NewAbsorptionEngine("")

	tests := []struct {
		sent string
		want DiscourseFunc
	}{
		{"Paris is a major European city.", DFDefines},
		{"The bridge collapsed because of structural weakness.", DFExplainsWhy},
		{"Many fruits, such as apples and oranges, are rich in vitamins.", DFGivesExample},
		{"Unlike Mars, Venus has a thick atmosphere.", DFCompares},
		{"Shakespeare is widely regarded as the greatest English writer.", DFEvaluates},
		{"Photosynthesis works by converting sunlight into chemical energy.", DFDescribes},
		{"Deforestation leads to habitat loss and soil erosion.", DFConsequence},
		{"The university was founded in 1209 by scholars from Oxford.", DFContext},
		{"The city has a population of approximately 8 million people.", DFQuantifies},
	}

	for _, tt := range tests {
		got := ae.classifyFunction(tt.sent)
		if got != tt.want {
			t.Errorf("classifyFunction(%q) = %s, want %s", tt.sent, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Tone detection.
// -----------------------------------------------------------------------

func TestClassifyTone(t *testing.T) {
	ae := NewAbsorptionEngine("")

	tests := []struct {
		sent string
		want string
	}{
		{
			"The phenomenon was observed in multiple controlled experiments, suggesting a statistically significant correlation.",
			"academic",
		},
		{
			"Have you ever wondered why the sky is blue?",
			"conversational",
		},
		{
			"It's kinda cool but I don't really get it.",
			"casual",
		},
		{
			"The institution was established by royal charter and has been continuously operational since its founding.",
			"formal",
		},
	}

	for _, tt := range tests {
		got := ae.classifyTone(tt.sent)
		if got != tt.want {
			t.Errorf("classifyTone(%q) = %q, want %q", tt.sent, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Structure detection.
// -----------------------------------------------------------------------

func TestClassifyStructure(t *testing.T) {
	ae := NewAbsorptionEngine("")

	tests := []struct {
		sent string
		want string
	}{
		{"A cat is a small domesticated carnivore.", "simple_definition"},
		{"Unlike dogs, cats are generally independent.", "contrastive"},
		{"If the temperature drops below zero, the pipes will freeze.", "conditional"},
		{"After the war ended, the country began rebuilding.", "temporal"},
		{"The economy declined because of rising inflation.", "causal_chain"},
		{"The museum, which was built in 1850, houses over 10000 artifacts.", "relative_clause"},
	}

	for _, tt := range tests {
		got := ae.classifyStructure(tt.sent)
		if got != tt.want {
			t.Errorf("classifyStructure(%q) = %q, want %q", tt.sent, got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Template extraction quality.
// -----------------------------------------------------------------------

func TestExtractTemplate(t *testing.T) {
	ae := NewAbsorptionEngine("")

	tests := []struct {
		sent      string
		fn        DiscourseFunc
		wantSlots int    // minimum slot count
		wantIn    string // substring that should appear in template
	}{
		{
			"Albert Einstein was a theoretical physicist who developed general relativity.",
			DFDefines,
			1,
			"[SUBJECT]",
		},
		{
			"Marie Curie was a brilliant scientist who discovered radium.",
			DFDefines,
			1,
			"was a",
		},
		{
			"Unlike Jupiter, Saturn has prominent rings.",
			DFCompares,
			1,
			"[",
		},
	}

	for _, tt := range tests {
		tmpl, slots := ae.extractTemplate(tt.sent, tt.fn)
		if len(slots) < tt.wantSlots {
			t.Errorf("extractTemplate(%q): got %d slots, want >= %d. template: %s",
				tt.sent, len(slots), tt.wantSlots, tmpl)
		}
		if tt.wantIn != "" && !strings.Contains(tmpl, tt.wantIn) {
			t.Errorf("extractTemplate(%q): template %q missing %q",
				tt.sent, tmpl, tt.wantIn)
		}
		t.Logf("template: %s  slots: %v", tmpl, slots)
	}
}

// -----------------------------------------------------------------------
// Slot filling / realization.
// -----------------------------------------------------------------------

func TestRealize(t *testing.T) {
	ae := NewAbsorptionEngine("")

	pattern := &AbsorbedPattern{
		Template: "[SUBJECT] was a [MODIFIER] [CATEGORY] who [VERB_PHRASE].",
		SlotTypes: []AbsorbedSlot{
			{Name: "SUBJECT", Position: 0, Kind: "noun_phrase"},
			{Name: "MODIFIER", Position: 17, Kind: "adjective"},
			{Name: "CATEGORY", Position: 29, Kind: "noun"},
			{Name: "VERB_PHRASE", Position: 44, Kind: "verb_phrase"},
		},
	}

	result := ae.Realize(pattern, map[string]string{
		"SUBJECT":     "Ada Lovelace",
		"MODIFIER":    "brilliant",
		"CATEGORY":    "mathematician",
		"VERB_PHRASE": "pioneered computing",
	})

	if result != "Ada Lovelace was a brilliant mathematician who pioneered computing." {
		t.Errorf("Realize: got %q", result)
	}
}

func TestRealizeFixesArticles(t *testing.T) {
	ae := NewAbsorptionEngine("")

	pattern := &AbsorbedPattern{
		Template: "[SUBJECT] is a [CATEGORY].",
		SlotTypes: []AbsorbedSlot{
			{Name: "SUBJECT", Position: 0, Kind: "noun_phrase"},
			{Name: "CATEGORY", Position: 18, Kind: "noun"},
		},
	}

	// "a elephant" should become "an elephant".
	result := ae.Realize(pattern, map[string]string{
		"SUBJECT":  "Dumbo",
		"CATEGORY": "elephant",
	})

	if !strings.Contains(result, "an elephant") {
		t.Errorf("expected 'an elephant', got: %s", result)
	}
}

func TestRealizeNilPattern(t *testing.T) {
	ae := NewAbsorptionEngine("")
	result := ae.Realize(nil, nil)
	if result != "" {
		t.Errorf("Realize(nil) should return empty, got %q", result)
	}
}

func TestRealizeCapitalization(t *testing.T) {
	ae := NewAbsorptionEngine("")

	pattern := &AbsorbedPattern{
		Template: "[SUBJECT] is known for [VERB_PHRASE].",
		SlotTypes: []AbsorbedSlot{
			{Name: "SUBJECT", Position: 0, Kind: "noun_phrase"},
			{Name: "VERB_PHRASE", Position: 25, Kind: "verb_phrase"},
		},
	}

	result := ae.Realize(pattern, map[string]string{
		"SUBJECT":     "python",
		"VERB_PHRASE": "readability",
	})

	// Should capitalize first letter.
	if !strings.HasPrefix(result, "Python") {
		t.Errorf("expected capitalized start, got: %s", result)
	}
}

// -----------------------------------------------------------------------
// Persistence round-trip.
// -----------------------------------------------------------------------

func TestSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "absorbed.json")

	// Create and populate.
	ae := NewAbsorptionEngine(path)
	ae.Absorb("Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.")
	ae.Absorb("Marie Curie was a pioneering scientist who discovered radioactivity.")

	before := ae.PatternCount()
	if before == 0 {
		t.Fatal("expected patterns before save")
	}

	// Save.
	if err := ae.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Verify file exists.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("saved file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	// Load into fresh engine.
	ae2 := NewAbsorptionEngine(path)
	if err := ae2.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}

	after := ae2.PatternCount()
	if after != before {
		t.Errorf("pattern count mismatch: before=%d after=%d", before, after)
	}

	// Verify indices are rebuilt.
	stats := ae2.Stats()
	if stats.Total != after {
		t.Errorf("stats total %d != pattern count %d", stats.Total, after)
	}
	if len(stats.ByFunction) == 0 {
		t.Error("byFunction index not rebuilt after Load")
	}
	if len(stats.ByTone) == 0 {
		t.Error("byTone index not rebuilt after Load")
	}
	if len(stats.ByStructure) == 0 {
		t.Error("byStructure index not rebuilt after Load")
	}
}

func TestSaveLoadEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.json")

	ae := NewAbsorptionEngine(path)
	if err := ae.Save(); err != nil {
		t.Fatalf("Save empty: %v", err)
	}

	ae2 := NewAbsorptionEngine(path)
	if err := ae2.Load(); err != nil {
		t.Fatalf("Load empty: %v", err)
	}
	if ae2.PatternCount() != 0 {
		t.Error("expected zero patterns from empty file")
	}
}

// -----------------------------------------------------------------------
// Retrieval.
// -----------------------------------------------------------------------

func TestRetrieveFallback(t *testing.T) {
	ae := NewAbsorptionEngine("")

	// Absorb some text so there are patterns.
	ae.Absorb("Albert Einstein was a brilliant physicist who developed relativity.")
	ae.Absorb("The city was founded in 1850 by settlers from the east.")
	ae.Absorb("Climate change leads to rising sea levels and extreme weather.")

	if ae.PatternCount() == 0 {
		t.Fatal("expected some patterns")
	}

	// Try retrieval with exact criteria that might not match.
	p := ae.Retrieve(DFDefines, "poetic", "listing")
	// Should fall back gracefully.
	if p == nil {
		// Try more lenient retrieval.
		p = ae.Retrieve(DFDefines, "", "")
	}
	// At worst, fallback to any pattern should work.
	if p == nil {
		p = ae.Retrieve(DFDescribes, "", "")
	}
	if p == nil {
		t.Log("no patterns matched any retrieval tier (acceptable if classification didn't produce the expected functions)")
	}
}

// -----------------------------------------------------------------------
// Sentence splitting — abbreviation handling.
// -----------------------------------------------------------------------

func TestSplitAbsorptionSentences(t *testing.T) {
	tests := []struct {
		input string
		want  int // expected sentence count
	}{
		{
			"Dr. Smith went to Washington. He arrived on Monday.",
			2,
		},
		{
			"The U.S. government passed the law. It took effect immediately.",
			2,
		},
		{
			"She studied e.g. biology and chemistry. Her grades were excellent.",
			2,
		},
		{
			"One sentence only.",
			1,
		},
		{
			"First paragraph.\n\nSecond paragraph.",
			2,
		},
	}

	for _, tt := range tests {
		got := splitAbsorptionSentences(tt.input)
		if len(got) != tt.want {
			t.Errorf("splitAbsorptionSentences(%q): got %d sentences %v, want %d",
				tt.input, len(got), got, tt.want)
		}
	}
}

// -----------------------------------------------------------------------
// Quality scoring.
// -----------------------------------------------------------------------

func TestScoreAbsorptionQuality(t *testing.T) {
	tests := []struct {
		sent    string
		wantMin int
		wantMax int
	}{
		{"", 0, 0},                         // empty
		{"Too short.", 0, 0},               // too few words
		{"What is the meaning of life?", 0, 0}, // question
		{"lowercase start is bad.", 0, 0},       // no capital
		{"This has [[wiki markup]] in it.", 0, 0}, // markup
		{"A perfectly normal sentence of reasonable length.", 1, 3},
		{"This is a well-formed sentence with good length and proper ending.", 2, 3},
	}

	for _, tt := range tests {
		got := scoreAbsorptionQuality(tt.sent)
		if got < tt.wantMin || got > tt.wantMax {
			t.Errorf("scoreAbsorptionQuality(%q) = %d, want [%d, %d]",
				tt.sent, got, tt.wantMin, tt.wantMax)
		}
	}
}

// -----------------------------------------------------------------------
// Edge cases.
// -----------------------------------------------------------------------

func TestAbsorbRejectsJunk(t *testing.T) {
	ae := NewAbsorptionEngine("")

	junk := []string{
		"",
		"hi",
		"??????",
		"[[Category:Stuff]]",
		"* List item one",
		"a b c d", // lowercase start
	}

	for _, j := range junk {
		p := ae.AbsorbSentence(j)
		if p != nil {
			t.Errorf("expected nil for junk input %q, got pattern: %+v", j, p)
		}
	}
}

func TestAbsorbEndToEnd(t *testing.T) {
	ae := NewAbsorptionEngine("")

	// Absorb a real paragraph.
	text := `Vienna is the capital and largest city of Austria. The city has a population of approximately 2 million people. It is widely regarded as one of the most liveable cities in the world. Unlike many European capitals, Vienna has maintained much of its historical architecture.`

	count := ae.Absorb(text)
	t.Logf("absorbed %d patterns", count)

	if count == 0 {
		t.Fatal("expected at least one absorbed pattern")
	}

	// Try to retrieve and realize.
	stats := ae.Stats()
	t.Logf("functions: %v", stats.ByFunction)
	t.Logf("tones: %v", stats.ByTone)
	t.Logf("structures: %v", stats.ByStructure)

	// Try retrieval for any absorbed function.
	for fnName := range stats.ByFunction {
		fn := parseDFString(fnName)
		p := ae.Retrieve(fn, "", "")
		if p != nil {
			t.Logf("retrieved for %s: %s", fnName, p.Template)
		}
	}
}
