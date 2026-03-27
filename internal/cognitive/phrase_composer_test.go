package cognitive

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------
// Template Decomposition Tests
// -----------------------------------------------------------------------

func TestDecomposeDefinition(t *testing.T) {
	tmpl := DecomposeToTemplate(
		"Marie Curie was a brilliant physicist.",
		"Marie Curie", RelIsA, "physicist",
	)
	if tmpl == nil {
		t.Fatal("expected a template, got nil")
	}
	if !strings.Contains(tmpl.Pattern, "[SUBJECT]") {
		t.Errorf("pattern should contain [SUBJECT], got: %s", tmpl.Pattern)
	}
	if !strings.Contains(tmpl.Pattern, "[OBJECT]") {
		t.Errorf("pattern should contain [OBJECT], got: %s", tmpl.Pattern)
	}
	if tmpl.Source != "Marie Curie was a brilliant physicist." {
		t.Errorf("source mismatch: %s", tmpl.Source)
	}
	if tmpl.Relation != RelIsA {
		t.Errorf("relation should be is_a, got: %s", tmpl.Relation)
	}
	// Should extract "brilliant" as a modifier slot.
	if !strings.Contains(tmpl.Pattern, "[MODIFIER]") {
		t.Errorf("expected MODIFIER slot for evaluative adjective, got: %s", tmpl.Pattern)
	}
	t.Logf("Pattern: %s", tmpl.Pattern)
}

func TestDecomposeLocation(t *testing.T) {
	tmpl := DecomposeToTemplate(
		"The Eiffel Tower is located in Paris.",
		"The Eiffel Tower", RelLocatedIn, "Paris",
	)
	if tmpl == nil {
		t.Fatal("expected a template, got nil")
	}
	if tmpl.Function != DFContext {
		t.Errorf("expected DFContext function, got: %s", tmpl.Function)
	}
	// Check slot kinds.
	for _, slot := range tmpl.Slots {
		if slot.Name == "OBJECT" && slot.Kind != PhraseSlotLocation {
			t.Errorf("object slot for located_in should be PhraseSlotLocation, got: %s", slot.Kind)
		}
	}
	t.Logf("Pattern: %s", tmpl.Pattern)
}

func TestDecomposeYear(t *testing.T) {
	tmpl := DecomposeToTemplate(
		"Wikipedia was founded in 2001.",
		"Wikipedia", RelFoundedIn, "2001",
	)
	if tmpl == nil {
		t.Fatal("expected a template, got nil")
	}
	for _, slot := range tmpl.Slots {
		if slot.Name == "OBJECT" && slot.Kind != PhraseSlotYear {
			t.Errorf("object slot for year should be PhraseSlotYear, got: %s", slot.Kind)
		}
	}
	t.Logf("Pattern: %s", tmpl.Pattern)
}

func TestDecomposeRejectsShort(t *testing.T) {
	tmpl := DecomposeToTemplate("X is Y.", "X", RelIsA, "Y")
	if tmpl != nil {
		t.Error("should reject very short sentences")
	}
}

func TestDecomposeRejectsMissing(t *testing.T) {
	tmpl := DecomposeToTemplate(
		"Albert Einstein was a physicist.",
		"Marie Curie", RelIsA, "physicist",
	)
	if tmpl != nil {
		t.Error("should reject when subject is not in sentence")
	}
}

// -----------------------------------------------------------------------
// Composition Tests
// -----------------------------------------------------------------------

func TestComposeFromTemplate(t *testing.T) {
	pc := NewPhraseComposer()

	// Seed with a few templates from real-sounding sentences.
	sentences := []struct {
		sent string
		subj string
		rel  RelType
		obj  string
	}{
		{"Marie Curie was a pioneering physicist.", "Marie Curie", RelIsA, "physicist"},
		{"London is located in England.", "London", RelLocatedIn, "England"},
		{"Wikipedia was founded in 2001.", "Wikipedia", RelFoundedIn, "2001"},
		{"Python is a popular programming language.", "Python", RelIsA, "programming language"},
		{"The Louvre is located in Paris.", "The Louvre", RelLocatedIn, "Paris"},
		{"Tesla was founded in 2003.", "Tesla", RelFoundedIn, "2003"},
	}

	for _, s := range sentences {
		tmpl := DecomposeToTemplate(s.sent, s.subj, s.rel, s.obj)
		if tmpl != nil {
			pc.AddTemplate(*tmpl)
		}
	}

	if pc.Size() == 0 {
		t.Fatal("no templates were added")
	}
	t.Logf("Loaded %d templates", pc.Size())

	// Compose a novel sentence about a different entity.
	result := pc.Compose("Ada Lovelace", RelIsA, "mathematician", DFDefines)
	if result == "" {
		t.Fatal("Compose returned empty string")
	}
	t.Logf("Composed: %s", result)

	// The result should contain the new entities.
	if !strings.Contains(result, "Ada Lovelace") {
		t.Errorf("result should contain subject 'Ada Lovelace': %s", result)
	}
	if !strings.Contains(result, "mathematician") {
		t.Errorf("result should contain object 'mathematician': %s", result)
	}

	// It should NOT contain the original entities.
	if strings.Contains(result, "Marie Curie") {
		t.Errorf("result should not contain original subject: %s", result)
	}

	// Should end properly.
	if !strings.HasSuffix(result, ".") && !strings.HasSuffix(result, "!") && !strings.HasSuffix(result, "?") {
		t.Errorf("result should end with punctuation: %s", result)
	}
}

func TestComposeLocation(t *testing.T) {
	pc := NewPhraseComposer()

	tmpl := DecomposeToTemplate(
		"The Eiffel Tower is located in Paris.",
		"The Eiffel Tower", RelLocatedIn, "Paris",
	)
	if tmpl != nil {
		pc.AddTemplate(*tmpl)
	}

	result := pc.Compose("The Colosseum", RelLocatedIn, "Rome", DFContext)
	if result == "" {
		t.Fatal("expected composed sentence")
	}
	t.Logf("Composed: %s", result)

	if !strings.Contains(result, "The Colosseum") {
		t.Errorf("missing subject: %s", result)
	}
	if !strings.Contains(result, "Rome") {
		t.Errorf("missing object: %s", result)
	}
}

func TestComposeNovelty(t *testing.T) {
	pc := NewPhraseComposer()

	src := "Marie Curie was a pioneering physicist."
	tmpl := DecomposeToTemplate(src, "Marie Curie", RelIsA, "physicist")
	if tmpl != nil {
		pc.AddTemplate(*tmpl)
	}

	result := pc.Compose("Alan Turing", RelIsA, "computer scientist", DFDefines)
	if result == "" {
		t.Skip("no composition possible")
	}
	t.Logf("Composed: %s", result)

	// The result must differ from the source — it's a NOVEL sentence.
	if result == src {
		t.Errorf("composed sentence is identical to source — not novel")
	}
}

func TestComposeSlotTypeMatching(t *testing.T) {
	pc := NewPhraseComposer()

	// Add templates with different object types.
	yearTmpl := DecomposeToTemplate(
		"Wikipedia was founded in 2001.",
		"Wikipedia", RelFoundedIn, "2001",
	)
	locTmpl := DecomposeToTemplate(
		"The Louvre is located in Paris.",
		"The Louvre", RelLocatedIn, "Paris",
	)
	if yearTmpl != nil {
		pc.AddTemplate(*yearTmpl)
	}
	if locTmpl != nil {
		pc.AddTemplate(*locTmpl)
	}

	// When composing with a year, should prefer the year template.
	result := pc.Compose("Google", RelFoundedIn, "1998", DFContext)
	if result == "" {
		t.Skip("no composition possible")
	}
	t.Logf("Year composition: %s", result)

	if !strings.Contains(result, "1998") {
		t.Errorf("year should appear in result: %s", result)
	}

	// When composing with a location, should prefer the location template.
	result2 := pc.Compose("Big Ben", RelLocatedIn, "London", DFContext)
	if result2 == "" {
		t.Skip("no composition possible")
	}
	t.Logf("Location composition: %s", result2)

	if !strings.Contains(result2, "London") {
		t.Errorf("location should appear in result: %s", result2)
	}
}

func TestComposeAllMultiple(t *testing.T) {
	pc := NewPhraseComposer()

	sentences := []struct {
		sent string
		subj string
		rel  RelType
		obj  string
	}{
		{"Albert Einstein was a renowned physicist.", "Albert Einstein", RelIsA, "physicist"},
		{"Marie Curie was a pioneering physicist.", "Marie Curie", RelIsA, "physicist"},
		{"Isaac Newton was a legendary physicist.", "Isaac Newton", RelIsA, "physicist"},
	}

	for _, s := range sentences {
		tmpl := DecomposeToTemplate(s.sent, s.subj, s.rel, s.obj)
		if tmpl != nil {
			pc.AddTemplate(*tmpl)
		}
	}

	results := pc.ComposeAll("Richard Feynman", RelIsA, "physicist", DFDefines, 3)
	t.Logf("Got %d compositions", len(results))
	for _, r := range results {
		t.Logf("  - %s", r)
		if !strings.Contains(r, "Richard Feynman") {
			t.Errorf("result missing subject: %s", r)
		}
	}
}

// -----------------------------------------------------------------------
// Article Extraction Tests
// -----------------------------------------------------------------------

func TestExtractTemplatesFromArticle(t *testing.T) {
	article := `Albert Einstein was a theoretical physicist. ` +
		`Einstein was born in Ulm. ` +
		`He developed the theory of relativity. ` +
		`Princeton University is located in New Jersey. ` +
		`The institute was founded in 1930.`

	templates := ExtractTemplatesFromArticle("Albert Einstein", article)
	t.Logf("Extracted %d templates", len(templates))
	for _, tmpl := range templates {
		t.Logf("  Pattern: %s  (rel=%s, fn=%s, q=%d)",
			tmpl.Pattern, tmpl.Relation, tmpl.Function, tmpl.Quality)
	}

	if len(templates) == 0 {
		t.Error("expected at least one template from the article")
	}

	// At least one should have SUBJECT and OBJECT slots.
	foundBoth := false
	for _, tmpl := range templates {
		hasSubj, hasObj := false, false
		for _, s := range tmpl.Slots {
			if s.Name == "SUBJECT" {
				hasSubj = true
			}
			if s.Name == "OBJECT" {
				hasObj = true
			}
		}
		if hasSubj && hasObj {
			foundBoth = true
			break
		}
	}
	if !foundBoth {
		t.Error("expected at least one template with both SUBJECT and OBJECT")
	}
}

// -----------------------------------------------------------------------
// Persistence Tests
// -----------------------------------------------------------------------

func TestPhraseSaveLoad(t *testing.T) {
	pc := NewPhraseComposer()

	tmpl := DecomposeToTemplate(
		"Marie Curie was a brilliant physicist.",
		"Marie Curie", RelIsA, "physicist",
	)
	if tmpl == nil {
		t.Fatal("failed to decompose")
	}
	pc.AddTemplate(*tmpl)

	tmpl2 := DecomposeToTemplate(
		"London is located in England.",
		"London", RelLocatedIn, "England",
	)
	if tmpl2 != nil {
		pc.AddTemplate(*tmpl2)
	}

	// Save.
	dir := t.TempDir()
	path := filepath.Join(dir, "phrases.json")
	if err := pc.Save(path); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify file exists and has content.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}

	// Load into a new composer.
	pc2 := NewPhraseComposer()
	if err := pc2.Load(path); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if pc2.Size() != pc.Size() {
		t.Errorf("size mismatch: saved %d, loaded %d", pc.Size(), pc2.Size())
	}

	// The loaded composer should produce the same composition.
	r1 := pc.Compose("Ada Lovelace", RelIsA, "mathematician", DFDefines)
	r2 := pc2.Compose("Ada Lovelace", RelIsA, "mathematician", DFDefines)
	if r1 != r2 {
		t.Errorf("composition mismatch after load:\n  original: %s\n  loaded:   %s", r1, r2)
	}
}

// -----------------------------------------------------------------------
// Edge Cases
// -----------------------------------------------------------------------

func TestComposeEmpty(t *testing.T) {
	pc := NewPhraseComposer()
	result := pc.Compose("Ada Lovelace", RelIsA, "mathematician", DFDefines)
	if result != "" {
		t.Errorf("expected empty result from empty composer, got: %s", result)
	}
}

func TestDecomposeEmptyInputs(t *testing.T) {
	if DecomposeToTemplate("", "X", RelIsA, "Y") != nil {
		t.Error("empty sentence should return nil")
	}
	if DecomposeToTemplate("Some sentence here.", "", RelIsA, "Y") != nil {
		t.Error("empty subject should return nil")
	}
}

func TestPhraseSlotKindStrings(t *testing.T) {
	tests := []struct {
		pk   PhraseSlotKind
		name string
	}{
		{PhraseSlotSubject, "subject"},
		{PhraseSlotObject, "object"},
		{PhraseSlotModifier, "modifier"},
		{PhraseSlotCategory, "category"},
		{PhraseSlotVerb, "verb"},
		{PhraseSlotLocation, "location"},
		{PhraseSlotYear, "year"},
		{PhraseSlotQuantity, "quantity"},
	}
	for _, tt := range tests {
		if tt.pk.String() != tt.name {
			t.Errorf("PhraseSlotKind(%d).String() = %s, want %s", tt.pk, tt.pk.String(), tt.name)
		}
		if parsePhraseSlotKind(tt.name) != tt.pk {
			t.Errorf("parsePhraseSlotKind(%s) = %d, want %d", tt.name, parsePhraseSlotKind(tt.name), tt.pk)
		}
	}
}

func TestModifierExtraction(t *testing.T) {
	modSentences := []struct {
		sent string
		subj string
		obj  string
		mod  string
	}{
		{"Isaac Newton was a renowned physicist.", "Isaac Newton", "physicist", "renowned"},
		{"Ada Lovelace was a pioneering mathematician.", "Ada Lovelace", "mathematician", "pioneering"},
		{"Albert Einstein was a legendary physicist.", "Albert Einstein", "physicist", "legendary"},
	}

	for _, ms := range modSentences {
		tmpl := DecomposeToTemplate(ms.sent, ms.subj, RelIsA, ms.obj)
		if tmpl == nil {
			t.Errorf("failed to decompose: %s", ms.sent)
			continue
		}

		foundMod := false
		for _, slot := range tmpl.Slots {
			if slot.Name == "MODIFIER" && strings.EqualFold(slot.Original, ms.mod) {
				foundMod = true
			}
		}
		if !foundMod {
			t.Errorf("expected MODIFIER=%s in template from %q, got pattern: %s",
				ms.mod, ms.sent, tmpl.Pattern)
		}
	}
}

func TestSameEntityRejection(t *testing.T) {
	pc := NewPhraseComposer()

	tmpl := DecomposeToTemplate(
		"Marie Curie was a pioneering physicist.",
		"Marie Curie", RelIsA, "physicist",
	)
	if tmpl != nil {
		pc.AddTemplate(*tmpl)
	}

	// Composing about the SAME entity should not use its own template.
	// With only one template from Marie Curie, result should be empty.
	result := pc.Compose("Marie Curie", RelIsA, "chemist", DFDefines)
	if result != "" {
		t.Errorf("should not compose using same entity's own template, got: %s", result)
	}
}

func TestCollapseSpaces(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"hello  world", "hello world"},
		{"a   b   c", "a b c"},
		{"no extra", "no extra"},
		{" leading", " leading"},
	}
	for _, tt := range tests {
		got := collapseSpaces(tt.in)
		if got != tt.want {
			t.Errorf("collapseSpaces(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
