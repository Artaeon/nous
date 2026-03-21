package cognitive

import (
	"math/rand"
	"path/filepath"
	"testing"
)

func TestTemplateInduction(t *testing.T) {
	ti := NewTemplateInducer()

	// Induce from a sentence.
	text := "The ancient philosophy shaped modern thinking."
	learned := ti.InduceFromText(text, "test")

	t.Logf("Templates learned: %d", learned)
	t.Logf("Total templates: %d", ti.Size())

	if ti.Size() == 0 {
		t.Fatal("expected at least one template")
	}

	templates := ti.BestTemplates(5)
	for _, tmpl := range templates {
		t.Logf("  Pattern: %s", tmpl.Pattern)
	}
}

func TestTemplateRealization(t *testing.T) {
	ti := NewTemplateInducer()

	// Induce a template.
	ti.InduceFromText("The remarkable scientist discovered fundamental principles.", "test")

	if ti.Size() == 0 {
		t.Skip("no templates induced")
	}

	templates := ti.BestTemplates(1)
	tmpl := &templates[0]

	// Realize with different words.
	fills := map[SlotType][]string{
		SlotAdj:  {"ancient", "profound"},
		SlotNoun: {"philosopher", "truth"},
		SlotVerb: {"explored"},
	}

	rng := rand.New(rand.NewSource(42))
	result := ti.Realize(tmpl, fills, rng)
	t.Logf("Template: %s", tmpl.Pattern)
	t.Logf("Realized: %s", result)

	if result == "" {
		t.Error("expected non-empty realized sentence")
	}
}

func TestTemplateDeduplication(t *testing.T) {
	ti := NewTemplateInducer()

	// Same sentence twice should increment count, not duplicate.
	ti.InduceFromText("The powerful engine drives remarkable innovation.", "test")
	ti.InduceFromText("The powerful engine drives remarkable innovation.", "test")

	if ti.Size() != 1 {
		t.Errorf("expected 1 template (deduped), got %d", ti.Size())
	}

	templates := ti.BestTemplates(1)
	if templates[0].SeenCount != 2 {
		t.Errorf("expected seen_count 2, got %d", templates[0].SeenCount)
	}
}

func TestTemplateFiltering(t *testing.T) {
	ti := NewTemplateInducer()

	// Too short (< 5 words) — should not be learned.
	ti.InduceFromText("Hello world.", "test")
	if ti.Size() != 0 {
		t.Errorf("expected 0 templates for short sentence, got %d", ti.Size())
	}

	// Good length — should be learned.
	ti.InduceFromText("The elegant framework enables powerful computational reasoning systems.", "test")
	if ti.Size() == 0 {
		t.Error("expected template for valid sentence")
	}
}

func TestTemplateSlotQuery(t *testing.T) {
	ti := NewTemplateInducer()

	ti.InduceFromText("The innovative technology transformed global communication networks.", "test")
	ti.InduceFromText("Ancient wisdom guided philosophical inquiry throughout history.", "test")

	// Query templates with both ADJ and NOUN slots.
	matching := ti.TemplatesWithSlots(SlotAdj, SlotNoun)
	t.Logf("Templates with ADJ+NOUN: %d", len(matching))
	for _, m := range matching {
		t.Logf("  %s", m.Pattern)
	}
}

func TestTemplatePersistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "templates.json")

	ti := NewTemplateInducer()
	ti.InduceFromText("The remarkable discovery changed scientific understanding forever.", "test")
	origSize := ti.Size()

	if err := ti.Save(path); err != nil {
		t.Fatal(err)
	}

	ti2 := NewTemplateInducer()
	if err := ti2.Load(path); err != nil {
		t.Fatal(err)
	}

	if ti2.Size() != origSize {
		t.Errorf("expected %d templates after load, got %d", origSize, ti2.Size())
	}
}

func TestTemplateQualityUpdate(t *testing.T) {
	ti := NewTemplateInducer()
	ti.InduceFromText("The powerful engine drives remarkable innovation forward.", "test")

	templates := ti.BestTemplates(1)
	pattern := templates[0].Pattern
	initialQuality := templates[0].Quality

	// Positive feedback should increase quality.
	ti.MarkUsed(pattern, true)
	templates = ti.BestTemplates(1)
	if templates[0].Quality <= initialQuality {
		t.Errorf("expected quality increase after positive feedback")
	}

	// Negative feedback should decrease quality.
	for i := 0; i < 5; i++ {
		ti.MarkUsed(pattern, false)
	}
	templates = ti.BestTemplates(1)
	if templates[0].Quality >= initialQuality {
		t.Errorf("expected quality decrease after negative feedback")
	}
}
