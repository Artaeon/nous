package federation

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// SharedCrystal creation and ID generation
// ---------------------------------------------------------------------------

func TestNewSharedCrystal(t *testing.T) {
	c := NewSharedCrystal("what is {topic}", "A {topic} is ...", "question", 0.85)

	// ID must be the SHA-256 of the pattern.
	h := sha256.Sum256([]byte("what is {topic}"))
	want := hex.EncodeToString(h[:])
	if c.ID != want {
		t.Fatalf("ID = %s, want %s", c.ID, want)
	}

	if c.Pattern != "what is {topic}" {
		t.Fatalf("Pattern = %q", c.Pattern)
	}
	if c.Response != "A {topic} is ..." {
		t.Fatalf("Response = %q", c.Response)
	}
	if c.Intent != "question" {
		t.Fatalf("Intent = %q", c.Intent)
	}
	if c.Quality != 0.85 {
		t.Fatalf("Quality = %f", c.Quality)
	}
	if c.Votes != 1 {
		t.Fatalf("Votes = %d", c.Votes)
	}
	if c.Source != "compiled" {
		t.Fatalf("Source = %q", c.Source)
	}
	if len(c.Tags) != 0 {
		t.Fatalf("Tags = %v", c.Tags)
	}
}

func TestSharedCrystalDeterministicID(t *testing.T) {
	a := NewSharedCrystal("hello {name}", "Hi!", "greeting", 0.9)
	b := NewSharedCrystal("hello {name}", "Hey!", "greeting", 0.7)

	if a.ID != b.ID {
		t.Fatal("same pattern must produce same ID")
	}

	c := NewSharedCrystal("goodbye {name}", "Bye!", "farewell", 0.8)
	if a.ID == c.ID {
		t.Fatal("different patterns must produce different IDs")
	}
}

// ---------------------------------------------------------------------------
// Bundle export/import round-trip
// ---------------------------------------------------------------------------

func TestBundleExportImportRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test-bundle.json")

	crystals := []SharedCrystal{
		NewSharedCrystal("what is {topic}", "A {topic} is ...", "question", 0.9),
		NewSharedCrystal("how to {task}", "To {task}, first ...", "howto", 0.8),
	}

	original := NewCrystalBundle("instance-abc", crystals)
	if err := original.Export(path); err != nil {
		t.Fatalf("Export: %v", err)
	}

	loaded, err := ImportBundle(path)
	if err != nil {
		t.Fatalf("ImportBundle: %v", err)
	}

	if loaded.Version != original.Version {
		t.Fatalf("Version = %d, want %d", loaded.Version, original.Version)
	}
	if loaded.Instance != original.Instance {
		t.Fatalf("Instance = %q, want %q", loaded.Instance, original.Instance)
	}
	if loaded.Checksum != original.Checksum {
		t.Fatalf("Checksum = %q, want %q", loaded.Checksum, original.Checksum)
	}
	if len(loaded.Crystals) != len(original.Crystals) {
		t.Fatalf("len(Crystals) = %d, want %d", len(loaded.Crystals), len(original.Crystals))
	}

	if err := loaded.Validate(); err != nil {
		t.Fatalf("Validate round-tripped bundle: %v", err)
	}
}

func TestBundleExportCreatesDirectories(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a", "b", "c", "bundle.json")

	bundle := NewCrystalBundle("inst", nil)
	if err := bundle.Export(path); err != nil {
		t.Fatalf("Export with nested dirs: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("file not created: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Bundle validation
// ---------------------------------------------------------------------------

func TestValidateCorruptChecksum(t *testing.T) {
	crystals := []SharedCrystal{
		NewSharedCrystal("pattern", "response", "intent", 0.9),
	}
	bundle := NewCrystalBundle("inst", crystals)
	bundle.Checksum = "bad"

	if err := bundle.Validate(); err == nil {
		t.Fatal("expected checksum mismatch error")
	}
}

func TestValidateEmptyPattern(t *testing.T) {
	crystal := NewSharedCrystal("", "response", "intent", 0.5)
	// Override the pattern to empty (the constructor uses it for ID but stores it).
	crystal.Pattern = ""
	bundle := NewCrystalBundle("inst", []SharedCrystal{crystal})
	// Fix checksum so only the empty-pattern check fires.
	bundle.Checksum = computeChecksum(bundle.Crystals)

	if err := bundle.Validate(); err == nil {
		t.Fatal("expected empty pattern error")
	}
}

func TestValidateBadVersion(t *testing.T) {
	bundle := NewCrystalBundle("inst", nil)
	bundle.Version = 99

	if err := bundle.Validate(); err == nil {
		t.Fatal("expected version error")
	}
}

func TestValidateGoodBundle(t *testing.T) {
	crystals := []SharedCrystal{
		NewSharedCrystal("hello {name}", "Hi!", "greeting", 0.9),
	}
	bundle := NewCrystalBundle("inst", crystals)

	if err := bundle.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Registry: publish, search, merge, stats
// ---------------------------------------------------------------------------

func TestRegistryPublishAndSearch(t *testing.T) {
	dir := t.TempDir()
	reg, err := NewRegistry(dir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	crystals := []SharedCrystal{
		NewSharedCrystal("what is {topic}", "A {topic} is ...", "question", 0.9),
		NewSharedCrystal("how to {task}", "To {task}, first ...", "howto", 0.8),
		NewSharedCrystal("define {word}", "{word} means ...", "question", 0.7),
	}
	bundle := NewCrystalBundle("inst-1", crystals)

	if err := reg.Publish(bundle); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	// Search by keyword in pattern.
	results := reg.Search("what", 10)
	if len(results) != 1 {
		t.Fatalf("Search 'what': got %d results, want 1", len(results))
	}
	if results[0].Pattern != "what is {topic}" {
		t.Fatalf("unexpected pattern: %q", results[0].Pattern)
	}

	// Search by intent keyword.
	results = reg.Search("question", 10)
	if len(results) != 2 {
		t.Fatalf("Search 'question': got %d results, want 2", len(results))
	}

	// Limit.
	results = reg.Search("question", 1)
	if len(results) != 1 {
		t.Fatalf("Search with limit 1: got %d", len(results))
	}

	// No match.
	results = reg.Search("nonexistent", 10)
	if len(results) != 0 {
		t.Fatalf("Search 'nonexistent': got %d", len(results))
	}
}

func TestRegistryMerge(t *testing.T) {
	dir := t.TempDir()
	reg, err := NewRegistry(dir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	c1 := NewSharedCrystal("hello {name}", "Hi!", "greeting", 0.8)
	c1.Votes = 2
	reg.Merge(c1)

	// Merge same crystal with different quality and votes.
	c2 := NewSharedCrystal("hello {name}", "Hey!", "greeting", 0.6)
	c2.Votes = 3
	reg.Merge(c2)

	merged := reg.Crystals[c1.ID]
	if merged.Votes != 5 {
		t.Fatalf("Votes = %d, want 5", merged.Votes)
	}

	// Weighted average: (0.8*2 + 0.6*3) / 5 = (1.6+1.8)/5 = 0.68
	expectedQ := 0.68
	if diff := merged.Quality - expectedQ; diff > 0.001 || diff < -0.001 {
		t.Fatalf("Quality = %f, want ~%f", merged.Quality, expectedQ)
	}
}

func TestRegistryTopCrystals(t *testing.T) {
	dir := t.TempDir()
	reg, err := NewRegistry(dir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	reg.Merge(NewSharedCrystal("low", "...", "misc", 0.3))
	reg.Merge(NewSharedCrystal("mid", "...", "misc", 0.6))
	reg.Merge(NewSharedCrystal("high", "...", "misc", 0.9))

	top := reg.TopCrystals(2)
	if len(top) != 2 {
		t.Fatalf("TopCrystals(2): got %d", len(top))
	}
	if top[0].Quality != 0.9 {
		t.Fatalf("top[0].Quality = %f, want 0.9", top[0].Quality)
	}
	if top[1].Quality != 0.6 {
		t.Fatalf("top[1].Quality = %f, want 0.6", top[1].Quality)
	}
}

func TestRegistryStats(t *testing.T) {
	dir := t.TempDir()
	reg, err := NewRegistry(dir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	crystals := []SharedCrystal{
		NewSharedCrystal("a", "...", "question", 0.8),
		NewSharedCrystal("b", "...", "question", 0.6),
		NewSharedCrystal("c", "...", "howto", 0.9),
	}
	bundle := NewCrystalBundle("inst", crystals)
	if err := reg.Publish(bundle); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	stats := reg.Stats()
	if stats.TotalCrystals != 3 {
		t.Fatalf("TotalCrystals = %d", stats.TotalCrystals)
	}
	if stats.TotalBundles != 1 {
		t.Fatalf("TotalBundles = %d", stats.TotalBundles)
	}

	// Average: (0.8+0.6+0.9)/3 ≈ 0.7667
	if stats.AvgQuality < 0.76 || stats.AvgQuality > 0.77 {
		t.Fatalf("AvgQuality = %f, want ~0.767", stats.AvgQuality)
	}

	if stats.TopIntents["question"] != 2 {
		t.Fatalf("TopIntents[question] = %d, want 2", stats.TopIntents["question"])
	}
	if stats.TopIntents["howto"] != 1 {
		t.Fatalf("TopIntents[howto] = %d, want 1", stats.TopIntents["howto"])
	}
}

func TestRegistryExportFiltered(t *testing.T) {
	dir := t.TempDir()
	reg, err := NewRegistry(dir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	reg.Merge(NewSharedCrystal("low", "...", "misc", 0.3))
	reg.Merge(NewSharedCrystal("high", "...", "misc", 0.9))

	exportPath := filepath.Join(dir, "export.json")
	if err := reg.Export(exportPath, 0.5); err != nil {
		t.Fatalf("Export: %v", err)
	}

	bundle, err := ImportBundle(exportPath)
	if err != nil {
		t.Fatalf("ImportBundle: %v", err)
	}
	if len(bundle.Crystals) != 1 {
		t.Fatalf("exported %d crystals, want 1", len(bundle.Crystals))
	}
	if bundle.Crystals[0].Pattern != "high" {
		t.Fatalf("unexpected exported pattern: %q", bundle.Crystals[0].Pattern)
	}
}

func TestRegistryLoad(t *testing.T) {
	dir := t.TempDir()

	// Write a bundle file directly.
	crystals := []SharedCrystal{
		NewSharedCrystal("persisted", "...", "test", 0.75),
	}
	bundle := NewCrystalBundle("pre-existing", crystals)
	if err := bundle.Export(filepath.Join(dir, "bundle-pre-existing-1.json")); err != nil {
		t.Fatalf("Export: %v", err)
	}

	// Open registry — should load the pre-existing bundle.
	reg, err := NewRegistry(dir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	if len(reg.Crystals) != 1 {
		t.Fatalf("loaded %d crystals, want 1", len(reg.Crystals))
	}

	stats := reg.Stats()
	if stats.TotalBundles != 1 {
		t.Fatalf("TotalBundles = %d, want 1", stats.TotalBundles)
	}
}

// ---------------------------------------------------------------------------
// Trust scoring
// ---------------------------------------------------------------------------

func TestTrustScorerDefaults(t *testing.T) {
	ts := NewTrustScorer()
	if ts.MinVotes != 3 {
		t.Fatalf("MinVotes = %d", ts.MinVotes)
	}
	if ts.MinQuality != 0.6 {
		t.Fatalf("MinQuality = %f", ts.MinQuality)
	}
	if ts.DecayDays != 90 {
		t.Fatalf("DecayDays = %d", ts.DecayDays)
	}
}

func TestTrustScoreHighQualityManyVotes(t *testing.T) {
	ts := NewTrustScorer()

	c := NewSharedCrystal("good pattern", "good response", "question", 0.95)
	c.Votes = 10
	c.LastVoted = time.Now() // recent

	score := ts.Score(c)
	// High quality, many votes, recent => should be high.
	if score < 0.7 {
		t.Fatalf("Score = %f, expected > 0.7 for high quality + many votes", score)
	}
}

func TestTrustScoreOldLowVotes(t *testing.T) {
	ts := NewTrustScorer()

	c := NewSharedCrystal("old pattern", "old response", "misc", 0.4)
	c.Votes = 1
	c.LastVoted = time.Now().AddDate(0, 0, -365) // a year ago

	score := ts.Score(c)
	// Low quality, few votes, old => should be low.
	if score > 0.4 {
		t.Fatalf("Score = %f, expected < 0.4 for low quality + old + few votes", score)
	}
}

func TestShouldImport(t *testing.T) {
	ts := NewTrustScorer()

	good := NewSharedCrystal("good", "...", "q", 0.9)
	good.Votes = 5
	good.LastVoted = time.Now()

	bad := NewSharedCrystal("bad", "...", "q", 0.2)
	bad.Votes = 1
	bad.LastVoted = time.Now().AddDate(0, 0, -200)

	if !ts.ShouldImport(good) {
		t.Fatal("ShouldImport(good) = false, want true")
	}
	if ts.ShouldImport(bad) {
		t.Fatal("ShouldImport(bad) = true, want false")
	}
}

func TestObserveBundle(t *testing.T) {
	ts := NewTrustScorer()

	ts.ObserveBundle("inst-a", 9, 1)
	if s := ts.BundleScores["inst-a"]; s != 0.9 {
		t.Fatalf("initial score = %f, want 0.9", s)
	}

	// Second observation: EMA with alpha=0.3.
	ts.ObserveBundle("inst-a", 1, 9) // 0.1 rate
	// 0.3*0.1 + 0.7*0.9 = 0.03 + 0.63 = 0.66
	expected := 0.66
	if s := ts.BundleScores["inst-a"]; s < expected-0.01 || s > expected+0.01 {
		t.Fatalf("updated score = %f, want ~%f", s, expected)
	}

	// No observations => no change.
	ts.ObserveBundle("inst-b", 0, 0)
	if _, ok := ts.BundleScores["inst-b"]; ok {
		t.Fatal("zero observations should not create entry")
	}
}

func TestTrustScorerSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trust.json")

	ts := NewTrustScorer()
	ts.MinVotes = 5
	ts.BundleScores["inst-x"] = 0.85

	if err := ts.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	ts2 := NewTrustScorer()
	if err := ts2.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}

	if ts2.MinVotes != 5 {
		t.Fatalf("MinVotes = %d, want 5", ts2.MinVotes)
	}
	if ts2.BundleScores["inst-x"] != 0.85 {
		t.Fatalf("BundleScores[inst-x] = %f, want 0.85", ts2.BundleScores["inst-x"])
	}
}

// ---------------------------------------------------------------------------
// Full workflow: create -> export -> import -> trust -> merge
// ---------------------------------------------------------------------------

func TestFullWorkflow(t *testing.T) {
	dir := t.TempDir()

	// Instance A creates crystals and exports a bundle.
	crystalsA := []SharedCrystal{
		NewSharedCrystal("what is {topic}", "{topic} is ...", "question", 0.9),
		NewSharedCrystal("how to {task}", "To {task}: ...", "howto", 0.85),
		NewSharedCrystal("junk pattern", "...", "spam", 0.2),
	}
	// Bump votes on good crystals.
	crystalsA[0].Votes = 5
	crystalsA[1].Votes = 4

	bundleA := NewCrystalBundle("instance-a", crystalsA)
	bundlePath := filepath.Join(dir, "bundle-a.json")
	if err := bundleA.Export(bundlePath); err != nil {
		t.Fatalf("Export: %v", err)
	}

	// Instance B imports the bundle.
	imported, err := ImportBundle(bundlePath)
	if err != nil {
		t.Fatalf("ImportBundle: %v", err)
	}
	if err := imported.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	// Instance B runs trust scoring.
	ts := NewTrustScorer()
	var accepted, rejected int
	registryDir := filepath.Join(dir, "registry")
	reg, err := NewRegistry(registryDir)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	for _, c := range imported.Crystals {
		if ts.ShouldImport(c) {
			reg.Merge(c)
			accepted++
		} else {
			rejected++
		}
	}

	// The high-quality crystals should be accepted, the junk rejected.
	if accepted != 2 {
		t.Fatalf("accepted = %d, want 2", accepted)
	}
	if rejected != 1 {
		t.Fatalf("rejected = %d, want 1", rejected)
	}

	// Update bundle trust.
	ts.ObserveBundle("instance-a", accepted, rejected)
	if s := ts.BundleScores["instance-a"]; s < 0.6 {
		t.Fatalf("bundle trust = %f, expected >= 0.6", s)
	}

	// Verify registry state.
	stats := reg.Stats()
	if stats.TotalCrystals != 2 {
		t.Fatalf("TotalCrystals = %d, want 2", stats.TotalCrystals)
	}
	if stats.AvgQuality < 0.85 {
		t.Fatalf("AvgQuality = %f, want >= 0.85", stats.AvgQuality)
	}

	top := reg.TopCrystals(1)
	if len(top) != 1 {
		t.Fatalf("TopCrystals(1): got %d", len(top))
	}
	if top[0].Quality != 0.9 {
		t.Fatalf("top crystal quality = %f, want 0.9", top[0].Quality)
	}
}
