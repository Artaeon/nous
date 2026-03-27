package cognitive

import (
	"strings"
	"testing"
)

func TestNewCommonSenseGraph(t *testing.T) {
	csg := NewCommonSenseGraph()
	if csg == nil {
		t.Fatal("NewCommonSenseGraph returned nil")
	}
	if csg.Size() < 500 {
		t.Errorf("expected at least 500 associations, got %d", csg.Size())
	}
	if csg.TopicCount() < 50 {
		t.Errorf("expected at least 50 topics, got %d", csg.TopicCount())
	}
	t.Logf("Common sense graph: %d associations across %d topics", csg.Size(), csg.TopicCount())
}

// -----------------------------------------------------------------------
// Core associations — verify everyday knowledge exists
// -----------------------------------------------------------------------

func TestCoreAssociations_DinnerFood(t *testing.T) {
	csg := NewCommonSenseGraph()

	assocs := csg.Lookup("dinner")
	if len(assocs) == 0 {
		t.Fatal("no associations for 'dinner'")
	}

	// Dinner should be related to food items
	found := map[string]bool{}
	for _, a := range assocs {
		found[strings.ToLower(a.Target)] = true
	}

	for _, want := range []string{"pasta", "salad", "soup", "pizza"} {
		if !found[want] {
			t.Errorf("dinner missing association to %q", want)
		}
	}

	// Should know dinner is a meal
	meals := csg.LookupByRelation("dinner", CSIsA)
	foundMeal := false
	for _, a := range meals {
		if strings.Contains(strings.ToLower(a.Target), "meal") {
			foundMeal = true
		}
	}
	if !foundMeal {
		t.Error("dinner should be classified as a meal")
	}
}

func TestCoreAssociations_BoredActivities(t *testing.T) {
	csg := NewCommonSenseGraph()

	assocs := csg.Lookup("bored")
	if len(assocs) == 0 {
		t.Fatal("no associations for 'bored'")
	}

	// Bored should cause desire for activities
	desires := csg.LookupByRelation("bored", CSCausesDesire)
	if len(desires) < 5 {
		t.Errorf("expected at least 5 activity suggestions for boredom, got %d", len(desires))
	}

	found := map[string]bool{}
	for _, a := range desires {
		found[strings.ToLower(a.Target)] = true
	}

	for _, want := range []string{"go for a walk", "read a book", "exercise"} {
		if !found[want] {
			t.Errorf("bored missing desire for %q", want)
		}
	}
}

func TestCoreAssociations_PromotedCelebration(t *testing.T) {
	csg := NewCommonSenseGraph()

	assocs := csg.Lookup("promoted")
	if len(assocs) == 0 {
		t.Fatal("no associations for 'promoted'")
	}

	found := map[string]bool{}
	for _, a := range assocs {
		found[strings.ToLower(a.Target)] = true
	}

	if !found["congratulations"] {
		t.Error("promoted should be associated with congratulations")
	}
	if !found["celebration"] {
		t.Error("promoted should cause desire for celebration")
	}

	// Should have work/career context
	contexts := csg.LookupByRelation("promoted", CSHasContext)
	if len(contexts) == 0 {
		t.Error("promoted should have work/career context")
	}
}

func TestCoreAssociations_SkyBlue(t *testing.T) {
	csg := NewCommonSenseGraph()

	props := csg.LookupByRelation("sky", CSHasProperty)
	foundBlue := false
	for _, a := range props {
		if strings.EqualFold(a.Target, "blue") {
			foundBlue = true
		}
	}
	if !foundBlue {
		t.Error("sky should have property 'blue'")
	}

	causes := csg.LookupByRelation("sky", CSCausedBy)
	foundRayleigh := false
	for _, a := range causes {
		if strings.Contains(strings.ToLower(a.Target), "rayleigh") {
			foundRayleigh = true
		}
	}
	if !foundRayleigh {
		t.Error("sky blue should be caused by Rayleigh scattering")
	}
}

func TestCoreAssociations_HeadacheRemedy(t *testing.T) {
	csg := NewCommonSenseGraph()

	desires := csg.LookupByRelation("headache", CSCausesDesire)
	if len(desires) < 2 {
		t.Errorf("expected at least 2 remedies for headache, got %d", len(desires))
	}

	found := map[string]bool{}
	for _, a := range desires {
		found[strings.ToLower(a.Target)] = true
	}

	if !found["rest"] {
		t.Error("headache should cause desire for rest")
	}
	if !found["water"] {
		t.Error("headache should cause desire for water")
	}
}

func TestCoreAssociations_RainUmbrella(t *testing.T) {
	csg := NewCommonSenseGraph()

	desires := csg.LookupByRelation("rain", CSCausesDesire)
	foundUmbrella := false
	for _, a := range desires {
		if strings.EqualFold(a.Target, "umbrella") {
			foundUmbrella = true
		}
	}
	if !foundUmbrella {
		t.Error("rain should cause desire for umbrella")
	}
}

func TestCoreAssociations_BookLiterature(t *testing.T) {
	csg := NewCommonSenseGraph()

	assocs := csg.Lookup("book")
	if len(assocs) == 0 {
		t.Fatal("no associations for 'book'")
	}

	// Book should be related to reading
	usedFor := csg.LookupByRelation("book", CSUsedFor)
	foundReading := false
	for _, a := range usedFor {
		if strings.EqualFold(a.Target, "reading") {
			foundReading = true
		}
	}
	if !foundReading {
		t.Error("book should be used for reading")
	}
}

func TestCoreAssociations_FriendSocial(t *testing.T) {
	csg := NewCommonSenseGraph()

	assocs := csg.Lookup("friend")
	if len(assocs) == 0 {
		t.Fatal("no associations for 'friend'")
	}

	found := map[string]bool{}
	for _, a := range assocs {
		found[strings.ToLower(a.Target)] = true
	}

	for _, want := range []string{"call", "meet up", "text"} {
		if !found[want] {
			t.Errorf("friend missing association to %q", want)
		}
	}
}

func TestCoreAssociations_CarDriving(t *testing.T) {
	csg := NewCommonSenseGraph()

	usedFor := csg.LookupByRelation("car", CSUsedFor)
	foundDriving := false
	for _, a := range usedFor {
		if strings.EqualFold(a.Target, "driving") {
			foundDriving = true
		}
	}
	if !foundDriving {
		t.Error("car should be used for driving")
	}

	isA := csg.LookupByRelation("car", CSIsA)
	foundVehicle := false
	for _, a := range isA {
		if strings.EqualFold(a.Target, "vehicle") {
			foundVehicle = true
		}
	}
	if !foundVehicle {
		t.Error("car should be a vehicle")
	}
}

func TestCoreAssociations_StressedRelief(t *testing.T) {
	csg := NewCommonSenseGraph()

	desires := csg.LookupByRelation("stressed", CSCausesDesire)
	if len(desires) < 3 {
		t.Errorf("expected at least 3 stress relief suggestions, got %d", len(desires))
	}

	found := map[string]bool{}
	for _, a := range desires {
		found[strings.ToLower(a.Target)] = true
	}

	if !found["take a break"] {
		t.Error("stressed should cause desire for taking a break")
	}
	if !found["talk to someone"] {
		t.Error("stressed should cause desire to talk to someone")
	}
}

func TestCoreAssociations_TiredSleep(t *testing.T) {
	csg := NewCommonSenseGraph()

	desires := csg.LookupByRelation("tired", CSCausesDesire)
	foundSleep := false
	for _, a := range desires {
		if strings.EqualFold(a.Target, "sleep") {
			foundSleep = true
		}
	}
	if !foundSleep {
		t.Error("tired should cause desire for sleep")
	}
}

// -----------------------------------------------------------------------
// Suggest — verify suggestions are composed from the graph
// -----------------------------------------------------------------------

func TestSuggest_Dinner(t *testing.T) {
	csg := NewCommonSenseGraph()

	suggestions := csg.Suggest("dinner", "recommendation")
	if len(suggestions) == 0 {
		t.Fatal("Suggest('dinner', 'recommendation') returned no suggestions")
	}

	t.Logf("Dinner suggestions: %v", suggestions)

	// Suggestions should mention actual food items from the graph
	combined := strings.ToLower(strings.Join(suggestions, " "))
	foodItems := []string{"pasta", "salad", "soup", "pizza", "curry", "stir-fry", "tacos"}
	foundAny := false
	for _, food := range foodItems {
		if strings.Contains(combined, food) {
			foundAny = true
			break
		}
	}
	if !foundAny {
		t.Error("dinner suggestions should mention food items from the association graph")
	}
}

func TestSuggest_Bored(t *testing.T) {
	csg := NewCommonSenseGraph()

	suggestions := csg.Suggest("bored", "activity")
	if len(suggestions) == 0 {
		t.Fatal("Suggest('bored', 'activity') returned no suggestions")
	}

	t.Logf("Boredom suggestions: %v", suggestions)

	// Should suggest actual activities
	combined := strings.ToLower(strings.Join(suggestions, " "))
	activities := []string{"walk", "book", "learn", "game", "movie", "exercise", "friend", "music"}
	foundAny := false
	for _, act := range activities {
		if strings.Contains(combined, act) {
			foundAny = true
			break
		}
	}
	if !foundAny {
		t.Error("boredom suggestions should mention activities from the association graph")
	}
}

func TestSuggest_Headache(t *testing.T) {
	csg := NewCommonSenseGraph()

	suggestions := csg.Suggest("headache", "remedy")
	if len(suggestions) == 0 {
		t.Fatal("Suggest('headache', 'remedy') returned no suggestions")
	}

	t.Logf("Headache suggestions: %v", suggestions)
}

func TestSuggest_UnknownTopic(t *testing.T) {
	csg := NewCommonSenseGraph()

	suggestions := csg.Suggest("xyznonexistent", "anything")
	if suggestions != nil {
		t.Errorf("expected nil for unknown topic, got %v", suggestions)
	}
}

func TestSuggest_EmptyTopic(t *testing.T) {
	csg := NewCommonSenseGraph()

	suggestions := csg.Suggest("", "anything")
	if suggestions != nil {
		t.Errorf("expected nil for empty topic, got %v", suggestions)
	}
}

// -----------------------------------------------------------------------
// Resolve — verify everyday query resolution
// -----------------------------------------------------------------------

func TestResolve_DinnerQuery(t *testing.T) {
	csg := NewCommonSenseGraph()

	tests := []struct {
		query   string
		topic   string
		context string
	}{
		{"what should I have for dinner?", "food", "meal suggestion"},
		{"What's for dinner tonight?", "food", "meal suggestion"},
		{"suggest something for lunch", "food", "meal suggestion"},
		{"recommend a meal", "food", "meal suggestion"},
	}

	for _, tt := range tests {
		resolved := csg.Resolve(tt.query)
		if resolved == nil {
			t.Errorf("Resolve(%q) returned nil, expected topic=%q", tt.query, tt.topic)
			continue
		}
		if resolved.Topic != tt.topic {
			t.Errorf("Resolve(%q).Topic = %q, want %q", tt.query, resolved.Topic, tt.topic)
		}
		if resolved.Context != tt.context {
			t.Errorf("Resolve(%q).Context = %q, want %q", tt.query, resolved.Context, tt.context)
		}
	}
}

func TestResolve_BoredQuery(t *testing.T) {
	csg := NewCommonSenseGraph()

	tests := []struct {
		query string
		topic string
	}{
		{"I'm bored", "activity"},
		{"I am bored", "activity"},
		{"feeling bored", "activity"},
	}

	for _, tt := range tests {
		resolved := csg.Resolve(tt.query)
		if resolved == nil {
			t.Errorf("Resolve(%q) returned nil, expected topic=%q", tt.query, tt.topic)
			continue
		}
		if resolved.Topic != tt.topic {
			t.Errorf("Resolve(%q).Topic = %q, want %q", tt.query, resolved.Topic, tt.topic)
		}
	}
}

func TestResolve_BookRecommendation(t *testing.T) {
	csg := NewCommonSenseGraph()

	tests := []struct {
		query   string
		topic   string
		context string
	}{
		{"recommend a book", "literature", "recommendation"},
		{"suggest a good book", "literature", "recommendation"},
		{"what's a good book to read?", "literature", "recommendation"},
	}

	for _, tt := range tests {
		resolved := csg.Resolve(tt.query)
		if resolved == nil {
			t.Errorf("Resolve(%q) returned nil, expected topic=%q", tt.query, tt.topic)
			continue
		}
		if resolved.Topic != tt.topic {
			t.Errorf("Resolve(%q).Topic = %q, want %q", tt.query, resolved.Topic, tt.topic)
		}
		if resolved.Context != tt.context {
			t.Errorf("Resolve(%q).Context = %q, want %q", tt.query, resolved.Context, tt.context)
		}
	}
}

func TestResolve_SkyBlue(t *testing.T) {
	csg := NewCommonSenseGraph()

	resolved := csg.Resolve("why is the sky blue?")
	if resolved == nil {
		t.Fatal("Resolve('why is the sky blue?') returned nil")
	}
	if resolved.Topic != "Rayleigh scattering" {
		t.Errorf("Topic = %q, want 'Rayleigh scattering'", resolved.Topic)
	}
	if resolved.Context != "explanation" {
		t.Errorf("Context = %q, want 'explanation'", resolved.Context)
	}
}

func TestResolve_Emotional(t *testing.T) {
	csg := NewCommonSenseGraph()

	tests := []struct {
		query string
		topic string
	}{
		{"I'm feeling stressed", "stress relief"},
		{"I'm sad", "emotional support"},
		{"I got promoted!", "career"},
	}

	for _, tt := range tests {
		resolved := csg.Resolve(tt.query)
		if resolved == nil {
			t.Errorf("Resolve(%q) returned nil, expected topic=%q", tt.query, tt.topic)
			continue
		}
		if resolved.Topic != tt.topic {
			t.Errorf("Resolve(%q).Topic = %q, want %q", tt.query, resolved.Topic, tt.topic)
		}
	}
}

func TestResolve_Health(t *testing.T) {
	csg := NewCommonSenseGraph()

	tests := []struct {
		query string
		topic string
	}{
		{"I have a headache", "headache"},
		{"I'm feeling tired", "fatigue"},
		{"I can't sleep", "insomnia"},
	}

	for _, tt := range tests {
		resolved := csg.Resolve(tt.query)
		if resolved == nil {
			t.Errorf("Resolve(%q) returned nil, expected topic=%q", tt.query, tt.topic)
			continue
		}
		if resolved.Topic != tt.topic {
			t.Errorf("Resolve(%q).Topic = %q, want %q", tt.query, resolved.Topic, tt.topic)
		}
	}
}

func TestResolve_MovieRecommendation(t *testing.T) {
	csg := NewCommonSenseGraph()

	resolved := csg.Resolve("recommend a movie")
	if resolved == nil {
		t.Fatal("Resolve('recommend a movie') returned nil")
	}
	if resolved.Topic != "film" {
		t.Errorf("Topic = %q, want 'film'", resolved.Topic)
	}
}

func TestResolve_UnknownQuery(t *testing.T) {
	csg := NewCommonSenseGraph()

	resolved := csg.Resolve("what is the capital of France?")
	if resolved != nil {
		t.Errorf("expected nil for factual query, got topic=%q", resolved.Topic)
	}
}

func TestResolve_EmptyQuery(t *testing.T) {
	csg := NewCommonSenseGraph()

	resolved := csg.Resolve("")
	if resolved != nil {
		t.Error("expected nil for empty query")
	}
}

// -----------------------------------------------------------------------
// ExtractAssociations — verify Wikipedia mining
// -----------------------------------------------------------------------

func TestExtractAssociations_SeeAlso(t *testing.T) {
	text := `Python is a programming language.

== See also ==
* [[Java (programming language)]]
* [[Ruby (programming language)]]
* [[Go (programming language)]]

== References ==
Some references here.`

	assocs := ExtractAssociations("Python", text)
	if len(assocs) == 0 {
		t.Fatal("ExtractAssociations returned no associations from See also section")
	}

	found := map[string]bool{}
	for _, a := range assocs {
		if a.Relation == CSRelatedTo {
			found[a.Target] = true
		}
	}

	for _, want := range []string{"Java (programming language)", "Ruby (programming language)", "Go (programming language)"} {
		if !found[want] {
			t.Errorf("missing See also association: %q", want)
		}
	}
}

func TestExtractAssociations_Categories(t *testing.T) {
	text := `Vienna is the capital of Austria.

[[Category:Capitals in Europe]]
[[Category:Cities in Austria]]
[[Category:Articles needing cleanup]]`

	assocs := ExtractAssociations("Vienna", text)

	foundCapitals := false
	foundCities := false
	for _, a := range assocs {
		if a.Relation == CSHasContext {
			if a.Target == "Capitals in Europe" {
				foundCapitals = true
			}
			if a.Target == "Cities in Austria" {
				foundCities = true
			}
		}
	}

	if !foundCapitals {
		t.Error("should extract 'Capitals in Europe' category")
	}
	if !foundCities {
		t.Error("should extract 'Cities in Austria' category")
	}

	// Should skip maintenance categories
	for _, a := range assocs {
		if strings.Contains(a.Target, "Articles needing") {
			t.Error("should skip maintenance category 'Articles needing cleanup'")
		}
	}
}

func TestExtractAssociations_FirstParagraphLinks(t *testing.T) {
	text := `A [[dog]] is a [[domesticated]] [[carnivore]] of the family [[Canidae]].

It is part of the wolf-like canids.`

	assocs := ExtractAssociations("Dog", text)

	found := map[string]bool{}
	for _, a := range assocs {
		found[a.Target] = true
	}

	if !found["domesticated"] {
		t.Error("should extract inline link 'domesticated' from first paragraph")
	}
	if !found["carnivore"] {
		t.Error("should extract inline link 'carnivore' from first paragraph")
	}
}

func TestExtractAssociations_Empty(t *testing.T) {
	assocs := ExtractAssociations("", "some text")
	if assocs != nil {
		t.Error("expected nil for empty title")
	}

	assocs = ExtractAssociations("Title", "")
	if assocs != nil {
		t.Error("expected nil for empty text")
	}
}

func TestExtractAssociations_SelfLink(t *testing.T) {
	text := `The [[Sun]] is a star. See also the [[Sun]] article.`

	assocs := ExtractAssociations("Sun", text)
	for _, a := range assocs {
		if strings.EqualFold(a.Target, "Sun") {
			t.Error("should not include self-links")
		}
	}
}

func TestExtractAssociations_DisplayLinks(t *testing.T) {
	text := `[[United States|US]] is a country. [[New York City|NYC]] is a city.`

	assocs := ExtractAssociations("Geography", text)
	found := map[string]bool{}
	for _, a := range assocs {
		found[a.Target] = true
	}

	if !found["United States"] {
		t.Error("should extract target from display links, got targets: " + strings.Join(targetList(assocs), ", "))
	}
}

// -----------------------------------------------------------------------
// LookupByRelation
// -----------------------------------------------------------------------

func TestLookupByRelation(t *testing.T) {
	csg := NewCommonSenseGraph()

	// All dinner CSRelatedTo should be food-related
	related := csg.LookupByRelation("dinner", CSRelatedTo)
	if len(related) == 0 {
		t.Fatal("no CSRelatedTo associations for dinner")
	}

	// All exercise CSUsedFor should be purpose-related
	usedFor := csg.LookupByRelation("exercise", CSUsedFor)
	if len(usedFor) == 0 {
		t.Fatal("no CSUsedFor associations for exercise")
	}
}

// -----------------------------------------------------------------------
// Add / AddBatch — verify graph manipulation
// -----------------------------------------------------------------------

func TestAdd_Deduplication(t *testing.T) {
	csg := NewCommonSenseGraph()

	initial := csg.Size()
	csg.Add("test_topic", "test_target", CSRelatedTo, 0.5)
	after1 := csg.Size()
	if after1 != initial+1 {
		t.Errorf("expected size to increase by 1, got %d → %d", initial, after1)
	}

	// Adding the same association should not increase size
	csg.Add("test_topic", "test_target", CSRelatedTo, 0.3)
	after2 := csg.Size()
	if after2 != after1 {
		t.Errorf("duplicate add should not increase size, got %d → %d", after1, after2)
	}

	// But higher weight should update
	csg.Add("test_topic", "test_target", CSRelatedTo, 0.9)
	assocs := csg.Lookup("test_topic")
	for _, a := range assocs {
		if a.Target == "test_target" && a.Weight != 0.9 {
			t.Errorf("expected weight to be updated to 0.9, got %f", a.Weight)
		}
	}
}

func TestAdd_WeightClamping(t *testing.T) {
	csg := &CommonSenseGraph{
		associations: make(map[string][]Association),
		rng:          nil,
	}

	csg.Add("a", "b", CSRelatedTo, 1.5)
	assocs := csg.Lookup("a")
	if len(assocs) != 1 || assocs[0].Weight != 1.0 {
		t.Error("weight should be clamped to 1.0")
	}

	csg.Add("c", "d", CSRelatedTo, -0.5)
	assocs = csg.Lookup("c")
	if len(assocs) != 1 || assocs[0].Weight != 0.0 {
		t.Error("weight should be clamped to 0.0")
	}
}

func TestAddBatch(t *testing.T) {
	csg := &CommonSenseGraph{
		associations: make(map[string][]Association),
		rng:          nil,
	}

	batch := []Association{
		{Target: "x", Relation: CSRelatedTo, Weight: 0.5},
		{Target: "y", Relation: CSIsA, Weight: 0.8},
		{Target: "z", Relation: CSUsedFor, Weight: 0.6},
	}

	csg.AddBatch("source", batch)
	assocs := csg.Lookup("source")
	if len(assocs) != 3 {
		t.Errorf("expected 3 associations, got %d", len(assocs))
	}
}

// -----------------------------------------------------------------------
// LoadIntoGraph — verify integration with CognitiveGraph
// -----------------------------------------------------------------------

func TestLoadIntoGraph(t *testing.T) {
	csg := NewCommonSenseGraph()
	graph := NewCognitiveGraph("")

	loaded := csg.LoadIntoGraph(graph)
	if loaded == 0 {
		t.Fatal("LoadIntoGraph loaded 0 edges")
	}
	if loaded != csg.Size() {
		t.Errorf("expected %d edges loaded, got %d", csg.Size(), loaded)
	}

	// Verify some associations made it into the graph
	desc := graph.LookupDescription("dinner")
	_ = desc // may or may not have a described_as; that's fine

	// The node should exist
	node := graph.GetNode("dinner")
	if node == nil {
		t.Error("'dinner' node should exist in graph after LoadIntoGraph")
	}

	t.Logf("Loaded %d common sense associations into cognitive graph", loaded)
}

func TestLoadIntoGraph_NilGraph(t *testing.T) {
	csg := NewCommonSenseGraph()
	loaded := csg.LoadIntoGraph(nil)
	if loaded != 0 {
		t.Error("LoadIntoGraph(nil) should return 0")
	}
}

// -----------------------------------------------------------------------
// Case insensitivity
// -----------------------------------------------------------------------

func TestCaseInsensitivity(t *testing.T) {
	csg := NewCommonSenseGraph()

	// Should work regardless of case
	a1 := csg.Lookup("Dinner")
	a2 := csg.Lookup("DINNER")
	a3 := csg.Lookup("dinner")

	if len(a1) == 0 || len(a2) == 0 || len(a3) == 0 {
		t.Error("lookup should be case-insensitive")
	}
	if len(a1) != len(a3) {
		t.Error("case-insensitive lookups should return same results")
	}
}

// -----------------------------------------------------------------------
// Coverage of all seeded domains
// -----------------------------------------------------------------------

func TestAllDomainsCovered(t *testing.T) {
	csg := NewCommonSenseGraph()

	domains := map[string][]string{
		"food":     {"dinner", "lunch", "breakfast", "food", "cooking", "snack"},
		"activity": {"bored", "activity", "hobby", "exercise"},
		"emotion":  {"stressed", "sad", "happy", "promoted", "anxious", "lonely"},
		"books":    {"book", "literature", "fiction", "science fiction"},
		"film":     {"film", "movie"},
		"music":    {"music"},
		"objects":  {"car", "phone", "computer"},
		"nature":   {"sky", "rain", "sun", "snow"},
		"health":   {"headache", "tired", "insomnia", "health", "sleep"},
		"social":   {"friend", "family", "birthday", "party"},
		"work":     {"work", "career", "meeting", "study"},
		"home":     {"home", "cleaning", "morning", "evening", "shopping", "money"},
		"travel":   {"travel", "vacation", "commute"},
	}

	for domain, topics := range domains {
		for _, topic := range topics {
			assocs := csg.Lookup(topic)
			if len(assocs) == 0 {
				t.Errorf("domain %q: topic %q has no associations", domain, topic)
			}
		}
	}
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func targetList(assocs []Association) []string {
	var targets []string
	for _, a := range assocs {
		targets = append(targets, a.Target)
	}
	return targets
}
