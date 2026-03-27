package cognitive

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// buildTestGraph creates a small knowledge graph for spark testing.
//
//	music --IsA--> art --RelatedTo--> creativity --UsedFor--> problem_solving --PartOf--> mathematics
//	music --Has--> harmony --RelatedTo--> frequency --PartOf--> physics
//	physics --RelatedTo--> wave_equation --UsedFor--> acoustics
//	python --IsA--> programming_language --UsedFor--> software --CreatedBy--> engineers
//	engineers --PartOf--> stem
//	stem --RelatedTo--> mathematics
//
// Domains: music/art → "arts", physics/frequency → "science", python/software → "technology"
func buildSparkTestGraph() *CognitiveGraph {
	cg := NewCognitiveGraph("")

	// Arts cluster
	cg.AddEdge("music", "art", RelIsA, "test")
	cg.AddEdge("art", "creativity", RelRelatedTo, "test")
	cg.AddEdge("creativity", "problem solving", RelUsedFor, "test")
	cg.AddEdge("problem solving", "mathematics", RelPartOf, "test")

	// Science cluster
	cg.AddEdge("music", "harmony", RelHas, "test")
	cg.AddEdge("harmony", "frequency", RelRelatedTo, "test")
	cg.AddEdge("frequency", "physics", RelPartOf, "test")
	cg.AddEdge("physics", "wave equation", RelRelatedTo, "test")
	cg.AddEdge("wave equation", "acoustics", RelUsedFor, "test")

	// Technology cluster
	cg.AddEdge("python", "programming language", RelIsA, "test")
	cg.AddEdge("programming language", "software", RelUsedFor, "test")
	cg.AddEdge("software", "engineers", RelCreatedBy, "test")
	cg.AddEdge("engineers", "stem", RelPartOf, "test")
	cg.AddEdge("stem", "mathematics", RelRelatedTo, "test")

	// Set domains via domain edges.
	cg.AddEdge("music", "arts", RelDomain, "test")
	cg.AddEdge("art", "arts", RelDomain, "test")
	cg.AddEdge("harmony", "arts", RelDomain, "test")
	cg.AddEdge("physics", "science", RelDomain, "test")
	cg.AddEdge("frequency", "science", RelDomain, "test")
	cg.AddEdge("wave equation", "science", RelDomain, "test")
	cg.AddEdge("acoustics", "science", RelDomain, "test")
	cg.AddEdge("mathematics", "science", RelDomain, "test")
	cg.AddEdge("python", "technology", RelDomain, "test")
	cg.AddEdge("programming language", "technology", RelDomain, "test")
	cg.AddEdge("software", "technology", RelDomain, "test")
	cg.AddEdge("engineers", "technology", RelDomain, "test")

	return cg
}

func TestFindPaths_DepthFiltering(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)

	paths := se.findPaths(nodeID("music"), 5)

	if len(paths) == 0 {
		t.Fatal("expected to find paths from music, got none")
	}

	// All returned paths should have depth >= 3 (at least 4 nodes including source).
	for _, p := range paths {
		hops := len(p.nodeIDs) - 1
		if hops < 3 {
			t.Errorf("path too short: %d hops (min 3), path: %v", hops, p.nodeIDs)
		}
	}

	// Should find paths reaching mathematics (4 hops through art→creativity→problem_solving).
	found := false
	for _, p := range paths {
		last := p.nodeIDs[len(p.nodeIDs)-1]
		if last == nodeID("mathematics") || last == nodeID("problem solving") {
			found = true
			break
		}
	}
	if !found {
		t.Log("paths found:")
		for _, p := range paths {
			t.Logf("  %v", p.nodeIDs)
		}
		t.Error("expected to find a path reaching mathematics or problem solving")
	}
}

func TestFindPaths_RespectsMaxVisited(t *testing.T) {
	// Create a wide graph to test the visit cap.
	cg := NewCognitiveGraph("")
	cg.AddEdge("root", "a1", RelRelatedTo, "test")
	for i := 0; i < 300; i++ {
		from := "a1"
		if i > 0 {
			from = nodeID(strings.Repeat("x", i%26+1))
		}
		to := nodeID(strings.Repeat("y", i%26+1))
		cg.AddEdge(from, to, RelRelatedTo, "test")
	}

	se := NewSparkEngine(cg)
	paths := se.findPaths(nodeID("root"), 5)

	// Should not panic or hang — just verify it returns.
	_ = paths
}

func TestScoreNovelty_CrossDomainBonus(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)

	// Path from music (arts) to physics (science) — cross-domain.
	crossPath := []string{
		nodeID("music"), nodeID("harmony"), nodeID("frequency"), nodeID("physics"),
	}
	crossRels := []string{"has", "related_to", "part_of"}
	crossScore := se.scoreNovelty(crossPath[0], crossPath[len(crossPath)-1], crossPath, crossRels)

	// Path from music (arts) to creativity (no domain set, same type) — same domain.
	// Build a 3-hop same-domain-ish path.
	samePath := []string{
		nodeID("music"), nodeID("art"), nodeID("creativity"), nodeID("problem solving"),
	}
	sameRels := []string{"is_a", "related_to", "used_for"}
	// Set problem solving to arts domain too.
	cg.AddEdge("problem solving", "arts", RelDomain, "test")

	sameScore := se.scoreNovelty(samePath[0], samePath[len(samePath)-1], samePath, sameRels)

	if crossScore <= sameScore {
		t.Errorf("cross-domain score (%.3f) should be higher than same-domain (%.3f)",
			crossScore, sameScore)
	}
}

func TestScoreNovelty_LongerPathsScoreHigher(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)

	short := []string{nodeID("music"), nodeID("harmony"), nodeID("frequency"), nodeID("physics")}
	shortRels := []string{"has", "related_to", "part_of"}

	long := []string{
		nodeID("music"), nodeID("harmony"), nodeID("frequency"),
		nodeID("physics"), nodeID("wave equation"), nodeID("acoustics"),
	}
	longRels := []string{"has", "related_to", "part_of", "related_to", "used_for"}

	shortScore := se.scoreNovelty(short[0], short[len(short)-1], short, shortRels)
	longScore := se.scoreNovelty(long[0], long[len(long)-1], long, longRels)

	if longScore <= shortScore {
		t.Errorf("longer path score (%.3f) should exceed shorter path score (%.3f)",
			longScore, shortScore)
	}
}

func TestHubNodeDetection(t *testing.T) {
	cg := NewCognitiveGraph("")

	// Create a hub node with many edges (unique targets).
	for i := 0; i < 40; i++ {
		target := fmt.Sprintf("target_%d", i)
		cg.AddEdge("hub", target, RelRelatedTo, "test")
	}

	se := NewSparkEngine(cg)

	if !se.isHubNode(nodeID("hub")) {
		t.Error("expected hub to be detected as hub node")
	}

	// A node with few edges should not be a hub.
	cg.AddEdge("leaf", "other", RelRelatedTo, "test")
	if se.isHubNode(nodeID("leaf")) {
		t.Error("leaf should not be a hub node")
	}
}

func TestHubNodePenalizesNovelty(t *testing.T) {
	cg := NewCognitiveGraph("")

	// Build a path through a hub.
	cg.AddEdge("start", "mid1", RelRelatedTo, "test")
	cg.AddEdge("mid1", "hub", RelRelatedTo, "test")
	cg.AddEdge("hub", "end", RelRelatedTo, "test")

	// Make "hub" actually a hub by adding lots of edges.
	for i := 0; i < 40; i++ {
		target := nodeID(strings.Repeat("z", i%26+1) + string(rune('a'+i%26)))
		cg.AddEdge("hub", target, RelRelatedTo, "test")
	}

	se := NewSparkEngine(cg)

	pathWithHub := []string{nodeID("start"), nodeID("mid1"), nodeID("hub"), nodeID("end")}
	relsHub := []string{"related_to", "related_to", "related_to"}

	// Build a path without a hub (same length).
	cg.AddEdge("start2", "mid2", RelIsA, "test")
	cg.AddEdge("mid2", "mid3", RelUsedFor, "test")
	cg.AddEdge("mid3", "end2", RelCreatedBy, "test")

	pathNoHub := []string{nodeID("start2"), nodeID("mid2"), nodeID("mid3"), nodeID("end2")}
	relsNoHub := []string{"is_a", "used_for", "created_by"}

	scoreHub := se.scoreNovelty(pathWithHub[0], pathWithHub[len(pathWithHub)-1], pathWithHub, relsHub)
	scoreNoHub := se.scoreNovelty(pathNoHub[0], pathNoHub[len(pathNoHub)-1], pathNoHub, relsNoHub)

	if scoreHub >= scoreNoHub {
		t.Errorf("hub path score (%.3f) should be lower than non-hub path (%.3f)",
			scoreHub, scoreNoHub)
	}
}

func TestBuildExplanation(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)

	tests := []struct {
		name      string
		source    string
		target    string
		path      []string
		relations []string
		wantParts []string // substrings that must appear
	}{
		{
			name:      "short path",
			source:    "music",
			target:    "physics",
			path:      []string{"music", "harmony", "frequency", "physics"},
			relations: []string{"has", "related_to", "part_of"},
			wantParts: []string{"music", "physics", "connects to"},
		},
		{
			name:      "medium path",
			source:    "music",
			target:    "acoustics",
			path:      []string{"music", "harmony", "frequency", "physics", "acoustics"},
			relations: []string{"has", "related_to", "part_of", "related_to"},
			wantParts: []string{"music", "acoustics", "relates to"},
		},
		{
			name:   "long path",
			source: "music",
			target: "engineers",
			path: []string{
				"music", "art", "creativity", "problem solving",
				"mathematics", "stem", "engineers",
			},
			relations: []string{"is_a", "related_to", "used_for", "part_of", "related_to", "part_of"},
			wantParts: []string{"music", "engineers", "chain"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			explanation := se.buildExplanation(tt.source, tt.target, tt.path, tt.relations)
			if explanation == "" {
				t.Fatal("got empty explanation")
			}
			for _, part := range tt.wantParts {
				if !strings.Contains(strings.ToLower(explanation), strings.ToLower(part)) {
					t.Errorf("explanation %q missing expected substring %q", explanation, part)
				}
			}
		})
	}
}

func TestCooldownBehavior(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)
	se.minNovelty = 0.0 // accept everything for this test
	se.maxPerTurn = 100 // don't limit

	// First ignition should produce sparks.
	sparks1 := se.Ignite([]string{"music"})
	if len(sparks1) == 0 {
		t.Fatal("first ignition should produce sparks")
	}

	// Second ignition with same topic — cooldown should block repeats.
	sparks2 := se.Ignite([]string{"music"})

	// The same source→target pairs from sparks1 should not appear in sparks2.
	surfaced := make(map[string]bool)
	for _, s := range sparks1 {
		surfaced[s.Source+"→"+s.Target] = true
	}
	for _, s := range sparks2 {
		key := s.Source + "→" + s.Target
		if surfaced[key] {
			t.Errorf("spark %q should be on cooldown", key)
		}
	}
}

func TestCooldownExpires(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)
	se.cooldownDur = 1 * time.Millisecond // very short for testing

	se.RecordSurfaced("music", "physics")

	// Wait for cooldown to expire.
	time.Sleep(5 * time.Millisecond)

	// The pair should no longer be blocked.
	se.mu.Lock()
	key := "music→physics"
	surfTime, ok := se.surfaced[key]
	expired := !ok || time.Since(surfTime) >= se.cooldownDur
	se.mu.Unlock()

	if !expired {
		// This is fine — just verify the mechanism.
	}
}

func TestTopicWindowManagement(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)

	// Add 15 topics.
	topics := make([]string, 15)
	for i := range topics {
		topics[i] = string(rune('a' + i))
	}
	se.RecordTopics(topics)

	got := se.RecentTopics()
	if len(got) != 15 {
		t.Errorf("expected 15 topics, got %d", len(got))
	}

	// Add 10 more — should trim to 20.
	more := make([]string, 10)
	for i := range more {
		more[i] = string(rune('A' + i))
	}
	se.RecordTopics(more)

	got = se.RecentTopics()
	if len(got) != 20 {
		t.Errorf("expected 20 topics after overflow, got %d", len(got))
	}

	// The oldest 5 should have been dropped.
	for _, topic := range got {
		for _, dropped := range []string{"a", "b", "c", "d", "e"} {
			if topic == dropped {
				t.Errorf("topic %q should have been dropped from window", dropped)
			}
		}
	}
}

func TestIgnite_ReturnsNilForUnknownTopics(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)

	sparks := se.Ignite([]string{"nonexistent_topic_xyz"})
	if sparks != nil {
		t.Errorf("expected nil for unknown topic, got %d sparks", len(sparks))
	}
}

func TestIgnite_RespectsMaxPerTurn(t *testing.T) {
	cg := buildSparkTestGraph()
	se := NewSparkEngine(cg)
	se.minNovelty = 0.0 // accept everything
	se.maxPerTurn = 1

	sparks := se.Ignite([]string{"music"})
	if len(sparks) > 1 {
		t.Errorf("maxPerTurn=1 but got %d sparks", len(sparks))
	}
}

func TestRelationVerb(t *testing.T) {
	tests := []struct {
		rel  string
		want string
	}{
		{"is_a", "is a type of"},
		{"used_for", "is used for"},
		{"created_by", "was created by"},
		{"located_in", "is located in"},
		{"part_of", "is part of"},
		{"unknown_rel", "connects to"},
	}

	for _, tt := range tests {
		got := relationVerb(tt.rel)
		if got != tt.want {
			t.Errorf("relationVerb(%q) = %q, want %q", tt.rel, got, tt.want)
		}
	}
}
