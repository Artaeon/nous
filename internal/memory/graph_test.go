package memory

import (
	"fmt"
	"path/filepath"
	"testing"
)

func TestKnowledgeGraph_AddNodeAndQuery(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	id := kg.AddNode("write_file", "tool")
	if id == "" {
		t.Fatal("AddNode returned empty ID")
	}

	// Adding same node should return same ID
	id2 := kg.AddNode("write_file", "tool")
	if id != id2 {
		t.Error("expected same ID for duplicate node")
	}

	// Query should find it
	results := kg.Query("write_file")
	if len(results) == 0 {
		t.Error("Query returned no results for known node")
	}
	if results[0].Label != "write_file" {
		t.Errorf("expected label 'write_file', got %q", results[0].Label)
	}

	// Weight should have increased
	if results[0].Weight < 2 {
		t.Error("expected weight >= 2 after adding same node twice")
	}
}

func TestKnowledgeGraph_AddEdgeAndNeighbors(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	id1 := kg.AddNode("main.go", "file")
	id2 := kg.AddNode("nous", "project")
	kg.AddEdge(id1, id2, "contains")

	neighbors := kg.Neighbors(id1)
	if len(neighbors) != 1 {
		t.Fatalf("expected 1 neighbor, got %d", len(neighbors))
	}
	if neighbors[0].Label != "nous" {
		t.Errorf("expected neighbor 'nous', got %q", neighbors[0].Label)
	}

	// Edge should strengthen on duplicate add
	kg.AddEdge(id1, id2, "contains")
	edges := kg.EdgesFor(id1)
	if len(edges) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(edges))
	}
	if edges[0].Weight < 2 {
		t.Error("expected edge weight >= 2 after duplicate add")
	}
}

func TestKnowledgeGraph_Stats(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	kg.AddNode("read_file", "tool")
	kg.AddNode("write_file", "tool")
	kg.AddNode("main.go", "file")
	kg.AddEdge("tool:read_file", "file:main.go", "uses")

	stats := kg.Stats()
	if stats.NodeCount != 3 {
		t.Errorf("expected 3 nodes, got %d", stats.NodeCount)
	}
	if stats.EdgeCount != 1 {
		t.Errorf("expected 1 edge, got %d", stats.EdgeCount)
	}
	if stats.ByType["tool"] != 2 {
		t.Errorf("expected 2 tool nodes, got %d", stats.ByType["tool"])
	}
}

func TestKnowledgeGraph_ExtractFromText(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	text := `I edited ./internal/main.go using read_file and write_file tools.
The package is github.com/artaeon/nous created by John Smith.`

	kg.ExtractFromText(text)

	stats := kg.Stats()
	if stats.NodeCount == 0 {
		t.Error("expected nodes to be extracted")
	}
	if stats.EdgeCount == 0 {
		t.Error("expected co-occurrence edges to be created")
	}

	// Should find file nodes
	files := kg.Query("main.go")
	if len(files) == 0 {
		t.Error("expected to find main.go file node")
	}

	// Should find tool nodes
	tools := kg.Query("read_file")
	if len(tools) == 0 {
		t.Error("expected to find read_file tool node")
	}
}

func TestKnowledgeGraph_Persistence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "graph.json")

	kg := NewKnowledgeGraph(path)
	kg.AddNode("test_node", "concept")
	kg.AddEdge("concept:test_node", "concept:test_node", "self")

	if err := kg.Save(); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	kg2 := NewKnowledgeGraph(path)
	results := kg2.Query("test_node")
	if len(results) == 0 {
		t.Error("expected to find node after reload")
	}
}

func TestKnowledgeGraph_ExtractFromTextGoFilePaths(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	text := `Edited files ./internal/server/server.go and /home/user/project/cmd/main.go`
	kg.ExtractFromText(text)

	files := kg.Query("server.go")
	if len(files) == 0 {
		t.Error("expected to find server.go file node")
	}

	files2 := kg.Query("main.go")
	if len(files2) == 0 {
		t.Error("expected to find main.go file node")
	}
}

func TestKnowledgeGraph_ExtractFromTextToolNames(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	text := `Used find_replace and write_file tools to modify the code`
	kg.ExtractFromText(text)

	tools := kg.Query("find_replace")
	if len(tools) == 0 {
		t.Error("expected to find find_replace tool node")
	}

	tools2 := kg.Query("write_file")
	if len(tools2) == 0 {
		t.Error("expected to find write_file tool node")
	}
}

func TestKnowledgeGraph_BidirectionalEdges(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	id1 := kg.AddNode("main.go", "file")
	id2 := kg.AddNode("test.go", "file")
	kg.AddEdge(id1, id2, "related_to")

	// Neighbors should be found from both directions
	neighbors1 := kg.Neighbors(id1)
	found := false
	for _, n := range neighbors1 {
		if n.Label == "test.go" {
			found = true
		}
	}
	if !found {
		t.Error("expected test.go as neighbor of main.go")
	}

	neighbors2 := kg.Neighbors(id2)
	found = false
	for _, n := range neighbors2 {
		if n.Label == "main.go" {
			found = true
		}
	}
	if !found {
		t.Error("expected main.go as neighbor of test.go (bidirectional)")
	}
}

func TestKnowledgeGraph_DuplicateNodePrevention(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	id1 := kg.AddNode("test_tool", "tool")
	id2 := kg.AddNode("test_tool", "tool")
	id3 := kg.AddNode("test_tool", "tool")

	if id1 != id2 || id2 != id3 {
		t.Error("expected same ID for duplicate nodes")
	}

	stats := kg.Stats()
	if stats.NodeCount != 1 {
		t.Errorf("expected 1 node (no duplicates), got %d", stats.NodeCount)
	}

	// Weight should be 3 (added 3 times)
	results := kg.Query("test_tool")
	if len(results) == 0 {
		t.Fatal("expected to find test_tool")
	}
	if results[0].Weight != 3.0 {
		t.Errorf("expected weight 3.0 after 3 adds, got %f", results[0].Weight)
	}
}

func TestKnowledgeGraph_PersistenceRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "graph.json")

	// Create graph with nodes and edges
	kg1 := NewKnowledgeGraph(path)
	id1 := kg1.AddNode("server.go", "file")
	id2 := kg1.AddNode("nous", "project")
	id3 := kg1.AddNode("grep_tool", "tool")
	kg1.AddEdge(id1, id2, "contains")
	kg1.AddEdge(id1, id3, "uses")

	if err := kg1.Save(); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Load into new graph
	kg2 := NewKnowledgeGraph(path)

	stats := kg2.Stats()
	if stats.NodeCount != 3 {
		t.Errorf("expected 3 nodes after reload, got %d", stats.NodeCount)
	}
	if stats.EdgeCount != 2 {
		t.Errorf("expected 2 edges after reload, got %d", stats.EdgeCount)
	}

	// Verify specific node
	results := kg2.Query("server.go")
	if len(results) == 0 {
		t.Error("expected to find server.go after reload")
	}
	if results[0].Type != "file" {
		t.Errorf("expected type 'file', got %q", results[0].Type)
	}

	// Verify neighbors still work
	neighbors := kg2.Neighbors(id1)
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors for server.go, got %d", len(neighbors))
	}
}

func TestKnowledgeGraph_QueryNoMatches(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	kg.AddNode("something", "concept")

	results := kg.Query("nonexistent_thing_xyz")
	if len(results) != 0 {
		t.Errorf("expected 0 results for non-matching query, got %d", len(results))
	}
}

func TestKnowledgeGraph_LargeGraphPerformance(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	// Add 1000 nodes
	ids := make([]string, 1000)
	for i := 0; i < 1000; i++ {
		label := fmt.Sprintf("node_%04d", i)
		ids[i] = kg.AddNode(label, "concept")
	}

	// Add edges between sequential nodes
	for i := 0; i < 999; i++ {
		kg.AddEdge(ids[i], ids[i+1], "related_to")
	}

	stats := kg.Stats()
	if stats.NodeCount != 1000 {
		t.Errorf("expected 1000 nodes, got %d", stats.NodeCount)
	}
	if stats.EdgeCount != 999 {
		t.Errorf("expected 999 edges, got %d", stats.EdgeCount)
	}

	// Query should still work and return max 20 results
	results := kg.Query("node_")
	if len(results) > 20 {
		t.Errorf("expected at most 20 results, got %d", len(results))
	}

	// Neighbors should work for middle node
	neighbors := kg.Neighbors(ids[500])
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors for middle node, got %d", len(neighbors))
	}
}

func TestKnowledgeGraph_EdgesForEmpty(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	kg.AddNode("lonely", "concept")
	edges := kg.EdgesFor("concept:lonely")
	if len(edges) != 0 {
		t.Errorf("expected 0 edges for isolated node, got %d", len(edges))
	}
}

func TestKnowledgeGraph_StatsEmpty(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	stats := kg.Stats()
	if stats.NodeCount != 0 {
		t.Errorf("expected 0 nodes in empty graph, got %d", stats.NodeCount)
	}
	if stats.EdgeCount != 0 {
		t.Errorf("expected 0 edges in empty graph, got %d", stats.EdgeCount)
	}
}

func TestKnowledgeGraph_DuplicateEdgePrevention(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	id1 := kg.AddNode("a", "concept")
	id2 := kg.AddNode("b", "concept")

	kg.AddEdge(id1, id2, "related_to")
	kg.AddEdge(id1, id2, "related_to")
	kg.AddEdge(id1, id2, "related_to")

	edges := kg.EdgesFor(id1)
	if len(edges) != 1 {
		t.Errorf("expected 1 edge (duplicates merged), got %d", len(edges))
	}
	if edges[0].Weight != 3.0 {
		t.Errorf("expected edge weight 3.0, got %f", edges[0].Weight)
	}
}

func TestKnowledgeGraph_QueryCaseInsensitive(t *testing.T) {
	dir := t.TempDir()
	kg := NewKnowledgeGraph(filepath.Join(dir, "graph.json"))

	kg.AddNode("BuildSystem", "concept")

	results := kg.Query("buildsystem")
	if len(results) == 0 {
		t.Error("expected case-insensitive match")
	}
}
