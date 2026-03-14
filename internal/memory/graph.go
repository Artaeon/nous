package memory

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/artaeon/nous/internal/safefile"
)

// Node is a vertex in the knowledge graph.
type Node struct {
	ID       string    `json:"id"`
	Label    string    `json:"label"`
	Type     string    `json:"type"` // "concept", "file", "person", "tool", "project"
	Weight   float64   `json:"weight"`
	LastSeen time.Time `json:"last_seen"`
}

// Edge is a directed relationship between two nodes.
type Edge struct {
	From     string  `json:"from"`
	To       string  `json:"to"`
	Relation string  `json:"relation"` // "uses", "contains", "depends_on", "related_to", "created_by"
	Weight   float64 `json:"weight"`
}

// GraphStats summarizes the knowledge graph.
type GraphStats struct {
	NodeCount int            `json:"node_count"`
	EdgeCount int            `json:"edge_count"`
	ByType    map[string]int `json:"by_type"`
	TopNodes  []Node         `json:"top_nodes"`
}

// KnowledgeGraph is a simple in-memory graph with JSON persistence.
type KnowledgeGraph struct {
	mu    sync.RWMutex
	nodes map[string]*Node
	edges []Edge
	file  string
}

// NewKnowledgeGraph creates or loads a knowledge graph.
func NewKnowledgeGraph(path string) *KnowledgeGraph {
	kg := &KnowledgeGraph{
		nodes: make(map[string]*Node),
		file:  path,
	}
	_ = kg.Load()
	return kg
}

// AddNode adds or updates a node, returning its ID.
func (kg *KnowledgeGraph) AddNode(label, nodeType string) string {
	kg.mu.Lock()
	defer kg.mu.Unlock()

	id := nodeID(label, nodeType)
	if existing, ok := kg.nodes[id]; ok {
		existing.Weight++
		existing.LastSeen = time.Now()
		return id
	}

	kg.nodes[id] = &Node{
		ID:       id,
		Label:    label,
		Type:     nodeType,
		Weight:   1.0,
		LastSeen: time.Now(),
	}
	return id
}

// AddEdge creates or strengthens a relationship between two nodes.
func (kg *KnowledgeGraph) AddEdge(from, to, relation string) {
	kg.mu.Lock()
	defer kg.mu.Unlock()

	// Check if edge already exists
	for i := range kg.edges {
		if kg.edges[i].From == from && kg.edges[i].To == to && kg.edges[i].Relation == relation {
			kg.edges[i].Weight++
			return
		}
	}
	kg.edges = append(kg.edges, Edge{
		From:     from,
		To:       to,
		Relation: relation,
		Weight:   1.0,
	})
}

// Query finds nodes whose label contains the query string (case-insensitive).
func (kg *KnowledgeGraph) Query(query string) []*Node {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	query = strings.ToLower(query)
	var results []*Node
	for _, n := range kg.nodes {
		if strings.Contains(strings.ToLower(n.Label), query) {
			cp := *n
			results = append(results, &cp)
		}
	}

	// Sort by weight descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Weight > results[j].Weight
	})

	if len(results) > 20 {
		results = results[:20]
	}
	return results
}

// Neighbors returns all nodes directly connected to the given node ID.
func (kg *KnowledgeGraph) Neighbors(id string) []*Node {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	connected := make(map[string]bool)
	for _, e := range kg.edges {
		if e.From == id {
			connected[e.To] = true
		}
		if e.To == id {
			connected[e.From] = true
		}
	}

	var results []*Node
	for nid := range connected {
		if n, ok := kg.nodes[nid]; ok {
			cp := *n
			results = append(results, &cp)
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Weight > results[j].Weight
	})
	return results
}

// EdgesFor returns all edges involving the given node.
func (kg *KnowledgeGraph) EdgesFor(id string) []Edge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	var results []Edge
	for _, e := range kg.edges {
		if e.From == id || e.To == id {
			results = append(results, e)
		}
	}
	return results
}

// Stats returns a summary of the knowledge graph.
func (kg *KnowledgeGraph) Stats() GraphStats {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	byType := make(map[string]int)
	var allNodes []Node
	for _, n := range kg.nodes {
		byType[n.Type]++
		allNodes = append(allNodes, *n)
	}

	sort.Slice(allNodes, func(i, j int) bool {
		return allNodes[i].Weight > allNodes[j].Weight
	})
	topCount := 10
	if len(allNodes) < topCount {
		topCount = len(allNodes)
	}

	return GraphStats{
		NodeCount: len(kg.nodes),
		EdgeCount: len(kg.edges),
		ByType:    byType,
		TopNodes:  allNodes[:topCount],
	}
}

// ExtractFromText uses heuristics to discover entities and relationships.
func (kg *KnowledgeGraph) ExtractFromText(text string) {
	var extracted []string

	// Extract file paths
	fileRe := regexp.MustCompile(`(?:^|[\s"'` + "`" + `])([/.][\w./\-]+\.[\w]+)`)
	for _, match := range fileRe.FindAllStringSubmatch(text, -1) {
		path := strings.TrimSpace(match[1])
		if len(path) > 3 {
			id := kg.AddNode(path, "file")
			extracted = append(extracted, id)
		}
	}

	// Extract tool names (word_word patterns typical of tool names)
	toolRe := regexp.MustCompile(`\b([a-z]+_[a-z_]+)\b`)
	for _, match := range toolRe.FindAllStringSubmatch(text, -1) {
		name := match[1]
		if len(name) > 3 && len(name) < 30 {
			id := kg.AddNode(name, "tool")
			extracted = append(extracted, id)
		}
	}

	// Extract Go package references (e.g. "package foo", "import foo/bar")
	pkgRe := regexp.MustCompile(`(?:package|import)\s+"?([a-zA-Z][\w/.\-]*)"?`)
	for _, match := range pkgRe.FindAllStringSubmatch(text, -1) {
		id := kg.AddNode(match[1], "project")
		extracted = append(extracted, id)
	}

	// Extract person names (capitalized word after "by", "from", "author")
	personRe := regexp.MustCompile(`(?i)(?:by|from|author)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)`)
	for _, match := range personRe.FindAllStringSubmatch(text, -1) {
		name := match[1]
		if isLikelyName(name) {
			id := kg.AddNode(name, "person")
			extracted = append(extracted, id)
		}
	}

	// Extract concepts: significant capitalized terms (2+ words)
	conceptRe := regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b`)
	for _, match := range conceptRe.FindAllStringSubmatch(text, -1) {
		concept := match[1]
		if len(concept) > 5 && !isLikelyName(concept) {
			id := kg.AddNode(concept, "concept")
			extracted = append(extracted, id)
		}
	}

	// Build co-occurrence edges
	for i := 0; i < len(extracted); i++ {
		for j := i + 1; j < len(extracted); j++ {
			if extracted[i] != extracted[j] {
				kg.AddEdge(extracted[i], extracted[j], "related_to")
			}
		}
	}
}

// Save persists the graph to disk.
func (kg *KnowledgeGraph) Save() error {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	state := struct {
		Nodes map[string]*Node `json:"nodes"`
		Edges []Edge           `json:"edges"`
	}{
		Nodes: kg.nodes,
		Edges: kg.edges,
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return safefile.WriteAtomic(kg.file, data, 0644)
}

// Load restores the graph from disk.
func (kg *KnowledgeGraph) Load() error {
	data, err := os.ReadFile(kg.file)
	if err != nil {
		return err
	}

	var state struct {
		Nodes map[string]*Node `json:"nodes"`
		Edges []Edge           `json:"edges"`
	}
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()
	if state.Nodes != nil {
		kg.nodes = state.Nodes
	}
	kg.edges = state.Edges
	return nil
}

// GraphPath returns the default path for graph persistence.
func GraphPath(basePath string) string {
	return filepath.Join(basePath, "knowledge_graph.json")
}

// --- helpers ---

func nodeID(label, nodeType string) string {
	clean := strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '/' || r == '.' || r == '_' || r == '-' {
			return r
		}
		return '_'
	}, strings.ToLower(label))
	return fmt.Sprintf("%s:%s", nodeType, clean)
}

func isLikelyName(s string) bool {
	words := strings.Fields(s)
	if len(words) < 1 || len(words) > 4 {
		return false
	}
	// Reject common non-name capitalized phrases
	lower := strings.ToLower(s)
	nonNames := []string{"the", "this", "that", "these", "those", "error", "warning", "note"}
	for _, nn := range nonNames {
		if strings.HasPrefix(lower, nn) {
			return false
		}
	}
	for _, w := range words {
		if len(w) < 2 {
			return false
		}
		if !unicode.IsUpper(rune(w[0])) {
			return false
		}
	}
	return true
}
