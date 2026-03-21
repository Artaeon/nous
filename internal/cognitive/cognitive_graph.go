package cognitive

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// -----------------------------------------------------------------------
// Cognitive Graph — a living knowledge structure that grows, connects,
// infers, and reasons. Not a database. Not an LLM. A thinking structure.
// -----------------------------------------------------------------------

// NodeType classifies what a node represents.
type NodeType string

const (
	NodeConcept    NodeType = "concept"    // abstract idea (philosophy, programming)
	NodeEntity     NodeType = "entity"     // named thing (Stoicera, Vienna, Go)
	NodeEvent      NodeType = "event"      // something that happened (founded, released)
	NodeProperty   NodeType = "property"   // attribute (fast, open-source, free)
	NodeAction     NodeType = "action"     // user action (journaled, searched, fetched)
	NodePreference NodeType = "preference" // user preference (likes, dislikes)
)

// RelType is the type of relationship between nodes.
type RelType string

const (
	RelIsA         RelType = "is_a"         // X is a Y (definition)
	RelLocatedIn   RelType = "located_in"   // X is in Y
	RelPartOf      RelType = "part_of"      // X is part of Y
	RelCreatedBy   RelType = "created_by"   // X was created by Y
	RelFoundedBy   RelType = "founded_by"   // X was founded by Y
	RelFoundedIn   RelType = "founded_in"   // X was founded in Y (year)
	RelHas         RelType = "has"          // X has Y
	RelOffers      RelType = "offers"       // X offers/provides Y
	RelUsedFor     RelType = "used_for"     // X is used for Y
	RelRelatedTo   RelType = "related_to"   // general association
	RelSimilarTo   RelType = "similar_to"   // X is similar to Y
	RelCauses      RelType = "causes"       // X causes Y
	RelContradicts RelType = "contradicts"  // X contradicts Y
	RelFollows     RelType = "follows"      // X happens after Y (temporal)
	RelPrefers     RelType = "prefers"      // user prefers X
	RelDislikes    RelType = "dislikes"     // user dislikes X
	RelDomain      RelType = "domain"       // X belongs to domain Y
	RelDescribedAs RelType = "described_as" // X is described as Y
)

// transitiveRels are relation types where A→B→C implies A→C.
var transitiveRels = map[RelType]bool{
	RelLocatedIn: true,
	RelPartOf:    true,
	RelIsA:       true,
}

// CogNode represents a concept in the cognitive graph.
type CogNode struct {
	ID         string            `json:"id"`
	Label      string            `json:"label"`       // human-readable
	Type       NodeType          `json:"type"`
	Properties map[string]string `json:"properties,omitempty"`

	// Activation — how relevant this node is RIGHT NOW
	Activation float64   `json:"activation"`
	LastActive time.Time `json:"last_active"`

	// Confidence — how certain we are this node is correct
	Confidence float64 `json:"confidence"`

	// Usage stats
	CreatedAt   time.Time `json:"created_at"`
	AccessCount int       `json:"access_count"`

	// Source — where we learned about this
	Source string `json:"source,omitempty"`
}

// CogEdge represents a typed relationship between two nodes.
type CogEdge struct {
	From      string  `json:"from"`
	To        string  `json:"to"`
	Relation  RelType `json:"relation"`
	Weight    float64 `json:"weight"`    // strength 0.0-1.0
	Confidence float64 `json:"confidence"` // how sure we are
	Source    string  `json:"source,omitempty"`
	Inferred  bool    `json:"inferred,omitempty"` // true if derived by inference
	CreatedAt time.Time `json:"created_at"`
}

// CognitiveGraph is the core thinking structure.
// Thread-safe, persistent, with spreading activation and inference.
type CognitiveGraph struct {
	nodes map[string]*CogNode
	edges []*CogEdge

	// Indexes for fast lookup
	outEdges map[string][]*CogEdge // node ID → outgoing edges
	inEdges  map[string][]*CogEdge // node ID → incoming edges
	byLabel  map[string][]string   // lowercase label → node IDs

	// Activation parameters
	activationDecay float64 // multiplier per hop (0.5 = halves each hop)
	maxSpreadDepth  int     // max hops for spreading activation

	// Persistence
	path     string
	modified bool
	mu       sync.RWMutex
}

// graphData is the JSON-serializable form.
type graphData struct {
	Nodes []*CogNode `json:"nodes"`
	Edges []*CogEdge `json:"edges"`
}

// NewCognitiveGraph creates a new graph, loading from disk if available.
func NewCognitiveGraph(path string) *CognitiveGraph {
	cg := &CognitiveGraph{
		nodes:           make(map[string]*CogNode),
		outEdges:        make(map[string][]*CogEdge),
		inEdges:         make(map[string][]*CogEdge),
		byLabel:         make(map[string][]string),
		activationDecay: 0.5,
		maxSpreadDepth:  3,
		path:            path,
	}
	if path != "" {
		cg.load()
	}
	return cg
}

// -----------------------------------------------------------------------
// Node operations
// -----------------------------------------------------------------------

// nodeID generates a stable ID from a label.
func nodeID(label string) string {
	return strings.ToLower(strings.TrimSpace(label))
}

// EnsureNode creates a node if it doesn't exist, returns its ID.
func (cg *CognitiveGraph) EnsureNode(label string, ntype NodeType) string {
	cg.mu.Lock()
	defer cg.mu.Unlock()
	return cg.ensureNodeLocked(label, ntype, "", 0.5)
}

func (cg *CognitiveGraph) ensureNodeLocked(label string, ntype NodeType, source string, confidence float64) string {
	id := nodeID(label)
	if id == "" {
		return ""
	}

	if existing, ok := cg.nodes[id]; ok {
		existing.AccessCount++
		// Boost confidence if seen again
		existing.Confidence = math.Min(1.0, existing.Confidence+0.05)
		return id
	}

	node := &CogNode{
		ID:         id,
		Label:      label,
		Type:       ntype,
		Properties: make(map[string]string),
		Confidence: confidence,
		CreatedAt:  time.Now(),
		Source:     source,
	}
	cg.nodes[id] = node

	// Index by label
	lower := strings.ToLower(label)
	cg.byLabel[lower] = append(cg.byLabel[lower], id)

	cg.modified = true
	return id
}

// GetNode returns a node by ID, or nil.
func (cg *CognitiveGraph) GetNode(id string) *CogNode {
	cg.mu.RLock()
	defer cg.mu.RUnlock()
	return cg.nodes[id]
}

// FindNodes finds nodes whose label contains the query.
func (cg *CognitiveGraph) FindNodes(query string) []*CogNode {
	cg.mu.RLock()
	defer cg.mu.RUnlock()

	lower := strings.ToLower(query)
	var results []*CogNode

	// Exact match first
	if ids, ok := cg.byLabel[lower]; ok {
		for _, id := range ids {
			if n, ok := cg.nodes[id]; ok {
				results = append(results, n)
			}
		}
	}

	// Substring match
	if len(results) == 0 {
		for _, node := range cg.nodes {
			if strings.Contains(strings.ToLower(node.Label), lower) {
				results = append(results, node)
			}
		}
	}

	return results
}

// -----------------------------------------------------------------------
// Edge operations
// -----------------------------------------------------------------------

// AddEdge adds a relationship between two nodes.
// Deduplicates: if an identical edge exists, boosts its weight.
func (cg *CognitiveGraph) AddEdge(fromLabel, toLabel string, rel RelType, source string) {
	cg.mu.Lock()
	defer cg.mu.Unlock()
	// Ensure nodes exist so the graph is fully connected
	fromID := cg.ensureNodeLocked(fromLabel, NodeConcept, source, 0.7)
	toID := cg.ensureNodeLocked(toLabel, NodeConcept, source, 0.7)
	cg.addEdgeLocked(fromID, toID, rel, source, 0.7, false)
}

func (cg *CognitiveGraph) addEdgeLocked(fromID, toID string, rel RelType, source string, confidence float64, inferred bool) {
	if fromID == "" || toID == "" || fromID == toID {
		return
	}

	// Check for duplicate
	for _, e := range cg.outEdges[fromID] {
		if e.To == toID && e.Relation == rel {
			// Boost existing edge
			e.Weight = math.Min(1.0, e.Weight+0.1)
			e.Confidence = math.Min(1.0, e.Confidence+0.05)
			return
		}
	}

	edge := &CogEdge{
		From:       fromID,
		To:         toID,
		Relation:   rel,
		Weight:     0.7,
		Confidence: confidence,
		Source:     source,
		Inferred:   inferred,
		CreatedAt:  time.Now(),
	}

	cg.edges = append(cg.edges, edge)
	cg.outEdges[fromID] = append(cg.outEdges[fromID], edge)
	cg.inEdges[toID] = append(cg.inEdges[toID], edge)
	cg.modified = true
}

// EdgesFrom returns all outgoing edges from a node.
func (cg *CognitiveGraph) EdgesFrom(label string) []*CogEdge {
	cg.mu.RLock()
	defer cg.mu.RUnlock()
	id := nodeID(label)
	return cg.outEdges[id]
}

// EdgesTo returns all incoming edges to a node.
func (cg *CognitiveGraph) EdgesTo(label string) []*CogEdge {
	cg.mu.RLock()
	defer cg.mu.RUnlock()
	id := nodeID(label)
	return cg.inEdges[id]
}

// -----------------------------------------------------------------------
// Spreading Activation — the core "thinking" mechanism
// -----------------------------------------------------------------------

// Activate sets a node's activation and spreads to connected nodes.
// This is how the graph "focuses" on relevant concepts.
func (cg *CognitiveGraph) Activate(id string, strength float64) {
	cg.mu.Lock()
	defer cg.mu.Unlock()

	node, ok := cg.nodes[id]
	if !ok {
		return
	}

	node.Activation = strength
	node.LastActive = time.Now()
	node.AccessCount++

	// BFS spreading activation
	type queueItem struct {
		id    string
		depth int
	}
	visited := map[string]bool{id: true}
	queue := []queueItem{{id, 0}}

	for len(queue) > 0 {
		item := queue[0]
		queue = queue[1:]

		if item.depth >= cg.maxSpreadDepth {
			continue
		}

		parentActivation := cg.nodes[item.id].Activation

		// Spread to outgoing edges
		for _, edge := range cg.outEdges[item.id] {
			if visited[edge.To] {
				continue
			}
			visited[edge.To] = true

			child := cg.nodes[edge.To]
			if child == nil {
				continue
			}

			// Activation = parent × edge weight × decay
			childActivation := parentActivation * edge.Weight * cg.activationDecay
			if childActivation > child.Activation {
				child.Activation = childActivation
				child.LastActive = time.Now()
			}

			queue = append(queue, queueItem{edge.To, item.depth + 1})
		}

		// Also spread backwards (incoming edges, weaker)
		for _, edge := range cg.inEdges[item.id] {
			if visited[edge.From] {
				continue
			}
			visited[edge.From] = true

			parent := cg.nodes[edge.From]
			if parent == nil {
				continue
			}

			backActivation := parentActivation * edge.Weight * cg.activationDecay * 0.5
			if backActivation > parent.Activation {
				parent.Activation = backActivation
				parent.LastActive = time.Now()
			}

			queue = append(queue, queueItem{edge.From, item.depth + 1})
		}
	}
}

// ActivateMulti activates multiple nodes and lets their activations overlap.
func (cg *CognitiveGraph) ActivateMulti(ids []string, strength float64) {
	for _, id := range ids {
		cg.Activate(id, strength)
	}
}

// DecayAll reduces all activations (called between interactions).
func (cg *CognitiveGraph) DecayAll(factor float64) {
	cg.mu.Lock()
	defer cg.mu.Unlock()
	for _, node := range cg.nodes {
		node.Activation *= factor
		if node.Activation < 0.01 {
			node.Activation = 0
		}
	}
}

// MostActive returns the N most activated nodes.
func (cg *CognitiveGraph) MostActive(n int) []*CogNode {
	cg.mu.RLock()
	defer cg.mu.RUnlock()

	var active []*CogNode
	for _, node := range cg.nodes {
		if node.Activation > 0.01 {
			active = append(active, node)
		}
	}

	sort.Slice(active, func(i, j int) bool {
		return active[i].Activation > active[j].Activation
	})

	if len(active) > n {
		active = active[:n]
	}
	return active
}

// -----------------------------------------------------------------------
// Graph-based Question Answering
// -----------------------------------------------------------------------

// GraphAnswer holds a structured answer from the graph.
type GraphAnswer struct {
	DirectFacts  []string  // facts directly from edges
	InferredFacts []string // facts derived by inference
	Related      []string  // related concepts (from activation)
	Confidence   float64   // overall confidence
	NodeCount    int       // how many nodes contributed
	Explanation  string    // how we got the answer (the path)
}

// Query answers a question by traversing the graph.
// Returns nil if the graph has insufficient knowledge.
func (cg *CognitiveGraph) Query(question string) *GraphAnswer {
	cg.mu.Lock()
	defer cg.mu.Unlock()

	// 1. Extract key concepts from the question
	qtype := classifyQuestion(question)
	keywords := tokenize(question)

	// 2. Find matching nodes
	var matchedIDs []string
	for _, kw := range keywords {
		if ids, ok := cg.byLabel[kw]; ok {
			matchedIDs = append(matchedIDs, ids...)
		}
		// Also check for substring matches in node labels
		for id, node := range cg.nodes {
			if strings.Contains(strings.ToLower(node.Label), kw) {
				matchedIDs = append(matchedIDs, id)
			}
		}
	}

	// Deduplicate
	matchedIDs = uniqueStrings(matchedIDs)
	if len(matchedIDs) == 0 {
		return nil
	}

	// 3. Activate matched nodes (spreading activation)
	for _, id := range matchedIDs {
		if node, ok := cg.nodes[id]; ok {
			node.Activation = 1.0
			node.LastActive = time.Now()
			node.AccessCount++
		}
	}

	// BFS spread from all matched nodes
	visited := make(map[string]bool)
	for _, id := range matchedIDs {
		visited[id] = true
	}
	queue := make([]string, len(matchedIDs))
	copy(queue, matchedIDs)
	depth := 0

	for len(queue) > 0 && depth < cg.maxSpreadDepth {
		var nextQueue []string
		for _, id := range queue {
			parentAct := cg.nodes[id].Activation
			for _, edge := range cg.outEdges[id] {
				if !visited[edge.To] {
					visited[edge.To] = true
					if child, ok := cg.nodes[edge.To]; ok {
						act := parentAct * edge.Weight * cg.activationDecay
						if act > child.Activation {
							child.Activation = act
						}
						nextQueue = append(nextQueue, edge.To)
					}
				}
			}
			for _, edge := range cg.inEdges[id] {
				if !visited[edge.From] {
					visited[edge.From] = true
					if parent, ok := cg.nodes[edge.From]; ok {
						act := parentAct * edge.Weight * cg.activationDecay * 0.5
						if act > parent.Activation {
							parent.Activation = act
						}
						nextQueue = append(nextQueue, edge.From)
					}
				}
			}
		}
		queue = nextQueue
		depth++
	}

	// 4. Collect facts from activated subgraph
	answer := &GraphAnswer{}
	var factEdges []*CogEdge

	// Collect edges involving activated nodes, sorted by relevance
	for _, edge := range cg.edges {
		fromNode := cg.nodes[edge.From]
		toNode := cg.nodes[edge.To]
		if fromNode == nil || toNode == nil {
			continue
		}
		if fromNode.Activation > 0.1 || toNode.Activation > 0.1 {
			factEdges = append(factEdges, edge)
		}
	}

	// Sort by combined activation of from+to nodes
	sort.Slice(factEdges, func(i, j int) bool {
		scoreI := cg.edgeRelevance(factEdges[i], qtype)
		scoreJ := cg.edgeRelevance(factEdges[j], qtype)
		return scoreI > scoreJ
	})

	// 5. Compose facts from edges
	seen := make(map[string]bool)
	for _, edge := range factEdges {
		fact := cg.edgeToFact(edge)
		if fact == "" || seen[fact] {
			continue
		}
		seen[fact] = true

		if edge.Inferred {
			answer.InferredFacts = append(answer.InferredFacts, fact)
		} else {
			answer.DirectFacts = append(answer.DirectFacts, fact)
		}
	}

	// Limit output
	if len(answer.DirectFacts) > 8 {
		answer.DirectFacts = answer.DirectFacts[:8]
	}
	if len(answer.InferredFacts) > 3 {
		answer.InferredFacts = answer.InferredFacts[:3]
	}

	// 6. Collect related concepts (activated but not directly answering)
	active := make([]*CogNode, 0)
	for _, node := range cg.nodes {
		if node.Activation > 0.05 {
			active = append(active, node)
		}
	}
	sort.Slice(active, func(i, j int) bool {
		return active[i].Activation > active[j].Activation
	})
	for i, node := range active {
		if i >= 5 {
			break
		}
		// Only include if not already in direct/inferred facts
		alreadyMentioned := false
		for _, f := range answer.DirectFacts {
			if strings.Contains(strings.ToLower(f), strings.ToLower(node.Label)) {
				alreadyMentioned = true
				break
			}
		}
		if !alreadyMentioned {
			answer.Related = append(answer.Related, node.Label)
		}
	}

	answer.NodeCount = len(matchedIDs)
	if len(answer.DirectFacts) > 0 {
		answer.Confidence = 0.8
	} else if len(answer.InferredFacts) > 0 {
		answer.Confidence = 0.5
	}

	return answer
}

// ComposeAnswer turns a GraphAnswer into a natural language response.
func (cg *CognitiveGraph) ComposeAnswer(question string, ga *GraphAnswer) string {
	if ga == nil || (len(ga.DirectFacts) == 0 && len(ga.InferredFacts) == 0) {
		return ""
	}

	var b strings.Builder

	// Lead with direct facts
	for i, fact := range ga.DirectFacts {
		if i > 0 {
			b.WriteString(" ")
		}
		b.WriteString(ensurePeriod(fact))
	}

	// Add inferred facts with a marker
	if len(ga.InferredFacts) > 0 && len(ga.DirectFacts) > 0 {
		b.WriteString(" Additionally, ")
		b.WriteString(strings.ToLower(ga.InferredFacts[0]))
		if !strings.HasSuffix(ga.InferredFacts[0], ".") {
			b.WriteString(".")
		}
		for _, fact := range ga.InferredFacts[1:] {
			b.WriteString(" ")
			b.WriteString(ensurePeriod(fact))
		}
	} else if len(ga.InferredFacts) > 0 {
		for i, fact := range ga.InferredFacts {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString(ensurePeriod(fact))
		}
	}

	// Add related concepts hint
	if len(ga.Related) > 0 && len(ga.Related) <= 4 {
		b.WriteString(fmt.Sprintf("\n\nRelated: %s.", strings.Join(ga.Related, ", ")))
	}

	return b.String()
}

// edgeRelevance scores how relevant an edge is to a question type.
func (cg *CognitiveGraph) edgeRelevance(edge *CogEdge, qtype QuestionType) float64 {
	fromAct := cg.nodes[edge.From].Activation
	toAct := cg.nodes[edge.To].Activation
	base := (fromAct + toAct) * edge.Weight * edge.Confidence

	// Boost edges that match the question type
	switch qtype {
	case QWhat:
		if edge.Relation == RelIsA || edge.Relation == RelDescribedAs {
			base *= 2.0
		}
	case QWho:
		if edge.Relation == RelCreatedBy || edge.Relation == RelFoundedBy {
			base *= 2.0
		}
	case QWhere:
		if edge.Relation == RelLocatedIn {
			base *= 2.0
		}
	case QWhen:
		if edge.Relation == RelFoundedIn {
			base *= 2.0
		}
	case QList:
		if edge.Relation == RelHas || edge.Relation == RelOffers {
			base *= 1.5
		}
	}

	return base
}

// edgeToFact converts an edge to a human-readable fact.
func (cg *CognitiveGraph) edgeToFact(edge *CogEdge) string {
	from := cg.nodes[edge.From]
	to := cg.nodes[edge.To]
	if from == nil || to == nil {
		return ""
	}

	// Use proper capitalization from node labels
	subj := from.Label
	obj := to.Label

	switch edge.Relation {
	case RelIsA:
		return fmt.Sprintf("%s is %s", subj, articleFor(obj))
	case RelLocatedIn:
		return fmt.Sprintf("%s is located in %s", subj, obj)
	case RelPartOf:
		return fmt.Sprintf("%s is part of %s", subj, obj)
	case RelCreatedBy:
		return fmt.Sprintf("%s was created by %s", subj, obj)
	case RelFoundedBy:
		return fmt.Sprintf("%s was founded by %s", subj, obj)
	case RelFoundedIn:
		return fmt.Sprintf("%s was founded in %s", subj, obj)
	case RelHas:
		return fmt.Sprintf("%s has %s", subj, obj)
	case RelOffers:
		return fmt.Sprintf("%s offers %s", subj, obj)
	case RelUsedFor:
		return fmt.Sprintf("%s is used for %s", subj, obj)
	case RelRelatedTo:
		return fmt.Sprintf("%s is related to %s", subj, obj)
	case RelSimilarTo:
		return fmt.Sprintf("%s is similar to %s", subj, obj)
	case RelCauses:
		return fmt.Sprintf("%s causes %s", subj, obj)
	case RelContradicts:
		return fmt.Sprintf("%s contradicts %s", subj, obj)
	case RelPrefers:
		return fmt.Sprintf("prefers %s", obj)
	case RelDislikes:
		return fmt.Sprintf("dislikes %s", obj)
	case RelDomain:
		return fmt.Sprintf("%s is in the %s domain", subj, obj)
	case RelDescribedAs:
		return fmt.Sprintf("%s is %s", subj, obj)
	default:
		return fmt.Sprintf("%s → %s → %s", subj, edge.Relation, obj)
	}
}

// articleFor adds "a" or "an" before a noun phrase if needed.
func articleFor(s string) string {
	if s == "" {
		return s
	}
	lower := strings.ToLower(s)
	// Don't add article if it already has one
	if strings.HasPrefix(lower, "a ") || strings.HasPrefix(lower, "an ") ||
		strings.HasPrefix(lower, "the ") {
		return s
	}
	return articleForWord(s) + " " + s
}

// -----------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------

// Stats returns graph statistics.
type GraphStats struct {
	Nodes        int
	Edges        int
	ActiveNodes  int // nodes with activation > 0.01
	Topics       int // unique entity nodes
	InferredEdges int
}

func (cg *CognitiveGraph) Stats() GraphStats {
	cg.mu.RLock()
	defer cg.mu.RUnlock()

	stats := GraphStats{
		Nodes: len(cg.nodes),
		Edges: len(cg.edges),
	}
	for _, n := range cg.nodes {
		if n.Activation > 0.01 {
			stats.ActiveNodes++
		}
		if n.Type == NodeEntity {
			stats.Topics++
		}
	}
	for _, e := range cg.edges {
		if e.Inferred {
			stats.InferredEdges++
		}
	}
	return stats
}

// NodeCount returns the number of nodes.
func (cg *CognitiveGraph) NodeCount() int {
	cg.mu.RLock()
	defer cg.mu.RUnlock()
	return len(cg.nodes)
}

// EdgeCount returns the number of edges.
func (cg *CognitiveGraph) EdgeCount() int {
	cg.mu.RLock()
	defer cg.mu.RUnlock()
	return len(cg.edges)
}

// -----------------------------------------------------------------------
// Persistence
// -----------------------------------------------------------------------

func (cg *CognitiveGraph) Save() error {
	cg.mu.RLock()
	defer cg.mu.RUnlock()

	if cg.path == "" || !cg.modified {
		return nil
	}

	data := graphData{
		Nodes: make([]*CogNode, 0, len(cg.nodes)),
		Edges: cg.edges,
	}
	for _, n := range cg.nodes {
		data.Nodes = append(data.Nodes, n)
	}

	b, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	tmpPath := cg.path + ".tmp"
	if err := os.WriteFile(tmpPath, b, 0644); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, cg.path); err != nil {
		return err
	}

	cg.modified = false
	return nil
}

func (cg *CognitiveGraph) load() {
	b, err := os.ReadFile(cg.path)
	if err != nil {
		return
	}

	var data graphData
	if err := json.Unmarshal(b, &data); err != nil {
		return
	}

	for _, n := range data.Nodes {
		if n.Properties == nil {
			n.Properties = make(map[string]string)
		}
		cg.nodes[n.ID] = n
		lower := strings.ToLower(n.Label)
		cg.byLabel[lower] = append(cg.byLabel[lower], n.ID)
	}

	for _, e := range data.Edges {
		cg.edges = append(cg.edges, e)
		cg.outEdges[e.From] = append(cg.outEdges[e.From], e)
		cg.inEdges[e.To] = append(cg.inEdges[e.To], e)
	}
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func uniqueStrings(ss []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}
