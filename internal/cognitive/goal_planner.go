package cognitive

import (
	"fmt"
	"strings"
)

// -----------------------------------------------------------------------
// Goal Planner — forward-chaining plan generation from the knowledge graph.
//
// Given a goal like "learn Go", the planner:
//   1. Finds the goal node in the graph
//   2. Collects related concepts via part_of, related_to, has edges
//   3. Orders them by dependency (foundational first, applications last)
//   4. Generates a human-readable step-by-step plan
//
// Pure code. No LLM. Plans from structured knowledge.
// -----------------------------------------------------------------------

// GoalPlanner generates step-by-step plans from the knowledge graph.
type GoalPlanner struct {
	Graph    *CognitiveGraph
	Semantic *SemanticEngine
}

// GoalPlan is a generated step-by-step plan.
type GoalPlan struct {
	Goal       string
	Steps      []GoalStep
	Confidence float64
	Trace      string // human-readable explanation
}

// GoalStep is one step in a plan.
type GoalStep struct {
	Order     int
	Action    string // "Learn X", "Practice Y", "Understand Z"
	Reason    string // why this step
	DependsOn []int  // which previous steps are prerequisites
}

// NewGoalPlanner creates a goal planning engine.
func NewGoalPlanner(graph *CognitiveGraph, semantic *SemanticEngine) *GoalPlanner {
	return &GoalPlanner{
		Graph:    graph,
		Semantic: semantic,
	}
}

// PlanFor generates a step-by-step plan to achieve a goal.
func (gp *GoalPlanner) PlanFor(goal string) *GoalPlan {
	gp.Graph.mu.RLock()
	defer gp.Graph.mu.RUnlock()

	// Find the goal node
	goalID := gp.findGoalNode(goal)
	if goalID == "" {
		return nil
	}

	goalNode := gp.Graph.nodes[goalID]
	goalLabel := goalID
	if goalNode != nil {
		goalLabel = goalNode.Label
	}

	// Collect related concepts organized by relationship type
	type relatedConcept struct {
		id       string
		label    string
		relation RelType
		depth    int // 0 = direct, 1 = indirect
	}

	var concepts []relatedConcept
	seen := map[string]bool{goalID: true}

	// Direct outgoing edges from the goal
	for _, edge := range gp.Graph.outEdges[goalID] {
		if seen[edge.To] {
			continue
		}
		seen[edge.To] = true

		label := edge.To
		if node, ok := gp.Graph.nodes[edge.To]; ok {
			label = node.Label
		}
		concepts = append(concepts, relatedConcept{
			id: edge.To, label: label, relation: edge.Relation, depth: 0,
		})
	}

	// Direct incoming edges (things that reference the goal)
	for _, edge := range gp.Graph.inEdges[goalID] {
		if seen[edge.From] {
			continue
		}
		seen[edge.From] = true

		label := edge.From
		if node, ok := gp.Graph.nodes[edge.From]; ok {
			label = node.Label
		}
		concepts = append(concepts, relatedConcept{
			id: edge.From, label: label, relation: edge.Relation, depth: 0,
		})
	}

	// One more hop for richer plans
	initialLen := len(concepts)
	for i := 0; i < initialLen; i++ {
		c := concepts[i]
		for _, edge := range gp.Graph.outEdges[c.id] {
			if seen[edge.To] {
				continue
			}
			if edge.Relation == RelPartOf || edge.Relation == RelRelatedTo ||
				edge.Relation == RelIsA || edge.Relation == RelHas {
				seen[edge.To] = true
				label := edge.To
				if node, ok := gp.Graph.nodes[edge.To]; ok {
					label = node.Label
				}
				concepts = append(concepts, relatedConcept{
					id: edge.To, label: label, relation: edge.Relation, depth: 1,
				})
			}
		}
	}

	if len(concepts) == 0 {
		return nil
	}

	// Organize into plan steps with ordering heuristics:
	// 1. is_a concepts first (understand the category)
	// 2. Foundational concepts (part_of, origins)
	// 3. Core features (has, described_as)
	// 4. Applications (used_for)
	// 5. Related exploration (related_to, similar_to)

	type bucket struct {
		priority int
		verb     string
		items    []relatedConcept
	}

	buckets := map[string]*bucket{
		"category":    {0, "Understand", nil},
		"foundation":  {1, "Learn", nil},
		"feature":     {2, "Explore", nil},
		"application": {3, "Practice", nil},
		"related":     {4, "Explore", nil},
	}

	for _, c := range concepts {
		switch c.relation {
		case RelIsA:
			buckets["category"].items = append(buckets["category"].items, c)
		case RelPartOf, RelFoundedBy, RelCreatedBy, RelFoundedIn:
			buckets["foundation"].items = append(buckets["foundation"].items, c)
		case RelHas, RelOffers, RelDescribedAs:
			buckets["feature"].items = append(buckets["feature"].items, c)
		case RelUsedFor:
			buckets["application"].items = append(buckets["application"].items, c)
		default:
			buckets["related"].items = append(buckets["related"].items, c)
		}
	}

	// Build ordered steps
	var steps []GoalStep
	stepNum := 0

	orderedBuckets := []*bucket{
		buckets["category"],
		buckets["foundation"],
		buckets["feature"],
		buckets["application"],
		buckets["related"],
	}

	for _, b := range orderedBuckets {
		if len(b.items) == 0 {
			continue
		}

		for i, item := range b.items {
			if i >= 3 { // max 3 items per bucket
				break
			}
			stepNum++
			reason := goalReasonForRelation(item.relation, goalLabel)

			var deps []int
			if stepNum > 1 {
				deps = []int{stepNum - 1}
			}

			steps = append(steps, GoalStep{
				Order:     stepNum,
				Action:    b.verb + " " + item.label,
				Reason:    reason,
				DependsOn: deps,
			})
		}
	}

	if len(steps) == 0 {
		return nil
	}

	// Compose trace
	var trace strings.Builder
	trace.WriteString(fmt.Sprintf("Plan to %s %s:\n", goalActionVerb(goalLabel), goalLabel))
	for _, s := range steps {
		trace.WriteString(fmt.Sprintf("  %d. %s — %s\n", s.Order, s.Action, s.Reason))
	}

	return &GoalPlan{
		Goal:       goalLabel,
		Steps:      steps,
		Confidence: 0.6,
		Trace:      trace.String(),
	}
}

// FormatGoalPlan produces a human-readable plan response.
func FormatGoalPlan(plan *GoalPlan) string {
	if plan == nil || len(plan.Steps) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("Here's a plan to %s %s:\n\n", goalActionVerb(plan.Goal), plan.Goal))

	for _, step := range plan.Steps {
		b.WriteString(fmt.Sprintf("%d. %s", step.Order, step.Action))
		if step.Reason != "" {
			b.WriteString(fmt.Sprintf(" — %s", step.Reason))
		}
		b.WriteString("\n")
	}

	return b.String()
}

// formatGenericLearningPlan generates a learning plan for a topic.
// Uses domain-specific advice when possible, generic structure otherwise.
func formatGenericLearningPlan(goal string) string {
	// Check for domain-specific learning advice
	lower := strings.ToLower(goal)
	if plan, ok := domainLearningPlans[lower]; ok {
		return plan
	}
	// Check partial matches
	for domain, plan := range domainLearningPlans {
		if strings.Contains(lower, domain) {
			return plan
		}
	}

	// Generic learning plan — varied structure, no robotic repetition
	return fmt.Sprintf(`Learning %s — a starting point:

1. Get oriented — read an overview of %s to understand the landscape. What are the main ideas? Who are the key figures?
2. Learn the vocabulary — every field has its own language. Identify the 10-20 terms you'll see everywhere.
3. Find a good resource — a well-regarded book, course, or tutorial series. One solid source beats ten scattered ones.
4. Practice actively — don't just read. Try exercises, build something small, or explain what you've learned to someone else.
5. Go deeper — once the basics click, pick a subtopic that interests you and explore it. Curiosity is the best teacher.`, goal, goal)
}

// domainLearningPlans provides specific advice for common learning goals.
var domainLearningPlans = map[string]string{
	"guitar": `Learning guitar — a practical roadmap:

1. Get a guitar that's comfortable to hold. Acoustic is great for beginners — no amp needed.
2. Learn basic open chords: G, C, D, Em, Am. These five chords unlock hundreds of songs.
3. Practice chord transitions — the hard part isn't the chords, it's switching between them smoothly.
4. Learn a simple strumming pattern (down-down-up-up-down-up) and play along to songs you like.
5. Start with easy songs — "Knockin' on Heaven's Door", "Horse With No Name", or "Wonderwall" are classics for beginners.
6. Build calluses and finger strength by playing a little every day. 15 minutes daily beats 2 hours once a week.
7. Once you're comfortable, explore barre chords, fingerpicking, or music theory to level up.`,

	"piano": `Learning piano — a practical roadmap:

1. Learn where the notes are — the pattern of black and white keys repeats every octave. Find middle C.
2. Practice scales — start with C major (all white keys). This builds finger independence and muscle memory.
3. Learn basic chords: C, F, G, Am. Play them with your left hand while your right hand plays melodies.
4. Start with simple songs — "Mary Had a Little Lamb", "Ode to Joy", or "Let It Be" are great first pieces.
5. Read sheet music — learn treble and bass clef. Start with just the right hand, then add the left.
6. Practice hands together — this is the hardest leap. Start very slowly, even one note at a time.
7. Explore what excites you — classical, jazz, pop. The best practice is playing music you love.`,

	"programming": `Learning programming — a practical roadmap:

1. Pick one language to start — Python is great for beginners. It reads almost like English.
2. Learn the fundamentals: variables, if/else, loops, functions. Every language has these.
3. Build tiny programs — a calculator, a to-do list, a number guessing game. Start small.
4. Learn to debug — reading error messages is a superpower. Google the error, read the stack trace.
5. Build a real project — something you actually want. A personal website, a Discord bot, a simple game.
6. Learn version control (Git) and how to read other people's code.
7. Join a community — Stack Overflow, Reddit, Discord servers. Programming is learned by doing and asking.`,

	"cooking": `Learning cooking — a practical roadmap:

1. Master the basics: boil water, cook rice, make scrambled eggs, sear meat. Technique matters more than recipes.
2. Learn knife skills — how to chop an onion, mince garlic, dice vegetables. This saves time and fingers.
3. Understand heat — high heat for searing, medium for sauteing, low for simmering. Most beginners cook too hot.
4. Start with simple recipes: pasta with garlic and olive oil, stir-fry, omelets, soups.
5. Season as you go — salt early and often. Taste before serving. Acid (lemon, vinegar) brightens everything.
6. Learn 5 mother sauces or master one cuisine you love.
7. Cook for others — feedback (and compliments) is the best motivator.`,
}

// findGoalNode finds the best matching node for a goal string.
func (gp *GoalPlanner) findGoalNode(goal string) string {
	lower := strings.ToLower(strings.TrimSpace(goal))

	// Strip action verbs
	prefixes := []string{
		"learn ", "master ", "understand ", "study ", "get into ",
		"start with ", "begin ", "get started with ", "improve at ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			lower = strings.TrimSpace(lower[len(p):])
			break
		}
	}

	// Exact match
	if _, ok := gp.Graph.nodes[lower]; ok {
		return lower
	}

	// Label index
	if ids, ok := gp.Graph.byLabel[lower]; ok && len(ids) > 0 {
		return ids[0]
	}

	// Substring match
	for id, node := range gp.Graph.nodes {
		if strings.Contains(strings.ToLower(node.Label), lower) {
			return id
		}
	}

	// Word-level match
	words := strings.Fields(lower)
	for _, word := range words {
		if len(word) < 3 {
			continue
		}
		for id, node := range gp.Graph.nodes {
			if strings.Contains(strings.ToLower(node.Label), word) {
				return id
			}
		}
	}

	return ""
}

// goalReasonForRelation generates a reason string for why a step matters.
func goalReasonForRelation(rel RelType, goal string) string {
	switch rel {
	case RelIsA:
		return "understand the category"
	case RelPartOf:
		return "a component of " + goal
	case RelHas, RelOffers:
		return "a key feature to know"
	case RelUsedFor:
		return "see practical applications"
	case RelFoundedBy, RelCreatedBy:
		return "understand the origins"
	case RelFoundedIn:
		return "know the historical context"
	case RelDescribedAs:
		return "understand core characteristics"
	case RelRelatedTo, RelSimilarTo:
		return "broaden your understanding"
	default:
		return "related to " + goal
	}
}

// goalActionVerb picks an appropriate verb for a goal.
func goalActionVerb(goal string) string {
	lower := strings.ToLower(goal)
	if strings.Contains(lower, "language") || strings.Contains(lower, "programming") {
		return "learn"
	}
	if strings.Contains(lower, "philosophy") || strings.Contains(lower, "theory") {
		return "study"
	}
	return "learn about"
}

// IsPlanningQuestion detects if a query is asking for a plan.
func IsPlanningQuestion(query string) bool {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "?!."))
	planSignals := []string{
		"how do i learn", "how to learn", "how can i learn",
		"how do i start", "how to start", "how to get into",
		"how to get started", "steps to", "plan for", "roadmap for",
		"guide to", "how do i get started", "how to begin",
		"teach me", "how do i improve", "how to master",
		"how to study", "how should i approach", "what should i learn",
	}
	for _, signal := range planSignals {
		if strings.Contains(lower, signal) {
			return true
		}
	}
	return false
}

// ExtractGoal extracts the learning goal from a planning question.
func ExtractGoal(query string) string {
	lower := strings.ToLower(strings.TrimRight(strings.TrimSpace(query), "?!."))
	prefixes := []string{
		"how do i learn ", "how to learn ", "how can i learn ",
		"how do i start ", "how to start with ", "how to get into ",
		"steps to learn ", "plan for learning ", "roadmap for ",
		"guide to ", "how do i get started with ", "how to begin ",
		"teach me ", "how do i improve at ", "how to master ",
		"how to study ", "how should i approach ",
		"what should i learn about ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return strings.TrimSpace(lower[len(p):])
		}
	}
	// Fallback: strip common question words
	for _, word := range []string{"how to ", "how do i ", "how can i "} {
		if strings.HasPrefix(lower, word) {
			return strings.TrimSpace(lower[len(word):])
		}
	}
	return lower
}
