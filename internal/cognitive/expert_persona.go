package cognitive

import (
	"fmt"
	"math"
	"strings"
)

// -----------------------------------------------------------------------
// Expert Personas — graph-constrained knowledge perspectives.
//
// A persona constrains which knowledge graph nodes are activated and
// how responses are framed. A "physicist" persona weights science nodes
// heavily and frames answers in terms of physics. A "historian" persona
// weights history nodes and frames answers chronologically.
//
// Unlike LLM-based personas that are just system prompts, these are
// REAL constraints on the knowledge retrieval pipeline — the persona
// determines which facts are gathered and how they're prioritized.
//
// Usage:
//   pe := NewPersonaEngine(graph)
//   result := pe.Answer("Is time travel possible?", "physicist")
//   // Returns physics-grounded answer from graph, not LLM fabrication.
// -----------------------------------------------------------------------

// PersonaEngine manages expert personas over the knowledge graph.
type PersonaEngine struct {
	graph    *CognitiveGraph
	personas map[string]*ExpertPersona
}

// ExpertPersona defines an expert perspective with domain constraints.
type ExpertPersona struct {
	Name        string             // "physicist", "historian", "economist"
	DisplayName string             // "Physicist", "Historian", "Economist"
	Description string             // one-line description
	Domains     []string           // knowledge graph domains to weight
	Relations   []RelType          // preferred relation types
	FrameVerbs  []string           // how this persona frames answers
	Bias        map[string]float64 // topic → weight boost (0.0-1.0)
}

// PersonaAnswer is a persona-constrained response.
type PersonaAnswer struct {
	Persona    string   // which persona answered
	Response   string   // the answer text
	Facts      []string // facts used (for citation)
	Confidence float64  // how well the persona covered this topic
	Domains    []string // which domains contributed
	Disclaimer string   // if the persona can't fully answer
}

// NewPersonaEngine creates a persona engine with built-in expert personas.
func NewPersonaEngine(graph *CognitiveGraph) *PersonaEngine {
	pe := &PersonaEngine{
		graph:    graph,
		personas: make(map[string]*ExpertPersona),
	}
	pe.registerDefaults()
	return pe
}

// Answer generates a persona-constrained response to a query.
func (pe *PersonaEngine) Answer(query, personaName string) *PersonaAnswer {
	persona := pe.personas[strings.ToLower(personaName)]
	if persona == nil {
		return &PersonaAnswer{
			Persona:    personaName,
			Response:   fmt.Sprintf("Unknown persona: %s. Available: %s", personaName, strings.Join(pe.ListPersonas(), ", ")),
			Confidence: 0,
		}
	}

	if pe.graph == nil {
		return &PersonaAnswer{
			Persona:    persona.Name,
			Response:   "Knowledge graph not available.",
			Confidence: 0,
		}
	}

	topic := extractExpertTopic(query)
	facts, domains := pe.gatherPersonaFacts(topic, persona)

	if len(facts) == 0 {
		// Try the graph description as fallback.
		desc := pe.graph.LookupDescription(topic)
		if desc != "" {
			return &PersonaAnswer{
				Persona:    persona.Name,
				Response:   fmt.Sprintf("From a %s perspective: %s", persona.DisplayName, desc),
				Facts:      []string{desc},
				Confidence: 0.5,
				Domains:    domains,
			}
		}
		return &PersonaAnswer{
			Persona:    persona.Name,
			Response:   fmt.Sprintf("As a %s, I don't have enough information about %s in my knowledge base.", persona.DisplayName, topic),
			Confidence: 0.1,
			Disclaimer: fmt.Sprintf("Topic '%s' not well covered in %s domain.", topic, persona.DisplayName),
		}
	}

	response := pe.composePersonaResponse(topic, facts, persona)
	confidence := math.Min(0.95, 0.3+float64(len(facts))*0.1)

	return &PersonaAnswer{
		Persona:    persona.Name,
		Response:   response,
		Facts:      facts,
		Confidence: confidence,
		Domains:    domains,
	}
}

// ListPersonas returns all available persona names.
func (pe *PersonaEngine) ListPersonas() []string {
	var names []string
	for name := range pe.personas {
		names = append(names, name)
	}
	return names
}

// GetPersona returns a persona by name.
func (pe *PersonaEngine) GetPersona(name string) *ExpertPersona {
	return pe.personas[strings.ToLower(name)]
}

// RegisterPersona adds a custom persona.
func (pe *PersonaEngine) RegisterPersona(p *ExpertPersona) {
	pe.personas[strings.ToLower(p.Name)] = p
}

// -----------------------------------------------------------------------
// Fact gathering with persona constraints
// -----------------------------------------------------------------------

func (pe *PersonaEngine) gatherPersonaFacts(topic string, persona *ExpertPersona) ([]string, []string) {
	var facts []string
	domainSet := make(map[string]bool)

	// Direct facts from the topic.
	edges := pe.graph.EdgesFrom(topic)
	for _, e := range edges {
		weight := pe.edgeWeightForPersona(e, persona)
		if weight < 0.2 {
			continue
		}
		label := pe.graph.NodeLabel(e.To)
		if label == "" {
			label = e.To
		}
		fact := fmt.Sprintf("%s %s %s", topic, humanizeRelation(string(e.Relation)), label)
		facts = append(facts, fact)

		for _, d := range persona.Domains {
			if strings.Contains(strings.ToLower(label), d) {
				domainSet[d] = true
			}
		}
	}

	// Gather from domain-related topics.
	for _, domain := range persona.Domains {
		domainEdges := pe.graph.EdgesFrom(domain)
		for _, e := range domainEdges {
			label := pe.graph.NodeLabel(e.To)
			if label == "" {
				label = e.To
			}
			lower := strings.ToLower(label)
			topicLower := strings.ToLower(topic)
			if strings.Contains(lower, topicLower) || strings.Contains(topicLower, lower) {
				fact := fmt.Sprintf("%s %s %s", domain, humanizeRelation(string(e.Relation)), label)
				facts = append(facts, fact)
				domainSet[domain] = true
			}
		}
	}

	// Deduplicate.
	seen := make(map[string]bool)
	var unique []string
	for _, f := range facts {
		if !seen[f] {
			seen[f] = true
			unique = append(unique, f)
		}
	}

	var domains []string
	for d := range domainSet {
		domains = append(domains, d)
	}
	return unique, domains
}

func (pe *PersonaEngine) edgeWeightForPersona(edge *CogEdge, persona *ExpertPersona) float64 {
	base := edge.Weight
	if base == 0 {
		base = 0.5
	}

	for _, rel := range persona.Relations {
		if edge.Relation == rel {
			base *= 1.5
			break
		}
	}

	targetLabel := strings.ToLower(pe.graph.NodeLabel(edge.To))
	for _, domain := range persona.Domains {
		if strings.Contains(targetLabel, domain) {
			base *= 1.3
			break
		}
	}

	for keyword, boost := range persona.Bias {
		if strings.Contains(targetLabel, keyword) {
			base *= (1.0 + boost)
			break
		}
	}

	return math.Min(1.0, base)
}

// -----------------------------------------------------------------------
// Response composition with persona framing
// -----------------------------------------------------------------------

func (pe *PersonaEngine) composePersonaResponse(topic string, facts []string, persona *ExpertPersona) string {
	var b strings.Builder

	// Opening frame.
	if len(persona.FrameVerbs) > 0 {
		frame := persona.FrameVerbs[len(topic)%len(persona.FrameVerbs)]
		fmt.Fprintf(&b, "From a %s perspective, %s %s. ", persona.DisplayName, topic, frame)
	} else {
		fmt.Fprintf(&b, "As a %s: ", persona.DisplayName)
	}

	maxFacts := 5
	if len(facts) < maxFacts {
		maxFacts = len(facts)
	}
	for i, f := range facts[:maxFacts] {
		if i > 0 {
			b.WriteString(" ")
		}
		if len(f) > 0 {
			b.WriteString(strings.ToUpper(f[:1]) + f[1:])
		}
		if !strings.HasSuffix(f, ".") {
			b.WriteString(".")
		}
	}

	if len(facts) > maxFacts {
		fmt.Fprintf(&b, " There are %d additional aspects to explore.", len(facts)-maxFacts)
	}

	return b.String()
}

// -----------------------------------------------------------------------
// Topic extraction for persona queries
// -----------------------------------------------------------------------

func extractExpertTopic(query string) string {
	lower := strings.ToLower(strings.TrimSpace(query))

	// Strip persona-specific prefixes.
	prefixes := []string{
		"as a physicist, ", "as a historian, ", "as an economist, ",
		"as a philosopher, ", "as a biologist, ", "as an engineer, ",
		"as a psychologist, ",
		"from a physics perspective, ", "from a history perspective, ",
		"from an economic perspective, ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			lower = lower[len(p):]
			break
		}
	}

	// Strip question patterns.
	qPrefixes := []string{
		"what is ", "what are ", "explain ", "tell me about ",
		"how does ", "why is ", "is ", "can ", "does ",
		"describe ", "define ",
	}
	for _, p := range qPrefixes {
		if strings.HasPrefix(lower, p) {
			lower = lower[len(p):]
			break
		}
	}

	return strings.TrimRight(strings.TrimSpace(lower), "?.!")
}

// -----------------------------------------------------------------------
// Default expert persona definitions
// -----------------------------------------------------------------------

func (pe *PersonaEngine) registerDefaults() {
	pe.personas["physicist"] = &ExpertPersona{
		Name:        "physicist",
		DisplayName: "Physicist",
		Description: "Analyzes through physics — forces, energy, spacetime, quantum mechanics.",
		Domains:     []string{"physics", "science", "quantum", "relativity", "energy", "force", "particle"},
		Relations:   []RelType{RelCauses, RelIsA, RelRelatedTo, RelDescribedAs},
		FrameVerbs:  []string{"can be understood through physical principles", "involves fundamental forces", "relates to energy and matter"},
		Bias:        map[string]float64{"einstein": 0.3, "newton": 0.3, "quantum": 0.4},
	}

	pe.personas["historian"] = &ExpertPersona{
		Name:        "historian",
		DisplayName: "Historian",
		Description: "Analyzes through historical context — timelines, causes, consequences.",
		Domains:     []string{"history", "war", "civilization", "empire", "revolution", "century"},
		Relations:   []RelType{RelFoundedIn, RelCreatedBy, RelFollows, RelInfluencedBy, RelCauses},
		FrameVerbs:  []string{"has deep historical roots", "evolved through centuries of development", "must be understood in historical context"},
		Bias:        map[string]float64{"ancient": 0.3, "medieval": 0.2, "modern": 0.1},
	}

	pe.personas["economist"] = &ExpertPersona{
		Name:        "economist",
		DisplayName: "Economist",
		Description: "Analyzes through economic theory — markets, incentives, trade-offs.",
		Domains:     []string{"economics", "market", "trade", "finance", "gdp", "inflation", "currency"},
		Relations:   []RelType{RelCauses, RelUsedFor, RelRelatedTo, RelInfluencedBy},
		FrameVerbs:  []string{"involves economic trade-offs", "can be analyzed through market dynamics", "has significant economic implications"},
		Bias:        map[string]float64{"supply": 0.3, "demand": 0.3, "price": 0.2},
	}

	pe.personas["philosopher"] = &ExpertPersona{
		Name:        "philosopher",
		DisplayName: "Philosopher",
		Description: "Analyzes through philosophical inquiry — ethics, meaning, logic.",
		Domains:     []string{"philosophy", "ethics", "logic", "metaphysics", "epistemology", "consciousness"},
		Relations:   []RelType{RelIsA, RelRelatedTo, RelCauses, RelDescribedAs, RelOppositeOf},
		FrameVerbs:  []string{"raises fundamental questions", "must be examined through careful reasoning", "challenges our assumptions about reality"},
		Bias:        map[string]float64{"socrates": 0.3, "plato": 0.3, "aristotle": 0.3, "kant": 0.2},
	}

	pe.personas["biologist"] = &ExpertPersona{
		Name:        "biologist",
		DisplayName: "Biologist",
		Description: "Analyzes through biology — evolution, genetics, ecosystems.",
		Domains:     []string{"biology", "evolution", "genetics", "cell", "organism", "ecology", "species"},
		Relations:   []RelType{RelIsA, RelPartOf, RelHas, RelCauses, RelRelatedTo},
		FrameVerbs:  []string{"can be understood through biological processes", "involves evolutionary mechanisms", "relates to living systems"},
		Bias:        map[string]float64{"dna": 0.4, "darwin": 0.3, "gene": 0.3},
	}

	pe.personas["engineer"] = &ExpertPersona{
		Name:        "engineer",
		DisplayName: "Engineer",
		Description: "Analyzes through engineering — systems, design, constraints, trade-offs.",
		Domains:     []string{"engineering", "technology", "system", "design", "software", "hardware"},
		Relations:   []RelType{RelUsedFor, RelPartOf, RelHas, RelCreatedBy, RelRelatedTo},
		FrameVerbs:  []string{"is a systems design problem", "involves engineering trade-offs", "can be built and optimized"},
		Bias:        map[string]float64{"algorithm": 0.3, "system": 0.2, "design": 0.2},
	}

	pe.personas["psychologist"] = &ExpertPersona{
		Name:        "psychologist",
		DisplayName: "Psychologist",
		Description: "Analyzes through psychology — behavior, cognition, motivation.",
		Domains:     []string{"psychology", "behavior", "cognitive", "emotion", "brain", "mental"},
		Relations:   []RelType{RelCauses, RelRelatedTo, RelInfluencedBy, RelDescribedAs},
		FrameVerbs:  []string{"reflects patterns in human behavior", "involves cognitive and emotional processes", "can be understood through psychological frameworks"},
		Bias:        map[string]float64{"freud": 0.2, "jung": 0.2, "cognitive": 0.3},
	}
}

// IsExpertPersonaQuery detects if a query asks for an expert perspective.
func IsExpertPersonaQuery(input string) (bool, string) {
	lower := strings.ToLower(strings.TrimSpace(input))

	prefixes := []string{
		"as a ", "as an ", "ask the ", "ask a ",
		"from a ", "from the ", "from an ",
		"what would a ", "what would an ",
	}

	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			rest := lower[len(p):]
			fields := strings.Fields(rest)
			if len(fields) > 0 {
				name := strings.TrimRight(fields[0], ",'s")
				personaWords := map[string]bool{
					"physicist": true, "historian": true, "economist": true,
					"philosopher": true, "biologist": true, "engineer": true,
					"psychologist": true, "scientist": true,
				}
				if personaWords[name] {
					return true, name
				}
				// Check common suffixes.
				if strings.HasSuffix(name, "ist") || strings.HasSuffix(name, "ian") {
					return true, name
				}
			}
		}
	}

	return false, ""
}
