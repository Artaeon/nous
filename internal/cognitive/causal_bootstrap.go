package cognitive

// -----------------------------------------------------------------------
// Causal Knowledge Bootstrap — seeds the knowledge graph with common-sense
// causal relationships for major topics.
//
// The simulation engine needs multi-hop causal chains to produce rich
// "what if" scenarios. Without these, simulations return "no effects found."
// This bootstrap adds ~200 causal edges covering infrastructure, nature,
// society, technology, and economy — the topics people most commonly
// simulate.
// -----------------------------------------------------------------------

// CausalFact is a single causal relationship to bootstrap.
type CausalFact struct {
	From     string
	To       string
	Relation RelType
}

// BootstrapCausalKnowledge adds common-sense causal edges to the graph.
// Only adds edges that don't already exist (idempotent).
func BootstrapCausalKnowledge(g *CognitiveGraph) int {
	if g == nil {
		return 0
	}

	added := 0
	for _, f := range causalBootstrapFacts() {
		fromID := g.EnsureNode(f.From, NodeConcept)
		toID := g.EnsureNode(f.To, NodeConcept)
		// Only add if no edge of this type exists.
		edges := g.EdgesFrom(fromID)
		exists := false
		for _, e := range edges {
			if e.To == toID && e.Relation == f.Relation {
				exists = true
				break
			}
		}
		if !exists {
			g.AddEdge(fromID, toID, f.Relation, "causal_bootstrap")
			added++
		}
	}
	return added
}

func causalBootstrapFacts() []CausalFact {
	return []CausalFact{
		// ── Internet infrastructure ──
		{"internet", "communication", RelEnables},
		{"internet", "e-commerce", RelEnables},
		{"internet", "social media", RelEnables},
		{"internet", "remote work", RelEnables},
		{"internet", "cloud computing", RelEnables},
		{"internet", "online education", RelEnables},
		{"internet", "digital economy", RelEnables},
		{"internet", "GPS navigation", RelEnables},
		{"communication", "emergency services", RelEnables},
		{"communication", "supply chain", RelEnables},
		{"e-commerce", "retail industry", RelEnables},
		{"social media", "news distribution", RelEnables},
		{"cloud computing", "data storage", RelEnables},
		{"supply chain", "food distribution", RelEnables},
		{"supply chain", "manufacturing", RelEnables},
		{"food distribution", "food security", RelEnables},
		{"digital economy", "financial markets", RelEnables},
		{"financial markets", "banking", RelEnables},
		{"banking", "loans", RelEnables},
		{"banking", "savings", RelEnables},

		// ── Electricity infrastructure ──
		{"electricity", "lighting", RelEnables},
		{"electricity", "heating", RelEnables},
		{"electricity", "refrigeration", RelEnables},
		{"electricity", "computing", RelEnables},
		{"electricity", "medical equipment", RelEnables},
		{"electricity", "transportation", RelEnables},
		{"electricity", "internet", RelRequires},
		{"refrigeration", "food preservation", RelEnables},
		{"food preservation", "food safety", RelEnables},
		{"computing", "internet", RelEnables},
		{"medical equipment", "hospitals", RelEnables},
		{"hospitals", "healthcare", RelEnables},

		// ── Gravity and physics ──
		{"gravity", "orbital motion", RelCauses},
		{"gravity", "tides", RelCauses},
		{"gravity", "atmospheric retention", RelCauses},
		{"gravity", "weight", RelCauses},
		{"gravity", "structural engineering", RelRequires},
		{"gravity", "water flow", RelCauses},
		{"orbital motion", "seasons", RelCauses},
		{"orbital motion", "day-night cycle", RelCauses},
		{"atmospheric retention", "breathable air", RelEnables},
		{"breathable air", "life on Earth", RelEnables},
		{"water flow", "rivers", RelCauses},
		{"rivers", "agriculture", RelEnables},
		{"agriculture", "food production", RelEnables},
		{"tides", "coastal ecosystems", RelEnables},

		// ── Water ──
		{"water", "agriculture", RelEnables},
		{"water", "drinking water", RelEnables},
		{"water", "sanitation", RelEnables},
		{"water", "hydroelectric power", RelEnables},
		{"water", "ecosystems", RelEnables},
		{"drinking water", "human health", RelEnables},
		{"sanitation", "disease prevention", RelEnables},
		{"agriculture", "food supply", RelEnables},

		// ── Climate and environment ──
		{"fossil fuels", "carbon dioxide", RelProduces},
		{"carbon dioxide", "greenhouse effect", RelCauses},
		{"greenhouse effect", "global warming", RelCauses},
		{"global warming", "sea level rise", RelCauses},
		{"global warming", "extreme weather", RelCauses},
		{"global warming", "biodiversity loss", RelCauses},
		{"sea level rise", "coastal flooding", RelCauses},
		{"deforestation", "carbon dioxide", RelProduces},
		{"deforestation", "habitat loss", RelCauses},
		{"habitat loss", "species extinction", RelCauses},
		{"renewable energy", "fossil fuels", RelPrevents},
		{"renewable energy", "clean air", RelEnables},

		// ── Economy ──
		{"employment", "income", RelProduces},
		{"income", "consumer spending", RelEnables},
		{"consumer spending", "economic growth", RelCauses},
		{"economic growth", "employment", RelEnables},
		{"inflation", "purchasing power", RelPrevents},
		{"recession", "unemployment", RelCauses},
		{"unemployment", "poverty", RelCauses},
		{"education", "employment", RelEnables},
		{"education", "innovation", RelEnables},
		{"innovation", "economic growth", RelCauses},
		{"trade", "economic growth", RelEnables},
		{"taxation", "public services", RelEnables},
		{"public services", "infrastructure", RelEnables},

		// ── Technology ──
		{"artificial intelligence", "automation", RelEnables},
		{"automation", "productivity", RelEnables},
		{"automation", "job displacement", RelCauses},
		{"machine learning", "pattern recognition", RelEnables},
		{"quantum computing", "cryptography", RelPrevents},
		{"quantum computing", "drug discovery", RelEnables},
		{"blockchain", "decentralization", RelEnables},
		{"blockchain", "cryptocurrency", RelEnables},
		{"5G", "autonomous vehicles", RelEnables},
		{"5G", "internet of things", RelEnables},
		{"semiconductor", "computing", RelEnables},
		{"semiconductor", "smartphones", RelEnables},

		// ── Society ──
		{"democracy", "free speech", RelEnables},
		{"democracy", "voting rights", RelEnables},
		{"free speech", "press freedom", RelEnables},
		{"press freedom", "government accountability", RelEnables},
		{"rule of law", "property rights", RelEnables},
		{"property rights", "investment", RelEnables},
		{"corruption", "economic growth", RelPrevents},
		{"corruption", "trust", RelPrevents},
		{"war", "displacement", RelCauses},
		{"war", "infrastructure destruction", RelCauses},
		{"peace", "trade", RelEnables},
		{"peace", "economic growth", RelEnables},

		// ── Health ──
		{"exercise", "cardiovascular health", RelEnables},
		{"exercise", "mental health", RelEnables},
		{"nutrition", "immune system", RelEnables},
		{"smoking", "lung cancer", RelCauses},
		{"smoking", "heart disease", RelCauses},
		{"vaccination", "disease prevention", RelEnables},
		{"antibiotics", "bacterial infection", RelPrevents},
		{"sleep", "cognitive function", RelEnables},
		{"stress", "mental health", RelPrevents},
		{"pollution", "respiratory disease", RelCauses},

		// ── Transportation ──
		{"oil", "transportation", RelEnables},
		{"oil", "plastics", RelProduces},
		{"transportation", "trade", RelEnables},
		{"transportation", "commuting", RelEnables},
		{"electric vehicles", "oil dependency", RelPrevents},
		{"electric vehicles", "clean air", RelEnables},
		{"railways", "freight transport", RelEnables},
		{"aviation", "global travel", RelEnables},

		// ── Sun and energy ──
		{"sun", "solar energy", RelProduces},
		{"sun", "photosynthesis", RelEnables},
		{"photosynthesis", "plant growth", RelEnables},
		{"plant growth", "food chain", RelEnables},
		{"food chain", "ecosystems", RelEnables},
		{"solar energy", "renewable energy", RelIsA},
		{"wind", "wind energy", RelProduces},
		{"wind energy", "renewable energy", RelIsA},

		// ── Education ──
		{"literacy", "education", RelEnables},
		{"education", "critical thinking", RelEnables},
		{"critical thinking", "informed decisions", RelEnables},
		{"research", "scientific progress", RelEnables},
		{"scientific progress", "technology", RelEnables},
		{"technology", "quality of life", RelEnables},

		// ── Cross-domain cascades ──
		{"pandemic", "economic recession", RelCauses},
		{"pandemic", "healthcare strain", RelCauses},
		{"pandemic", "remote work", RelCauses},
		{"earthquake", "infrastructure damage", RelCauses},
		{"infrastructure damage", "supply chain disruption", RelCauses},
		{"supply chain disruption", "price increases", RelCauses},
		{"volcanic eruption", "climate cooling", RelCauses},
		{"asteroid impact", "mass extinction", RelCauses},
	}
}
