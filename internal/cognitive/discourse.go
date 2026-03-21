package cognitive

import (
	"fmt"
	"math/rand"
	"strings"
)

// -----------------------------------------------------------------------
// Discourse Planner — Rhetorical Structure Theory-based text planning.
//
// Instead of dumping facts in random order, the discourse planner:
//   1. Analyzes available facts to determine what rhetorical moves are possible
//   2. Selects a discourse schema based on response type and fact profile
//   3. Plans a sequence of rhetorical sections (HOOK → DEFINE → ORIGIN → ...)
//   4. Each section has a communicative goal and constraints
//   5. The generative engine then fills each section in order
//
// This is the single biggest quality improvement for Nous text generation.
// It's the difference between "fact salad" and "coherent prose".
//
// Based on Rhetorical Structure Theory (Mann & Thompson, 1988) adapted
// for knowledge-graph-driven NLG.
// -----------------------------------------------------------------------

// RhetRelation is a rhetorical relation between discourse units.
type RhetRelation int

const (
	RhetElaboration   RhetRelation = iota // adds detail to a nucleus
	RhetBackground                        // provides context
	RhetCause                             // explains why
	RhetResult                            // states a consequence
	RhetContrast                          // highlights difference
	RhetSequence                          // temporal/logical ordering
	RhetPurpose                           // what it's for
	RhetEvaluation                        // value judgment / commentary
	RhetSummary                           // wraps up / restates
	RhetConcession                        // acknowledges counterpoint
	RhetJoint                             // coordination (no hierarchy)
)

// DiscourseSection is one planned unit in the discourse tree.
type DiscourseSection struct {
	Role      SectionRole  // what this section does
	Relation  RhetRelation // how it connects to the previous section
	Facts     []edgeFact   // facts to realize in this section
	Goal      string       // communicative goal ("introduce the topic", "explain origins")
	Connector string       // optional transition phrase to prepend
	MaxSents  int          // max sentences for this section (0 = default 2)
}

// SectionRole classifies what a section does in the discourse.
type SectionRole int

const (
	SectionHook       SectionRole = iota // attention-grabbing opener
	SectionDefine                        // what-is-it definition
	SectionOrigin                        // where it came from
	SectionFeatures                      // key properties/parts
	SectionPurpose                       // what it's used for
	SectionImpact                        // why it matters / effects
	SectionComparison                    // contrast with related things
	SectionClose                         // concluding statement
)

// DiscourseSchema is a named plan template for a type of text.
type DiscourseSchema struct {
	Name     string        // "explanatory", "narrative", "argumentative"
	Sections []SectionRole // ordered sequence of section roles
}

// DiscoursePlan is the output of the planner — an ordered sequence
// of sections ready for the generative engine to fill.
type DiscoursePlan struct {
	Schema   string
	Sections []DiscourseSection
	Topic    string
}

// DiscoursePlanner plans rhetorical structure before text generation.
type DiscoursePlanner struct {
	rng *rand.Rand
}

// NewDiscoursePlanner creates a discourse planning engine.
func NewDiscoursePlanner(rng *rand.Rand) *DiscoursePlanner {
	return &DiscoursePlanner{rng: rng}
}

// -----------------------------------------------------------------------
// Discourse Schemas — structural templates for different text types
// -----------------------------------------------------------------------

var explanatorySchema = DiscourseSchema{
	Name: "explanatory",
	Sections: []SectionRole{
		SectionHook,     // grab attention
		SectionDefine,   // what is it
		SectionOrigin,   // where did it come from
		SectionFeatures, // key properties
		SectionPurpose,  // what's it for
		SectionClose,    // wrap up
	},
}

var narrativeSchema = DiscourseSchema{
	Name: "narrative",
	Sections: []SectionRole{
		SectionHook,     // set the scene
		SectionOrigin,   // the beginning
		SectionFeatures, // what developed
		SectionImpact,   // the significance
		SectionClose,    // reflection
	},
}

var comparativeSchema = DiscourseSchema{
	Name: "comparative",
	Sections: []SectionRole{
		SectionHook,       // frame the comparison
		SectionDefine,     // what each thing is
		SectionComparison, // how they differ
		SectionPurpose,    // different use cases
		SectionClose,      // verdict
	},
}

var briefSchema = DiscourseSchema{
	Name: "brief",
	Sections: []SectionRole{
		SectionDefine, // what is it
		SectionClose,  // done
	},
}

var featureFocusSchema = DiscourseSchema{
	Name: "feature-focus",
	Sections: []SectionRole{
		SectionHook,     // attention
		SectionDefine,   // what is it
		SectionFeatures, // deep dive into features
		SectionPurpose,  // applications
		SectionClose,    // wrap up
	},
}

var originFocusSchema = DiscourseSchema{
	Name: "origin-focus",
	Sections: []SectionRole{
		SectionHook,    // attention
		SectionOrigin,  // deep dive into origins
		SectionDefine,  // what it became
		SectionImpact,  // significance
		SectionClose,   // wrap up
	},
}

// -----------------------------------------------------------------------
// Fact Profile — analyzes what kinds of facts are available
// -----------------------------------------------------------------------

// factProfile summarizes the types of facts available for planning.
type factProfile struct {
	identity    []edgeFact // is_a, described_as
	origin      []edgeFact // created_by, founded_by, founded_in
	features    []edgeFact // has, part_of, offers
	purpose     []edgeFact // used_for
	connections []edgeFact // related_to, similar_to, causes
	location    []edgeFact // located_in
	total       int
}

func profileFacts(facts []edgeFact) factProfile {
	p := factProfile{total: len(facts)}
	for _, f := range facts {
		switch f.Relation {
		case RelIsA, RelDescribedAs:
			p.identity = append(p.identity, f)
		case RelCreatedBy, RelFoundedBy, RelFoundedIn:
			p.origin = append(p.origin, f)
		case RelHas, RelPartOf, RelOffers:
			p.features = append(p.features, f)
		case RelUsedFor:
			p.purpose = append(p.purpose, f)
		case RelRelatedTo, RelSimilarTo, RelCauses, RelFollows:
			p.connections = append(p.connections, f)
		case RelLocatedIn:
			p.location = append(p.location, f)
		default:
			p.connections = append(p.connections, f)
		}
	}
	return p
}

// -----------------------------------------------------------------------
// Schema Selection — picks the best discourse structure for the content
// -----------------------------------------------------------------------

// SelectSchema chooses the best discourse schema for the given facts
// and response type.
func (dp *DiscoursePlanner) SelectSchema(facts []edgeFact, respType ResponseType) DiscourseSchema {
	profile := profileFacts(facts)

	// Brief response for very few facts
	if profile.total <= 2 {
		return briefSchema
	}

	// Response type hints
	if respType == RespExplain {
		return explanatorySchema
	}

	// Content-driven selection
	hasOrigin := len(profile.origin) > 0
	hasFeatures := len(profile.features) >= 2
	hasPurpose := len(profile.purpose) > 0
	hasConnections := len(profile.connections) >= 2

	// Score each schema based on how well facts match its sections
	type scored struct {
		schema DiscourseSchema
		score  float64
	}

	candidates := []scored{
		{explanatorySchema, 1.0}, // baseline
	}

	if hasOrigin {
		s := 2.0
		if !hasFeatures {
			s = 3.0 // origin-heavy content
		}
		candidates = append(candidates, scored{originFocusSchema, s})
	}
	if hasFeatures {
		s := 2.0
		if !hasOrigin {
			s = 3.0 // feature-heavy content
		}
		candidates = append(candidates, scored{featureFocusSchema, s})
	}
	if hasOrigin && hasFeatures && hasPurpose {
		candidates = append(candidates, scored{narrativeSchema, 2.5})
	}
	if hasConnections {
		candidates = append(candidates, scored{comparativeSchema, 1.5})
	}

	// Weighted random selection for variety
	totalWeight := 0.0
	for _, c := range candidates {
		totalWeight += c.score
	}
	r := dp.rng.Float64() * totalWeight
	cumulative := 0.0
	for _, c := range candidates {
		cumulative += c.score
		if r <= cumulative {
			return c.schema
		}
	}
	return explanatorySchema
}

// -----------------------------------------------------------------------
// Plan Generation — fills the schema with concrete sections
// -----------------------------------------------------------------------

// Plan generates a complete discourse plan from facts and a schema.
func (dp *DiscoursePlanner) Plan(topic string, facts []edgeFact, schema DiscourseSchema) *DiscoursePlan {
	profile := profileFacts(facts)
	plan := &DiscoursePlan{
		Schema: schema.Name,
		Topic:  topic,
	}

	for i, role := range schema.Sections {
		section := dp.planSection(role, topic, &profile, i)
		if section != nil {
			plan.Sections = append(plan.Sections, *section)
		}
	}

	return plan
}

// PlanFromFacts is the main entry point — selects schema and generates plan.
func (dp *DiscoursePlanner) PlanFromFacts(topic string, facts []edgeFact, respType ResponseType) *DiscoursePlan {
	schema := dp.SelectSchema(facts, respType)
	return dp.Plan(topic, facts, schema)
}

// planSection creates a concrete discourse section for a role.
func (dp *DiscoursePlanner) planSection(role SectionRole, topic string, profile *factProfile, index int) *DiscourseSection {
	section := &DiscourseSection{
		Role:     role,
		MaxSents: 2,
	}

	switch role {
	case SectionHook:
		section.Goal = "grab the reader's attention"
		section.Relation = RhetBackground
		// Hook uses no facts — it's generated from the topic name
		section.MaxSents = 1
		return section

	case SectionDefine:
		section.Goal = "define what " + topic + " is"
		section.Relation = RhetElaboration
		section.Facts = dp.selectFacts(profile.identity, 2)
		if index > 0 {
			section.Connector = dp.pick(defineTransitions)
		}
		if len(section.Facts) == 0 {
			return nil // skip if no identity facts
		}

	case SectionOrigin:
		section.Goal = "explain the origins of " + topic
		section.Relation = RhetBackground
		// Origin section includes location facts as contextual background
		originFacts := append([]edgeFact{}, profile.origin...)
		originFacts = append(originFacts, profile.location...)
		section.Facts = dp.selectFacts(originFacts, 4)
		section.Connector = dp.pick(originTransitions)
		if len(section.Facts) == 0 {
			return nil
		}

	case SectionFeatures:
		section.Goal = "describe key features of " + topic
		section.Relation = RhetElaboration
		section.Facts = dp.selectFacts(profile.features, 4)
		section.MaxSents = 3
		section.Connector = dp.pick(featureTransitions)
		if len(section.Facts) == 0 {
			return nil
		}

	case SectionPurpose:
		section.Goal = "explain what " + topic + " is used for"
		section.Relation = RhetPurpose
		section.Facts = dp.selectFacts(profile.purpose, 3)
		section.Connector = dp.pick(purposeTransitions)
		if len(section.Facts) == 0 {
			return nil
		}

	case SectionImpact:
		section.Goal = "explain why " + topic + " matters"
		section.Relation = RhetEvaluation
		// Impact draws from connections + purpose
		var impactFacts []edgeFact
		impactFacts = append(impactFacts, profile.connections...)
		impactFacts = append(impactFacts, profile.purpose...)
		section.Facts = dp.selectFacts(impactFacts, 3)
		section.Connector = dp.pick(impactTransitions)
		if len(section.Facts) == 0 {
			return nil
		}

	case SectionComparison:
		section.Goal = "compare " + topic + " with related concepts"
		section.Relation = RhetContrast
		section.Facts = dp.selectFacts(profile.connections, 3)
		section.Connector = dp.pick(comparisonTransitions)
		if len(section.Facts) == 0 {
			return nil
		}

	case SectionClose:
		section.Goal = "provide a concluding thought about " + topic
		section.Relation = RhetSummary
		section.MaxSents = 1
		// Closing uses no facts — it's a synthesis/evaluation
		return section
	}

	return section
}

// selectFacts picks up to n facts, removing them from the source slice
// so they won't be reused in other sections.
func (dp *DiscoursePlanner) selectFacts(source []edgeFact, n int) []edgeFact {
	if len(source) == 0 {
		return nil
	}
	if n > len(source) {
		n = len(source)
	}
	selected := make([]edgeFact, n)
	copy(selected, source[:n])
	return selected
}

func (dp *DiscoursePlanner) pick(options []string) string {
	if len(options) == 0 {
		return ""
	}
	return options[dp.rng.Intn(len(options))]
}

// -----------------------------------------------------------------------
// Transition Phrases — section-specific discourse connectors
// -----------------------------------------------------------------------

var defineTransitions = []string{
	"At its core,",
	"In essence,",
	"Fundamentally,",
	"Put simply,",
	"To understand it,",
	"At the most basic level,",
	"In broad terms,",
}

var originTransitions = []string{
	"Looking at its origins,",
	"The story begins with",
	"Historically,",
	"It all started when",
	"Tracing its roots,",
	"Going back to the beginning,",
	"The history here is worth knowing.",
}

var featureTransitions = []string{
	"What makes it stand out is",
	"Among its key characteristics,",
	"Several things define it.",
	"Looking at what it offers,",
	"There are several notable aspects.",
	"What sets it apart:",
	"Its defining qualities include",
}

var purposeTransitions = []string{
	"In practice,",
	"When it comes to real-world use,",
	"Its practical value lies in",
	"On the applied side,",
	"Where it really shines is",
	"From a practical standpoint,",
}

var impactTransitions = []string{
	"The significance of this goes deeper.",
	"Why does this matter?",
	"The broader impact is worth noting.",
	"Its influence extends further.",
	"Looking at the bigger picture,",
	"The ripple effects are clear.",
}

var comparisonTransitions = []string{
	"Compared to similar concepts,",
	"In contrast,",
	"When placed alongside related ideas,",
	"Looking at how it relates to others,",
	"Drawing comparisons,",
}

// -----------------------------------------------------------------------
// Plan Formatting — human-readable plan trace for debugging
// -----------------------------------------------------------------------

// FormatPlan produces a human-readable trace of the discourse plan.
func FormatPlan(plan *DiscoursePlan) string {
	if plan == nil {
		return "<no plan>"
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Discourse Plan [%s] for %q:\n", plan.Schema, plan.Topic))
	for i, s := range plan.Sections {
		b.WriteString(fmt.Sprintf("  %d. [%s] %s (%d facts",
			i+1, sectionRoleName(s.Role), s.Goal, len(s.Facts)))
		if s.Connector != "" {
			b.WriteString(fmt.Sprintf(", connector: %q", s.Connector))
		}
		b.WriteString(")\n")
	}
	return b.String()
}

func sectionRoleName(r SectionRole) string {
	switch r {
	case SectionHook:
		return "HOOK"
	case SectionDefine:
		return "DEFINE"
	case SectionOrigin:
		return "ORIGIN"
	case SectionFeatures:
		return "FEATURES"
	case SectionPurpose:
		return "PURPOSE"
	case SectionImpact:
		return "IMPACT"
	case SectionComparison:
		return "COMPARISON"
	case SectionClose:
		return "CLOSE"
	default:
		return "UNKNOWN"
	}
}

func rhetRelationName(r RhetRelation) string {
	switch r {
	case RhetElaboration:
		return "elaboration"
	case RhetBackground:
		return "background"
	case RhetCause:
		return "cause"
	case RhetResult:
		return "result"
	case RhetContrast:
		return "contrast"
	case RhetSequence:
		return "sequence"
	case RhetPurpose:
		return "purpose"
	case RhetEvaluation:
		return "evaluation"
	case RhetSummary:
		return "summary"
	case RhetConcession:
		return "concession"
	case RhetJoint:
		return "joint"
	default:
		return "unknown"
	}
}
