package cognitive

import (
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------
// NLG Engine — structural natural language generation from knowledge facts.
//
// This engine produces flowing prose from edgeFact slices by performing
// real linguistic operations rather than filling printf templates:
//
//   - Sentence fusion: combine related facts into complex sentences
//     using relative clauses, participial phrases, and coordination.
//   - Aggregation: group same-type facts and express them as lists
//     within a single clause rather than repeating the subject.
//   - Referring expressions: pronouns, definite descriptions, and
//     zero-anaphora to avoid robotic subject repetition.
//   - Discourse planning: semantic connectors chosen by the relationship
//     between adjacent fact groups, not generic filler words.
//   - Information structure: given-before-new ordering so that each
//     sentence builds on what the reader already knows.
//
// The pipeline: group -> order -> fuse -> connect -> pronominalize -> polish.
// -----------------------------------------------------------------------

// semanticRole classifies a fact group by its communicative function.
type semanticRole int

const (
	roleIdentity  semanticRole = iota // IsA, DescribedAs, KnownFor — what the thing IS
	roleOrigin                        // CreatedBy, FoundedBy, FoundedIn — where it came from
	roleProperties                    // Has, Offers — what it has or provides
	roleUsage                         // UsedFor — what it's for
	roleRelations                     // RelatedTo, PartOf, SimilarTo — connections
	roleLocation                      // LocatedIn — where it is
)

// factGroup bundles facts that share a semantic role.
type factGroup struct {
	Role  semanticRole
	Facts []edgeFact
}

// NLGEngine generates flowing natural language from structured knowledge.
// Instead of template filling, it uses linguistic operations:
//   - Sentence fusion: combine related facts into complex sentences
//   - Aggregation: group facts by type and express concisely
//   - Referring expressions: pronouns, synonyms, definite descriptions
//   - Discourse planning: semantic connectors based on fact relationships
//   - Information structure: given->new ordering for natural flow
type NLGEngine struct{}

// NewNLGEngine creates an NLG engine ready for realization.
func NewNLGEngine() *NLGEngine {
	return &NLGEngine{}
}

// -----------------------------------------------------------------------
// Realize — main entry point
// -----------------------------------------------------------------------

// Realize takes a subject and its facts and returns flowing prose.
// Pipeline: group -> order -> fuse (position-aware) -> join -> pronominalize -> polish.
func (e *NLGEngine) Realize(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	groups := e.groupFacts(facts)
	groups = e.orderGroups(groups)

	// Fuse each group with awareness of its position in the discourse.
	// The first group (identity) uses the full subject name. Subsequent
	// groups use pronouns, participial phrases, or semantic openers to
	// avoid robotic repetition and create natural flow.
	var sentences []string
	for i, g := range groups {
		isFirst := (i == 0)
		text := e.fuseGroupPositional(subject, g, isFirst, facts)
		if text != "" {
			sentences = append(sentences, text)
		}
	}

	if len(sentences) == 0 {
		return ""
	}

	joined := strings.Join(sentences, " ")
	pronominalized := e.pronominalize(joined, subject, facts)
	polished := e.polish(pronominalized)
	return polished
}

// fusedSegment is the output of fusing one group, carrying its role
// so that connectGroups can pick semantically appropriate connectors.
type fusedSegment struct {
	role semanticRole
	text string
}

// -----------------------------------------------------------------------
// groupFacts — classify facts into semantic roles
// -----------------------------------------------------------------------

func (e *NLGEngine) groupFacts(facts []edgeFact) []factGroup {
	buckets := make(map[semanticRole][]edgeFact)

	for _, f := range facts {
		role := e.classifyRole(f.Relation)
		buckets[role] = append(buckets[role], f)
	}

	var groups []factGroup
	for role, fs := range buckets {
		groups = append(groups, factGroup{Role: role, Facts: fs})
	}
	return groups
}

func (e *NLGEngine) classifyRole(rel RelType) semanticRole {
	switch rel {
	case RelIsA, RelDescribedAs, RelKnownFor:
		return roleIdentity
	case RelCreatedBy, RelFoundedBy, RelFoundedIn:
		return roleOrigin
	case RelHas, RelOffers:
		return roleProperties
	case RelUsedFor:
		return roleUsage
	case RelRelatedTo, RelPartOf, RelSimilarTo, RelOppositeOf,
		RelCauses, RelFollows, RelInfluencedBy, RelDerivedFrom:
		return roleRelations
	case RelLocatedIn:
		return roleLocation
	default:
		return roleRelations
	}
}

// -----------------------------------------------------------------------
// orderGroups — information structure: identity first, then origin, etc.
// -----------------------------------------------------------------------

func (e *NLGEngine) orderGroups(groups []factGroup) []factGroup {
	// Canonical order: identity -> origin -> location -> properties -> usage -> relations
	order := []semanticRole{
		roleIdentity,
		roleOrigin,
		roleLocation,
		roleProperties,
		roleUsage,
		roleRelations,
	}

	byRole := make(map[semanticRole]factGroup)
	for _, g := range groups {
		byRole[g.Role] = g
	}

	var ordered []factGroup
	for _, r := range order {
		if g, ok := byRole[r]; ok {
			ordered = append(ordered, g)
		}
	}
	return ordered
}

// -----------------------------------------------------------------------
// fuseGroup — the core NLG: build complex sentences from fact clusters
// -----------------------------------------------------------------------

func (e *NLGEngine) fuseGroup(subject string, group factGroup) string {
	return e.fuseGroupPositional(subject, group, true, nil)
}

// fuseGroupPositional fuses a fact group with awareness of its discourse
// position. When isFirst is true, the group leads with the full subject.
// When false, it uses pronouns, participial phrases, or semantic openers
// to create natural flow from the preceding group.
func (e *NLGEngine) fuseGroupPositional(subject string, group factGroup, isFirst bool, allFacts []edgeFact) string {
	if len(group.Facts) == 0 {
		return ""
	}

	// Determine the pronoun for continuation sentences
	pronoun := "it"
	if !isFirst && allFacts != nil {
		pronoun = e.genderPronoun(subject, allFacts)
	}

	switch group.Role {
	case roleIdentity:
		return e.fuseIdentity(subject, group.Facts)
	case roleOrigin:
		return e.fuseOrigin(subject, group.Facts)
	case roleProperties:
		if isFirst {
			return e.fuseProperties(subject, group.Facts)
		}
		return e.fusePropertiesContinuation(pronoun, group.Facts)
	case roleUsage:
		if isFirst {
			return e.fuseUsage(subject, group.Facts)
		}
		return e.fuseUsageContinuation(pronoun, group.Facts)
	case roleRelations:
		if isFirst {
			return e.fuseRelations(subject, group.Facts)
		}
		return e.fuseRelationsContinuation(pronoun, group.Facts)
	case roleLocation:
		return e.fuseLocation(subject, group.Facts)
	default:
		return e.fuseGeneric(subject, group.Facts)
	}
}

// fuseIdentity combines IsA, DescribedAs, and KnownFor facts into a
// complex noun phrase with relative clauses and participial modifiers.
//
// 1 IsA:                    "Python is a programming language."
// 1 IsA + DescribedAs:      "Python is a programming language described as versatile."
// 1 IsA + KnownFor:         "Python is a programming language known for its simplicity."
// 1 IsA + DescribedAs + KnownFor: "Python is a programming language, described as
//
//	versatile and known for its simplicity."
func (e *NLGEngine) fuseIdentity(subject string, facts []edgeFact) string {
	var isAFacts []edgeFact
	var describedAs []edgeFact
	var knownFor []edgeFact

	for _, f := range facts {
		switch f.Relation {
		case RelIsA:
			isAFacts = append(isAFacts, f)
		case RelDescribedAs:
			describedAs = append(describedAs, f)
		case RelKnownFor:
			knownFor = append(knownFor, f)
		}
	}

	var b strings.Builder
	sub := nlgCapFirst(subject)

	// Start with the primary IsA classification
	if len(isAFacts) > 0 {
		b.WriteString(sub)
		b.WriteString(" is ")
		b.WriteString(articleFor(strings.TrimSpace(isAFacts[0].Object)))

		// If there are additional IsA facts, attach with "and" coordination
		for i := 1; i < len(isAFacts); i++ {
			if i == len(isAFacts)-1 {
				b.WriteString(" and ")
			} else {
				b.WriteString(", ")
			}
			b.WriteString(articleFor(strings.TrimSpace(isAFacts[i].Object)))
		}
	} else if len(describedAs) > 0 {
		// No IsA — lead with description
		b.WriteString(sub)
		b.WriteString(" is ")
		b.WriteString(strings.TrimSpace(describedAs[0].Object))
		describedAs = describedAs[1:]
	} else if len(knownFor) > 0 {
		b.WriteString(sub)
		b.WriteString(" is known for ")
		b.WriteString(strings.TrimSpace(knownFor[0].Object))
		knownFor = knownFor[1:]
		if len(knownFor) == 0 {
			b.WriteString(".")
			return b.String()
		}
	}

	// Attach participial phrases for remaining modifiers
	var modifiers []string
	for _, f := range describedAs {
		modifiers = append(modifiers, "described as "+strings.TrimSpace(f.Object))
	}
	for _, f := range knownFor {
		modifiers = append(modifiers, "known for "+strings.TrimSpace(f.Object))
	}

	if len(modifiers) > 0 {
		if len(isAFacts) > 0 || b.Len() > 0 {
			b.WriteString(", ")
		}
		b.WriteString(joinCoordinated(modifiers))
	}

	b.WriteString(".")
	return b.String()
}

// fuseOrigin combines CreatedBy, FoundedBy, and FoundedIn into a single
// origin sentence using participial phrases.
//
// CreatedBy + FoundedIn: "Created by Guido van Rossum in 1991, ..."
// FoundedBy + FoundedIn: "Founded by Larry Page and Sergey Brin in 1998, ..."
// Just FoundedIn:        "Established in 1991, ..."
func (e *NLGEngine) fuseOrigin(subject string, facts []edgeFact) string {
	var creators []string
	var founders []string
	var dates []string

	for _, f := range facts {
		obj := strings.TrimSpace(f.Object)
		switch f.Relation {
		case RelCreatedBy:
			creators = append(creators, obj)
		case RelFoundedBy:
			founders = append(founders, obj)
		case RelFoundedIn:
			dates = append(dates, obj)
		}
	}

	var b strings.Builder

	// Build the participial origin clause
	hasAgent := len(creators) > 0 || len(founders) > 0
	hasDate := len(dates) > 0

	if hasAgent || hasDate {
		// Choose verb: "created by" vs "founded by"
		if len(creators) > 0 {
			b.WriteString("Created by ")
			b.WriteString(joinCoordinated(creators))
		} else if len(founders) > 0 {
			b.WriteString("Founded by ")
			b.WriteString(joinCoordinated(founders))
		}

		if hasDate {
			if hasAgent {
				b.WriteString(" in ")
			} else {
				b.WriteString("Established in ")
			}
			b.WriteString(joinCoordinated(dates))
		}

		// Close with a continuation that references the actual origin.
		b.WriteString(", ")
		b.WriteString(subject)
		if hasDate && hasAgent {
			b.WriteString(" has continued to develop since then.")
		} else if hasDate {
			b.WriteString(" has continued to develop since " + dates[0] + ".")
		} else {
			b.WriteString(" continues to build on that foundation.")
		}
	}

	return b.String()
}

// fuseProperties aggregates Has and Offers facts into a list sentence.
//
// 1 fact:  "It features readable syntax."
// N facts: "It features readable syntax, dynamic typing, and extensive libraries."
func (e *NLGEngine) fuseProperties(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	objects := make([]string, 0, len(facts))
	for _, f := range facts {
		objects = append(objects, strings.TrimSpace(f.Object))
	}

	var b strings.Builder
	b.WriteString(nlgCapFirst(subject))
	switch len(objects) {
	case 1:
		b.WriteString(" features ")
		b.WriteString(objects[0])
	case 2:
		b.WriteString(" has ")
		b.WriteString(joinCoordinated(objects))
	default:
		b.WriteString(" includes ")
		b.WriteString(joinCoordinated(objects))
	}
	b.WriteString(".")
	return b.String()
}

// fusePropertiesContinuation generates a properties sentence that flows
// naturally after a preceding identity or origin sentence. Uses "it" or
// a participial phrase instead of repeating the subject.
func (e *NLGEngine) fusePropertiesContinuation(pronoun string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	objects := make([]string, 0, len(facts))
	for _, f := range facts {
		objects = append(objects, strings.TrimSpace(f.Object))
	}

	var b strings.Builder
	b.WriteString(nlgCapFirst(pronoun))
	switch len(objects) {
	case 1:
		b.WriteString(" has ")
		b.WriteString(objects[0])
	case 2:
		b.WriteString(" has ")
		b.WriteString(joinCoordinated(objects))
	default:
		b.WriteString(" includes ")
		b.WriteString(joinCoordinated(objects))
	}
	b.WriteString(".")
	return b.String()
}

// fuseUsageContinuation generates a usage sentence that flows naturally
// after preceding text, starting with a pronominal reference.
func (e *NLGEngine) fuseUsageContinuation(pronoun string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	objects := make([]string, 0, len(facts))
	for _, f := range facts {
		objects = append(objects, strings.TrimSpace(f.Object))
	}

	var b strings.Builder
	b.WriteString(nlgCapFirst(pronoun))
	switch len(objects) {
	case 1:
		b.WriteString(" is used for ")
	case 2:
		b.WriteString(" is used for ")
	default:
		b.WriteString(" is applied in ")
	}
	b.WriteString(joinCoordinated(objects))
	b.WriteString(".")
	return b.String()
}

// fuseRelationsContinuation generates a relations sentence that flows
// after preceding text.
func (e *NLGEngine) fuseRelationsContinuation(pronoun string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Sub-group by relation
	byRel := make(map[RelType][]string)
	relOrder := make([]RelType, 0)
	for _, f := range facts {
		if _, seen := byRel[f.Relation]; !seen {
			relOrder = append(relOrder, f.Relation)
		}
		byRel[f.Relation] = append(byRel[f.Relation], strings.TrimSpace(f.Object))
	}

	var sentences []string
	for _, rel := range relOrder {
		objs := byRel[rel]
		s := e.buildRelationClauseContinuation(pronoun, rel, objs)
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	return strings.Join(sentences, " ")
}

// buildRelationClauseContinuation builds a relation clause starting
// with "it" or a participial phrase rather than the full subject.
func (e *NLGEngine) buildRelationClauseContinuation(pronoun string, rel RelType, objects []string) string {
	if len(objects) == 0 {
		return ""
	}

	var b strings.Builder
	pro := nlgCapFirst(pronoun)
	list := joinCoordinated(objects)

	switch rel {
	case RelRelatedTo:
		b.WriteString(pro)
		b.WriteString(" relates to ")
		b.WriteString(list)
	case RelPartOf:
		b.WriteString(pro)
		b.WriteString(" is part of ")
		b.WriteString(list)
	case RelSimilarTo:
		b.WriteString(pro)
		b.WriteString(" is similar to ")
		b.WriteString(list)
	case RelOppositeOf:
		b.WriteString(pro)
		b.WriteString(" contrasts with ")
		b.WriteString(list)
	case RelCauses:
		b.WriteString(pro)
		b.WriteString(" leads to ")
		b.WriteString(list)
	case RelFollows:
		b.WriteString(pro)
		b.WriteString(" follows from ")
		b.WriteString(list)
	case RelInfluencedBy:
		b.WriteString(pro)
		b.WriteString(" is influenced by ")
		b.WriteString(list)
	case RelDerivedFrom:
		b.WriteString(pro)
		b.WriteString(" derives from ")
		b.WriteString(list)
	default:
		b.WriteString(pro)
		b.WriteString(" is connected to ")
		b.WriteString(list)
	}
	b.WriteString(".")
	return b.String()
}

// fuseUsage aggregates UsedFor facts into a single sentence with a list.
//
// 1 fact:  "It is widely used for data science."
// N facts: "It is widely used for data science, web development, and automation."
func (e *NLGEngine) fuseUsage(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	objects := make([]string, 0, len(facts))
	for _, f := range facts {
		objects = append(objects, strings.TrimSpace(f.Object))
	}

	var b strings.Builder
	b.WriteString(nlgCapFirst(subject))
	switch len(objects) {
	case 1:
		b.WriteString(" is used for ")
	case 2:
		b.WriteString(" is used for ")
	default:
		b.WriteString(" is applied in ")
	}
	b.WriteString(joinCoordinated(objects))
	b.WriteString(".")
	return b.String()
}

// fuseRelations produces sentences about connections, using varied verb
// phrases chosen by the specific relation type.
func (e *NLGEngine) fuseRelations(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Sub-group by relation to fuse same-type relations together
	byRel := make(map[RelType][]string)
	relOrder := make([]RelType, 0)
	for _, f := range facts {
		if _, seen := byRel[f.Relation]; !seen {
			relOrder = append(relOrder, f.Relation)
		}
		byRel[f.Relation] = append(byRel[f.Relation], strings.TrimSpace(f.Object))
	}

	var sentences []string
	for _, rel := range relOrder {
		objs := byRel[rel]
		s := e.buildRelationClause(subject, rel, objs)
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	return strings.Join(sentences, " ")
}

// buildRelationClause constructs a clause for a specific relation type
// and its objects, using a verb phrase appropriate to the semantics.
func (e *NLGEngine) buildRelationClause(subject string, rel RelType, objects []string) string {
	if len(objects) == 0 {
		return ""
	}

	var b strings.Builder
	sub := nlgCapFirst(subject)
	list := joinCoordinated(objects)

	switch rel {
	case RelRelatedTo:
		b.WriteString(sub)
		b.WriteString(" relates to ")
		b.WriteString(list)
	case RelPartOf:
		b.WriteString(sub)
		b.WriteString(" is part of ")
		b.WriteString(list)
	case RelSimilarTo:
		b.WriteString(sub)
		b.WriteString(" is similar to ")
		b.WriteString(list)
	case RelOppositeOf:
		b.WriteString(sub)
		b.WriteString(" contrasts with ")
		b.WriteString(list)
	case RelCauses:
		b.WriteString(sub)
		b.WriteString(" leads to ")
		b.WriteString(list)
	case RelFollows:
		b.WriteString(sub)
		b.WriteString(" follows from ")
		b.WriteString(list)
	case RelInfluencedBy:
		b.WriteString(sub)
		b.WriteString(" is influenced by ")
		b.WriteString(list)
	case RelDerivedFrom:
		b.WriteString(sub)
		b.WriteString(" derives from ")
		b.WriteString(list)
	default:
		b.WriteString(sub)
		b.WriteString(" is connected to ")
		b.WriteString(list)
	}
	b.WriteString(".")
	return b.String()
}

// fuseLocation combines LocatedIn facts.
func (e *NLGEngine) fuseLocation(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	locations := make([]string, 0, len(facts))
	for _, f := range facts {
		locations = append(locations, strings.TrimSpace(f.Object))
	}

	var b strings.Builder
	b.WriteString(nlgCapFirst(subject))
	if len(locations) == 1 {
		b.WriteString(" is based in ")
	} else {
		b.WriteString(" has a presence in ")
	}
	b.WriteString(joinCoordinated(locations))
	b.WriteString(".")
	return b.String()
}

// fuseGeneric is the fallback for any unrecognized role.
func (e *NLGEngine) fuseGeneric(subject string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	// Aggregate all objects into a single sentence referencing the content.
	objects := make([]string, 0, len(facts))
	for _, f := range facts {
		objects = append(objects, strings.TrimSpace(f.Object))
	}
	var b strings.Builder
	b.WriteString(nlgCapFirst(subject))
	b.WriteString(" relates to ")
	b.WriteString(joinCoordinated(objects))
	b.WriteString(".")
	return b.String()
}

// -----------------------------------------------------------------------
// connectGroups — join fused segments with discourse connectors
// -----------------------------------------------------------------------

func (e *NLGEngine) connectGroups(segments []fusedSegment) string {
	if len(segments) == 0 {
		return ""
	}
	if len(segments) == 1 {
		return segments[0].text
	}

	var b strings.Builder
	b.WriteString(segments[0].text)

	for i := 1; i < len(segments); i++ {
		b.WriteString(" ")
		seg := segments[i].text
		connector := e.discourseConnector(segments[i-1].role, segments[i].role)
		if connector == "" {
			b.WriteString(seg)
		} else {
			// Connector is a standalone transition sentence.
			// Append it, then the segment text unchanged.
			b.WriteString(connector)
			b.WriteString(" ")
			b.WriteString(seg)
		}
	}

	return b.String()
}

// discourseConnector selects a semantically appropriate transition phrase
// based on the roles of two adjacent groups. Returns empty string for
// transitions that flow naturally without a connector.
func (e *NLGEngine) discourseConnector(from, to semanticRole) string {
	switch {
	// Identity -> Origin flows naturally (the origin sentence already
	// starts with a participial phrase like "Created by...")
	case from == roleIdentity && to == roleOrigin:
		return ""

	// Identity -> Properties: the continuation methods handle the transition
	case from == roleIdentity && to == roleProperties:
		return ""

	// Identity -> Usage: natural flow
	case from == roleIdentity && to == roleUsage:
		return ""

	// Origin -> Properties: natural flow
	case from == roleOrigin && to == roleProperties:
		return ""

	// Origin -> Usage: natural flow
	case from == roleOrigin && to == roleUsage:
		return ""

	// Properties -> Usage: natural flow
	case from == roleProperties && to == roleUsage:
		return ""

	// Any -> Location: natural flow
	case to == roleLocation:
		return ""

	// Any -> Relations: natural flow
	case to == roleRelations:
		return ""

	default:
		return ""
	}
}

// -----------------------------------------------------------------------
// pronominalize — replace repeated subject mentions with referring exprs
// -----------------------------------------------------------------------

// pronominalize replaces repeated subject mentions with pronouns and
// definite descriptions, cycling through forms to create natural text.
//
// Occurrence 1: full name       ("Python is a ...")
// Occurrence 2: pronoun         ("it", "he", "she")
// Occurrence 3: description     ("the language", "the framework")
// Occurrence 4+: alternate      (pronoun, description, pronoun, ...)
func (e *NLGEngine) pronominalize(text, subject string, facts []edgeFact) string {
	if subject == "" || text == "" {
		return text
	}

	pronoun := e.genderPronoun(subject, facts)
	description := e.definitDescription(subject, facts)

	// We operate on whole-word occurrences of the subject to avoid
	// replacing partial matches (e.g. "Go" inside "Google").
	// Split into segments around the subject.
	occurrences := nlgCountSubject(text, subject)
	if occurrences <= 1 {
		return text
	}

	var result strings.Builder
	remaining := text
	count := 0

	for {
		idx := nlgFindSubject(remaining, subject)
		if idx < 0 {
			result.WriteString(remaining)
			break
		}

		// Write everything before this occurrence
		result.WriteString(remaining[:idx])
		count++

		// Determine the replacement
		var replacement string
		switch {
		case count == 1:
			// First mention: keep the full name
			replacement = subject
		case count == 2:
			// Second mention: pronoun
			replacement = pronoun
		case count == 3:
			// Third mention: definite description
			replacement = description
		default:
			// Alternate between pronoun and description
			if count%2 == 0 {
				replacement = pronoun
			} else {
				replacement = description
			}
		}

		// Preserve capitalization: if the original was at sentence start,
		// capitalize the replacement
		if idx == 0 || (idx >= 2 && remaining[idx-2] == '.' && remaining[idx-1] == ' ') {
			replacement = nlgCapFirst(replacement)
		}

		result.WriteString(replacement)
		remaining = remaining[idx+len(subject):]
	}

	return result.String()
}

// genderPronoun determines the appropriate pronoun for a subject based
// on its name and facts. Uses the existing gender detection infrastructure.
func (e *NLGEngine) genderPronoun(subject string, facts []edgeFact) string {
	// Check if it's a person via IsA facts
	category := ""
	for _, f := range facts {
		if f.Relation == RelIsA {
			category = strings.ToLower(f.Object)
			break
		}
	}

	// If category suggests a group/plural entity, use "they"
	if isPlural(category) {
		return "they"
	}

	// If category suggests a person, use gendered pronoun
	if isPerson(inferCategoryFromString(category)) {
		g := detectGender(subject)
		switch g {
		case GenderFemale:
			return "she"
		case GenderMale:
			return "he"
		default:
			return "he"
		}
	}

	// Default: inanimate thing
	return "it"
}

// definitDescription builds a definite description like "the language"
// from the subject's IsA facts. Falls back to "the" + lowercased last
// word of subject for multi-word subjects.
func (e *NLGEngine) definitDescription(subject string, facts []edgeFact) string {
	// Try to extract a short category from IsA facts
	for _, f := range facts {
		if f.Relation == RelIsA {
			cat := extractShortCategory(f.Object)
			if cat != "" {
				return "the " + cat
			}
		}
	}

	// For multi-word subjects, try to use the last word as a description
	words := strings.Fields(subject)
	if len(words) > 1 {
		return "the " + strings.ToLower(words[len(words)-1])
	}

	// Fallback: "it"
	return "it"
}

// -----------------------------------------------------------------------
// isProperNoun — check if a word is a proper noun
// -----------------------------------------------------------------------

// isProperNoun returns true if the string begins with an uppercase letter
// and is not a sentence-initial word (heuristic: not preceded by ". ").
func isProperNoun(s string) bool {
	if s == "" {
		return false
	}
	r := []rune(s)
	return unicode.IsUpper(r[0])
}

// -----------------------------------------------------------------------
// RealizeComparison — generate comparison prose from two sets of facts
// -----------------------------------------------------------------------

// RealizeComparison generates comparison prose for two subjects, focusing
// on shared properties and meaningful differences.
func (e *NLGEngine) RealizeComparison(a, b string, factsA, factsB []edgeFact) string {
	if len(factsA) == 0 && len(factsB) == 0 {
		return ""
	}

	var sections []string

	// Section 1: Brief introduction of each item
	introA := e.realizeIntro(a, factsA)
	introB := e.realizeIntro(b, factsB)

	if introA != "" && introB != "" {
		sections = append(sections, introA+" "+introB)
	} else if introA != "" {
		sections = append(sections, introA)
	} else if introB != "" {
		sections = append(sections, introB)
	}

	// Section 2: Shared properties
	shared := e.findSharedProperties(factsA, factsB)
	if len(shared) > 0 {
		var b2 strings.Builder
		b2.WriteString("Both ")
		b2.WriteString(a)
		b2.WriteString(" and ")
		b2.WriteString(b)
		b2.WriteString(" share ")
		b2.WriteString(joinCoordinated(shared))
		b2.WriteString(".")
		sections = append(sections, b2.String())
	}

	// Section 3: Differences — reference what each uniquely has.
	diffA, diffB := e.findDifferences(a, b, factsA, factsB)
	if len(diffA) > 0 || len(diffB) > 0 {
		var diffParts []string
		if len(diffA) > 0 {
			var d strings.Builder
			d.WriteString(nlgCapFirst(a))
			d.WriteString(" uniquely offers ")
			d.WriteString(joinCoordinated(diffA))
			d.WriteString(".")
			diffParts = append(diffParts, d.String())
		}
		if len(diffB) > 0 {
			var d strings.Builder
			d.WriteString(nlgCapFirst(b))
			d.WriteString(" uniquely offers ")
			d.WriteString(joinCoordinated(diffB))
			d.WriteString(".")
			diffParts = append(diffParts, d.String())
		}
		sections = append(sections, strings.Join(diffParts, " "))
	}

	return e.polish(strings.Join(sections, " "))
}

// realizeIntro builds a brief fused introduction from IsA + origin facts.
func (e *NLGEngine) realizeIntro(subject string, facts []edgeFact) string {
	var isA string
	var origin string

	for _, f := range facts {
		switch f.Relation {
		case RelIsA:
			if isA == "" {
				isA = strings.TrimSpace(f.Object)
			}
		case RelCreatedBy:
			if origin == "" {
				origin = "created by " + strings.TrimSpace(f.Object)
			}
		case RelFoundedBy:
			if origin == "" {
				origin = "founded by " + strings.TrimSpace(f.Object)
			}
		case RelFoundedIn:
			if origin == "" {
				origin = "established in " + strings.TrimSpace(f.Object)
			}
		}
	}

	if isA == "" && origin == "" {
		return ""
	}

	var b strings.Builder
	b.WriteString(nlgCapFirst(subject))

	if isA != "" {
		b.WriteString(" is ")
		b.WriteString(articleFor(isA))
		if origin != "" {
			b.WriteString(" ")
			b.WriteString(origin)
		}
	} else {
		b.WriteString(" was ")
		b.WriteString(origin)
	}

	b.WriteString(".")
	return b.String()
}

// findSharedProperties returns object labels that appear in both fact sets.
func (e *NLGEngine) findSharedProperties(factsA, factsB []edgeFact) []string {
	setA := make(map[string]bool)
	for _, f := range factsA {
		if f.Relation == RelHas || f.Relation == RelOffers || f.Relation == RelUsedFor {
			setA[strings.ToLower(strings.TrimSpace(f.Object))] = true
		}
	}

	var shared []string
	seen := make(map[string]bool)
	for _, f := range factsB {
		if f.Relation == RelHas || f.Relation == RelOffers || f.Relation == RelUsedFor {
			key := strings.ToLower(strings.TrimSpace(f.Object))
			if setA[key] && !seen[key] {
				shared = append(shared, strings.TrimSpace(f.Object))
				seen[key] = true
			}
		}
	}
	return shared
}

// findDifferences returns properties unique to each subject.
func (e *NLGEngine) findDifferences(a, b string, factsA, factsB []edgeFact) (uniqueA, uniqueB []string) {
	setA := make(map[string]bool)
	setB := make(map[string]bool)

	propertyRels := map[RelType]bool{
		RelHas: true, RelOffers: true, RelUsedFor: true, RelKnownFor: true,
	}

	for _, f := range factsA {
		if propertyRels[f.Relation] {
			setA[strings.ToLower(strings.TrimSpace(f.Object))] = true
		}
	}
	for _, f := range factsB {
		if propertyRels[f.Relation] {
			setB[strings.ToLower(strings.TrimSpace(f.Object))] = true
		}
	}

	seenA := make(map[string]bool)
	for _, f := range factsA {
		if propertyRels[f.Relation] {
			key := strings.ToLower(strings.TrimSpace(f.Object))
			if !setB[key] && !seenA[key] {
				uniqueA = append(uniqueA, strings.TrimSpace(f.Object))
				seenA[key] = true
			}
		}
	}

	seenB := make(map[string]bool)
	for _, f := range factsB {
		if propertyRels[f.Relation] {
			key := strings.ToLower(strings.TrimSpace(f.Object))
			if !setA[key] && !seenB[key] {
				uniqueB = append(uniqueB, strings.TrimSpace(f.Object))
				seenB[key] = true
			}
		}
	}

	return uniqueA, uniqueB
}

// -----------------------------------------------------------------------
// RealizeExplanation — generate explanatory prose in pedagogical order
// -----------------------------------------------------------------------

// RealizeExplanation generates explanation prose following:
// definition -> mechanism -> examples -> caveats.
func (e *NLGEngine) RealizeExplanation(topic string, facts []edgeFact) string {
	if len(facts) == 0 {
		return ""
	}

	var definition []edgeFact
	var mechanism []edgeFact
	var examples []edgeFact
	var caveats []edgeFact

	for _, f := range facts {
		switch f.Relation {
		case RelIsA, RelDescribedAs:
			definition = append(definition, f)
		case RelCreatedBy, RelFoundedBy, RelFoundedIn, RelCauses,
			RelFollows, RelInfluencedBy, RelDerivedFrom:
			mechanism = append(mechanism, f)
		case RelUsedFor, RelHas, RelOffers, RelKnownFor:
			examples = append(examples, f)
		case RelContradicts, RelOppositeOf:
			caveats = append(caveats, f)
		default:
			// PartOf, RelatedTo, SimilarTo, LocatedIn -> mechanism/context
			mechanism = append(mechanism, f)
		}
	}

	var sections []string

	// Definition section
	if len(definition) > 0 {
		defText := e.fuseIdentity(topic, definition)
		if defText != "" {
			sections = append(sections, defText)
		}
	}

	// Mechanism section — how/why it works.
	// No generic connector; the fused sentences flow from the definition.
	if len(mechanism) > 0 {
		groups := e.groupFacts(mechanism)
		groups = e.orderGroups(groups)
		var mechParts []string
		for _, g := range groups {
			t := e.fuseGroup(topic, g)
			if t != "" {
				mechParts = append(mechParts, t)
			}
		}
		if len(mechParts) > 0 {
			sections = append(sections, strings.Join(mechParts, " "))
		}
	}

	// Examples section — practical manifestations.
	// No generic connector; the content speaks for itself.
	if len(examples) > 0 {
		groups := e.groupFacts(examples)
		groups = e.orderGroups(groups)
		var exParts []string
		for _, g := range groups {
			t := e.fuseGroup(topic, g)
			if t != "" {
				exParts = append(exParts, t)
			}
		}
		if len(exParts) > 0 {
			sections = append(sections, strings.Join(exParts, " "))
		}
	}

	// Caveats section — contradictions and opposites.
	// Reference the actual contrasting concept instead of generic filler.
	if len(caveats) > 0 {
		var caveatParts []string
		for _, f := range caveats {
			obj := strings.TrimSpace(f.Object)
			var cb strings.Builder
			switch f.Relation {
			case RelContradicts:
				cb.WriteString(nlgCapFirst(topic))
				cb.WriteString(" contrasts with ")
				cb.WriteString(obj)
			case RelOppositeOf:
				cb.WriteString("Unlike ")
				cb.WriteString(obj)
				cb.WriteString(", ")
				cb.WriteString(topic)
				cb.WriteString(" takes a different approach")
			default:
				cb.WriteString(nlgCapFirst(obj))
				cb.WriteString(" also relates to ")
				cb.WriteString(topic)
			}
			cb.WriteString(".")
			caveatParts = append(caveatParts, cb.String())
		}
		sections = append(sections, strings.Join(caveatParts, " "))
	}

	if len(sections) == 0 {
		return ""
	}

	result := strings.Join(sections, " ")
	result = e.pronominalize(result, topic, facts)
	return e.polish(result)
}

// -----------------------------------------------------------------------
// polish — final text cleanup
// -----------------------------------------------------------------------

func (e *NLGEngine) polish(text string) string {
	// Normalize whitespace
	text = strings.Join(strings.Fields(text), " ")

	// Fix double periods
	text = strings.ReplaceAll(text, "..", ".")

	// Fix space before period
	text = strings.ReplaceAll(text, " .", ".")

	// Fix double spaces that may remain
	for strings.Contains(text, "  ") {
		text = strings.ReplaceAll(text, "  ", " ")
	}

	// Ensure the text ends with a period
	text = strings.TrimSpace(text)
	if text != "" && !strings.HasSuffix(text, ".") && !strings.HasSuffix(text, "!") && !strings.HasSuffix(text, "?") {
		text += "."
	}

	return text
}

// -----------------------------------------------------------------------
// Linguistic utilities — shared building blocks
// -----------------------------------------------------------------------

// joinCoordinated joins items with commas and "and" before the last:
// ["a"] -> "a"
// ["a", "b"] -> "a and b"
// ["a", "b", "c"] -> "a, b, and c"
func joinCoordinated(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	case 2:
		return items[0] + " and " + items[1]
	default:
		return strings.Join(items[:len(items)-1], ", ") + ", and " + items[len(items)-1]
	}
}

// nlgCapFirst capitalizes the first letter of a string.
func nlgCapFirst(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

// nlgLowerFirst lowercases the first letter unless the word looks like
// a proper noun (has uppercase letters after the first position).
func nlgLowerFirst(s string) string {
	if s == "" {
		return s
	}
	runes := []rune(s)
	// Don't lowercase proper nouns (e.g. "Python", "Google")
	if len(runes) > 1 && unicode.IsUpper(runes[1]) {
		return s
	}
	runes[0] = unicode.ToLower(runes[0])
	return string(runes)
}

// nlgCountSubject counts whole-word occurrences of subject in text.
func nlgCountSubject(text, subject string) int {
	count := 0
	remaining := text
	for {
		idx := nlgFindSubject(remaining, subject)
		if idx < 0 {
			break
		}
		count++
		remaining = remaining[idx+len(subject):]
	}
	return count
}

// nlgFindSubject finds the next whole-word occurrence of subject in text.
// Returns the byte index or -1 if not found. "Whole-word" means the
// character before and after the match is not a letter.
func nlgFindSubject(text, subject string) int {
	start := 0
	for {
		idx := strings.Index(text[start:], subject)
		if idx < 0 {
			return -1
		}
		absIdx := start + idx

		// Check word boundary before
		if absIdx > 0 {
			before := rune(text[absIdx-1])
			if unicode.IsLetter(before) {
				start = absIdx + len(subject)
				continue
			}
		}

		// Check word boundary after
		afterIdx := absIdx + len(subject)
		if afterIdx < len(text) {
			after := rune(text[afterIdx])
			if unicode.IsLetter(after) {
				start = absIdx + len(subject)
				continue
			}
		}

		return absIdx
	}
}

// extractShortCategory extracts the last noun from a category phrase.
// "programming language" -> "language"
// "open-source framework" -> "framework"
// "philosopher" -> "philosopher"
func extractShortCategory(obj string) string {
	obj = strings.ToLower(strings.TrimSpace(obj))
	obj = strings.TrimPrefix(obj, "a ")
	obj = strings.TrimPrefix(obj, "an ")
	obj = strings.TrimPrefix(obj, "the ")
	parts := strings.Fields(obj)
	if len(parts) == 0 {
		return ""
	}
	return parts[len(parts)-1]
}

// inferCategoryFromString extracts the last word from a category string
// (same as inferCategory but works on a plain string).
func inferCategoryFromString(category string) string {
	category = strings.TrimSpace(strings.ToLower(category))
	category = strings.TrimPrefix(category, "a ")
	category = strings.TrimPrefix(category, "an ")
	parts := strings.Fields(category)
	if len(parts) == 0 {
		return ""
	}
	return parts[len(parts)-1]
}

// isPlural returns true if a category string suggests a plural/group entity.
func isPlural(category string) bool {
	cat := strings.ToLower(strings.TrimSpace(category))
	for _, suffix := range []string{"team", "group", "band", "committee", "organization", "collective", "ensemble"} {
		if strings.HasSuffix(cat, suffix) {
			return true
		}
	}
	return false
}
