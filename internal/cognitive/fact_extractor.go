package cognitive

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// -----------------------------------------------------------------------
// Automated Fact Extractor — mines typed relations from Wikipedia-style
// text using deterministic pattern matching on sentence structure.
// Designed to scale from 432 hand-curated facts to 5,000+ extracted facts.
// -----------------------------------------------------------------------

// WikiFactExtractor extracts typed relations from natural language text
// using deterministic pattern matching on sentence structure.
type WikiFactExtractor struct {
	patterns []ExtractionPattern
}

// ExtractionPattern matches a sentence structure and extracts a typed fact.
type ExtractionPattern struct {
	Regex        *regexp.Regexp
	Relation     RelType
	SubjectGroup int // which regex capture group contains the subject
	ObjectGroup  int // which regex capture group contains the object
	Confidence   float64
}

// ExtractedFact is one fact extracted from text.
type ExtractedFact struct {
	Subject    string
	Relation   RelType
	Object     string
	Source     string  // the sentence it came from
	Confidence float64 // 0.9 = exact match, 0.7 = partial, 0.5 = inferred
}

// NewWikiFactExtractor creates a fact extractor preloaded with 30+ extraction
// patterns covering all relation types.
func NewWikiFactExtractor() *WikiFactExtractor {
	fe := &WikiFactExtractor{}
	fe.patterns = buildExtractionPatterns()
	return fe
}

// buildExtractionPatterns constructs all extraction regexes.
// Order matters: more specific patterns before general ones.
func buildExtractionPatterns() []ExtractionPattern {
	var ps []ExtractionPattern
	add := func(pat string, rel RelType, subj, obj int, conf float64) {
		ps = append(ps, ExtractionPattern{
			Regex:        regexp.MustCompile(pat),
			Relation:     rel,
			SubjectGroup: subj,
			ObjectGroup:  obj,
			Confidence:   conf,
		})
	}

	// -----------------------------------------------------------------
	// FoundedBy / CreatedBy — specific verb patterns (before IsA)
	// -----------------------------------------------------------------

	// "X was founded/created/developed/invented/designed/discovered/built by Y"
	add(`(?i)^(.+?)\s+was\s+(?:created|developed|built|designed|invented|discovered)\s+by\s+(.+?)$`,
		RelCreatedBy, 1, 2, 0.9)
	add(`(?i)^(.+?)\s+was\s+(?:founded|co-?founded)\s+by\s+(.+?)$`,
		RelFoundedBy, 1, 2, 0.9)

	// "Y created/developed/founded/invented/discovered X"
	add(`(?i)^(.+?)\s+(?:created|developed|invented|discovered|built)\s+(.+?)$`,
		RelCreatedBy, 2, 1, 0.7)
	add(`(?i)^(.+?)\s+(?:founded|co-?founded|established)\s+(.+?)$`,
		RelFoundedBy, 2, 1, 0.7)

	// "developed/created by Y" participle clause with implicit subject
	add(`(?i)^(.+?),?\s+(?:created|developed|built|designed|invented|discovered)\s+by\s+(.+?)$`,
		RelCreatedBy, 1, 2, 0.9)

	// "elucidated/proposed/formulated by Y"
	add(`(?i)^(.+?),?\s+(?:elucidated|proposed|formulated|described|introduced|conceived|devised)\s+by\s+(.+?)$`,
		RelCreatedBy, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// FoundedIn / date patterns
	// -----------------------------------------------------------------

	// "X was founded/established in YEAR"
	add(`(?i)^(.+?)\s+was\s+(?:founded|established|incorporated)\s+in\s+(\d{4})`,
		RelFoundedIn, 1, 2, 0.9)

	// "X ... released/published/introduced in YEAR"
	add(`(?i)^(.+?)\s+(?:was\s+)?(?:released|published|introduced|launched|unveiled|announced)\s+in\s+(\d{4})`,
		RelFoundedIn, 1, 2, 0.9)

	// "X began/started in YEAR"
	add(`(?i)^(.+?)\s+(?:began|started|originated|commenced|emerged)\s+in\s+(\d{4})`,
		RelFoundedIn, 1, 2, 0.9)

	// "X ... in YEAR" with founding context — looser date extraction
	add(`(?i)^(.+?)\s+(?:was\s+)?(?:formed|created|built|developed|founded|opened)\s+.*?in\s+(\d{4})`,
		RelFoundedIn, 1, 2, 0.7)

	// "published/first appeared in YEAR" with preceding subject
	add(`(?i)^(.+?),?\s+(?:first\s+)?(?:published|appeared|discovered|observed|demonstrated|proposed)\s+in\s+(\d{4})`,
		RelFoundedIn, 1, 2, 0.7)

	// "dating to YEAR"
	add(`(?i)^(.+?),?\s+dating\s+to\s+(?:roughly\s+|approximately\s+)?(\d{3,4}\s*(?:BCE?|CE)?)`,
		RelFoundedIn, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// LocatedIn patterns
	// -----------------------------------------------------------------

	// "X is located/based/headquartered/situated in Y"
	add(`(?i)^(.+?)\s+(?:is|are|was|were)\s+(?:located|based|headquartered|situated)\s+in\s+(.+?)$`,
		RelLocatedIn, 1, 2, 0.9)

	// "X, located in Y,"
	add(`(?i)^(.+?),\s+located\s+(?:in|between|along)\s+(.+?),`,
		RelLocatedIn, 1, 2, 0.9)

	// "X ... in modern-day Y" (geography/history pattern)
	add(`(?i)^(.+?),?\s+(?:located\s+)?in\s+modern-day\s+(.+?)$`,
		RelLocatedIn, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// PartOf patterns
	// -----------------------------------------------------------------

	// "X is part of Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+(?:a\s+)?part\s+of\s+(.+?)$`,
		RelPartOf, 1, 2, 0.9)

	// "X belongs to Y"
	add(`(?i)^(.+?)\s+belongs?\s+to\s+(.+?)$`,
		RelPartOf, 1, 2, 0.9)

	// "X is a component/subset/member of Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+(?:a\s+)?(?:component|subset|member|branch|division|section|subfield|subdiscipline)\s+of\s+(.+?)$`,
		RelPartOf, 1, 2, 0.9)

	// -----------------------------------------------------------------
	// UsedFor patterns
	// -----------------------------------------------------------------

	// "X is used for/in Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+used\s+(?:for|in|to)\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.9)

	// "X is applied to Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+applied\s+to\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.9)

	// "X enables/facilitates Y"
	add(`(?i)^(.+?)\s+(?:enables?|facilitates?|allows?|permits?)\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.7)

	// "X is designed for Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+designed\s+(?:for|to)\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.9)

	// "X serves as Y"
	add(`(?i)^(.+?)\s+serves?\s+as\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.9)

	// -----------------------------------------------------------------
	// Has patterns
	// -----------------------------------------------------------------

	// "X has/have Y"
	add(`(?i)^(.+?)\s+(?:has|have)\s+(.+?)$`,
		RelHas, 1, 2, 0.7)

	// "X features/includes/offers/provides/supports/contains Y"
	add(`(?i)^(.+?)\s+(?:features?|includes?|offers?|provides?|supports?|contains?)\s+(.+?)$`,
		RelHas, 1, 2, 0.9)

	// "X possesses Y"
	add(`(?i)^(.+?)\s+possesse?s?\s+(.+?)$`,
		RelHas, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// IsA patterns — broad, after more specific patterns
	// -----------------------------------------------------------------

	// "X is a/an Y" — the core definition pattern
	add(`(?i)^(.+?)\s+(?:is|was)\s+(?:a|an)\s+(.+?)$`,
		RelIsA, 1, 2, 0.9)

	// "X are Y" — plural definition
	add(`(?i)^(.+?)\s+(?:are|were)\s+(.+?)$`,
		RelIsA, 1, 2, 0.7)

	// "X is one of the Y"
	add(`(?i)^(.+?)\s+is\s+one\s+of\s+the\s+(.+?)$`,
		RelIsA, 1, 2, 0.9)

	// "X, a/an Y," — appositive definition
	add(`(?i)^(.+?),\s+(?:a|an)\s+(.+?),`,
		RelIsA, 1, 2, 0.9)

	// "X is the Y" — definite article definition
	add(`(?i)^(.+?)\s+(?:is|was)\s+the\s+(.+?)$`,
		RelIsA, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// RelatedTo patterns
	// -----------------------------------------------------------------

	// "X is related/connected/linked to Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+(?:related|connected|linked)\s+to\s+(.+?)$`,
		RelRelatedTo, 1, 2, 0.9)

	// "X is associated with Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+associated\s+with\s+(.+?)$`,
		RelRelatedTo, 1, 2, 0.9)

	// "X influenced Y"
	add(`(?i)^(.+?)\s+(?:influenced|inspired|affected|shaped|transformed)\s+(.+?)$`,
		RelRelatedTo, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// DescribedAs patterns
	// -----------------------------------------------------------------

	// "X is known for Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+(?:known|famous|renowned|recognized|noted|celebrated)\s+for\s+(.+?)$`,
		RelKnownFor, 1, 2, 0.9)

	// "X is characterized by Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+(?:characterized|defined|distinguished|marked)\s+by\s+(.+?)$`,
		RelDescribedAs, 1, 2, 0.7)

	// "X is described as Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+described\s+as\s+(.+?)$`,
		RelDescribedAs, 1, 2, 0.9)

	// -----------------------------------------------------------------
	// InfluencedBy
	// -----------------------------------------------------------------
	add(`(?i)^(.+?)\s+(?:was|were)\s+influenced\s+by\s+(.+?)$`,
		RelInfluencedBy, 1, 2, 0.9)

	// -----------------------------------------------------------------
	// DerivedFrom
	// -----------------------------------------------------------------
	add(`(?i)^(.+?)\s+(?:is|are)\s+derived\s+from\s+(.+?)$`,
		RelDerivedFrom, 1, 2, 0.9)

	// "X evolved from Y" / "X originates from Y"
	add(`(?i)^(.+?)\s+(?:evolved|originates?|stems?|comes?|arose)\s+from\s+(.+?)$`,
		RelDerivedFrom, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// Causes
	// -----------------------------------------------------------------
	add(`(?i)^(.+?)\s+(?:causes?|leads?\s+to|results?\s+in|produces?|generates?)\s+(.+?)$`,
		RelCauses, 1, 2, 0.7)

	// -----------------------------------------------------------------
	// Mid-sentence patterns (not anchored to ^) — lower confidence
	// These catch facts embedded in subordinate clauses.
	// -----------------------------------------------------------------

	// "... invented/discovered/created by Y in YEAR"
	add(`(?i)(.+?)\s+(?:invented|discovered|created|developed|proposed|formulated)\s+by\s+(.+?)\s+in\s+\d{4}`,
		RelCreatedBy, 1, 2, 0.7)

	// "... founded by Y"
	add(`(?i)(.+?)\s+(?:co-?)?founded\s+by\s+(.+?)(?:\s+in\s+\d{4})?$`,
		RelFoundedBy, 1, 2, 0.7)

	// "... in YEAR" with establishment verbs (mid-sentence date extraction)
	add(`(?i)(.+?)\s+(?:published|discovered|established|introduced|released)\s+(?:.*?\s+)?in\s+(\d{4})`,
		RelFoundedIn, 1, 2, 0.5)

	// "... responsible for Y" — describes function/role
	add(`(?i)(.+?)\s+responsible\s+for\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.5)

	// "... essential for/to Y"
	add(`(?i)(.+?)\s+(?:essential|critical|crucial|vital|necessary|important)\s+(?:for|to)\s+(.+?)$`,
		RelUsedFor, 1, 2, 0.5)

	// "... plays a role in Y" / "... plays roles in Y"
	add(`(?i)(.+?)\s+plays?\s+(?:a\s+)?(?:key\s+|central\s+|major\s+|important\s+|critical\s+)?roles?\s+in\s+(.+?)$`,
		RelRelatedTo, 1, 2, 0.5)

	// "... the study of Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+the\s+study\s+of\s+(.+?)$`,
		RelIsA, 1, 2, 0.9)

	// "... the science of Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+the\s+(?:science|field|discipline|branch|area)\s+of\s+(.+?)$`,
		RelIsA, 1, 2, 0.9)

	// "... the process by which Y"
	add(`(?i)^(.+?)\s+(?:is|are)\s+the\s+(?:process|mechanism|method|technique|procedure)\s+(?:by which|through which|whereby)\s+(.+?)$`,
		RelIsA, 1, 2, 0.9)

	// "... governs/underpins/enables Y"
	add(`(?i)(.+?)\s+(?:governs?|underpins?|enables?|powers?|drives?)\s+(.+?)$`,
		RelRelatedTo, 1, 2, 0.5)

	// "... the basis for/of Y"
	add(`(?i)(.+?)\s+(?:is|are|forms?)\s+the\s+(?:basis|foundation|cornerstone|bedrock)\s+(?:for|of)\s+(.+?)$`,
		RelRelatedTo, 1, 2, 0.5)

	// "... classified into Y"
	add(`(?i)(.+?)\s+(?:is|are)\s+(?:classified|divided|categorized|organized|grouped)\s+into\s+(.+?)$`,
		RelHas, 1, 2, 0.5)

	// "... composed of Y" / "made up of Y" / "consists of Y"
	add(`(?i)(.+?)\s+(?:is|are)\s+(?:composed|made\s+up|comprised)\s+of\s+(.+?)$`,
		RelHas, 1, 2, 0.7)
	add(`(?i)(.+?)\s+consists?\s+of\s+(.+?)$`,
		RelHas, 1, 2, 0.7)

	// "... gave rise to Y" / "... led to Y"
	add(`(?i)(.+?)\s+(?:gave\s+rise\s+to|led\s+to|contributed\s+to)\s+(.+?)$`,
		RelCauses, 1, 2, 0.5)

	return ps
}

// -----------------------------------------------------------------------
// Sentence splitting (local to this extractor for self-contained usage)
// -----------------------------------------------------------------------

var (
	// Split on paragraph boundaries and list markers.
	factSentenceBreakRe = regexp.MustCompile(`\n\n+|\n[-*•]\s+|\n\d+[.)]\s+`)
	// Split on sentence-ending punctuation followed by space or end of string.
	factSentenceEndRe = regexp.MustCompile(`(?:[.!?])(?:\s+|$)`)
)

// factSplitSentences splits text into individual sentences, including
// sub-clauses separated by semicolons, for maximum extraction coverage.
func factSplitSentences(text string) []string {
	blocks := factSentenceBreakRe.Split(text, -1)

	var sentences []string
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		parts := factSentenceEndRe.Split(block, -1)
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if len(p) <= 10 {
				continue
			}
			sentences = append(sentences, p)

			// Also split on semicolons to get sub-clauses.
			if strings.Contains(p, ";") {
				for _, sub := range strings.Split(p, ";") {
					sub = strings.TrimSpace(sub)
					if len(sub) > 15 {
						sentences = append(sentences, sub)
					}
				}
			}

			// Split at ", and " / ", while " / ", where " to extract
			// clauses that have their own subject-verb pairs.
			for _, conj := range []string{", and ", ", while ", ", where ", ", whereas "} {
				if idx := strings.Index(p, conj); idx > 0 {
					after := strings.TrimSpace(p[idx+len(conj):])
					if len(after) > 15 {
						sentences = append(sentences, after)
					}
				}
			}
		}
	}
	return sentences
}

// -----------------------------------------------------------------------
// Object cleaning — strip trailing noise from extracted object phrases
// -----------------------------------------------------------------------

// cleanObject strips trailing subordinate clauses and normalizes whitespace.
func cleanObject(s string) string {
	s = strings.TrimSpace(s)

	// Strip trailing clause starters.
	for _, sep := range []string{
		", which ", ", that ", ", where ", ", when ", ", who ",
		", although ", ", though ", ", while ", ", because ",
		", including ", ", such as ",
		" which ", " that is ",
	} {
		if idx := strings.Index(strings.ToLower(s), sep); idx > 0 {
			s = s[:idx]
		}
	}

	// Strip trailing punctuation and whitespace.
	s = strings.TrimRight(s, " .,;:!?")

	// Collapse internal whitespace early so suffix matching works reliably.
	s = strings.Join(strings.Fields(s), " ")

	// Strip trailing preposition phrases that indicate incomplete extraction.
	// Run in a loop — multiple trailing prepositions can stack.
	changed := true
	for changed {
		changed = false
		low := strings.ToLower(s)
		for _, suffix := range []string{
			" by which", " through which", " whereby", " in which", " for which",
			" by", " through", " from", " with", " and a", " and the", " or a", " or the",
			" in", " on", " at", " for", " to", " as", " that", " which", " where", " when",
		} {
			if strings.HasSuffix(low, suffix) {
				s = s[:len(s)-len(suffix)]
				s = strings.TrimRight(s, " .,;:!?")
				changed = true
				break // restart the loop with the shortened string
			}
		}
	}

	// Cap length — objects longer than 120 chars are likely full sentences.
	if len(s) > 120 {
		if idx := strings.Index(s[60:], ", "); idx > 0 {
			s = s[:60+idx]
		} else {
			// Truncate at word boundary to avoid cutting mid-word.
			cut := 120
			for cut > 0 && s[cut-1] != ' ' {
				cut--
			}
			if cut > 20 { // don't over-truncate
				s = strings.TrimSpace(s[:cut])
			}
		}
		s = strings.TrimRight(s, " ,")
	}

	// Strip objects that are too short to be meaningful after all cleaning.
	if len(s) < 3 {
		return ""
	}

	return s
}

// cleanSubject normalizes and title-cases the subject.
func cleanSubject(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimRight(s, " .,;:!?")
	s = strings.Join(strings.Fields(s), " ")

	// Cap length — subjects should be concise.
	if len(s) > 80 {
		if idx := strings.Index(s[20:], ", "); idx > 0 {
			s = s[:20+idx]
		} else {
			// Truncate at word boundary to avoid cutting mid-word.
			cut := 80
			for cut > 0 && s[cut-1] != ' ' {
				cut--
			}
			if cut > 20 { // don't over-truncate
				s = strings.TrimSpace(s[:cut])
			}
		}
		s = strings.TrimRight(s, " ,")
	}

	return s
}

// -----------------------------------------------------------------------
// Core extraction
// -----------------------------------------------------------------------

// ExtractFromText splits text into sentences and extracts typed facts
// using all registered patterns. Returns deduplicated facts.
func (fe *WikiFactExtractor) ExtractFromText(text string) []ExtractedFact {
	sentences := factSplitSentences(text)
	seen := make(map[string]bool)
	var facts []ExtractedFact

	for _, sent := range sentences {
		extracted := fe.extractFromSentence(sent)
		for _, f := range extracted {
			key := dedupeKey(f.Subject, f.Relation, f.Object)
			if seen[key] {
				continue
			}
			seen[key] = true
			facts = append(facts, f)
		}
	}

	// Paragraph-level extraction: each paragraph's topic provides context
	// for mid-sentence patterns with implicit subjects.
	paragraphs := strings.Split(text, "\n\n")
	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		// The first sentence of each paragraph defines the topic.
		firstSent := extractFirstSentence(para)
		if len(firstSent) < 20 {
			continue
		}
		subj := extractParagraphSubject(firstSent)
		if subj == "" {
			continue
		}
		key := dedupeKey(subj, RelDescribedAs, firstSent)
		if !seen[key] {
			seen[key] = true
			facts = append(facts, ExtractedFact{
				Subject:    subj,
				Relation:   RelDescribedAs,
				Object:     firstSent,
				Source:      firstSent,
				Confidence: 0.5,
			})
		}

		// Extract topic-scoped facts from the full paragraph.
		topicFacts := fe.extractTopicScopedFacts(para, subj)
		for _, f := range topicFacts {
			key := dedupeKey(f.Subject, f.Relation, f.Object)
			if seen[key] {
				continue
			}
			seen[key] = true
			facts = append(facts, f)
		}
	}

	return facts
}

// extractFromSentence tries all patterns against a single sentence.
func (fe *WikiFactExtractor) extractFromSentence(sent string) []ExtractedFact {
	var facts []ExtractedFact

	for _, p := range fe.patterns {
		m := p.Regex.FindStringSubmatch(sent)
		if m == nil || len(m) <= p.SubjectGroup || len(m) <= p.ObjectGroup {
			continue
		}

		subj := cleanSubject(m[p.SubjectGroup])
		obj := cleanObject(m[p.ObjectGroup])

		if subj == "" || obj == "" || len(subj) < 2 || len(obj) < 1 {
			continue
		}
		// Skip if subject and object are identical.
		if strings.EqualFold(subj, obj) {
			continue
		}

		facts = append(facts, ExtractedFact{
			Subject:    subj,
			Relation:   p.Relation,
			Object:     obj,
			Source:      sent,
			Confidence: p.Confidence,
		})
	}

	return facts
}

// Compiled patterns for topic-scoped extraction (mid-sentence, no anchor).
var topicScopedPatterns = []struct {
	re  *regexp.Regexp
	rel RelType
	obj int // capture group index for the object
}{
	// "by [Person]" — created_by or founded_by
	{regexp.MustCompile(`(?i)\b(?:created|developed|invented|designed|discovered|proposed|formulated|introduced|devised|conceived)\s+by\s+([A-Z][\w]+(?:\s+(?:and\s+)?[A-Z][\w]+){0,4})`), RelCreatedBy, 1},
	{regexp.MustCompile(`(?i)\b(?:founded|co-?founded|established)\s+by\s+([A-Z][\w]+(?:\s+(?:and\s+)?[A-Z][\w]+){0,4})`), RelFoundedBy, 1},

	// "in YEAR" with establishment-type context
	{regexp.MustCompile(`(?i)\b(?:founded|established|created|published|released|introduced|launched|discovered|proposed|built|opened|invented)\s+(?:.*?\s+)?in\s+(\d{4})\b`), RelFoundedIn, 1},
	{regexp.MustCompile(`(?i)\bin\s+(\d{4})\b.*\b(?:founded|established|created|published|released|introduced|launched|discovered)\b`), RelFoundedIn, 1},

	// "used in/for Y" — used_for
	{regexp.MustCompile(`(?i)\bused\s+(?:in|for)\s+(.{5,60}?)(?:\.|,|;|$)`), RelUsedFor, 1},

	// "applications in Y" — used_for
	{regexp.MustCompile(`(?i)\bapplications?\s+(?:in|including)\s+(.{5,80}?)(?:\.|;|$)`), RelUsedFor, 1},

	// "enables/allows Y"
	{regexp.MustCompile(`(?i)\b(?:enabling|allowing|permitting|facilitating)\s+(.{5,60}?)(?:\.|,|;|$)`), RelUsedFor, 1},

	// "including X, Y, and Z" — has components
	{regexp.MustCompile(`(?i)\bincluding\s+(.{5,100}?)(?:\.|;|$)`), RelHas, 1},

	// "such as X, Y, and Z" — has examples
	{regexp.MustCompile(`(?i)\bsuch\s+as\s+(.{5,100}?)(?:\.|;|$)`), RelHas, 1},

	// "consists of X"
	{regexp.MustCompile(`(?i)\bconsists?\s+of\s+(.{5,80}?)(?:\.|;|$)`), RelHas, 1},

	// "composed of X"
	{regexp.MustCompile(`(?i)\b(?:composed|comprised|made\s+up)\s+of\s+(.{5,80}?)(?:\.|;|$)`), RelHas, 1},

	// "known as Y" — is_a / alias
	{regexp.MustCompile(`(?i)\b(?:also\s+)?known\s+as\s+(.{3,60}?)(?:\.|,|;|$)`), RelIsA, 1},

	// "referred to as Y"
	{regexp.MustCompile(`(?i)\breferred\s+to\s+as\s+(.{3,60}?)(?:\.|,|;|$)`), RelIsA, 1},

	// "called Y"
	{regexp.MustCompile(`(?i)\bcalled\s+(.{3,60}?)(?:\.|,|;|$)`), RelIsA, 1},

	// "essential for/to Y"
	{regexp.MustCompile(`(?i)\b(?:essential|critical|crucial|vital|fundamental|necessary|important)\s+(?:for|to)\s+(.{5,60}?)(?:\.|,|;|$)`), RelUsedFor, 1},

	// "the basis/foundation of/for Y"
	{regexp.MustCompile(`(?i)\bthe\s+(?:basis|foundation|cornerstone|underpinning)\s+(?:of|for)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "part of Y"
	{regexp.MustCompile(`(?i)\bpart\s+of\s+(.{3,60}?)(?:\.|,|;|$)`), RelPartOf, 1},

	// "located in Y" / "based in Y"
	{regexp.MustCompile(`(?i)\b(?:located|based|headquartered|situated)\s+in\s+([A-Z][\w]+(?:\s+[A-Z][\w]+){0,3})`), RelLocatedIn, 1},

	// "drives/powers/underpins Y"
	{regexp.MustCompile(`(?i)\b(?:drives?|powers?|underpins?|governs?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "contributed to Y"
	{regexp.MustCompile(`(?i)\bcontributed?\s+to\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "influenced Y" / "inspired Y"
	{regexp.MustCompile(`(?i)\b(?:influenced|inspired|shaped|transformed|revolutionized|advanced)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "from X to Y" range pattern — extracts both as related
	{regexp.MustCompile(`(?i)\bfrom\s+(.{3,40}?)\s+to\s+(.{3,40}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "through Y" mechanism
	{regexp.MustCompile(`(?i)\bthrough\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "key/major/significant/important X" — has
	{regexp.MustCompile(`(?i)\b(?:key|major|significant|important|primary|principal)\s+(?:features?|aspects?|characteristics?|properties|elements?|components?|factors?|principles?)\s+(?:include|are)\s+(.{5,100}?)(?:\.|;|$)`), RelHas, 1},

	// "produces/generates Y"
	{regexp.MustCompile(`(?i)\b(?:produces?|generates?|yields?|creates?|synthesizes?)\s+(.{3,60}?)(?:\.|,|;|$)`), RelHas, 1},

	// "derives from Y" / "originates from Y"
	{regexp.MustCompile(`(?i)\b(?:derives?|originates?|stems?|evolved?|descended?)\s+from\s+(.{3,60}?)(?:\.|,|;|$)`), RelDerivedFrom, 1},

	// "leads to Y" / "results in Y"
	{regexp.MustCompile(`(?i)\b(?:leads?\s+to|results?\s+in|causing)\s+(.{5,60}?)(?:\.|,|;|$)`), RelCauses, 1},

	// "type/form/kind of Y"
	{regexp.MustCompile(`(?i)\b(?:a\s+)?(?:type|form|kind|class|category|variety|species)\s+of\s+(.{3,60}?)(?:\.|,|;|$)`), RelIsA, 1},

	// "example of Y"
	{regexp.MustCompile(`(?i)\b(?:an?\s+)?example\s+of\s+(.{3,60}?)(?:\.|,|;|$)`), RelIsA, 1},

	// "X and Y" explicit subject-verb patterns that appear mid-sentence
	{regexp.MustCompile(`(?i)\bpredicts?\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},
	{regexp.MustCompile(`(?i)\bexplains?\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},
	{regexp.MustCompile(`(?i)\bdescribes?\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},
	{regexp.MustCompile(`(?i)\bdemonstrates?\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "pioneered/invented by" (past participle forms)
	{regexp.MustCompile(`(?i)\bpioneered\s+by\s+([A-Z][\w]+(?:\s+(?:and\s+)?[A-Z][\w]+){0,4})`), RelCreatedBy, 1},

	// "X ... Nobel Prize" — known_for
	{regexp.MustCompile(`(?i)\b(?:Nobel|Pulitzer|Fields|Turing)\s+(?:Prize|Medal|Award)\s+(?:in\s+)?(.{3,40}?)(?:\.|,|;|$)`), RelKnownFor, 1},

	// "Y won/earned/received"
	{regexp.MustCompile(`(?i)\b(?:won|earned|received|awarded)\s+(?:the\s+)?(.{5,60}?\b(?:prize|award|medal|honor))\b`), RelKnownFor, 1},

	// "applied in/to Y"
	{regexp.MustCompile(`(?i)\bapplied\s+(?:in|to)\s+(.{5,60}?)(?:\.|,|;|$)`), RelUsedFor, 1},

	// "ranges from X to Y"
	{regexp.MustCompile(`(?i)\branges?\s+from\s+(.{3,40}?)\s+to\s+(.{3,40}?)(?:\.|,|;|$)`), RelHas, 1},

	// "characteristic of Y"
	{regexp.MustCompile(`(?i)\b(?:a\s+)?characteristic\s+of\s+(.{3,60}?)(?:\.|,|;|$)`), RelDescribedAs, 1},

	// "defined as Y"
	{regexp.MustCompile(`(?i)\bdefined\s+as\s+(.{5,80}?)(?:\.|,|;|$)`), RelDescribedAs, 1},

	// "employs/utilizes/uses Y" — has
	{regexp.MustCompile(`(?i)\b(?:employs?|utilizes?|uses?|relies?\s+on|depends?\s+on)\s+(.{5,60}?)(?:\.|,|;|$)`), RelHas, 1},

	// "requires Y"
	{regexp.MustCompile(`(?i)\brequires?\s+(.{5,60}?)(?:\.|,|;|$)`), RelHas, 1},

	// "combines Y" / "integrates Y" / "incorporates Y"
	{regexp.MustCompile(`(?i)\b(?:combines?|integrates?|incorporates?|merges?|unifies?)\s+(.{5,80}?)(?:\.|,|;|$)`), RelHas, 1},

	// "involves Y" / "encompasses Y" / "covers Y"
	{regexp.MustCompile(`(?i)\b(?:involves?|encompasses?|covers?|spans?|addresses?|handles?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelHas, 1},

	// "studied by Y" / "researched by Y"
	{regexp.MustCompile(`(?i)\b(?:studied|researched|investigated|examined|analyzed|explored)\s+by\s+([A-Z][\w]+(?:\s+(?:and\s+)?[A-Z][\w]+){0,4})`), RelCreatedBy, 1},

	// "measure of Y" / "indicator of Y"
	{regexp.MustCompile(`(?i)\b(?:a\s+)?(?:measure|indicator|marker|sign|symptom)\s+of\s+(.{3,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "prevents/inhibits Y"
	{regexp.MustCompile(`(?i)\b(?:prevents?|inhibits?|blocks?|suppresses?|reduces?|limits?|restricts?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "enhances/improves/increases Y"
	{regexp.MustCompile(`(?i)\b(?:enhances?|improves?|increases?|boosts?|strengthens?|promotes?|accelerates?|amplifies?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "replaces/succeeds Y"
	{regexp.MustCompile(`(?i)\b(?:replaces?|succeeds?|supersedes?|supplants?)\s+(.{3,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "depends on Y" / "relies on Y" (extracted separately from the "uses" pattern)
	{regexp.MustCompile(`(?i)\b(?:depends?\s+on|relies?\s+on|based\s+on|built\s+(?:on|upon))\s+(.{5,60}?)(?:\.|,|;|$)`), RelDerivedFrom, 1},

	// "introduced by Y" (people name)
	{regexp.MustCompile(`(?i)\bintroduced\s+by\s+([A-Z][\w]+(?:\s+(?:and\s+)?[A-Z][\w]+){0,4})`), RelCreatedBy, 1},

	// "in the X century" — temporal context
	{regexp.MustCompile(`(?i)\bin\s+the\s+((?:\w+\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty-first)\s+century)`), RelFoundedIn, 1},

	// "dates? to/from YEAR"
	{regexp.MustCompile(`(?i)\bdates?\s+(?:to|from|back\s+to)\s+(?:(?:approximately|roughly|about|around)\s+)?(\d{3,4}\s*(?:BCE?|CE)?)`), RelFoundedIn, 1},

	// "X ... affect/impact Y"
	{regexp.MustCompile(`(?i)\b(?:affects?|impacts?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "connects/links X to Y"
	{regexp.MustCompile(`(?i)\b(?:connects?|links?|ties?|bridges?|binds?|joins?|attaches?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "determines Y" / "controls Y"
	{regexp.MustCompile(`(?i)\b(?:determines?|controls?|regulates?|modulates?|mediates?|constrains?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "associated with Y"
	{regexp.MustCompile(`(?i)\bassociated\s+with\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "converts Y" / "transforms Y"
	{regexp.MustCompile(`(?i)\b(?:converts?|transforms?|translates?|changes?|turns?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "X measured in Y" / "expressed in Y"
	{regexp.MustCompile(`(?i)\b(?:measured|expressed|quantified|calculated|represented)\s+(?:in|as|by)\s+(.{5,60}?)(?:\.|,|;|$)`), RelDescribedAs, 1},

	// "extends/expands Y"
	{regexp.MustCompile(`(?i)\b(?:extends?|expands?|broadens?|widens?|deepens?)\s+(.{5,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "opposite of Y" / "contrast to Y"
	{regexp.MustCompile(`(?i)\b(?:the\s+)?(?:opposite|contrast|reverse|inverse|antithesis)\s+(?:of|to)\s+(.{3,60}?)(?:\.|,|;|$)`), RelRelatedTo, 1},

	// "named after Y" / "named for Y"
	{regexp.MustCompile(`(?i)\bnamed\s+(?:after|for)\s+(.{3,60}?)(?:\.|,|;|$)`), RelDerivedFrom, 1},

	// "formalized by Y" / "codified by Y"
	{regexp.MustCompile(`(?i)\b(?:formalized|codified|standardized|systematized)\s+by\s+([A-Z][\w]+(?:\s+(?:and\s+)?[A-Z][\w]+){0,4})`), RelCreatedBy, 1},

	// "enables/supports Y" — used_for (mid-sentence)
	{regexp.MustCompile(`(?i)\b(?:enables?|supports?|facilitates?|permits?|provides?\s+for)\s+(.{5,60}?)(?:\.|,|;|$)`), RelUsedFor, 1},
}

// listSplittingRels tracks which relations should have their objects split
// when containing list items (e.g., "including X, Y, and Z" → 3 facts).
var listSplittingRels = map[RelType]bool{
	RelHas:       true,
	RelUsedFor:   true,
	RelRelatedTo: true,
	RelCauses:    true,
}

// extractTopicScopedFacts scans a full paragraph for mid-sentence patterns,
// using the paragraph's topic as the implicit subject when no explicit
// subject is captured. List objects are split into individual facts.
func (fe *WikiFactExtractor) extractTopicScopedFacts(para, topic string) []ExtractedFact {
	sentences := factSplitSentences(para)
	var facts []ExtractedFact

	for _, sent := range sentences {
		for _, tp := range topicScopedPatterns {
			matches := tp.re.FindAllStringSubmatch(sent, -1)
			for _, m := range matches {
				if len(m) <= tp.obj {
					continue
				}
				rawObj := m[tp.obj]

				// For list-bearing relations, split "X, Y, and Z" into items.
				if listSplittingRels[tp.rel] {
					items := splitListItems(rawObj)
					if len(items) > 1 {
						for _, item := range items {
							obj := cleanObject(item)
							if obj == "" || len(obj) < 2 || strings.EqualFold(obj, topic) {
								continue
							}
							facts = append(facts, ExtractedFact{
								Subject:    topic,
								Relation:   tp.rel,
								Object:     obj,
								Source:      sent,
								Confidence: 0.5,
							})
						}
						continue
					}
				}

				obj := cleanObject(rawObj)
				if obj == "" || len(obj) < 2 {
					continue
				}
				// Skip if object is same as topic.
				if strings.EqualFold(obj, topic) {
					continue
				}
				facts = append(facts, ExtractedFact{
					Subject:    topic,
					Relation:   tp.rel,
					Object:     obj,
					Source:      sent,
					Confidence: 0.5,
				})
			}
		}
	}

	return facts
}

// extractFirstSentence returns the first sentence from a paragraph.
func extractFirstSentence(para string) string {
	// Look for sentence-ending punctuation followed by a space.
	re := regexp.MustCompile(`[.!?]\s`)
	loc := re.FindStringIndex(para)
	if loc != nil {
		return strings.TrimSpace(para[:loc[0]+1])
	}
	return strings.TrimSpace(para)
}

// extractParagraphSubject extracts the main subject from the first sentence
// of a paragraph. Typically everything before the first "is/are/was/were".
func extractParagraphSubject(sentence string) string {
	// Match subject up to "is/are/was/were"
	re := regexp.MustCompile(`(?i)^(.+?)\s+(?:is|are|was|were)\s+`)
	m := re.FindStringSubmatch(sentence)
	if m != nil {
		subj := cleanSubject(m[1])
		if len(subj) >= 2 && len(subj) <= 80 {
			return subj
		}
	}
	return ""
}

// dedupeKey builds a canonical string for deduplication.
func dedupeKey(subj string, rel RelType, obj string) string {
	return strings.ToLower(subj) + "|" + string(rel) + "|" + strings.ToLower(obj)
}

// -----------------------------------------------------------------------
// File and directory extraction
// -----------------------------------------------------------------------

// ExtractFromFile reads a file and extracts facts from it.
func (fe *WikiFactExtractor) ExtractFromFile(path string) ([]ExtractedFact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	facts := fe.ExtractFromText(string(data))
	// Tag facts with source file.
	base := filepath.Base(path)
	for i := range facts {
		if facts[i].Source == "" {
			facts[i].Source = base
		}
	}
	return facts, nil
}

// ExtractFromDirectory processes all .txt files in a directory.
func (fe *WikiFactExtractor) ExtractFromDirectory(dir string) ([]ExtractedFact, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	seen := make(map[string]bool)
	var allFacts []ExtractedFact

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".txt") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		facts, err := fe.ExtractFromFile(path)
		if err != nil {
			continue // skip files that can't be read
		}
		for _, f := range facts {
			key := dedupeKey(f.Subject, f.Relation, f.Object)
			if seen[key] {
				continue
			}
			seen[key] = true
			allFacts = append(allFacts, f)
		}
	}

	return allFacts, nil
}

// -----------------------------------------------------------------------
// Graph ingestion
// -----------------------------------------------------------------------

// IngestIntoGraph adds extracted facts to a cognitive graph.
// Returns the count of facts added (edges created).
func (fe *WikiFactExtractor) IngestIntoGraph(graph *CognitiveGraph, facts []ExtractedFact) int {
	added := 0
	seen := make(map[string]bool)

	for _, f := range facts {
		// Quality filter: reject bad facts before ingestion
		if !isQualityFact(f) {
			continue
		}

		key := dedupeKey(f.Subject, f.Relation, f.Object)
		if seen[key] {
			continue
		}
		seen[key] = true

		// Determine node types.
		subjType := guessNodeType(f.Subject)
		objType := guessNodeType(f.Object)

		// For described_as, the object is a full sentence — store as property.
		if f.Relation == RelDescribedAs && len(f.Object) > 100 {
			graph.mu.Lock()
			id := graph.ensureNodeLocked(f.Subject, subjType, f.Source, f.Confidence)
			if n, ok := graph.nodes[id]; ok {
				propKey := "description"
				if _, exists := n.Properties[propKey]; exists {
					propKey = "fact_" + strings.ToLower(strings.ReplaceAll(f.Subject, " ", "_"))
				}
				n.Properties[propKey] = f.Object
			}
			graph.mu.Unlock()
			added++
			continue
		}

		graph.mu.Lock()
		fromID := graph.ensureNodeLocked(f.Subject, subjType, f.Source, f.Confidence)
		toID := graph.ensureNodeLocked(f.Object, objType, f.Source, f.Confidence*0.9)
		graph.addEdgeLocked(fromID, toID, f.Relation, f.Source, f.Confidence, false)
		graph.mu.Unlock()
		added++
	}

	return added
}

// isQualityFact rejects extracted facts that would produce garbled NLG output.
func isQualityFact(f ExtractedFact) bool {
	obj := strings.TrimSpace(f.Object)
	subj := strings.TrimSpace(f.Subject)

	// Reject empty or very short
	if len(obj) < 3 || len(subj) < 2 {
		return false
	}

	// Reject objects that are clearly fragments
	lower := strings.ToLower(obj)

	// Starts with a preposition or conjunction — fragment
	badStarts := []string{"and ", "or ", "the ", "a ", "an ", "in ", "on ", "at ", "to ", "for ", "with ", "from ", "by ", "as "}
	for _, bs := range badStarts {
		if strings.HasPrefix(lower, bs) && len(obj) < 30 {
			return false
		}
	}

	// Ends with a preposition — incomplete clause
	badEnds := []string{" by", " in", " on", " at", " to", " for", " with", " from", " as", " and", " or", " the", " a"}
	for _, be := range badEnds {
		if strings.HasSuffix(lower, be) {
			return false
		}
	}

	// Contains sentence-breaking markers in the middle — extraction went too far
	if strings.Contains(obj, ". ") && f.Relation != RelDescribedAs {
		return false
	}

	// Object is too long for non-description relations (likely captured a whole clause)
	if f.Relation != RelDescribedAs && len(obj) > 100 {
		return false
	}

	// Subject contains verbs or is a sentence fragment
	subjLower := strings.ToLower(subj)
	if strings.Contains(subjLower, " produces ") || strings.Contains(subjLower, " creates ") ||
		strings.Contains(subjLower, " makes ") || strings.Contains(subjLower, " leads ") {
		return false
	}

	// Object is just a common word that's not informative
	trivial := map[string]bool{
		"outcomes": true, "accuracy": true, "vector machines": true,
		"trial and error": true, "tightly": true,
	}
	if trivial[lower] {
		return false
	}

	return true
}
