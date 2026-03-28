package cognitive

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"time"
)

// FreshnessConfig controls dynamic content retrieval behaviour.
type FreshnessConfig struct {
	MaxAge      time.Duration // max acceptable age of content (24h default)
	TimeoutMs   int64         // retrieval timeout in milliseconds (2000 default)
	MinTrust    float64       // minimum trust score for a source (0.5)
	PreferFresh bool          // prefer fresher results over higher-scoring ones
}

// DefaultFreshnessConfig returns sensible production defaults.
func DefaultFreshnessConfig() *FreshnessConfig {
	return &FreshnessConfig{
		MaxAge:      24 * time.Hour,
		TimeoutMs:   2000,
		MinTrust:    0.5,
		PreferFresh: false,
	}
}

// TopicFreshness classifies whether a topic needs fresh data.
type TopicFreshness int

const (
	FreshnessStatic   TopicFreshness = iota // "what is photosynthesis" -- stable
	FreshnessSlow                            // "population of France" -- changes yearly
	FreshnessDynamic                         // "weather today" -- changes hourly
	FreshnessRealtime                        // "stock price of AAPL" -- changes by second
)

// String returns a human-readable name for the freshness level.
func (tf TopicFreshness) String() string {
	switch tf {
	case FreshnessStatic:
		return "static"
	case FreshnessSlow:
		return "slow"
	case FreshnessDynamic:
		return "dynamic"
	case FreshnessRealtime:
		return "realtime"
	default:
		return "unknown"
	}
}

// maxAgeFor returns the acceptable age for a given freshness class.
func maxAgeFor(f TopicFreshness) time.Duration {
	switch f {
	case FreshnessRealtime:
		return 1 * time.Minute
	case FreshnessDynamic:
		return 1 * time.Hour
	case FreshnessSlow:
		return 30 * 24 * time.Hour // 30 days
	default:
		return 365 * 24 * time.Hour // 1 year
	}
}

// FreshnessClassifier determines whether a query needs fresh data.
type FreshnessClassifier struct {
	dynamicPatterns  []*regexp.Regexp
	realtimePatterns []*regexp.Regexp
	slowPatterns     []*regexp.Regexp
	staticPatterns   []*regexp.Regexp
}

// NewFreshnessClassifier creates a classifier with standard patterns.
func NewFreshnessClassifier() *FreshnessClassifier {
	compile := func(patterns []string) []*regexp.Regexp {
		out := make([]*regexp.Regexp, 0, len(patterns))
		for _, p := range patterns {
			out = append(out, regexp.MustCompile(`(?i)`+p))
		}
		return out
	}

	return &FreshnessClassifier{
		realtimePatterns: compile([]string{
			`\bright now\b`,
			`\blive (?:stream|updates?|feed|results?|scores?|coverage|broadcast|election)\b`,
			`\breal[ -]?time\b`,
			`\bcurrent(?:ly)? (?:price|value|rate|score)`,
			`\bstock (?:price|ticker|quote)`,
			`\b[A-Z]{1,5} (?:stock|share|price|ticker)\b`,
			`\bbreaking news\b`,
			`\bright this (?:moment|second|instant)\b`,
		}),
		dynamicPatterns: compile([]string{
			`\btoday\b`,
			`\bcurrent\b`,
			`\blatest\b`,
			`\bnow\b`,
			`\bthis week\b`,
			`\bthis month\b`,
			`\btonight\b`,
			`\byesterday\b`,
			`\bweather\b`,
			`\bnews\b`,
			`\bscore[s]?\b`,
			`\bsports?\b.*\b(?:score|result|game|match)\b`,
			`\belection\b.*\b(?:result|poll|count)\b`,
			`\btrending\b`,
			`\bupdate[s]?\b`,
			`\brecent(?:ly)?\b`,
		}),
		slowPatterns: compile([]string{
			`\bpopulation\b`,
			`\bstatistic[s]?\b`,
			`\bhow many\b`,
			`\bgdp\b`,
			`\branking[s]?\b`,
			`\bcensus\b`,
			`\bdemographic[s]?\b`,
			`\bannual\b`,
			`\byearly\b`,
			`\bper capita\b`,
			`\brate of\b`,
			`\bnumber of\b`,
		}),
		staticPatterns: compile([]string{
			`\bwhat is\b`,
			`\bdefinition\b`,
			`\bhistory of\b`,
			`\bscien(?:ce|tific)\b`,
			`\bmath(?:ematics?)?\b`,
			`\btheorem\b`,
			`\blaw of\b`,
			`\bprinciple of\b`,
			`\bphilosophy\b`,
			`\bancient\b`,
		}),
	}
}

// Classify determines the freshness needs of a query.
func (fc *FreshnessClassifier) Classify(query string) TopicFreshness {
	query = strings.TrimSpace(query)
	if query == "" {
		return FreshnessStatic
	}

	// Check in order from most urgent to least.
	for _, re := range fc.realtimePatterns {
		if re.MatchString(query) {
			return FreshnessRealtime
		}
	}
	for _, re := range fc.dynamicPatterns {
		if re.MatchString(query) {
			return FreshnessDynamic
		}
	}
	for _, re := range fc.slowPatterns {
		if re.MatchString(query) {
			return FreshnessSlow
		}
	}

	return FreshnessStatic
}

// TrustScore evaluates how trustworthy a source is.
type TrustScore struct {
	Source  string
	Score   float64            // 0-1 composite
	Factors map[string]float64 // "recency", "authority", "consistency"
}

// ScoreTrust evaluates a source's trustworthiness based on age and
// cross-reference count.
func ScoreTrust(source string, age time.Duration, crossRefCount int) TrustScore {
	ts := TrustScore{
		Source:  source,
		Factors: make(map[string]float64, 3),
	}

	// Recency factor: exponential decay with half-life of 24 hours.
	hours := age.Hours()
	if hours < 0 {
		hours = 0
	}
	recency := math.Exp(-0.029 * hours) // half-life ~ 24h
	ts.Factors["recency"] = recency

	// Authority factor: based on source prefix conventions.
	authority := sourceAuthority(source)
	ts.Factors["authority"] = authority

	// Consistency factor: log-scaled cross-reference count.
	// 0 refs = 0.3, 1 ref = 0.6, 3+ refs = 0.9+
	consistency := 0.3
	if crossRefCount > 0 {
		consistency = math.Min(1.0, 0.3+0.3*math.Log2(float64(crossRefCount)+1))
	}
	ts.Factors["consistency"] = consistency

	// Composite: weighted average.
	ts.Score = 0.35*recency + 0.35*authority + 0.30*consistency
	return ts
}

// sourceAuthority returns an authority score based on source naming conventions.
func sourceAuthority(source string) float64 {
	lower := strings.ToLower(source)

	// Knowledge base sources are highly trusted.
	if strings.HasPrefix(lower, "kb:") || strings.HasPrefix(lower, "knowledge") {
		return 0.95
	}
	if strings.HasPrefix(lower, "graph:") {
		return 0.85
	}
	if strings.HasPrefix(lower, "wiki") || strings.Contains(lower, "encyclopedia") {
		return 0.80
	}
	if strings.Contains(lower, "official") || strings.Contains(lower, "gov") {
		return 0.90
	}
	if strings.HasPrefix(lower, "inferred") || strings.HasPrefix(lower, "generated") {
		return 0.40
	}
	if source == "" {
		return 0.30
	}
	return 0.60 // default for unknown sources
}

// FreshnessRetriever wraps retrieval with freshness awareness.
type FreshnessRetriever struct {
	Base       *TwoTierRetriever
	Classifier *FreshnessClassifier
	Config     *FreshnessConfig
}

// NewFreshnessRetriever creates a freshness-aware retriever.
func NewFreshnessRetriever(base *TwoTierRetriever) *FreshnessRetriever {
	return &FreshnessRetriever{
		Base:       base,
		Classifier: NewFreshnessClassifier(),
		Config:     DefaultFreshnessConfig(),
	}
}

// RetrieveWithFreshness retrieves results considering topic freshness needs.
// It classifies the query's freshness requirements, retrieves from the base
// retriever, then filters and re-ranks results by freshness and trust.
func (fr *FreshnessRetriever) RetrieveWithFreshness(query string, topK int) ([]RetrievalResult, TopicFreshness) {
	freshness := fr.Classifier.Classify(query)

	// Retrieve more candidates than needed so we can filter.
	candidates := fr.Base.Retrieve(query, topK*3)
	if len(candidates) == 0 {
		return nil, freshness
	}

	maxAge := maxAgeFor(freshness)
	if fr.Config.MaxAge > 0 && fr.Config.MaxAge < maxAge {
		maxAge = fr.Config.MaxAge
	}

	now := time.Now()

	// Score each candidate by combining retrieval score with trust.
	type scoredResult struct {
		result     RetrievalResult
		trustScore TrustScore
		finalScore float64
	}

	scored := make([]scoredResult, 0, len(candidates))
	for _, c := range candidates {
		age := now.Sub(c.Freshness)
		if age < 0 {
			age = 0
		}

		// For dynamic/realtime topics, skip results that are too old.
		if freshness >= FreshnessDynamic && age > maxAge {
			continue
		}

		trust := ScoreTrust(c.Source, age, 1) // default cross-ref = 1
		if trust.Score < fr.Config.MinTrust {
			continue
		}

		finalScore := c.Score
		if fr.Config.PreferFresh || freshness >= FreshnessDynamic {
			// Blend retrieval score with recency.
			recencyBoost := trust.Factors["recency"]
			finalScore = 0.6*c.Score + 0.4*recencyBoost
		}

		scored = append(scored, scoredResult{
			result:     c,
			trustScore: trust,
			finalScore: finalScore,
		})
	}

	// Sort by final score descending.
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].finalScore > scored[j].finalScore
	})

	if topK > len(scored) {
		topK = len(scored)
	}

	results := make([]RetrievalResult, topK)
	for i := 0; i < topK; i++ {
		results[i] = scored[i].result
		results[i].Score = scored[i].finalScore
	}
	return results, freshness
}
