package cognitive

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Wikidata SPARQL Importer — pulls structured knowledge from Wikidata
// and converts it into Nous KnowledgePackages for the cognitive graph.
//
// Uses the public Wikidata Query Service SPARQL endpoint. Respects
// rate limits and sets a proper User-Agent as required by Wikidata.
// -----------------------------------------------------------------------

// WikidataImporter fetches knowledge from Wikidata via SPARQL.
type WikidataImporter struct {
	Endpoint  string // SPARQL endpoint URL
	UserAgent string // required by Wikidata
	client    *http.Client
	lastReq   time.Time // for rate limiting
}

// NewWikidataImporter creates an importer with sensible defaults.
func NewWikidataImporter() *WikidataImporter {
	return &WikidataImporter{
		Endpoint:  "https://query.wikidata.org/sparql",
		UserAgent: "Nous/1.0 (cognitive engine)",
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// sparqlResult mirrors the SPARQL JSON results format.
type sparqlResult struct {
	Results struct {
		Bindings []map[string]struct {
			Value string `json:"value"`
			Type  string `json:"type"`
		} `json:"bindings"`
	} `json:"results"`
}

// domainQueries maps domain names to Wikidata class QIDs used in queries.
var domainQueries = map[string]struct {
	classes     []string // Q-IDs to search (instance of / subclass of)
	description string
}{
	"philosophy":  {[]string{"Q4964182", "Q5891", "Q1387659"}, "Philosophers, philosophical concepts, and schools of thought"},
	"science":     {[]string{"Q901", "Q336", "Q11862829"}, "Scientists, scientific fields, and discoveries"},
	"history":     {[]string{"Q198", "Q5", "Q11514315"}, "Historical events, figures, and periods"},
	"technology":  {[]string{"Q11016", "Q4830453", "Q39546"}, "Technology concepts, companies, and inventions"},
	"geography":   {[]string{"Q6256", "Q515", "Q570116"}, "Countries, cities, and landmarks"},
	"art":         {[]string{"Q483501", "Q968159", "Q838948"}, "Artists, art movements, and artworks"},
	"music":       {[]string{"Q639669", "Q188451", "Q34379"}, "Musicians, genres, and instruments"},
	"literature":  {[]string{"Q36180", "Q7725634", "Q5185279"}, "Authors, books, and literary movements"},
	"mathematics": {[]string{"Q170790", "Q65943", "Q1936384"}, "Mathematicians, theorems, and fields"},
	"biology":     {[]string{"Q7239", "Q2996394", "Q712378"}, "Organisms, biological processes, and body parts"},
}

// ImportDomain queries Wikidata for a domain and returns a KnowledgePackage.
func (wi *WikidataImporter) ImportDomain(domain string, limit int) (*KnowledgePackage, error) {
	domain = strings.ToLower(strings.TrimSpace(domain))
	if _, ok := domainQueries[domain]; !ok {
		return nil, fmt.Errorf("unsupported domain %q; supported: %s", domain, wi.SupportedDomains())
	}

	if limit <= 0 {
		limit = 100
	}

	query := wi.sparqlForDomain(domain, limit)
	data, err := wi.executeSPARQL(query)
	if err != nil {
		return nil, fmt.Errorf("sparql query for domain %s: %w", domain, err)
	}

	facts, memories, err := wi.parseResults(data, domain)
	if err != nil {
		return nil, fmt.Errorf("parse results for domain %s: %w", domain, err)
	}

	info := domainQueries[domain]
	pkg := &KnowledgePackage{
		Name:        "wikidata-" + domain,
		Version:     "1.0.0",
		Description: info.description + " (imported from Wikidata)",
		Author:      "Wikidata SPARQL Importer",
		Domain:      domain,
		Facts:       facts,
		Memories:    memories,
	}

	return pkg, nil
}

// ImportEntity imports facts about a single named entity from Wikidata.
func (wi *WikidataImporter) ImportEntity(entityName string) (*KnowledgePackage, error) {
	entityName = strings.TrimSpace(entityName)
	if entityName == "" {
		return nil, fmt.Errorf("entity name cannot be empty")
	}

	query := wi.sparqlForEntity(entityName)
	data, err := wi.executeSPARQL(query)
	if err != nil {
		return nil, fmt.Errorf("sparql query for entity %q: %w", entityName, err)
	}

	facts, memories, err := wi.parseResults(data, "entity")
	if err != nil {
		return nil, fmt.Errorf("parse results for entity %q: %w", entityName, err)
	}

	slug := strings.ToLower(strings.ReplaceAll(entityName, " ", "_"))
	pkg := &KnowledgePackage{
		Name:        "wikidata-entity-" + slug,
		Version:     "1.0.0",
		Description: fmt.Sprintf("Knowledge about %s (imported from Wikidata)", entityName),
		Author:      "Wikidata SPARQL Importer",
		Domain:      "entity",
		Facts:       facts,
		Memories:    memories,
	}

	return pkg, nil
}

// SupportedDomains returns a comma-separated list of supported domains.
func (wi *WikidataImporter) SupportedDomains() string {
	domains := make([]string, 0, len(domainQueries))
	for d := range domainQueries {
		domains = append(domains, d)
	}
	return strings.Join(domains, ", ")
}

// sparqlForDomain generates a SPARQL query for a domain.
func (wi *WikidataImporter) sparqlForDomain(domain string, limit int) string {
	info := domainQueries[domain]

	// Build VALUES clause for the domain's class QIDs
	var values []string
	for _, qid := range info.classes {
		values = append(values, fmt.Sprintf("wd:%s", qid))
	}
	valuesClause := strings.Join(values, " ")

	return fmt.Sprintf(`SELECT ?itemLabel ?propLabel ?valueLabel WHERE {
  VALUES ?class { %s }
  ?item wdt:P31/wdt:P279* ?class .
  {
    ?item wdt:P31 ?value .
    BIND("instance of" AS ?propLabel)
  } UNION {
    ?item wdt:P279 ?value .
    BIND("subclass of" AS ?propLabel)
  } UNION {
    ?item wdt:P131 ?value .
    BIND("located in administrative territorial entity" AS ?propLabel)
  } UNION {
    ?item wdt:P17 ?value .
    BIND("country" AS ?propLabel)
  } UNION {
    ?item wdt:P361 ?value .
    BIND("part of" AS ?propLabel)
  } UNION {
    ?item wdt:P170 ?value .
    BIND("creator" AS ?propLabel)
  } UNION {
    ?item wdt:P61 ?value .
    BIND("discoverer or inventor" AS ?propLabel)
  } UNION {
    ?item wdt:P112 ?value .
    BIND("founded by" AS ?propLabel)
  } UNION {
    ?item wdt:P571 ?value .
    BIND("inception" AS ?propLabel)
  } UNION {
    ?item wdt:P1542 ?value .
    BIND("has effect" AS ?propLabel)
  } UNION {
    ?item wdt:P527 ?value .
    BIND("has part" AS ?propLabel)
  } UNION {
    ?item wdt:P106 ?value .
    BIND("occupation" AS ?propLabel)
  } UNION {
    ?item wdt:P101 ?value .
    BIND("field of work" AS ?propLabel)
  } UNION {
    ?item wdt:P19 ?value .
    BIND("place of birth" AS ?propLabel)
  } UNION {
    ?item wdt:P20 ?value .
    BIND("place of death" AS ?propLabel)
  } UNION {
    ?item wdt:P800 ?value .
    BIND("notable work" AS ?propLabel)
  } UNION {
    ?item wdt:P135 ?value .
    BIND("movement" AS ?propLabel)
  } UNION {
    ?item wdt:P1366 ?value .
    BIND("replaced by" AS ?propLabel)
  } UNION {
    ?item wdt:P1365 ?value .
    BIND("replaces" AS ?propLabel)
  } UNION {
    ?item wdt:P569 ?value .
    BIND("date of birth" AS ?propLabel)
  } UNION {
    ?item wdt:P570 ?value .
    BIND("date of death" AS ?propLabel)
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT %d`, valuesClause, limit)
}

// sparqlForEntity generates a SPARQL query for a single entity by label.
func (wi *WikidataImporter) sparqlForEntity(name string) string {
	escaped := strings.ReplaceAll(name, `"`, `\"`)
	return fmt.Sprintf(`SELECT ?itemLabel ?propLabel ?valueLabel WHERE {
  ?item rdfs:label "%s"@en .
  {
    ?item wdt:P31 ?value .
    BIND("instance of" AS ?propLabel)
  } UNION {
    ?item wdt:P279 ?value .
    BIND("subclass of" AS ?propLabel)
  } UNION {
    ?item wdt:P131 ?value .
    BIND("located in administrative territorial entity" AS ?propLabel)
  } UNION {
    ?item wdt:P17 ?value .
    BIND("country" AS ?propLabel)
  } UNION {
    ?item wdt:P361 ?value .
    BIND("part of" AS ?propLabel)
  } UNION {
    ?item wdt:P170 ?value .
    BIND("creator" AS ?propLabel)
  } UNION {
    ?item wdt:P61 ?value .
    BIND("discoverer or inventor" AS ?propLabel)
  } UNION {
    ?item wdt:P112 ?value .
    BIND("founded by" AS ?propLabel)
  } UNION {
    ?item wdt:P571 ?value .
    BIND("inception" AS ?propLabel)
  } UNION {
    ?item wdt:P1542 ?value .
    BIND("has effect" AS ?propLabel)
  } UNION {
    ?item wdt:P527 ?value .
    BIND("has part" AS ?propLabel)
  } UNION {
    ?item wdt:P106 ?value .
    BIND("occupation" AS ?propLabel)
  } UNION {
    ?item wdt:P101 ?value .
    BIND("field of work" AS ?propLabel)
  } UNION {
    ?item wdt:P19 ?value .
    BIND("place of birth" AS ?propLabel)
  } UNION {
    ?item wdt:P20 ?value .
    BIND("place of death" AS ?propLabel)
  } UNION {
    ?item wdt:P800 ?value .
    BIND("notable work" AS ?propLabel)
  } UNION {
    ?item wdt:P135 ?value .
    BIND("movement" AS ?propLabel)
  } UNION {
    ?item wdt:P1366 ?value .
    BIND("replaced by" AS ?propLabel)
  } UNION {
    ?item wdt:P1365 ?value .
    BIND("replaces" AS ?propLabel)
  } UNION {
    ?item wdt:P569 ?value .
    BIND("date of birth" AS ?propLabel)
  } UNION {
    ?item wdt:P570 ?value .
    BIND("date of death" AS ?propLabel)
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 200`, escaped)
}

// executeSPARQL sends a SPARQL query to the endpoint and returns the raw JSON.
func (wi *WikidataImporter) executeSPARQL(query string) ([]byte, error) {
	// Rate limiting: at most 1 request per second
	if !wi.lastReq.IsZero() {
		elapsed := time.Since(wi.lastReq)
		if elapsed < time.Second {
			time.Sleep(time.Second - elapsed)
		}
	}

	reqURL := wi.Endpoint + "?query=" + url.QueryEscape(query)
	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Accept", "application/sparql-results+json")
	req.Header.Set("User-Agent", wi.UserAgent)

	// Retry with backoff on rate limiting
	var resp *http.Response
	for attempt := 0; attempt < 3; attempt++ {
		wi.lastReq = time.Now()
		resp, err = wi.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("http request: %w", err)
		}

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable {
			resp.Body.Close()
			backoff := time.Duration(1<<uint(attempt)) * time.Second
			time.Sleep(backoff)
			continue
		}
		break
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("sparql endpoint returned %d: %s", resp.StatusCode, string(body))
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	return data, nil
}

// parseResults converts SPARQL JSON results into PackageFacts and PackageMemories.
func (wi *WikidataImporter) parseResults(data []byte, domain string) ([]PackageFact, []PackageMemory, error) {
	var result sparqlResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, nil, fmt.Errorf("unmarshal sparql results: %w", err)
	}

	var facts []PackageFact
	var memories []PackageMemory
	seen := make(map[string]bool) // dedup key: "subject|relation|object"

	for _, binding := range result.Results.Bindings {
		itemLabel := binding["itemLabel"].Value
		propLabel := binding["propLabel"].Value
		valueLabel := binding["valueLabel"].Value

		if itemLabel == "" || propLabel == "" || valueLabel == "" {
			continue
		}

		// Skip results that are still Q-IDs (unlabelled)
		if strings.HasPrefix(itemLabel, "http://") || strings.HasPrefix(valueLabel, "http://") {
			continue
		}

		rel := wikidataPropToRelation(propLabel)

		// For "notable work", the entity created the work — reverse direction
		if propLabel == "notable work" {
			fact := PackageFact{
				Subject:  valueLabel,
				Relation: "created_by",
				Object:   itemLabel,
			}
			key := fact.Subject + "|" + fact.Relation + "|" + fact.Object
			if !seen[key] {
				seen[key] = true
				facts = append(facts, fact)
			}
			continue
		}

		fact := PackageFact{
			Subject:  itemLabel,
			Relation: rel,
			Object:   valueLabel,
		}
		key := fact.Subject + "|" + fact.Relation + "|" + fact.Object
		if seen[key] {
			continue
		}
		seen[key] = true
		facts = append(facts, fact)

		// Generate memories for certain relation types
		if rel == "is_a" {
			mem := PackageMemory{
				Key:      itemLabel,
				Value:    fmt.Sprintf("%s is %s", itemLabel, valueLabel),
				Category: "definition",
			}
			memKey := mem.Key + "|" + mem.Value
			if !seen[memKey] {
				seen[memKey] = true
				memories = append(memories, mem)
			}
		}
	}

	return facts, memories, nil
}

// wikidataPropToRelation maps Wikidata property labels to Nous relation strings.
func wikidataPropToRelation(prop string) string {
	prop = strings.ToLower(strings.TrimSpace(prop))
	switch prop {
	case "instance of", "subclass of":
		return "is_a"
	case "located in administrative territorial entity", "country":
		return "located_in"
	case "part of":
		return "part_of"
	case "creator", "discoverer or inventor":
		return "created_by"
	case "founded by":
		return "founded_by"
	case "inception", "date of birth", "date of death":
		return "founded_in"
	case "has effect":
		return "causes"
	case "has part":
		return "has"
	case "replaced by", "replaces":
		return "related_to"
	case "occupation", "field of work":
		return "domain"
	case "place of birth", "place of death":
		return "located_in"
	case "notable work":
		return "created_by"
	case "movement":
		return "part_of"
	default:
		return "related_to"
	}
}

// SavePackage writes a KnowledgePackage as pretty-printed JSON.
func (wi *WikidataImporter) SavePackage(pkg *KnowledgePackage, outputDir string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}

	filename := fmt.Sprintf("wikidata_%s.json", strings.ReplaceAll(pkg.Domain, " ", "_"))
	path := filepath.Join(outputDir, filename)

	data, err := json.MarshalIndent(pkg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal package: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write package to %s: %w", path, err)
	}

	return nil
}
