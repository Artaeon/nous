package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// dictEntry represents a single entry from the dictionary API response.
type dictEntry struct {
	Word     string        `json:"word"`
	Meanings []dictMeaning `json:"meanings"`
}

type dictMeaning struct {
	PartOfSpeech string           `json:"partOfSpeech"`
	Definitions  []dictDefinition `json:"definitions"`
	Synonyms     []string         `json:"synonyms"`
	Antonyms     []string         `json:"antonyms"`
}

type dictDefinition struct {
	Definition string   `json:"definition"`
	Example    string   `json:"example"`
	Synonyms   []string `json:"synonyms"`
	Antonyms   []string `json:"antonyms"`
}

// dictHTTPClient is the HTTP client used for dictionary lookups (can be overridden in tests).
var dictHTTPClient = &http.Client{Timeout: 5 * time.Second}

// LookupWord queries the free dictionary API and returns formatted results.
func LookupWord(word, action string) (string, error) {
	word = strings.TrimSpace(word)
	if word == "" {
		return "", fmt.Errorf("dict: word is required")
	}

	action = strings.ToLower(strings.TrimSpace(action))
	if action == "" {
		action = "define"
	}

	entries, err := fetchDictEntries(word)
	if err != nil {
		return "", err
	}

	switch action {
	case "define":
		return formatDefinitions(entries), nil
	case "synonyms":
		return formatSynonyms(entries), nil
	case "antonyms":
		return formatAntonyms(entries), nil
	default:
		return "", fmt.Errorf("dict: unknown action %q — supported: define, synonyms, antonyms", action)
	}
}

func fetchDictEntries(word string) ([]dictEntry, error) {
	url := fmt.Sprintf("https://api.dictionaryapi.dev/api/v2/entries/en/%s", word)
	resp, err := dictHTTPClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("dict: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) // 1MB limit
	if err != nil {
		return nil, fmt.Errorf("dict: failed to read response: %w", err)
	}

	if resp.StatusCode == 404 {
		return nil, fmt.Errorf("dict: word %q not found", word)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("dict: API returned status %d", resp.StatusCode)
	}

	var entries []dictEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		return nil, fmt.Errorf("dict: failed to parse response: %w", err)
	}

	return entries, nil
}

// ParseDictResponse parses raw JSON from the dictionary API into entries.
func ParseDictResponse(data []byte) ([]dictEntry, error) {
	var entries []dictEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("dict: failed to parse response: %w", err)
	}
	return entries, nil
}

// FormatDefinitions formats dictionary entries into a readable string.
func FormatDefinitions(entries []dictEntry) string {
	return formatDefinitions(entries)
}

func formatDefinitions(entries []dictEntry) string {
	var sb strings.Builder
	for _, entry := range entries {
		for _, meaning := range entry.Meanings {
			for i, def := range meaning.Definitions {
				if i >= 3 { // limit to 3 definitions per part of speech
					break
				}
				fmt.Fprintf(&sb, "%s (%s): %s", entry.Word, meaning.PartOfSpeech, def.Definition)
				if def.Example != "" {
					fmt.Fprintf(&sb, " Example: %q", def.Example)
				}
				sb.WriteString("\n")
			}
		}
	}

	result := strings.TrimSpace(sb.String())
	if result == "" {
		return "No definitions found."
	}
	return result
}

// FormatSynonyms formats synonym data from dictionary entries.
func FormatSynonyms(entries []dictEntry) string {
	return formatSynonyms(entries)
}

func formatSynonyms(entries []dictEntry) string {
	var allSyns []string
	seen := map[string]bool{}

	for _, entry := range entries {
		for _, meaning := range entry.Meanings {
			for _, s := range meaning.Synonyms {
				if !seen[s] {
					seen[s] = true
					allSyns = append(allSyns, s)
				}
			}
			for _, def := range meaning.Definitions {
				for _, s := range def.Synonyms {
					if !seen[s] {
						seen[s] = true
						allSyns = append(allSyns, s)
					}
				}
			}
		}
	}

	if len(allSyns) == 0 {
		return "No synonyms found."
	}
	return fmt.Sprintf("Synonyms: %s", strings.Join(allSyns, ", "))
}

// FormatAntonyms formats antonym data from dictionary entries.
func FormatAntonyms(entries []dictEntry) string {
	return formatAntonyms(entries)
}

func formatAntonyms(entries []dictEntry) string {
	var allAnts []string
	seen := map[string]bool{}

	for _, entry := range entries {
		for _, meaning := range entry.Meanings {
			for _, a := range meaning.Antonyms {
				if !seen[a] {
					seen[a] = true
					allAnts = append(allAnts, a)
				}
			}
			for _, def := range meaning.Definitions {
				for _, a := range def.Antonyms {
					if !seen[a] {
						seen[a] = true
						allAnts = append(allAnts, a)
					}
				}
			}
		}
	}

	if len(allAnts) == 0 {
		return "No antonyms found."
	}
	return fmt.Sprintf("Antonyms: %s", strings.Join(allAnts, ", "))
}

// RegisterDictTools adds the dict tool to the registry.
func RegisterDictTools(r *Registry) {
	r.Register(Tool{
		Name:        "dict",
		Description: "Dictionary and thesaurus. Args: word (required), action (define/synonyms/antonyms, default define).",
		Execute: func(args map[string]string) (string, error) {
			return LookupWord(args["word"], args["action"])
		},
	})
}
