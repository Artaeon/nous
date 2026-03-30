package micromodel

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// GenerateTrainingPairs creates (triple, sentence) pairs for training.
// Reads knowledge text files and generates multiple expression patterns
// for each fact type.
func GenerateTrainingPairs(knowledgeDir string) []TrainingExample {
	var examples []TrainingExample

	// Add template-based pairs for each relation type
	examples = append(examples, templatePairs()...)

	// Add pairs from knowledge text files
	if knowledgeDir != "" {
		examples = append(examples, extractFromKnowledge(knowledgeDir)...)
	}

	return examples
}

// templatePairs generates diverse expression patterns for each relation type.
// These teach the model HOW to express facts, not the facts themselves.
func templatePairs() []TrainingExample {
	type relTemplate struct {
		relation  string
		templates []string // %s = subject, %s = object
	}

	patterns := []relTemplate{
		{"is_a", []string{
			"%s is a %s.",
			"%s is a type of %s.",
			"%s is classified as a %s.",
			"%s, a %s, is widely recognized.",
			"As a %s, %s has distinct characteristics.",
		}},
		{"created_by", []string{
			"%s was created by %s.",
			"%s was developed by %s.",
			"%s was built by %s.",
			"%s, created by %s, has had significant impact.",
			"The creation of %s is attributed to %s.",
		}},
		{"founded_in", []string{
			"%s was founded in %s.",
			"%s was established in %s.",
			"%s originated in %s.",
			"The founding of %s dates back to %s.",
			"Since its establishment in %s, %s has grown significantly.",
		}},
		{"has", []string{
			"%s has %s.",
			"%s features %s.",
			"%s includes %s.",
			"Among the characteristics of %s is %s.",
			"One notable aspect of %s is its %s.",
		}},
		{"used_for", []string{
			"%s is used for %s.",
			"%s is commonly applied to %s.",
			"%s enables %s.",
			"In practice, %s is used for %s.",
			"A key application of %s is %s.",
		}},
		{"located_in", []string{
			"%s is located in %s.",
			"%s is based in %s.",
			"%s can be found in %s.",
			"Situated in %s, %s is well known.",
		}},
		{"part_of", []string{
			"%s is part of %s.",
			"%s belongs to %s.",
			"%s is a component of %s.",
			"As a part of %s, %s plays an important role.",
		}},
		{"related_to", []string{
			"%s is related to %s.",
			"%s is connected to %s.",
			"%s has connections to %s.",
			"There is a relationship between %s and %s.",
		}},
		{"known_for", []string{
			"%s is known for %s.",
			"%s is famous for %s.",
			"%s is recognized for %s.",
			"%s is celebrated for its %s.",
		}},
		{"influenced_by", []string{
			"%s was influenced by %s.",
			"%s drew inspiration from %s.",
			"The development of %s was influenced by %s.",
		}},
	}

	// Sample subjects and objects for each relation
	entities := map[string][][2]string{
		"is_a": {
			{"Python", "programming language"},
			{"Bitcoin", "cryptocurrency"},
			{"DNA", "molecule"},
			{"jazz", "music genre"},
			{"democracy", "form of government"},
			{"photosynthesis", "biological process"},
			{"the internet", "global network"},
			{"calculus", "branch of mathematics"},
		},
		"created_by": {
			{"Linux", "Linus Torvalds"},
			{"Python", "Guido van Rossum"},
			{"Bitcoin", "Satoshi Nakamoto"},
			{"relativity", "Albert Einstein"},
			{"the telephone", "Alexander Graham Bell"},
			{"the printing press", "Johannes Gutenberg"},
		},
		"founded_in": {
			{"Google", "1998"},
			{"Wikipedia", "2001"},
			{"the United Nations", "1945"},
			{"NASA", "1958"},
			{"Apple", "1976"},
			{"Tesla", "2003"},
		},
		"has": {
			{"Python", "list comprehensions"},
			{"the human body", "206 bones"},
			{"water", "two hydrogen atoms"},
			{"JavaScript", "dynamic typing"},
			{"the solar system", "eight planets"},
		},
		"used_for": {
			{"machine learning", "pattern recognition"},
			{"solar panels", "generating electricity"},
			{"antibiotics", "treating infections"},
			{"cryptography", "securing communications"},
			{"satellites", "global communication"},
		},
		"located_in": {
			{"the Eiffel Tower", "Paris"},
			{"Mount Everest", "the Himalayas"},
			{"the Colosseum", "Rome"},
			{"Silicon Valley", "California"},
		},
		"part_of": {
			{"the heart", "the circulatory system"},
			{"electrons", "atoms"},
			{"the judiciary", "the government"},
		},
		"related_to": {
			{"physics", "mathematics"},
			{"music", "emotion"},
			{"economics", "psychology"},
		},
		"known_for": {
			{"Einstein", "the theory of relativity"},
			{"Mozart", "his musical compositions"},
			{"Shakespeare", "his plays and sonnets"},
		},
		"influenced_by": {
			{"modern art", "African sculpture"},
			{"jazz", "blues and ragtime"},
		},
	}

	var examples []TrainingExample
	for _, pat := range patterns {
		ents, ok := entities[pat.relation]
		if !ok {
			continue
		}
		for _, ent := range ents {
			for _, tmpl := range pat.templates {
				input := fmt.Sprintf("%s <sep> %s <sep> %s", ent[0], pat.relation, ent[1])
				target := fmt.Sprintf(tmpl, ent[0], ent[1])
				examples = append(examples, TrainingExample{Input: input, Target: target})
			}
		}
	}

	return examples
}

// extractFromKnowledge reads knowledge text files and extracts
// (subject, sentence) pairs for training.
func extractFromKnowledge(dir string) []TrainingExample {
	var examples []TrainingExample

	files, err := filepath.Glob(filepath.Join(dir, "*.txt"))
	if err != nil || len(files) == 0 {
		return nil
	}

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}

		paragraphs := strings.Split(string(data), "\n\n")
		for _, para := range paragraphs {
			para = strings.TrimSpace(para)
			if len(para) < 50 {
				continue
			}

			// Extract the topic from the first sentence
			sentences := splitKnowledgeSentences(para)
			if len(sentences) == 0 {
				continue
			}

			topic := extractSentenceTopic(sentences[0])
			if topic == "" {
				continue
			}

			// Create training pairs from each sentence
			for _, sent := range sentences {
				sent = strings.TrimSpace(sent)
				if len(sent) < 20 || len(sent) > 200 {
					continue
				}

				// Determine the relation from the sentence
				rel := detectSentenceRelation(sent)
				obj := extractSentenceObject(sent, topic)
				if obj == "" {
					obj = "various aspects"
				}

				input := fmt.Sprintf("%s <sep> %s <sep> %s", topic, rel, obj)
				examples = append(examples, TrainingExample{Input: input, Target: sent})
			}
		}
	}

	return examples
}

func splitKnowledgeSentences(text string) []string {
	var sentences []string
	remaining := text
	for len(remaining) > 0 {
		best := -1
		for _, p := range []string{". ", "! ", "? "} {
			idx := strings.Index(remaining, p)
			if idx >= 0 && (best < 0 || idx < best) {
				best = idx
			}
		}
		if best < 0 {
			s := strings.TrimSpace(remaining)
			if s != "" {
				sentences = append(sentences, s)
			}
			break
		}
		s := strings.TrimSpace(remaining[:best+1])
		if s != "" {
			sentences = append(sentences, s)
		}
		remaining = remaining[best+2:]
	}
	return sentences
}

func extractSentenceTopic(sentence string) string {
	// First few words before "is", "was", "are"
	lower := strings.ToLower(sentence)
	for _, verb := range []string{" is ", " was ", " are ", " were "} {
		idx := strings.Index(lower, verb)
		if idx > 0 && idx < 50 {
			return strings.TrimSpace(sentence[:idx])
		}
	}
	// First 3 words
	words := strings.Fields(sentence)
	if len(words) >= 3 {
		return strings.Join(words[:3], " ")
	}
	return ""
}

func detectSentenceRelation(sentence string) string {
	lower := strings.ToLower(sentence)
	if strings.Contains(lower, " is a ") || strings.Contains(lower, " is an ") {
		return "is_a"
	}
	if strings.Contains(lower, "created by") || strings.Contains(lower, "developed by") {
		return "created_by"
	}
	if strings.Contains(lower, "founded in") || strings.Contains(lower, "established in") {
		return "founded_in"
	}
	if strings.Contains(lower, " has ") || strings.Contains(lower, "features") {
		return "has"
	}
	if strings.Contains(lower, "used for") || strings.Contains(lower, "used in") {
		return "used_for"
	}
	if strings.Contains(lower, "located in") || strings.Contains(lower, "based in") {
		return "located_in"
	}
	return "described_as"
}

func extractSentenceObject(sentence, topic string) string {
	lower := strings.ToLower(sentence)
	topicLower := strings.ToLower(topic)

	// Find the part after the verb phrase
	for _, pattern := range []string{" is a ", " is an ", " was a ", " are "} {
		idx := strings.Index(lower, pattern)
		if idx >= 0 {
			rest := strings.TrimSpace(sentence[idx+len(pattern):])
			// Take first clause
			for _, delim := range []string{",", ".", ";", " that ", " which "} {
				if di := strings.Index(rest, delim); di > 0 && di < 60 {
					rest = rest[:di]
				}
			}
			if rest != "" && !strings.EqualFold(rest, topicLower) {
				return rest
			}
		}
	}

	return ""
}
