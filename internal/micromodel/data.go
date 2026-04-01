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

	// Add multi-sentence training pairs for richer generation
	examples = append(examples, multiSentencePairs()...)

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
			"Categorized as a %s, %s stands out in its domain.",
			"%s represents a %s.",
			"%s serves as a %s.",
			"%s falls under the category of %s.",
			"%s can be described as a %s.",
			"Known widely as a %s, %s has unique properties.",
			"In essence, %s is a %s.",
		}},
		{"created_by", []string{
			"%s was created by %s.",
			"%s was developed by %s.",
			"%s was built by %s.",
			"%s, created by %s, has had significant impact.",
			"The creation of %s is attributed to %s.",
			"The work of %s resulted in %s.",
			"%s owes its creation to %s.",
			"It was %s who brought %s into existence.",
			"%s is the original creator of %s.",
			"%s was designed and built by %s.",
			"The invention of %s is credited to %s.",
			"Behind %s stands its creator, %s.",
		}},
		{"founded_in", []string{
			"%s was founded in %s.",
			"%s was established in %s.",
			"%s originated in %s.",
			"The founding of %s dates back to %s.",
			"Since its establishment in %s, %s has grown significantly.",
			"Dating back to %s, %s has a long history.",
			"The year %s marked the beginning of %s.",
			"%s traces its roots to %s.",
			"%s came into being in %s.",
			"It was in %s that %s was first established.",
			"%s has been operating since %s.",
			"The origins of %s go back to %s.",
		}},
		{"has", []string{
			"%s has %s.",
			"%s features %s.",
			"%s includes %s.",
			"Among the characteristics of %s is %s.",
			"One notable aspect of %s is its %s.",
			"%s is characterized by %s.",
			"A defining feature of %s is %s.",
			"%s possesses %s.",
			"%s contains %s.",
			"%s is equipped with %s.",
			"An important property of %s is %s.",
			"Within %s one can find %s.",
		}},
		{"used_for", []string{
			"%s is used for %s.",
			"%s is commonly applied to %s.",
			"%s enables %s.",
			"In practice, %s is used for %s.",
			"A key application of %s is %s.",
			"%s serves the purpose of %s.",
			"One can employ %s for %s.",
			"The primary function of %s is %s.",
			"%s plays an important role in %s.",
			"People rely on %s for %s.",
			"%s is widely utilized for %s.",
			"%s finds its main use in %s.",
		}},
		{"located_in", []string{
			"%s is located in %s.",
			"%s is based in %s.",
			"%s can be found in %s.",
			"Situated in %s, %s is well known.",
			"%s lies within %s.",
			"%s is positioned in %s.",
			"One can visit %s in %s.",
			"The location of %s is %s.",
			"%s stands in the heart of %s.",
			"Geographically, %s belongs to %s.",
		}},
		{"part_of", []string{
			"%s is part of %s.",
			"%s belongs to %s.",
			"%s is a component of %s.",
			"As a part of %s, %s plays an important role.",
			"%s is included in %s.",
			"%s forms a key element of %s.",
			"%s is one of the constituents of %s.",
			"%s is integral to %s.",
			"Within %s, %s serves a vital function.",
			"%s is embedded within %s.",
		}},
		{"related_to", []string{
			"%s is related to %s.",
			"%s is connected to %s.",
			"%s has connections to %s.",
			"There is a relationship between %s and %s.",
			"%s is closely linked to %s.",
			"%s shares common ground with %s.",
			"%s and %s are intertwined.",
			"The fields of %s and %s overlap.",
			"A strong connection exists between %s and %s.",
			"%s intersects with %s in many ways.",
		}},
		{"known_for", []string{
			"%s is known for %s.",
			"%s is famous for %s.",
			"%s is recognized for %s.",
			"%s is celebrated for its %s.",
			"%s gained renown through %s.",
			"The legacy of %s rests on %s.",
			"%s made a lasting mark with %s.",
			"%s is widely admired for %s.",
			"People remember %s for %s.",
			"%s earned its reputation through %s.",
		}},
		{"influenced_by", []string{
			"%s was influenced by %s.",
			"%s drew inspiration from %s.",
			"The development of %s was influenced by %s.",
			"%s owes much of its character to %s.",
			"%s was shaped by the ideas of %s.",
			"The roots of %s can be traced to %s.",
			"%s bears the imprint of %s.",
			"%s evolved under the influence of %s.",
			"Without %s, %s would look very different.",
			"The impact of %s on %s is unmistakable.",
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
			{"machine learning", "field of computer science"},
			{"chess", "board game"},
			{"the Sahara", "desert"},
			{"electricity", "form of energy"},
			{"philosophy", "discipline"},
			{"the internet", "communication network"},
			{"a violin", "string instrument"},
		},
		"created_by": {
			{"Linux", "Linus Torvalds"},
			{"Python", "Guido van Rossum"},
			{"Bitcoin", "Satoshi Nakamoto"},
			{"relativity", "Albert Einstein"},
			{"the telephone", "Alexander Graham Bell"},
			{"the printing press", "Johannes Gutenberg"},
			{"the C programming language", "Dennis Ritchie"},
			{"general relativity", "Albert Einstein"},
			{"the World Wide Web", "Tim Berners-Lee"},
			{"penicillin", "Alexander Fleming"},
			{"the lightbulb", "Thomas Edison"},
			{"the theory of evolution", "Charles Darwin"},
			{"the periodic table", "Dmitri Mendeleev"},
			{"dynamite", "Alfred Nobel"},
		},
		"founded_in": {
			{"Google", "1998"},
			{"Wikipedia", "2001"},
			{"the United Nations", "1945"},
			{"NASA", "1958"},
			{"Apple", "1976"},
			{"Tesla", "2003"},
			{"Microsoft", "1975"},
			{"SpaceX", "2002"},
			{"the European Union", "1993"},
			{"Stanford University", "1885"},
			{"Amazon", "1994"},
			{"the Red Cross", "1863"},
			{"MIT", "1861"},
			{"the World Health Organization", "1948"},
		},
		"has": {
			{"Python", "list comprehensions"},
			{"the human body", "206 bones"},
			{"water", "two hydrogen atoms"},
			{"JavaScript", "dynamic typing"},
			{"the solar system", "eight planets"},
			{"the Earth", "one moon"},
			{"chess", "64 squares"},
			{"the human brain", "billions of neurons"},
			{"a computer", "a central processing unit"},
			{"an atom", "a nucleus"},
			{"a cell", "a membrane"},
			{"the English alphabet", "26 letters"},
			{"a year", "365 days"},
			{"DNA", "a double helix structure"},
		},
		"used_for": {
			{"machine learning", "pattern recognition"},
			{"solar panels", "generating electricity"},
			{"antibiotics", "treating infections"},
			{"cryptography", "securing communications"},
			{"satellites", "global communication"},
			{"telescopes", "observing distant objects"},
			{"vaccines", "preventing diseases"},
			{"electricity", "powering devices"},
			{"GPS", "navigation"},
			{"microscopes", "examining small organisms"},
			{"radar", "detecting aircraft"},
			{"concrete", "constructing buildings"},
			{"anesthesia", "pain management"},
			{"compilers", "translating source code"},
		},
		"located_in": {
			{"the Eiffel Tower", "Paris"},
			{"Mount Everest", "the Himalayas"},
			{"the Colosseum", "Rome"},
			{"Silicon Valley", "California"},
			{"the Great Wall", "China"},
			{"the Statue of Liberty", "New York"},
			{"the Kremlin", "Moscow"},
			{"Machu Picchu", "Peru"},
			{"the Taj Mahal", "India"},
			{"the Amazon rainforest", "South America"},
			{"the Parthenon", "Athens"},
			{"the Great Barrier Reef", "Australia"},
		},
		"part_of": {
			{"the heart", "the circulatory system"},
			{"electrons", "atoms"},
			{"the judiciary", "the government"},
			{"the CPU", "a computer"},
			{"a chapter", "a book"},
			{"a wheel", "a vehicle"},
			{"the nucleus", "a cell"},
			{"a pixel", "a digital image"},
			{"a stanza", "a poem"},
			{"the roots", "a plant"},
			{"a string", "an orchestra"},
			{"the cortex", "the brain"},
		},
		"related_to": {
			{"physics", "mathematics"},
			{"music", "emotion"},
			{"economics", "psychology"},
			{"biology", "chemistry"},
			{"linguistics", "cognitive science"},
			{"architecture", "engineering"},
			{"astronomy", "physics"},
			{"nutrition", "health"},
			{"statistics", "data science"},
			{"ethics", "philosophy"},
			{"genetics", "evolution"},
			{"robotics", "artificial intelligence"},
		},
		"known_for": {
			{"Einstein", "the theory of relativity"},
			{"Mozart", "his musical compositions"},
			{"Shakespeare", "his plays and sonnets"},
			{"Marie Curie", "her research on radioactivity"},
			{"Leonardo da Vinci", "the Mona Lisa"},
			{"Nikola Tesla", "his work on alternating current"},
			{"Isaac Newton", "the laws of motion"},
			{"Beethoven", "his symphonies"},
			{"Aristotle", "his contributions to philosophy"},
			{"Ada Lovelace", "the first computer program"},
			{"Galileo", "his astronomical observations"},
			{"Darwin", "the theory of natural selection"},
		},
		"influenced_by": {
			{"modern art", "African sculpture"},
			{"jazz", "blues and ragtime"},
			{"the Renaissance", "classical antiquity"},
			{"impressionism", "Japanese woodblock prints"},
			{"machine learning", "statistics"},
			{"quantum mechanics", "classical mechanics"},
			{"rock music", "blues and gospel"},
			{"existentialism", "phenomenology"},
			{"modern architecture", "the Bauhaus movement"},
			{"hip hop", "funk and soul music"},
			{"cognitive science", "philosophy and neuroscience"},
			{"Python", "the ABC programming language"},
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

// multiSentencePairs generates training pairs where the target is 2-3
// sentences that combine multiple facts about a topic. This teaches the
// model to produce fluent, multi-sentence descriptions rather than
// single template expansions.
func multiSentencePairs() []TrainingExample {
	return []TrainingExample{
		{
			Input:  "Python <sep> described_as <sep> programming language and ecosystem",
			Target: "Python is a versatile programming language. It was created by Guido van Rossum and is widely used for data science, web development, and automation.",
		},
		{
			Input:  "quantum mechanics <sep> described_as <sep> fundamental physics theory",
			Target: "Quantum mechanics is a fundamental theory in physics. It describes the behavior of matter and energy at atomic scales, replacing classical deterministic models with probabilistic wave functions.",
		},
		{
			Input:  "the internet <sep> described_as <sep> global communication network",
			Target: "The internet is a global communication network connecting billions of devices. It evolved from ARPANET, a research project funded by the U.S. Department of Defense, and has transformed commerce, education, and social interaction.",
		},
		{
			Input:  "DNA <sep> described_as <sep> biological molecule",
			Target: "DNA is a molecule that carries genetic information in living organisms. Its double helix structure was described by Watson and Crick in 1953, building on X-ray work by Rosalind Franklin.",
		},
		{
			Input:  "machine learning <sep> described_as <sep> field of artificial intelligence",
			Target: "Machine learning is a branch of artificial intelligence focused on learning from data. Algorithms identify patterns in large datasets and improve their performance without being explicitly programmed for each task.",
		},
		{
			Input:  "the solar system <sep> described_as <sep> planetary system",
			Target: "The solar system is a planetary system centered on the Sun. It contains eight planets, dozens of moons, and countless smaller bodies including asteroids and comets.",
		},
		{
			Input:  "Bitcoin <sep> described_as <sep> digital currency",
			Target: "Bitcoin is a decentralized digital currency introduced in 2009. It relies on blockchain technology to record transactions without the need for a central authority.",
		},
		{
			Input:  "photosynthesis <sep> described_as <sep> biological process",
			Target: "Photosynthesis is the process by which plants convert sunlight into chemical energy. Using carbon dioxide and water, chloroplasts produce glucose and release oxygen as a byproduct.",
		},
		{
			Input:  "general relativity <sep> described_as <sep> physics theory",
			Target: "General relativity is Albert Einstein's theory of gravitation published in 1915. It describes gravity not as a force but as the curvature of spacetime caused by mass and energy.",
		},
		{
			Input:  "the Renaissance <sep> described_as <sep> cultural movement",
			Target: "The Renaissance was a cultural movement that began in Italy in the 14th century. It marked a renewed interest in classical art, science, and philosophy, producing figures like Leonardo da Vinci and Michelangelo.",
		},
		{
			Input:  "electricity <sep> described_as <sep> form of energy",
			Target: "Electricity is a form of energy resulting from the movement of charged particles. It powers modern civilization, from household lighting to industrial machinery and digital communications.",
		},
		{
			Input:  "the human brain <sep> described_as <sep> organ",
			Target: "The human brain is the central organ of the nervous system. It contains roughly 86 billion neurons and is responsible for thought, memory, emotion, and coordination of body functions.",
		},
		{
			Input:  "democracy <sep> described_as <sep> system of government",
			Target: "Democracy is a system of government in which power is vested in the people. Citizens exercise authority through voting and elected representatives, a concept rooted in ancient Athens.",
		},
		{
			Input:  "penicillin <sep> described_as <sep> antibiotic",
			Target: "Penicillin is an antibiotic discovered by Alexander Fleming in 1928. Its mass production during World War II saved countless lives and launched the era of modern antibiotics.",
		},
		{
			Input:  "the Amazon rainforest <sep> described_as <sep> tropical forest",
			Target: "The Amazon rainforest is the largest tropical rainforest on Earth. Spanning much of South America, it harbors extraordinary biodiversity and plays a critical role in regulating the global climate.",
		},
		{
			Input:  "calculus <sep> described_as <sep> branch of mathematics",
			Target: "Calculus is a branch of mathematics dealing with rates of change and accumulation. Independently developed by Newton and Leibniz in the 17th century, it underpins modern physics and engineering.",
		},
		{
			Input:  "the World Wide Web <sep> described_as <sep> information system",
			Target: "The World Wide Web is an information system built on top of the internet. Invented by Tim Berners-Lee in 1989, it uses hyperlinks and URLs to connect documents across the globe.",
		},
		{
			Input:  "chess <sep> described_as <sep> strategy game",
			Target: "Chess is a two-player strategy board game played on a 64-square grid. Originating in India around the 6th century, it has become one of the most studied and widely played games in the world.",
		},
		{
			Input:  "vaccines <sep> described_as <sep> medical intervention",
			Target: "Vaccines are biological preparations that provide immunity against specific diseases. By stimulating the immune system to recognize pathogens, they have eradicated smallpox and drastically reduced polio.",
		},
		{
			Input:  "the Sahara <sep> described_as <sep> desert",
			Target: "The Sahara is the largest hot desert in the world, covering much of North Africa. Despite its arid conditions, it supports diverse ecosystems and has been inhabited by humans for thousands of years.",
		},
		{
			Input:  "philosophy <sep> described_as <sep> academic discipline",
			Target: "Philosophy is an academic discipline concerned with fundamental questions about existence, knowledge, and ethics. Its traditions stretch from ancient Greece through modern analytic and continental schools of thought.",
		},
		{
			Input:  "the printing press <sep> described_as <sep> invention",
			Target: "The printing press is an invention attributed to Johannes Gutenberg around 1440. By enabling the mass production of books, it democratized knowledge and accelerated the spread of the Renaissance and Reformation.",
		},
		{
			Input:  "GPS <sep> described_as <sep> navigation system",
			Target: "GPS is a satellite-based navigation system originally developed by the U.S. military. It provides precise location and time data to receivers anywhere on Earth, enabling applications from mapping to aviation.",
		},
		{
			Input:  "Linux <sep> described_as <sep> operating system",
			Target: "Linux is an open-source operating system kernel created by Linus Torvalds in 1991. It powers servers, smartphones, and supercomputers, and forms the foundation of distributions like Ubuntu and Fedora.",
		},
		{
			Input:  "jazz <sep> described_as <sep> music genre",
			Target: "Jazz is a music genre that originated in the African American communities of New Orleans in the late 19th century. It is characterized by swing, blue notes, and improvisation, and has influenced countless other genres.",
		},
		{
			Input:  "SpaceX <sep> described_as <sep> aerospace company",
			Target: "SpaceX is an American aerospace company founded by Elon Musk in 2002. It pioneered reusable orbital rockets and has become a major provider of commercial launch services.",
		},
		{
			Input:  "the periodic table <sep> described_as <sep> chemical classification",
			Target: "The periodic table is a tabular arrangement of chemical elements organized by atomic number. First published by Dmitri Mendeleev in 1869, it reveals recurring patterns in element properties.",
		},
		{
			Input:  "the Eiffel Tower <sep> described_as <sep> landmark",
			Target: "The Eiffel Tower is an iron lattice structure located in Paris, France. Built for the 1889 World's Fair, it stands 330 meters tall and is one of the most visited monuments in the world.",
		},
		{
			Input:  "natural selection <sep> described_as <sep> evolutionary mechanism",
			Target: "Natural selection is a mechanism of evolution proposed by Charles Darwin. Organisms with traits better suited to their environment tend to survive and reproduce, gradually shaping the diversity of life.",
		},
		{
			Input:  "cryptography <sep> described_as <sep> field of study",
			Target: "Cryptography is the practice of securing information through encoding techniques. From ancient ciphers to modern public-key algorithms, it underpins digital security, banking, and private communications.",
		},
	}
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
