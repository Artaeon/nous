package micromodel

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

	// Add clause-reordered variations to teach flexible word order
	examples = append(examples, clauseReorderedPairs()...)

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

// clauseReorderedPairs generates training pairs that express the same
// fact with different clause orderings. This teaches the model that a
// single triple can be rendered in multiple syntactic arrangements,
// improving fluency and reducing template-sounding output.
func clauseReorderedPairs() []TrainingExample {
	// Each entry maps one input triple to several reworded targets.
	type reorderGroup struct {
		input    string
		variants []string
	}

	groups := []reorderGroup{
		{
			input: "Python <sep> is_a <sep> programming language",
			variants: []string{
				"Python is a programming language.",
				"A programming language, Python is widely used.",
				"As a programming language, Python is versatile and popular.",
				"Among programming languages, Python stands out.",
			},
		},
		{
			input: "Bitcoin <sep> created_by <sep> Satoshi Nakamoto",
			variants: []string{
				"Bitcoin was created by Satoshi Nakamoto.",
				"Satoshi Nakamoto is the creator of Bitcoin.",
				"It was Satoshi Nakamoto who created Bitcoin.",
				"The creator behind Bitcoin is Satoshi Nakamoto.",
			},
		},
		{
			input: "Google <sep> founded_in <sep> 1998",
			variants: []string{
				"Google was founded in 1998.",
				"In 1998, Google was founded.",
				"The year 1998 saw the founding of Google.",
				"Founded in 1998, Google has grown into a tech giant.",
			},
		},
		{
			input: "the solar system <sep> has <sep> eight planets",
			variants: []string{
				"The solar system has eight planets.",
				"Eight planets make up the solar system.",
				"Within the solar system, there are eight planets.",
				"There are eight planets in the solar system.",
			},
		},
		{
			input: "machine learning <sep> used_for <sep> pattern recognition",
			variants: []string{
				"Machine learning is used for pattern recognition.",
				"Pattern recognition is one application of machine learning.",
				"For pattern recognition, machine learning is commonly employed.",
				"One can apply machine learning to pattern recognition.",
			},
		},
		{
			input: "the Eiffel Tower <sep> located_in <sep> Paris",
			variants: []string{
				"The Eiffel Tower is located in Paris.",
				"In Paris stands the Eiffel Tower.",
				"Paris is home to the Eiffel Tower.",
				"Located in Paris, the Eiffel Tower attracts millions of visitors.",
			},
		},
		{
			input: "Einstein <sep> known_for <sep> the theory of relativity",
			variants: []string{
				"Einstein is known for the theory of relativity.",
				"The theory of relativity is the work of Einstein.",
				"It is for the theory of relativity that Einstein is best known.",
				"When people think of Einstein, they think of the theory of relativity.",
			},
		},
		{
			input: "the heart <sep> part_of <sep> the circulatory system",
			variants: []string{
				"The heart is part of the circulatory system.",
				"The circulatory system includes the heart.",
				"As a component of the circulatory system, the heart pumps blood.",
				"Within the circulatory system, the heart plays a central role.",
			},
		},
		{
			input: "physics <sep> related_to <sep> mathematics",
			variants: []string{
				"Physics is related to mathematics.",
				"Mathematics and physics are closely related.",
				"There is a deep connection between physics and mathematics.",
				"Physics and mathematics share a long intertwined history.",
			},
		},
		{
			input: "jazz <sep> influenced_by <sep> blues and ragtime",
			variants: []string{
				"Jazz was influenced by blues and ragtime.",
				"Blues and ragtime were major influences on jazz.",
				"The roots of jazz lie in blues and ragtime.",
				"Without blues and ragtime, jazz would not exist as we know it.",
			},
		},
		{
			input: "Linux <sep> created_by <sep> Linus Torvalds",
			variants: []string{
				"Linux was created by Linus Torvalds.",
				"Linus Torvalds is the creator of Linux.",
				"It was Linus Torvalds who developed Linux.",
				"The development of Linux began with Linus Torvalds.",
			},
		},
		{
			input: "DNA <sep> is_a <sep> molecule",
			variants: []string{
				"DNA is a molecule.",
				"A molecule known as DNA carries genetic information.",
				"As a molecule, DNA is essential to all known life.",
				"Among biological molecules, DNA is perhaps the most important.",
			},
		},
		{
			input: "solar panels <sep> used_for <sep> generating electricity",
			variants: []string{
				"Solar panels are used for generating electricity.",
				"Generating electricity is the primary purpose of solar panels.",
				"For generating electricity, solar panels harness sunlight.",
				"Electricity can be generated using solar panels.",
			},
		},
		{
			input: "Shakespeare <sep> known_for <sep> his plays and sonnets",
			variants: []string{
				"Shakespeare is known for his plays and sonnets.",
				"The plays and sonnets of Shakespeare are celebrated worldwide.",
				"It is his plays and sonnets that make Shakespeare immortal.",
				"Among literary figures, Shakespeare is known for his plays and sonnets.",
			},
		},
		{
			input: "Mount Everest <sep> located_in <sep> the Himalayas",
			variants: []string{
				"Mount Everest is located in the Himalayas.",
				"In the Himalayas rises Mount Everest.",
				"The Himalayas are home to Mount Everest.",
				"Towering above the Himalayas, Mount Everest is the tallest peak on Earth.",
			},
		},
		{
			input: "the human body <sep> has <sep> 206 bones",
			variants: []string{
				"The human body has 206 bones.",
				"There are 206 bones in the human body.",
				"A total of 206 bones compose the human body.",
				"Within the human body, one finds 206 bones.",
			},
		},
		{
			input: "NASA <sep> founded_in <sep> 1958",
			variants: []string{
				"NASA was founded in 1958.",
				"In 1958, NASA was established.",
				"The founding of NASA took place in 1958.",
				"Since its founding in 1958, NASA has led space exploration.",
			},
		},
		{
			input: "electrons <sep> part_of <sep> atoms",
			variants: []string{
				"Electrons are part of atoms.",
				"Atoms contain electrons.",
				"Among the constituents of atoms are electrons.",
				"Every atom includes electrons orbiting its nucleus.",
			},
		},
		{
			input: "modern art <sep> influenced_by <sep> African sculpture",
			variants: []string{
				"Modern art was influenced by African sculpture.",
				"African sculpture had a profound influence on modern art.",
				"The influence of African sculpture reshaped modern art.",
				"Without African sculpture, modern art would have developed very differently.",
			},
		},
		{
			input: "cryptography <sep> used_for <sep> securing communications",
			variants: []string{
				"Cryptography is used for securing communications.",
				"Securing communications is a primary use of cryptography.",
				"To secure communications, one relies on cryptography.",
				"Communications are protected through the use of cryptography.",
			},
		},
	}

	var examples []TrainingExample
	for _, g := range groups {
		for _, v := range g.variants {
			examples = append(examples, TrainingExample{Input: g.input, Target: v})
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

// ExtractParagraphPairs creates (topic_prompt, paragraph) training pairs
// for freeform paragraph generation. Instead of decomposing into triples,
// this preserves the full paragraph as the training target.
// Input format: "<bos> topic <sep> describe"
// Target: full paragraph text (truncated to maxSeqLen tokens)
func ExtractParagraphPairs(knowledgeDir string, maxTargetLen int) []TrainingExample {
	var examples []TrainingExample

	if maxTargetLen <= 0 {
		maxTargetLen = 250 // ~200 words
	}

	files, err := filepath.Glob(filepath.Join(knowledgeDir, "*.txt"))
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
			if len(para) < 100 {
				continue // skip short paragraphs
			}

			// Extract topic from first sentence
			sentences := splitKnowledgeSentences(para)
			if len(sentences) == 0 {
				continue
			}
			topic := extractSentenceTopic(sentences[0])
			if topic == "" {
				continue
			}

			// Truncate paragraph to maxTargetLen characters
			target := para
			if len(target) > maxTargetLen {
				// Truncate at sentence boundary
				truncated := target[:maxTargetLen]
				if lastDot := strings.LastIndex(truncated, ". "); lastDot > 100 {
					truncated = truncated[:lastDot+1]
				}
				target = truncated
			}

			// Create the freeform training pair
			input := fmt.Sprintf("<bos> %s <sep> describe", strings.ToLower(topic))
			examples = append(examples, TrainingExample{Input: input, Target: target})

			// Also create a "what is" variant
			input2 := fmt.Sprintf("<bos> what is %s <sep> explain", strings.ToLower(topic))
			examples = append(examples, TrainingExample{Input: input2, Target: target})
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

// ExtractCausalEdgesFromKnowledge scans knowledge text files for causal
// relationships and returns them as adjacency lists keyed by subject.
// Used by GenerateCausalChainPairs to create multi-hop training sequences.
func ExtractCausalEdgesFromKnowledge(knowledgeDir string) map[string][][3]string {
	edges := make(map[string][][3]string)

	files, err := filepath.Glob(filepath.Join(knowledgeDir, "*.txt"))
	if err != nil || len(files) == 0 {
		return edges
	}

	causalPatterns := []struct {
		re       *regexp.Regexp
		relation string
	}{
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+causes?\s+(.{3,40}?)(?:\.|,|;)`), "causes"},
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+enables?\s+(.{3,40}?)(?:\.|,|;)`), "enables"},
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+produces?\s+(.{3,40}?)(?:\.|,|;)`), "produces"},
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+prevents?\s+(.{3,40}?)(?:\.|,|;)`), "prevents"},
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+requires?\s+(.{3,40}?)(?:\.|,|;)`), "requires"},
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+leads?\s+to\s+(.{3,40}?)(?:\.|,|;)`), "causes"},
		{regexp.MustCompile(`(?i)(.{3,40}?)\s+results?\s+in\s+(.{3,40}?)(?:\.|,|;)`), "causes"},
	}

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		text := string(data)
		for _, pat := range causalPatterns {
			matches := pat.re.FindAllStringSubmatch(text, -1)
			for _, m := range matches {
				if len(m) >= 3 {
					subj := strings.TrimSpace(m[1])
					obj := strings.TrimSpace(m[2])
					if len(subj) > 2 && len(obj) > 2 {
						triple := [3]string{subj, pat.relation, obj}
						edges[subj] = append(edges[subj], triple)
					}
				}
			}
		}
	}

	return edges
}

// -----------------------------------------------------------------------
// Causal Chain Training Data — graph-walk sequences for Mamba SSM.
//
// Instead of isolated (triple, sentence) pairs, these are multi-hop
// causal sequences that teach the Mamba's hidden state to model
// causal propagation as temporal sequence dynamics.
//
// Three types:
//   1. Forward chains:      A causes B causes C → multi-sentence explanation
//   2. Counterfactual:      negate(A) → B doesn't happen → C doesn't happen
//   3. Branching:           A causes {B, C, D} → explains branching effects
//
// The SSM's h_t = Abar * h_{t-1} + Bbar * x_t naturally models state
// evolution — a causal chain IS a state evolution. Training on graph-walk
// sequences teaches the hidden state to carry "world state" through
// causal transitions.
// -----------------------------------------------------------------------

// CausalChainTriple represents one link in a causal chain.
type CausalChainTriple struct {
	Subject  string
	Relation string
	Object   string
}

// GenerateCausalChainPairs creates multi-hop causal training sequences
// from graph edges. Takes (subject, relation, object) triples organized
// as adjacency lists keyed by subject.
func GenerateCausalChainPairs(edges map[string][][3]string) []TrainingExample {
	var examples []TrainingExample

	// Build adjacency for causal edges.
	causalRels := map[string]bool{
		"causes": true, "enables": true, "produces": true,
		"prevents": true, "requires": true, "follows": true,
	}

	adj := make(map[string][][3]string)
	for _, triples := range edges {
		for _, t := range triples {
			if causalRels[t[1]] {
				adj[t[0]] = append(adj[t[0]], t)
			}
		}
	}

	// Generate forward chains (2-4 hops).
	for start, startEdges := range adj {
		for _, e1 := range startEdges {
			target1 := e1[2]

			// 2-hop chain.
			chain2Input := fmt.Sprintf("%s <sep> %s <sep> %s", e1[0], e1[1], e1[2])
			chain2Target := composeChainSentence([]CausalChainTriple{
				{e1[0], e1[1], e1[2]},
			})
			examples = append(examples, TrainingExample{Input: chain2Input, Target: chain2Target})

			// 3-hop.
			if hop2Edges, ok := adj[target1]; ok {
				for _, e2 := range hop2Edges {
					chain3Input := fmt.Sprintf("%s <sep> %s <sep> %s <sep> %s <sep> %s",
						e1[0], e1[1], e1[2], e2[1], e2[2])
					chain3Target := composeChainSentence([]CausalChainTriple{
						{e1[0], e1[1], e1[2]},
						{e2[0], e2[1], e2[2]},
					})
					examples = append(examples, TrainingExample{Input: chain3Input, Target: chain3Target})

					// 4-hop.
					if hop3Edges, ok := adj[e2[2]]; ok {
						for _, e3 := range hop3Edges[:min(len(hop3Edges), 2)] {
							chain4Input := fmt.Sprintf("%s <sep> %s <sep> %s <sep> %s <sep> %s <sep> %s <sep> %s",
								e1[0], e1[1], e1[2], e2[1], e2[2], e3[1], e3[2])
							chain4Target := composeChainSentence([]CausalChainTriple{
								{e1[0], e1[1], e1[2]},
								{e2[0], e2[1], e2[2]},
								{e3[0], e3[1], e3[2]},
							})
							examples = append(examples, TrainingExample{Input: chain4Input, Target: chain4Target})
						}
					}
				}
			}

			// Counterfactual: negate the first cause.
			counterInput := fmt.Sprintf("<negate> %s <sep> %s <sep> %s", e1[0], e1[1], e1[2])
			counterTarget := composeCounterfactualSentence(CausalChainTriple{e1[0], e1[1], e1[2]})
			examples = append(examples, TrainingExample{Input: counterInput, Target: counterTarget})

			_ = start
		}

		// Branching: 2+ causal outgoing edges.
		if len(startEdges) >= 2 {
			var chains []CausalChainTriple
			for _, e := range startEdges[:min(len(startEdges), 4)] {
				chains = append(chains, CausalChainTriple{e[0], e[1], e[2]})
			}
			branchInput := fmt.Sprintf("<branch> %s <sep> %d effects", start, len(chains))
			branchTarget := composeBranchSentence(start, chains)
			examples = append(examples, TrainingExample{Input: branchInput, Target: branchTarget})
		}
	}

	return examples
}

func composeChainSentence(chain []CausalChainTriple) string {
	var parts []string
	for i, link := range chain {
		verb := humanizeCausalVerb(link.Relation)
		if i == 0 {
			parts = append(parts, fmt.Sprintf("%s %s %s.", causalCapitalize(link.Subject), verb, link.Object))
		} else {
			parts = append(parts, fmt.Sprintf("This in turn %s %s.", verb, link.Object))
		}
	}
	return strings.Join(parts, " ")
}

func composeCounterfactualSentence(link CausalChainTriple) string {
	return fmt.Sprintf("Without %s, %s would not occur. This would remove a key factor affecting %s.",
		link.Subject, link.Object, link.Object)
}

func composeBranchSentence(source string, effects []CausalChainTriple) string {
	var parts []string
	parts = append(parts, fmt.Sprintf("%s has multiple effects.", causalCapitalize(source)))
	for _, e := range effects {
		verb := humanizeCausalVerb(e.Relation)
		parts = append(parts, fmt.Sprintf("It %s %s.", verb, e.Object))
	}
	return strings.Join(parts, " ")
}

func humanizeCausalVerb(rel string) string {
	switch rel {
	case "causes":
		return "causes"
	case "enables":
		return "enables"
	case "produces":
		return "produces"
	case "prevents":
		return "prevents"
	case "requires":
		return "requires"
	case "follows":
		return "leads to"
	default:
		return "affects"
	}
}

func causalCapitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
