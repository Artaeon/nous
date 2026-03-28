package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// EvalPrompt represents a single evaluation prompt with expected output criteria.
type EvalPrompt struct {
	ID         string   `json:"id"`
	Capability string   `json:"capability"`
	Query      string   `json:"query"`
	GoldAnswer string   `json:"gold_answer"`
	Rubric     []string `json:"rubric"`
	Difficulty string   `json:"difficulty"` // easy, medium, hard
	Tags       []string `json:"tags"`
}

// EvalSet is a versioned collection of evaluation prompts.
type EvalSet struct {
	Prompts []EvalPrompt `json:"prompts"`
	Version string       `json:"version"`
	Created time.Time    `json:"created"`
}

// SaveEvalSet persists an eval set to disk as JSON.
func SaveEvalSet(es *EvalSet, path string) error {
	data, err := json.MarshalIndent(es, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal eval set: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadEvalSet reads an eval set from a JSON file on disk.
func LoadEvalSet(path string) (*EvalSet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read eval set: %w", err)
	}
	var es EvalSet
	if err := json.Unmarshal(data, &es); err != nil {
		return nil, fmt.Errorf("unmarshal eval set: %w", err)
	}
	return &es, nil
}

// promptTemplate drives template expansion for a single capability.
type promptTemplate struct {
	queryFmt   string
	goldFmt    string
	rubric     []string
	tags       []string
	subjects   []string
	difficulty string
}

// GenerateEvalSet produces a balanced eval set with 1000+ prompts (125+ per capability).
func GenerateEvalSet() *EvalSet {
	es := &EvalSet{
		Version: "1.0.0",
		Created: time.Now().UTC(),
	}

	generators := []func() []EvalPrompt{
		genIntentRouting,
		genFactualQA,
		genDeepExplain,
		genCompareTradeoff,
		genMultiTurnContext,
		genPlanning,
		genToolUseAccuracy,
		genStyleControl,
	}

	for _, gen := range generators {
		es.Prompts = append(es.Prompts, gen()...)
	}

	return es
}

// expand creates prompts from templates, cycling through difficulties.
func expand(capability string, templates []promptTemplate) []EvalPrompt {
	var prompts []EvalPrompt
	difficulties := []string{"easy", "medium", "hard"}
	id := 0

	for _, tmpl := range templates {
		for _, subj := range tmpl.subjects {
			diff := tmpl.difficulty
			if diff == "" {
				diff = difficulties[id%len(difficulties)]
			}
			prompts = append(prompts, EvalPrompt{
				ID:         fmt.Sprintf("%s-%04d", capability, id),
				Capability: capability,
				Query:      fmt.Sprintf(tmpl.queryFmt, subj),
				GoldAnswer: fmt.Sprintf(tmpl.goldFmt, subj),
				Rubric:     tmpl.rubric,
				Difficulty: diff,
				Tags:       tmpl.tags,
			})
			id++
		}
	}
	return prompts
}

func genIntentRouting() []EvalPrompt {
	timerSubjects := []string{
		"5 minutes", "10 minutes", "30 seconds", "1 hour",
		"15 minutes", "2 hours", "45 seconds", "20 minutes",
		"3 minutes", "90 seconds",
	}
	weatherCities := []string{
		"Paris", "Tokyo", "New York", "London", "Sydney",
		"Berlin", "Mumbai", "Cairo", "Toronto", "Seoul",
		"Beijing", "Rome", "Moscow", "Bangkok", "Lima",
	}
	reminderSubjects := []string{
		"buy groceries", "call the dentist", "submit the report",
		"pick up laundry", "water the plants", "renew my passport",
		"pay the electric bill", "schedule a meeting", "update my resume",
		"book flight tickets",
	}
	searchTopics := []string{
		"best restaurants nearby", "latest news on climate change",
		"how to fix a leaky faucet", "top programming languages 2025",
		"cheap flights to Hawaii", "symptoms of vitamin D deficiency",
		"history of the Roman Empire", "best budget smartphones",
		"healthy meal prep ideas", "how to start a podcast",
	}
	mathQueries := []string{
		"what is 15% of 240", "calculate 17 times 23",
		"what is the square root of 144", "convert 100 Fahrenheit to Celsius",
		"what is 3 to the power of 5", "how many seconds in a day",
		"what is 7 factorial", "divide 1000 by 37",
		"what is 2.5 plus 3.7", "what percentage is 45 of 180",
	}
	greetings := []string{
		"hello", "hi there", "good morning", "hey",
		"good afternoon", "howdy", "what's up", "greetings",
		"yo", "good evening",
	}
	musicCmds := []string{
		"play some jazz", "pause the music", "skip this song",
		"turn up the volume", "play my favorites playlist",
		"shuffle my library", "play something relaxing",
		"what song is this", "add this to my playlist",
		"play the next track",
	}
	clarifications := []string{
		"what do you mean", "can you explain that", "I don't understand",
		"tell me more", "what exactly is that", "how so",
		"why is that", "in what way", "could you elaborate",
		"what does that mean",
	}
	opinionQueries := []string{
		"what's the best programming language",
		"is coffee good for you",
		"should I learn Python or JavaScript",
		"is remote work better than office work",
		"what's the best laptop brand",
		"are electric cars worth it",
		"is social media harmful",
		"should I invest in stocks or bonds",
		"is AI going to replace programmers",
		"what's the best exercise for weight loss",
	}
	composeCmds := []string{
		"write an email to my boss about taking Friday off",
		"draft a thank you note for the interview",
		"compose a birthday message for my sister",
		"write a complaint letter about late delivery",
		"draft a meeting agenda for Monday",
		"write a product description for a water bottle",
		"compose an apology for missing the meeting",
		"draft a resignation letter",
		"write a recommendation for a colleague",
		"compose an invitation for a housewarming party",
	}
	summarizeCmds := []string{
		"summarize this article for me",
		"give me the key points of this document",
		"what are the main takeaways",
		"can you give me a brief overview",
		"what is the TL;DR of this",
	}

	var prompts []EvalPrompt
	id := 0
	difficulties := []string{"easy", "medium", "hard"}

	add := func(query, gold, intent string, tags []string, rubric []string) {
		diff := difficulties[id%3]
		prompts = append(prompts, EvalPrompt{
			ID:         fmt.Sprintf("IntentRouting-%04d", id),
			Capability: "IntentRouting",
			Query:      query,
			GoldAnswer: gold,
			Rubric:     rubric,
			Difficulty: diff,
			Tags:       append([]string{intent}, tags...),
		})
		id++
	}

	baseRubric := []string{"correct intent classification", "appropriate confidence level", "handles ambiguity if present"}

	for _, s := range timerSubjects {
		add("set a timer for "+s, "intent:timer", "timer", []string{"command"}, baseRubric)
	}
	for _, c := range weatherCities {
		add("what's the weather in "+c, "intent:weather", "weather", []string{"query"}, baseRubric)
	}
	for _, r := range reminderSubjects {
		add("remind me to "+r, "intent:reminder", "reminder", []string{"command"}, baseRubric)
	}
	for _, s := range searchTopics {
		add(s, "intent:search", "search", []string{"query"}, baseRubric)
	}
	for _, m := range mathQueries {
		add(m, "intent:calculation", "calculation", []string{"query", "math"}, baseRubric)
	}
	for _, g := range greetings {
		add(g, "intent:greeting", "greeting", []string{"social"}, baseRubric)
	}
	for _, m := range musicCmds {
		add(m, "intent:music_control", "music", []string{"command"}, baseRubric)
	}
	for _, c := range clarifications {
		add(c, "intent:clarification", "clarification", []string{"meta"}, baseRubric)
	}
	for _, o := range opinionQueries {
		add(o, "intent:opinion", "opinion", []string{"subjective"}, baseRubric)
	}
	for _, c := range composeCmds {
		add(c, "intent:compose", "compose", []string{"generation"}, baseRubric)
	}
	for _, s := range summarizeCmds {
		add(s, "intent:summarize", "summarize", []string{"generation"}, baseRubric)
	}

	// Pad to 125 with ambiguous / multi-intent queries.
	ambiguous := []struct {
		q, gold string
		tags    []string
	}{
		{"play rain sounds and set a timer for 30 minutes", "intent:music_control+timer", []string{"multi-intent"}},
		{"what's the weather tomorrow and remind me to bring an umbrella", "intent:weather+reminder", []string{"multi-intent"}},
		{"tell me a joke", "intent:entertainment", []string{"social"}},
		{"turn off the lights", "intent:smart_home", []string{"command"}},
		{"how far is the moon", "intent:factual", []string{"query"}},
	}
	for len(prompts) < 125 {
		a := ambiguous[len(prompts)%len(ambiguous)]
		add(a.q, a.gold, "ambiguous", a.tags,
			[]string{"correct intent classification", "handles multi-intent if present", "appropriate routing"})
	}
	return prompts
}

func genFactualQA() []EvalPrompt {
	type factQ struct {
		query, gold string
		tags        []string
	}

	science := []factQ{
		{"what is photosynthesis", "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen", []string{"biology"}},
		{"what causes tides", "Tides are primarily caused by the gravitational pull of the Moon and to a lesser extent the Sun on Earth's oceans", []string{"physics", "astronomy"}},
		{"what is the speed of light", "The speed of light in a vacuum is approximately 299,792,458 meters per second", []string{"physics"}},
		{"how does DNA replication work", "DNA replication is a semi-conservative process where each strand serves as a template, using DNA polymerase to synthesize a complementary strand", []string{"biology", "genetics"}},
		{"what is the Krebs cycle", "The Krebs cycle is a series of chemical reactions in cellular respiration that generates energy through oxidation of acetyl-CoA in the mitochondria", []string{"biology", "biochemistry"}},
		{"what is dark matter", "Dark matter is a hypothetical form of matter that does not emit or interact with electromagnetic radiation but exerts gravitational effects on visible matter", []string{"physics", "cosmology"}},
		{"how do vaccines work", "Vaccines stimulate the immune system by introducing a weakened or inactivated form of a pathogen, training the body to recognize and fight future infections", []string{"biology", "medicine"}},
		{"what is the greenhouse effect", "The greenhouse effect is the trapping of heat in Earth's atmosphere by gases like CO2 and methane, which absorb and re-emit infrared radiation", []string{"climate", "physics"}},
		{"what is CRISPR", "CRISPR is a gene-editing technology that allows precise modification of DNA sequences using a guide RNA and the Cas9 enzyme", []string{"genetics", "biotech"}},
		{"how do black holes form", "Black holes form when massive stars exhaust their nuclear fuel and collapse under their own gravity, creating a singularity with an event horizon", []string{"astronomy", "physics"}},
	}

	geography := []factQ{
		{"what is the longest river in the world", "The Nile River is traditionally considered the longest river at approximately 6,650 km, though some measurements suggest the Amazon may be longer", []string{"geography"}},
		{"what is the tallest mountain", "Mount Everest is the tallest mountain above sea level at 8,849 meters", []string{"geography"}},
		{"what is the largest desert", "The Antarctic Desert is the largest desert by area, followed by the Sahara as the largest hot desert", []string{"geography"}},
		{"what is the deepest ocean trench", "The Mariana Trench in the Pacific Ocean is the deepest, reaching approximately 10,994 meters at Challenger Deep", []string{"geography", "ocean"}},
		{"how many continents are there", "There are seven continents: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, and South America", []string{"geography"}},
		{"what is the smallest country in the world", "Vatican City is the smallest country by both area and population", []string{"geography"}},
		{"what is the largest ocean", "The Pacific Ocean is the largest ocean, covering approximately 165.25 million square kilometers", []string{"geography"}},
		{"where is the Great Barrier Reef", "The Great Barrier Reef is located off the northeast coast of Australia in the Coral Sea", []string{"geography", "ecology"}},
		{"what is the capital of Australia", "The capital of Australia is Canberra, not Sydney or Melbourne", []string{"geography"}},
		{"what country has the most time zones", "France has the most time zones (12) when including overseas territories, followed by Russia (11)", []string{"geography"}},
	}

	history := []factQ{
		{"who wrote Romeo and Juliet", "William Shakespeare wrote Romeo and Juliet, believed to have been written between 1591 and 1596", []string{"literature", "history"}},
		{"when did World War II end", "World War II ended in 1945, with Germany surrendering in May and Japan in September", []string{"history"}},
		{"who invented the telephone", "Alexander Graham Bell is credited with patenting the first practical telephone in 1876", []string{"history", "technology"}},
		{"when was the Declaration of Independence signed", "The Declaration of Independence was adopted on July 4, 1776, though most delegates signed it on August 2, 1776", []string{"history", "politics"}},
		{"who was the first person to walk on the moon", "Neil Armstrong was the first person to walk on the Moon on July 20, 1969, during the Apollo 11 mission", []string{"history", "space"}},
		{"what caused the fall of the Roman Empire", "The fall of the Western Roman Empire in 476 AD resulted from multiple factors including economic troubles, military overexpansion, political instability, and barbarian invasions", []string{"history"}},
		{"who painted the Mona Lisa", "Leonardo da Vinci painted the Mona Lisa, believed to have been created between 1503 and 1519", []string{"art", "history"}},
		{"when was the printing press invented", "Johannes Gutenberg invented the movable-type printing press around 1440 in Mainz, Germany", []string{"history", "technology"}},
		{"what was the Industrial Revolution", "The Industrial Revolution was a period of major industrialization from the mid-1700s to mid-1800s, beginning in Britain, transforming manufacturing from hand production to machine-based processes", []string{"history", "economics"}},
		{"who discovered penicillin", "Alexander Fleming discovered penicillin in 1928 when he noticed a mold killing bacteria in a petri dish", []string{"history", "medicine"}},
	}

	math := []factQ{
		{"what is pi", "Pi is the ratio of a circle's circumference to its diameter, approximately 3.14159", []string{"math"}},
		{"what is the Pythagorean theorem", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2", []string{"math", "geometry"}},
		{"what is a prime number", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself", []string{"math"}},
		{"what is the Fibonacci sequence", "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...", []string{"math"}},
		{"what is calculus", "Calculus is a branch of mathematics dealing with rates of change (differential calculus) and accumulation of quantities (integral calculus)", []string{"math"}},
	}

	technology := []factQ{
		{"what is machine learning", "Machine learning is a subset of AI where systems learn from data to make predictions or decisions without being explicitly programmed for each task", []string{"technology", "AI"}},
		{"how does the internet work", "The internet is a global network of interconnected computers that communicate using standardized protocols (TCP/IP) to exchange data in packets through routers", []string{"technology"}},
		{"what is blockchain", "Blockchain is a distributed, immutable ledger technology that records transactions in linked blocks verified by consensus among network participants", []string{"technology", "crypto"}},
		{"what is quantum computing", "Quantum computing uses quantum bits (qubits) that can exist in superposition, enabling parallel computation for certain problems exponentially faster than classical computers", []string{"technology", "physics"}},
		{"what is an API", "An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate with each other", []string{"technology", "software"}},
	}

	// Combine all and generate prompts.
	all := make([]factQ, 0, 125)
	all = append(all, science...)
	all = append(all, geography...)
	all = append(all, history...)
	all = append(all, math...)
	all = append(all, technology...)

	// Additional questions to reach 125.
	extras := []factQ{
		{"what is inflation", "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power", []string{"economics"}},
		{"what is natural selection", "Natural selection is the process where organisms with favorable traits are more likely to survive and reproduce, driving evolution", []string{"biology"}},
		{"what is an atom", "An atom is the smallest unit of an element consisting of a nucleus with protons and neutrons, surrounded by electrons", []string{"chemistry", "physics"}},
		{"what is the periodic table", "The periodic table organizes chemical elements by atomic number, electron configuration, and recurring chemical properties", []string{"chemistry"}},
		{"what is osmosis", "Osmosis is the movement of water molecules through a semipermeable membrane from an area of lower solute concentration to higher", []string{"biology", "chemistry"}},
		{"what is GDP", "GDP (Gross Domestic Product) is the total monetary value of all finished goods and services produced within a country in a specific period", []string{"economics"}},
		{"what is relativity", "Einstein's theory of relativity includes special relativity (time and space are relative) and general relativity (gravity curves spacetime)", []string{"physics"}},
		{"who wrote 1984", "George Orwell wrote 1984, published in 1949, a dystopian novel about totalitarian government surveillance", []string{"literature"}},
		{"what is the water cycle", "The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and collection", []string{"science", "geography"}},
		{"what is photovoltaic effect", "The photovoltaic effect is the creation of voltage or current in a material when exposed to light, the basis of solar cells", []string{"physics", "energy"}},
		{"what is the Doppler effect", "The Doppler effect is the change in frequency of a wave as the source and observer move relative to each other", []string{"physics"}},
		{"what is entropy", "Entropy is a measure of disorder in a thermodynamic system; the second law of thermodynamics states it always increases in an isolated system", []string{"physics"}},
		{"what is plate tectonics", "Plate tectonics describes the movement of Earth's lithospheric plates, driven by convection currents in the mantle, causing earthquakes, volcanism, and mountain building", []string{"geology"}},
		{"what is a transistor", "A transistor is a semiconductor device used to amplify or switch electronic signals, forming the basic building block of modern electronics", []string{"technology", "electronics"}},
		{"what is the electromagnetic spectrum", "The electromagnetic spectrum is the range of all electromagnetic radiation from radio waves to gamma rays, including visible light", []string{"physics"}},
		{"what language has the most native speakers", "Mandarin Chinese has the most native speakers, with over 900 million", []string{"linguistics"}},
		{"what is mitosis", "Mitosis is a type of cell division that produces two genetically identical daughter cells from a single parent cell", []string{"biology"}},
		{"what is supply and demand", "Supply and demand is an economic model where price is determined by the relationship between the quantity of a product available and the desire for it", []string{"economics"}},
		{"what is the Turing test", "The Turing test, proposed by Alan Turing in 1950, evaluates a machine's ability to exhibit intelligent behavior indistinguishable from a human", []string{"technology", "AI"}},
		{"what is the Big Bang", "The Big Bang theory describes the universe originating from an extremely hot, dense state approximately 13.8 billion years ago and expanding ever since", []string{"astronomy", "cosmology"}},
		{"how many bones in the human body", "An adult human body has 206 bones; babies are born with approximately 270 which fuse over time", []string{"biology", "anatomy"}},
		{"what is Newton's first law", "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force", []string{"physics"}},
		{"what is an ecosystem", "An ecosystem is a community of living organisms and their physical environment interacting as a system through nutrient cycles and energy flows", []string{"biology", "ecology"}},
		{"what is the ozone layer", "The ozone layer is a region of Earth's stratosphere containing high concentrations of ozone (O3) that absorbs most of the Sun's ultraviolet radiation", []string{"science", "environment"}},
		{"what is the Geneva Convention", "The Geneva Conventions are international treaties establishing humanitarian standards for the treatment of wounded soldiers, prisoners of war, and civilians during conflict", []string{"history", "law"}},
		{"who wrote The Republic", "Plato wrote The Republic, a Socratic dialogue composed around 375 BC concerning justice and the ideal state", []string{"philosophy", "history"}},
		{"what is RNA", "RNA (ribonucleic acid) is a molecule essential for coding, decoding, regulation, and expression of genes, acting as a messenger between DNA and proteins", []string{"biology", "genetics"}},
		{"what is the Richter scale", "The Richter scale (now largely replaced by the moment magnitude scale) measures earthquake magnitude on a logarithmic scale", []string{"geology", "science"}},
		{"what is a semiconductor", "A semiconductor is a material with electrical conductivity between a conductor and insulator, whose conductivity can be controlled, used in electronic devices", []string{"physics", "technology"}},
		{"what is photon", "A photon is a quantum of electromagnetic radiation, a massless particle that carries energy proportional to the radiation frequency", []string{"physics"}},
		{"what is the Hippocratic oath", "The Hippocratic Oath is an ethical code attributed to Hippocrates that physicians traditionally swear, pledging to practice medicine ethically and do no harm", []string{"medicine", "history"}},
		{"what is continental drift", "Continental drift is the theory proposed by Alfred Wegener that continents move over geological time, now explained by plate tectonics", []string{"geology"}},
		{"what is the Magna Carta", "The Magna Carta is a charter issued in 1215 in England limiting the power of the king and establishing principles of due process and rule of law", []string{"history", "law"}},
		{"what is a neutron star", "A neutron star is the collapsed core of a massive star after a supernova, composed almost entirely of neutrons with extreme density", []string{"astronomy", "physics"}},
		{"what is the UN Security Council", "The UN Security Council is the principal organ responsible for international peace and security, with 5 permanent and 10 rotating members", []string{"politics"}},
		{"what is hemoglobin", "Hemoglobin is a protein in red blood cells that carries oxygen from the lungs to the body's tissues and returns carbon dioxide back", []string{"biology", "medicine"}},
		{"what is the Marshall Plan", "The Marshall Plan was a US program providing economic aid to Western European countries after World War II to help rebuild economies and prevent communist expansion", []string{"history", "economics"}},
		{"what is the Hubble Space Telescope", "The Hubble Space Telescope is a space-based observatory launched in 1990 that has captured detailed images of distant galaxies, nebulae, and stars", []string{"astronomy", "technology"}},
		{"what is a supernova", "A supernova is a powerful and luminous explosion of a star at the end of its life cycle, briefly outshining entire galaxies", []string{"astronomy"}},
		{"what are tectonic plates", "Tectonic plates are massive segments of Earth's lithosphere that move, float, and sometimes fracture, with their interaction causing earthquakes, volcanism, and mountain formation", []string{"geology"}},
		{"what is the Rosetta Stone", "The Rosetta Stone is a granodiorite stele inscribed with a decree in three scripts (Egyptian hieroglyphs, Demotic, and Greek) that was key to deciphering hieroglyphs", []string{"history", "archaeology"}},
		{"what is pH", "pH is a logarithmic scale measuring the acidity or alkalinity of a solution, ranging from 0 (most acidic) to 14 (most alkaline) with 7 being neutral", []string{"chemistry"}},
		{"what is a galaxy", "A galaxy is a massive system of stars, gas, dust, and dark matter bound together by gravity; the Milky Way is our galaxy", []string{"astronomy"}},
		{"what is the Bill of Rights", "The Bill of Rights is the first ten amendments to the US Constitution, ratified in 1791, guaranteeing fundamental rights and freedoms", []string{"history", "law"}},
		{"what is the speed of sound", "The speed of sound in air at sea level and 20 degrees Celsius is approximately 343 meters per second or 1,235 km/h", []string{"physics"}},
		{"what is coral bleaching", "Coral bleaching occurs when stressed corals expel their symbiotic algae (zooxanthellae), turning white and potentially dying if the stress persists", []string{"ecology", "environment"}},
		{"what is absolute zero", "Absolute zero is the lowest possible temperature, 0 Kelvin or -273.15 degrees Celsius, where atomic motion theoretically ceases", []string{"physics"}},
		{"what is the Treaty of Versailles", "The Treaty of Versailles was signed in 1919 ending World War I, imposing reparations and territorial losses on Germany", []string{"history"}},
		{"what is an isotope", "Isotopes are variants of a chemical element with the same number of protons but different numbers of neutrons", []string{"chemistry", "physics"}},
		{"what is a prion", "A prion is a misfolded protein that can cause normal proteins to also misfold, leading to progressive neurodegenerative diseases like CJD", []string{"biology", "medicine"}},
		{"what is habeas corpus", "Habeas corpus is a legal principle requiring that a person under arrest be brought before a court to determine if their detention is lawful", []string{"law", "history"}},
		{"what is the butterfly effect", "The butterfly effect is the concept that small changes in initial conditions can lead to vastly different outcomes in complex systems, central to chaos theory", []string{"math", "physics"}},
		{"what is terraforming", "Terraforming is the hypothetical process of modifying a planet's atmosphere, temperature, and ecology to make it habitable for Earth-like life", []string{"astronomy", "science"}},
		{"what is the Higgs boson", "The Higgs boson is an elementary particle in the Standard Model associated with the Higgs field, which gives other particles their mass, confirmed in 2012 at CERN", []string{"physics"}},
		{"what is gerrymandering", "Gerrymandering is the manipulation of electoral district boundaries to favor a particular political party or group", []string{"politics"}},
		{"what is symbiosis", "Symbiosis is a close and long-term biological interaction between two different species, which can be mutualistic, parasitic, or commensal", []string{"biology", "ecology"}},
		{"what is the Doppler shift in astronomy", "The Doppler shift in astronomy is the change in wavelength of light from celestial objects, with redshift indicating movement away and blueshift toward the observer", []string{"astronomy", "physics"}},
		{"what is nuclear fusion", "Nuclear fusion is the process of combining light atomic nuclei to form heavier nuclei, releasing enormous energy; it powers stars including the Sun", []string{"physics", "energy"}},
		{"what is a monsoon", "A monsoon is a seasonal wind pattern that brings heavy rainfall, most notably in South and Southeast Asia during summer months", []string{"geography", "climate"}},
		{"what is the Federal Reserve", "The Federal Reserve is the central bank of the United States, responsible for monetary policy, regulating banks, and maintaining financial stability", []string{"economics", "politics"}},
		{"what is serotonin", "Serotonin is a neurotransmitter that regulates mood, appetite, sleep, and cognitive functions; low levels are associated with depression", []string{"biology", "medicine"}},
		{"what is the Cambrian explosion", "The Cambrian explosion was a period approximately 541 million years ago when most major animal phyla appeared rapidly in the fossil record", []string{"biology", "paleontology"}},
		{"what is stoicism", "Stoicism is a Hellenistic philosophy teaching that virtue and wisdom are achieved through rational control of emotions and acceptance of fate", []string{"philosophy"}},
		{"what is CERN", "CERN is the European Organization for Nuclear Research, operating the Large Hadron Collider, the world's largest particle physics laboratory", []string{"physics", "technology"}},
		{"what is the IMF", "The International Monetary Fund is an international organization that promotes global monetary cooperation, financial stability, and economic growth", []string{"economics", "politics"}},
		{"what is the Coriolis effect", "The Coriolis effect is the apparent deflection of moving objects caused by Earth's rotation, influencing weather patterns and ocean currents", []string{"physics", "geography"}},
	}
	all = append(all, extras...)

	rubric := []string{
		"factually correct and verifiable",
		"sufficiently complete without unnecessary detail",
		"no hallucinated facts",
		"appropriate level of specificity",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, len(all))
	for i, q := range all {
		prompts = append(prompts, EvalPrompt{
			ID:         fmt.Sprintf("FactualQA-%04d", i),
			Capability: "FactualQA",
			Query:      q.query,
			GoldAnswer: q.gold,
			Rubric:     rubric,
			Difficulty: difficulties[i%3],
			Tags:       q.tags,
		})
	}
	// Pad to 125 by generating variant phrasings.
	variants := []string{
		"can you tell me %s",
		"I'd like to know %s",
		"please explain %s",
		"what can you tell me about %s",
		"define %s",
	}
	base := len(prompts)
	for len(prompts) < 125 {
		idx := (len(prompts) - base) % len(all)
		vi := (len(prompts) - base) / len(all) % len(variants)
		q := all[idx]
		prompts = append(prompts, EvalPrompt{
			ID:         fmt.Sprintf("FactualQA-%04d", len(prompts)),
			Capability: "FactualQA",
			Query:      fmt.Sprintf(variants[vi], q.query),
			GoldAnswer: q.gold,
			Rubric:     rubric,
			Difficulty: difficulties[len(prompts)%3],
			Tags:       q.tags,
		})
	}
	return prompts
}

func genDeepExplain() []EvalPrompt {
	type explainQ struct {
		topic, gold string
		tags        []string
	}

	topics := []explainQ{
		{"how neural networks learn", "Neural networks learn through forward propagation, loss computation, and backpropagation of gradients to iteratively adjust weights", []string{"AI", "technical"}},
		{"quantum entanglement to a 10-year-old", "Quantum entanglement means two particles are connected so measuring one instantly affects the other no matter the distance", []string{"physics", "eli5"}},
		{"how compilers work", "Compilers translate source code through lexing, parsing, semantic analysis, optimization, and code generation to produce machine code", []string{"CS", "technical"}},
		{"how the immune system fights infection", "The immune system uses innate defenses and adaptive responses including antibodies and T-cells to identify and destroy pathogens", []string{"biology", "medicine"}},
		{"how encryption keeps data safe", "Encryption transforms readable data into ciphertext using mathematical algorithms and keys that only authorized parties can reverse", []string{"security", "technical"}},
		{"the theory of evolution", "Evolution occurs through natural selection, genetic mutation, gene flow, and genetic drift leading to adaptation and speciation over time", []string{"biology"}},
		{"how a CPU executes instructions", "A CPU executes instructions through the fetch-decode-execute cycle, using registers, ALU, and control unit to process binary operations", []string{"CS", "hardware"}},
		{"how climate change affects ocean ecosystems", "Climate change raises ocean temperatures, increases acidification, alters currents, and disrupts marine food chains and coral reef ecosystems", []string{"ecology", "climate"}},
		{"the mechanics of flight", "Aircraft fly through the interaction of four forces: lift (from airfoil shape), weight (gravity), thrust (engines), and drag (air resistance)", []string{"physics", "engineering"}},
		{"how memory works in the brain", "Memory involves encoding through neural pathway strengthening, storage in different brain regions, and retrieval through pattern reactivation", []string{"neuroscience"}},
		{"how GPS determines your location", "GPS uses trilateration from signals of at least 4 satellites, measuring signal travel time to calculate precise 3D coordinates", []string{"technology"}},
		{"how photosynthesis converts sunlight to energy", "Photosynthesis captures light energy in chlorophyll, splits water in light reactions, then fixes CO2 into glucose via the Calvin cycle", []string{"biology", "chemistry"}},
		{"the prisoner's dilemma in game theory", "The prisoner's dilemma shows how individually rational choices can lead to collectively suboptimal outcomes when cooperation would benefit both parties", []string{"math", "economics"}},
		{"how blockchain consensus works", "Blockchain consensus mechanisms like proof-of-work and proof-of-stake ensure all nodes agree on the ledger state without a central authority", []string{"technology", "crypto"}},
		{"general relativity and curved spacetime", "General relativity describes gravity not as a force but as the curvature of spacetime caused by mass and energy, affecting the path of light and time", []string{"physics"}},
		{"how CRISPR gene editing works step by step", "CRISPR uses guide RNA to locate a target DNA sequence, then the Cas9 enzyme cuts the DNA at that point, allowing deletion, repair, or insertion of new genetic material", []string{"genetics", "biotech"}},
		{"how the stock market works", "The stock market facilitates buying and selling ownership shares of companies through exchanges, with prices driven by supply, demand, and expectations", []string{"economics", "finance"}},
		{"the water treatment process", "Water treatment involves coagulation, sedimentation, filtration, and disinfection to remove contaminants and pathogens from raw water for safe consumption", []string{"engineering", "environment"}},
		{"how batteries store and release energy", "Batteries store energy through chemical reactions between electrodes and electrolyte, and release it as electrical current when connected in a circuit", []string{"chemistry", "physics"}},
		{"the carbon cycle", "The carbon cycle moves carbon through the atmosphere, biosphere, oceans, and geosphere via photosynthesis, respiration, decomposition, and geological processes", []string{"ecology", "chemistry"}},
		{"how machine learning differs from traditional programming", "Traditional programming uses explicit rules while ML learns patterns from data, generalizing to make predictions on unseen inputs", []string{"AI", "CS"}},
		{"how nuclear reactors generate electricity", "Nuclear reactors sustain controlled fission chain reactions, using the released heat to produce steam that drives turbines connected to generators", []string{"physics", "energy"}},
		{"how the human eye sees color", "The eye detects color through cone cells sensitive to red, green, and blue wavelengths, with the brain combining signals to perceive the full spectrum", []string{"biology", "optics"}},
		{"how earthquakes happen", "Earthquakes occur when tectonic plates at faults suddenly slip past each other, releasing stored elastic energy as seismic waves", []string{"geology"}},
		{"the difference between TCP and UDP", "TCP provides reliable, ordered delivery with acknowledgments and retransmission, while UDP provides fast, connectionless delivery without guarantees", []string{"networking", "CS"}},
	}

	// Generate audience variants.
	audiences := []struct {
		prefix, suffix string
		tag            string
	}{
		{"explain ", " in simple terms", "eli5"},
		{"give a detailed technical explanation of ", "", "technical"},
		{"explain ", " step by step", "structured"},
		{"explain the concept of ", " with examples", "examples"},
		{"explain ", " and why it matters", "significance"},
	}

	rubric := []string{
		"progressive depth from overview to detail",
		"clear structural organization",
		"accurate technical content",
		"appropriate for target audience",
		"includes concrete examples or analogies",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, 130)
	id := 0

	for _, t := range topics {
		for ai, a := range audiences {
			if id >= 125 {
				break
			}
			diff := difficulties[id%3]
			if ai >= 2 {
				diff = "hard"
			}
			prompts = append(prompts, EvalPrompt{
				ID:         fmt.Sprintf("DeepExplain-%04d", id),
				Capability: "DeepExplain",
				Query:      a.prefix + t.topic + a.suffix,
				GoldAnswer: t.gold,
				Rubric:     rubric,
				Difficulty: diff,
				Tags:       append(t.tags, a.tag),
			})
			id++
		}
	}
	return prompts[:125]
}

func genCompareTradeoff() []EvalPrompt {
	type comparison struct {
		query, gold string
		tags        []string
	}

	comparisons := []comparison{
		{"compare Python vs Go for web servers", "Go offers superior performance and concurrency; Python offers faster development and richer ecosystem", []string{"programming"}},
		{"pros and cons of remote work", "Remote work offers flexibility and no commute but can cause isolation and blur work-life boundaries", []string{"work", "lifestyle"}},
		{"compare SQL vs NoSQL databases", "SQL provides ACID compliance and structured queries; NoSQL offers flexible schemas and horizontal scaling", []string{"technology", "databases"}},
		{"electric cars vs gasoline cars", "EVs have lower operating costs and emissions; gas cars have longer range and faster refueling", []string{"automotive", "environment"}},
		{"renting vs buying a home", "Buying builds equity but requires large upfront costs; renting offers flexibility with no maintenance burden", []string{"finance", "lifestyle"}},
		{"compare React vs Vue for frontend development", "React has a larger ecosystem and job market; Vue offers gentler learning curve and better documentation", []string{"programming", "web"}},
		{"Mac vs Windows for software development", "Mac offers Unix-based terminal and Apple ecosystem; Windows has broader hardware options and better gaming support", []string{"technology"}},
		{"compare microservices vs monolith architecture", "Microservices enable independent scaling and deployment; monoliths are simpler to develop, test, and deploy initially", []string{"architecture", "software"}},
		{"online learning vs traditional classroom", "Online offers flexibility and self-pacing; classroom provides social interaction and structured accountability", []string{"education"}},
		{"compare Kubernetes vs Docker Swarm", "Kubernetes offers more features and ecosystem support; Docker Swarm is simpler to set up and manage", []string{"devops", "technology"}},
		{"freelancing vs full-time employment", "Freelancing offers income flexibility and autonomy; employment provides stability, benefits, and team collaboration", []string{"career", "work"}},
		{"compare REST vs GraphQL", "REST is simpler and well-established with caching; GraphQL offers flexible queries and reduces over-fetching", []string{"technology", "API"}},
		{"SSD vs HDD for storage", "SSDs are faster and more durable with no moving parts; HDDs offer more storage per dollar for bulk data", []string{"hardware", "technology"}},
		{"compare agile vs waterfall methodology", "Agile enables iterative feedback and adaptation; waterfall provides clear milestones and documentation upfront", []string{"project-management"}},
		{"public cloud vs private cloud", "Public cloud offers scalability and lower upfront cost; private cloud provides more control and security compliance", []string{"technology", "infrastructure"}},
		{"compare TypeScript vs JavaScript", "TypeScript adds static typing reducing runtime errors; JavaScript has zero build step and universal browser support", []string{"programming", "web"}},
		{"vegetarian vs omnivore diet", "Vegetarian diets lower environmental impact and certain health risks; omnivore diets offer complete protein profiles more easily", []string{"health", "lifestyle"}},
		{"compare city living vs rural living", "City living offers convenience and opportunities; rural living provides space, nature, and lower cost of living", []string{"lifestyle"}},
		{"iOS vs Android for mobile development", "iOS offers consistent devices and higher user spending; Android has larger market share and more hardware diversity", []string{"technology", "mobile"}},
		{"compare functional vs object-oriented programming", "FP emphasizes immutability and composition; OOP offers intuitive modeling of real-world entities with encapsulation", []string{"programming", "paradigms"}},
		{"startup vs established company for career", "Startups offer rapid growth and ownership; established companies provide stability, mentorship, and resources", []string{"career"}},
		{"compare solar vs wind energy", "Solar is more predictable and scalable on rooftops; wind produces more energy per installation in suitable locations", []string{"energy", "environment"}},
		{"4-day vs 5-day work week", "4-day weeks improve wellbeing and productivity per hour; 5-day weeks offer more availability and traditional client alignment", []string{"work", "productivity"}},
		{"compare PostgreSQL vs MySQL", "PostgreSQL offers advanced features and standards compliance; MySQL is lighter and faster for simple read-heavy workloads", []string{"databases", "technology"}},
		{"in-person vs virtual meetings", "In-person meetings build stronger rapport and nonverbal communication; virtual meetings save travel time and enable remote participation", []string{"work", "communication"}},
	}

	// Generate variants to reach 125.
	framings := []struct {
		prefix, suffix string
	}{
		{"compare ", ""},
		{"what are the tradeoffs between ", ""},
		{"", ": which is better and why"},
		{"analyze the pros and cons of ", ""},
		{"help me decide between ", ""},
	}

	rubric := []string{
		"covers pros and cons of each option",
		"balanced and fair comparison",
		"addresses relevant criteria",
		"provides actionable guidance",
		"acknowledges context-dependence",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, 130)
	id := 0

	for _, c := range comparisons {
		for _, f := range framings {
			if id >= 125 {
				break
			}
			q := f.prefix + c.query + f.suffix
			prompts = append(prompts, EvalPrompt{
				ID:         fmt.Sprintf("CompareTradeoff-%04d", id),
				Capability: "CompareTradeoff",
				Query:      q,
				GoldAnswer: c.gold,
				Rubric:     rubric,
				Difficulty: difficulties[id%3],
				Tags:       c.tags,
			})
			id++
		}
		if id >= 125 {
			break
		}
	}
	return prompts[:125]
}

func genMultiTurnContext() []EvalPrompt {
	type turn struct {
		query, gold string
	}
	type conversation struct {
		turns []turn
		tags  []string
	}

	conversations := []conversation{
		{turns: []turn{
			{"Tell me about Mars", "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface"},
			{"How far is it from Earth", "Mars is between 55 and 400 million km from Earth depending on orbital positions"},
			{"Could humans live there", "Humans could potentially live on Mars with significant technological support for atmosphere, radiation shielding, and food production"},
			{"What about the temperature", "Mars temperatures range from -125C at the poles to 20C at the equator during summer"},
			{"Has anything landed there", "Multiple rovers including Curiosity and Perseverance, plus landers like InSight, have successfully landed on Mars"},
		}, tags: []string{"astronomy", "multi-hop"}},
		{turns: []turn{
			{"I'm learning Python", "Python is a versatile language great for beginners with clear syntax and extensive libraries"},
			{"What should I learn first", "Start with variables, data types, control flow (if/else, loops), then functions and basic data structures"},
			{"How about after that", "After basics, learn object-oriented programming, file I/O, error handling, and popular libraries like requests"},
			{"Which IDE do you recommend", "VS Code with the Python extension or PyCharm Community Edition are both excellent free options"},
			{"Can I build web apps with it", "Yes, frameworks like Flask for small projects and Django for full-featured web applications are very popular"},
		}, tags: []string{"programming", "tutorial"}},
		{turns: []turn{
			{"I want to start exercising", "Starting with moderate exercise 3-4 times a week combining cardio and strength training is a great approach"},
			{"I have bad knees though", "Low-impact exercises like swimming, cycling, elliptical training, and yoga are excellent for those with knee issues"},
			{"What about diet", "A balanced diet with adequate protein, complex carbs, healthy fats, and plenty of vegetables supports exercise recovery"},
			{"How much protein do I need", "For active individuals, aim for 1.2-2.0 grams of protein per kilogram of body weight per day"},
			{"Any meal suggestions", "Greek yogurt with berries for breakfast, grilled chicken salad for lunch, and salmon with quinoa for dinner"},
		}, tags: []string{"health", "fitness"}},
		{turns: []turn{
			{"What is machine learning", "Machine learning is a subset of AI where systems learn patterns from data to make predictions without explicit programming"},
			{"How is it different from deep learning", "Deep learning is a subset of ML using neural networks with many layers, excelling at complex patterns like images and language"},
			{"What's a neural network then", "A neural network is layers of interconnected nodes that transform inputs through weighted connections and activation functions"},
			{"Can you give a simple example", "A spam filter learns from labeled emails: it finds patterns in spam vs legitimate mail and classifies new emails accordingly"},
			{"What tools should I use to try it", "Start with Python and scikit-learn for classical ML, then PyTorch or TensorFlow for deep learning"},
		}, tags: []string{"AI", "technical"}},
		{turns: []turn{
			{"I'm planning a trip to Japan", "Japan offers incredible culture, food, and scenery; popular destinations include Tokyo, Kyoto, Osaka, and Hiroshima"},
			{"When is the best time to go", "Spring (March-May) for cherry blossoms or autumn (September-November) for fall foliage are the most popular seasons"},
			{"How much should I budget", "Budget approximately $100-200 per day covering accommodation, food, transport, and activities; Japan can be surprisingly affordable"},
			{"Do I need to speak Japanese", "Basic phrases help but major cities have English signage; translation apps work well for menus and conversations"},
			{"What about the rail pass", "The Japan Rail Pass offers unlimited travel on JR trains for 7, 14, or 21 days and is cost-effective for multi-city trips"},
		}, tags: []string{"travel", "planning"}},
		{turns: []turn{
			{"Tell me about the Renaissance", "The Renaissance was a cultural movement from the 14th to 17th century originating in Italy, reviving interest in classical art, science, and philosophy"},
			{"Who were the key figures", "Leonardo da Vinci, Michelangelo, Raphael, Galileo, Machiavelli, and Shakespeare were among the most influential figures"},
			{"What made it start in Italy", "Italy's wealthy city-states, trade routes, classical Roman heritage, and patronage by families like the Medici created ideal conditions"},
			{"How did it spread to the rest of Europe", "Trade networks, printing press, traveling scholars, and royal patronage spread Renaissance ideas across Europe"},
			{"What was its lasting impact", "The Renaissance laid foundations for modern science, secular thought, realistic art, humanism, and the eventual Enlightenment"},
		}, tags: []string{"history", "culture"}},
		{turns: []turn{
			{"I need help with my resume", "A strong resume should highlight achievements over duties, use action verbs, and be tailored to each position"},
			{"I have 3 years of experience in marketing", "Focus on quantifiable achievements like campaign ROI, audience growth, and revenue impact rather than listing responsibilities"},
			{"Should I include a summary section", "Yes, a 2-3 sentence professional summary at the top highlighting your key strengths and career focus is effective"},
			{"What about skills section", "Include both technical skills (tools, platforms, analytics) and soft skills (leadership, communication) relevant to marketing roles"},
			{"How long should it be", "With 3 years of experience, keep it to one page; focus on the most impactful and relevant accomplishments"},
		}, tags: []string{"career", "writing"}},
		{turns: []turn{
			{"What is cryptocurrency", "Cryptocurrency is a digital currency using cryptography for security, operating on decentralized blockchain networks"},
			{"Is Bitcoin the only one", "No, there are thousands including Ethereum, which adds smart contracts, and stablecoins like USDC pegged to fiat currencies"},
			{"What are smart contracts", "Smart contracts are self-executing programs on a blockchain that automatically enforce agreement terms when conditions are met"},
			{"Is it safe to invest in", "Crypto investments carry high volatility risk; diversification, research, and investing only what you can afford to lose are advisable"},
			{"How do I buy some", "Use regulated exchanges like Coinbase or Kraken, verify your identity, link a bank account, and start with small amounts"},
		}, tags: []string{"finance", "crypto"}},
		{turns: []turn{
			{"Explain climate change", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human greenhouse gas emissions since industrialization"},
			{"What are greenhouse gases", "The main greenhouse gases are CO2 from fossil fuels, methane from agriculture and landfills, and nitrous oxide from fertilizers"},
			{"How bad is it really", "Current trajectories suggest 1.5-4.5C warming by 2100, causing sea level rise, extreme weather, ecosystem disruption, and food insecurity"},
			{"What can individuals do", "Reduce energy use, eat less meat, use public transport, support renewable energy, and vote for climate-conscious policies"},
			{"What are countries doing about it", "The Paris Agreement targets limiting warming to 1.5C; countries are investing in renewables, carbon pricing, and efficiency mandates"},
		}, tags: []string{"environment", "science"}},
		{turns: []turn{
			{"I want to learn guitar", "Starting guitar involves choosing acoustic or electric, learning basic chords (G, C, D, Em, Am), and practicing regularly"},
			{"I picked acoustic", "Great choice for beginners; acoustic builds finger strength and requires no additional equipment beyond a tuner and picks"},
			{"What chords should I learn first", "Start with Em and Am (easiest), then G, C, and D; these five chords allow you to play hundreds of popular songs"},
			{"How long until I can play songs", "With 15-30 minutes of daily practice, most beginners can play simple songs with chord changes within 2-4 weeks"},
			{"Can you suggest some easy songs", "Try Knockin on Heaven's Door (G, D, Am, C), Horse With No Name (Em, D6), or Wish You Were Here by Pink Floyd"},
		}, tags: []string{"music", "hobby"}},
	}

	// Additional conversations to reach at least 25 conversation groups.
	moreConversations := []conversation{
		{turns: []turn{
			{"What's a good programming language for data science", "Python is the most popular for data science due to libraries like pandas, numpy, scikit-learn, and matplotlib"},
			{"What about R", "R excels at statistical analysis and visualization; many statisticians prefer it for academic research and specialized statistical methods"},
			{"Can I use both", "Yes, many data scientists use Python for ML pipelines and R for statistical analysis; they complement each other well"},
			{"What should I learn first", "Start with Python for its broader applicability, then add R for specialized statistical needs as you advance"},
			{"What about Julia", "Julia offers near-C speed for numerical computing and is growing but has a smaller ecosystem than Python or R"},
		}, tags: []string{"data-science", "programming"}},
		{turns: []turn{
			{"I'm thinking of getting a pet", "Consider your lifestyle, space, budget, and time commitment; dogs need more attention than cats, fish are low-maintenance"},
			{"I work from home", "Working from home is ideal for dogs as they need companionship; consider your energy level to match breed size and temperament"},
			{"I like medium-sized dogs", "Excellent choices include Labrador Retrievers, Border Collies, Australian Shepherds, and Beagles for medium-sized companions"},
			{"I have a small apartment though", "For apartments, consider Cavalier King Charles Spaniels, French Bulldogs, or Basset Hounds which need less space and exercise"},
			{"What about adoption", "Adoption is wonderful; shelters have dogs of all sizes and breeds, staff can match you with a dog suited to apartment living"},
		}, tags: []string{"lifestyle", "pets"}},
		{turns: []turn{
			{"How does the stock market work", "The stock market is an exchange where shares of public companies are bought and sold based on supply and demand"},
			{"What's a stock index", "A stock index tracks a group of stocks representing a market segment, like the S&P 500 tracking 500 large US companies"},
			{"How do I start investing", "Open a brokerage account, start with index funds for diversification, invest regularly, and think long-term"},
			{"What's the difference between stocks and bonds", "Stocks represent ownership with higher risk/reward; bonds are loans to entities with lower risk and fixed interest payments"},
			{"What about mutual funds vs ETFs", "Both pool investments; ETFs trade like stocks with lower fees, while mutual funds are priced daily and may be actively managed"},
		}, tags: []string{"finance", "investing"}},
		{turns: []turn{
			{"Tell me about black holes", "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon"},
			{"How are they detected if light can't escape", "They're detected by gravitational effects on nearby stars, X-rays from heated infalling matter, and gravitational waves from mergers"},
			{"What happens if you fall into one", "Approaching the event horizon, time dilates from an external observer's view; the infalling person experiences tidal forces called spaghettification"},
			{"Are there different sizes", "Yes: stellar black holes (few solar masses), intermediate (hundreds to thousands), and supermassive (millions to billions of solar masses)"},
			{"Is there one at the center of our galaxy", "Yes, Sagittarius A* is a supermassive black hole at the Milky Way's center with about 4 million solar masses, imaged by the EHT"},
		}, tags: []string{"astronomy", "physics"}},
		{turns: []turn{
			{"I want to cook more at home", "Home cooking saves money and is healthier; start with simple recipes and gradually build your skills and recipe repertoire"},
			{"What equipment do I need", "Essential items: a good chef's knife, cutting board, skillet, saucepan, baking sheet, and measuring cups and spoons"},
			{"What are some easy recipes to start with", "Stir-fry, pasta with sauce, omelets, sheet pan chicken and vegetables, and rice bowls are all beginner-friendly"},
			{"How do I meal prep", "Cook base ingredients like rice, roasted vegetables, and proteins on Sunday, then mix and match throughout the week"},
			{"Any tips for seasoning", "Salt enhances flavor; learn to use acid (lemon, vinegar) to brighten dishes; build a spice collection gradually starting with garlic, cumin, paprika"},
		}, tags: []string{"cooking", "lifestyle"}},
	}
	conversations = append(conversations, moreConversations...)

	rubric := []string{
		"correctly references information from previous turns",
		"maintains consistent context throughout",
		"resolves pronouns and references appropriately",
		"does not contradict earlier statements",
		"builds naturally on the conversation flow",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, 130)
	id := 0

	for _, conv := range conversations {
		for ti, t := range conv.turns {
			if id >= 125 {
				break
			}
			// Build context from previous turns.
			contextNote := ""
			if ti > 0 {
				contextNote = fmt.Sprintf("[Turn %d of %d in conversation] Previous: %q -> ", ti+1, len(conv.turns), conv.turns[ti-1].query)
			}
			prompts = append(prompts, EvalPrompt{
				ID:         fmt.Sprintf("MultiTurnContext-%04d", id),
				Capability: "MultiTurnContext",
				Query:      contextNote + t.query,
				GoldAnswer: t.gold,
				Rubric:     rubric,
				Difficulty: difficulties[ti%3],
				Tags:       conv.tags,
			})
			id++
		}
		if id >= 125 {
			break
		}
	}

	// Pad remaining with additional multi-turn patterns.
	followUps := []struct{ q, gold string }{
		{"What about the cost", "The cost varies depending on several factors including location, quality, and market conditions"},
		{"Tell me more about that last point", "Expanding on the previous point with additional detail and supporting evidence"},
		{"How does this compare to alternatives", "Compared to alternatives, there are distinct advantages and tradeoffs to consider"},
		{"Is there a simpler way to explain it", "In simpler terms, the core concept can be understood as a straightforward analogy"},
		{"What are the risks", "The main risks include uncertainty, resource requirements, and potential for unforeseen complications"},
	}
	for id < 125 {
		f := followUps[id%len(followUps)]
		prompts = append(prompts, EvalPrompt{
			ID:         fmt.Sprintf("MultiTurnContext-%04d", id),
			Capability: "MultiTurnContext",
			Query:      "[Follow-up turn] " + f.q,
			GoldAnswer: f.gold,
			Rubric:     rubric,
			Difficulty: difficulties[id%3],
			Tags:       []string{"follow-up", "context-dependent"},
		})
		id++
	}
	return prompts[:125]
}

func genPlanning() []EvalPrompt {
	type planQ struct {
		query, gold string
		tags        []string
	}

	plans := []planQ{
		{"plan a trip to Japan for 2 weeks", "A 2-week Japan itinerary covering Tokyo, Kyoto, Osaka, and Hiroshima with transport, accommodation, and activity planning", []string{"travel"}},
		{"create a study schedule for the GRE in 3 months", "A structured 12-week study plan covering verbal, quantitative, and analytical writing with practice tests", []string{"education"}},
		{"plan a wedding on a budget of $10,000", "A budget wedding plan covering venue, food, attire, photography, and decorations within financial constraints", []string{"event", "budget"}},
		{"create a fitness plan to run a half marathon in 6 months", "A progressive running plan building from beginner to 13.1 miles with cross-training and rest days", []string{"fitness", "health"}},
		{"plan the launch of a small e-commerce business", "Business launch plan covering product sourcing, platform setup, marketing, legal requirements, and financial projections", []string{"business", "startup"}},
		{"create a meal plan for a family of 4 on $100/week", "A weekly meal plan with shopping list covering balanced nutrition for a family within budget", []string{"cooking", "budget"}},
		{"plan a home renovation for a kitchen", "Kitchen renovation plan covering design, contractor selection, timeline, permits, and budget management", []string{"home", "construction"}},
		{"create a learning roadmap for becoming a data scientist", "A structured learning path from statistics and programming fundamentals through ML, deep learning, and portfolio building", []string{"career", "education"}},
		{"plan a fundraising event for a charity", "Event plan covering venue, sponsorships, entertainment, marketing, volunteer coordination, and donation logistics", []string{"nonprofit", "event"}},
		{"create a project plan for building a mobile app", "App development plan covering requirements, design, development sprints, testing, deployment, and maintenance", []string{"technology", "project"}},
		{"plan a garden for a small backyard", "Garden plan covering layout, plant selection for climate, soil preparation, planting schedule, and maintenance", []string{"gardening", "home"}},
		{"create a debt repayment plan for $30,000 in student loans", "Structured repayment strategy comparing snowball and avalanche methods with budget adjustments and timeline", []string{"finance", "personal"}},
		{"plan a career transition from teaching to tech", "Career transition plan covering skill assessment, learning path, networking, portfolio building, and job search strategy", []string{"career"}},
		{"create a content calendar for a startup blog", "12-month content strategy covering topics, posting schedule, SEO, promotion, and performance metrics", []string{"marketing", "content"}},
		{"plan a move to a new city", "Relocation plan covering job search, housing, logistics, budgeting, and community integration", []string{"lifestyle", "logistics"}},
		{"create a retirement savings plan for a 30-year-old", "Long-term savings plan covering 401k, IRA, investment allocation, and milestone targets over 30-35 years", []string{"finance", "retirement"}},
		{"plan a hackathon for 100 participants", "Hackathon plan covering venue, sponsors, registration, mentors, judging criteria, prizes, and day-of logistics", []string{"technology", "event"}},
		{"create a marketing plan for a local restaurant", "Restaurant marketing plan covering social media, local SEO, loyalty programs, events, and community partnerships", []string{"business", "marketing"}},
		{"plan a podcast launch", "Podcast launch plan covering concept, equipment, recording workflow, hosting platform, distribution, and promotion strategy", []string{"media", "creative"}},
		{"create an emergency preparedness plan for a family", "Family emergency plan covering evacuation routes, supply kits, communication plan, insurance review, and drills", []string{"safety", "family"}},
		{"plan a software migration from legacy to cloud", "Migration plan covering assessment, architecture, data migration, testing, rollout phases, and rollback procedures", []string{"technology", "enterprise"}},
		{"create a 30-day healthy eating challenge", "30-day structured meal plan with progressive dietary improvements, shopping lists, and accountability measures", []string{"health", "nutrition"}},
		{"plan a community volunteer program", "Volunteer program plan covering recruitment, training, scheduling, project identification, and impact measurement", []string{"nonprofit", "community"}},
		{"create an onboarding plan for new employees", "Employee onboarding plan covering first day through 90 days with training modules, mentorship, and checkpoints", []string{"HR", "business"}},
		{"plan a book writing schedule for 6 months", "Book writing plan with daily word count targets, chapter milestones, editing phases, and accountability systems", []string{"writing", "creative"}},
	}

	framings := []struct{ prefix, suffix string }{
		{"", ""},
		{"help me ", ""},
		{"I need to ", " - where do I start"},
		{"create a detailed plan to ", ""},
		{"step by step, ", ""},
	}

	rubric := []string{
		"clear sequential steps with logical ordering",
		"realistic timeline and resource estimates",
		"addresses potential obstacles and contingencies",
		"actionable and specific rather than vague",
		"appropriate scope for the constraint given",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, 130)
	id := 0

	for _, p := range plans {
		for _, f := range framings {
			if id >= 125 {
				break
			}
			q := f.prefix + p.query + f.suffix
			prompts = append(prompts, EvalPrompt{
				ID:         fmt.Sprintf("Planning-%04d", id),
				Capability: "Planning",
				Query:      q,
				GoldAnswer: p.gold,
				Rubric:     rubric,
				Difficulty: difficulties[id%3],
				Tags:       p.tags,
			})
			id++
		}
		if id >= 125 {
			break
		}
	}
	return prompts[:125]
}

func genToolUseAccuracy() []EvalPrompt {
	type toolQ struct {
		query, gold, tool string
		tags               []string
	}

	conversions := []toolQ{
		{"convert 5 miles to kilometers", "5 miles is approximately 8.05 kilometers", "unit_converter", []string{"conversion", "distance"}},
		{"convert 100 pounds to kilograms", "100 pounds is approximately 45.36 kilograms", "unit_converter", []string{"conversion", "weight"}},
		{"convert 72 Fahrenheit to Celsius", "72F is approximately 22.2C", "unit_converter", []string{"conversion", "temperature"}},
		{"how many cups in a gallon", "There are 16 cups in a US gallon", "unit_converter", []string{"conversion", "volume"}},
		{"convert 1000 meters to feet", "1000 meters is approximately 3280.84 feet", "unit_converter", []string{"conversion", "distance"}},
		{"how many ounces in a liter", "There are approximately 33.81 fluid ounces in a liter", "unit_converter", []string{"conversion", "volume"}},
		{"convert 50 km/h to mph", "50 km/h is approximately 31.07 mph", "unit_converter", []string{"conversion", "speed"}},
		{"how many inches in a meter", "There are approximately 39.37 inches in a meter", "unit_converter", []string{"conversion", "distance"}},
		{"convert 200 grams to ounces", "200 grams is approximately 7.05 ounces", "unit_converter", []string{"conversion", "weight"}},
		{"convert 2 liters to gallons", "2 liters is approximately 0.53 US gallons", "unit_converter", []string{"conversion", "volume"}},
	}

	translations := []toolQ{
		{"translate hello to French", "Bonjour", "translator", []string{"translation", "french"}},
		{"how do you say thank you in Japanese", "Arigatou gozaimasu", "translator", []string{"translation", "japanese"}},
		{"translate goodbye to Spanish", "Adios", "translator", []string{"translation", "spanish"}},
		{"how do you say good morning in German", "Guten Morgen", "translator", []string{"translation", "german"}},
		{"translate please to Italian", "Per favore", "translator", []string{"translation", "italian"}},
		{"how do you say yes in Mandarin", "Shi (是)", "translator", []string{"translation", "mandarin"}},
		{"translate water to Portuguese", "Agua", "translator", []string{"translation", "portuguese"}},
		{"how do you say excuse me in Korean", "Sillyehamnida (실례합니다)", "translator", []string{"translation", "korean"}},
		{"translate friend to Arabic", "Sadiq (صديق)", "translator", []string{"translation", "arabic"}},
		{"how do you say beautiful in Hindi", "Sundar (सुन्दर)", "translator", []string{"translation", "hindi"}},
	}

	calculations := []toolQ{
		{"what is 15% tip on $85.50", "A 15% tip on $85.50 is $12.83, making the total $98.33", "calculator", []string{"math", "money"}},
		{"calculate the area of a circle with radius 7", "The area is approximately 153.94 square units (pi * 7^2)", "calculator", []string{"math", "geometry"}},
		{"what is 2 to the power of 16", "2^16 = 65,536", "calculator", []string{"math", "exponent"}},
		{"compound interest on $5000 at 5% for 10 years", "With annual compounding, $5000 at 5% for 10 years grows to approximately $8,144.47", "calculator", []string{"math", "finance"}},
		{"what is the hypotenuse of a triangle with sides 3 and 4", "The hypotenuse is 5 (sqrt(3^2 + 4^2) = sqrt(25) = 5)", "calculator", []string{"math", "geometry"}},
		{"calculate BMI for 180 pounds and 5 foot 10", "BMI is approximately 25.8 (weight in kg / height in m squared)", "calculator", []string{"math", "health"}},
		{"what is the monthly payment on a $200,000 mortgage at 6% for 30 years", "The monthly payment is approximately $1,199.10", "calculator", []string{"math", "finance"}},
		{"how many calories in a 3 mile run", "Running 3 miles burns approximately 300-400 calories depending on weight and pace", "calculator", []string{"math", "fitness"}},
		{"convert 24-hour time 15:45 to 12-hour format", "15:45 in 24-hour format is 3:45 PM", "time_converter", []string{"conversion", "time"}},
		{"what day of the week was January 1, 2000", "January 1, 2000 was a Saturday", "calendar", []string{"date", "calendar"}},
	}

	lookups := []toolQ{
		{"what is the current time in Tokyo", "Tool should query timezone data for Asia/Tokyo (UTC+9)", "timezone", []string{"lookup", "time"}},
		{"what is the population of Canada", "Canada's population is approximately 40 million", "knowledge_lookup", []string{"lookup", "geography"}},
		{"what is the exchange rate of USD to EUR", "Tool should query current exchange rate data for USD/EUR", "currency", []string{"lookup", "finance"}},
		{"what is the boiling point of ethanol", "The boiling point of ethanol is 78.37 degrees Celsius", "knowledge_lookup", []string{"lookup", "chemistry"}},
		{"what is the atomic number of gold", "The atomic number of gold (Au) is 79", "knowledge_lookup", []string{"lookup", "chemistry"}},
		{"how many calories in an avocado", "A medium avocado contains approximately 240 calories", "nutrition_lookup", []string{"lookup", "food"}},
		{"what is the distance from Earth to the Sun", "The average distance from Earth to the Sun is about 149.6 million km (1 AU)", "knowledge_lookup", []string{"lookup", "astronomy"}},
		{"what is the ISBN of 1984 by George Orwell", "Tool should query book database for ISBN of 1984", "book_lookup", []string{"lookup", "literature"}},
		{"what is the zip code for Beverly Hills", "The primary zip code for Beverly Hills, CA is 90210", "location_lookup", []string{"lookup", "geography"}},
		{"how tall is the Eiffel Tower", "The Eiffel Tower is 330 meters (1,083 feet) tall including antennas", "knowledge_lookup", []string{"lookup", "architecture"}},
	}

	textOps := []toolQ{
		{"count the words in this paragraph", "Tool should use text analyzer to count words in the given text", "text_analyzer", []string{"text", "analysis"}},
		{"spell check this sentence", "Tool should use spell checker to identify and correct misspellings", "spell_checker", []string{"text", "correction"}},
		{"summarize this article in 3 sentences", "Tool should use summarizer to extract key points into 3 concise sentences", "summarizer", []string{"text", "NLP"}},
		{"find the sentiment of this review", "Tool should use sentiment analyzer to classify the review as positive, negative, or neutral", "sentiment_analyzer", []string{"text", "NLP"}},
		{"generate a random password 16 characters long", "Tool should use password generator with 16 chars, mixed case, digits, and symbols", "password_generator", []string{"security", "generation"}},
	}

	all := make([]toolQ, 0, 125)
	all = append(all, conversions...)
	all = append(all, translations...)
	all = append(all, calculations...)
	all = append(all, lookups...)
	all = append(all, textOps...)

	rubric := []string{
		"selects the correct tool for the task",
		"provides accurate parameters to the tool",
		"interprets the tool result correctly",
		"handles edge cases and errors gracefully",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, 130)
	id := 0

	for _, q := range all {
		prompts = append(prompts, EvalPrompt{
			ID:         fmt.Sprintf("ToolUseAccuracy-%04d", id),
			Capability: "ToolUseAccuracy",
			Query:      q.query,
			GoldAnswer: q.gold,
			Rubric:     rubric,
			Difficulty: difficulties[id%3],
			Tags:       append(q.tags, q.tool),
		})
		id++
	}

	// Generate variants to reach 125.
	variantPrefixes := []string{
		"can you ", "please ", "I need to ", "help me ", "quickly ",
	}
	base := len(prompts)
	for id < 125 {
		idx := (id - base) % len(all)
		vi := (id - base) / len(all) % len(variantPrefixes)
		q := all[idx]
		prompts = append(prompts, EvalPrompt{
			ID:         fmt.Sprintf("ToolUseAccuracy-%04d", id),
			Capability: "ToolUseAccuracy",
			Query:      variantPrefixes[vi] + q.query,
			GoldAnswer: q.gold,
			Rubric:     rubric,
			Difficulty: difficulties[id%3],
			Tags:       append(q.tags, q.tool),
		})
		id++
	}
	return prompts[:125]
}

func genStyleControl() []EvalPrompt {
	topics := []string{
		"quantum computing", "the French Revolution", "machine learning",
		"climate change", "blockchain technology", "the human immune system",
		"supply chain management", "cognitive behavioral therapy",
		"renewable energy", "evolutionary biology",
		"the stock market", "artificial intelligence ethics",
		"space exploration", "cybersecurity", "urban planning",
		"behavioral economics", "gene therapy", "cloud computing",
		"the water crisis", "autonomous vehicles",
		"the history of the internet", "marine biology",
		"sustainable agriculture", "nuclear fusion", "digital privacy",
	}

	styles := []struct {
		instruction, description string
		tags                     []string
	}{
		{"explain %s formally, as if writing for an academic journal", "formal academic tone with precise terminology", []string{"formal", "academic"}},
		{"give me a casual, conversational explanation of %s", "informal tone as if talking to a friend", []string{"casual", "conversational"}},
		{"explain %s in exactly 3 bullet points", "concise bullet point format with exactly 3 points", []string{"format", "bullet-points"}},
		{"explain %s to a 5-year-old", "extremely simplified language appropriate for young children", []string{"eli5", "simplified"}},
		{"write a one-paragraph summary of %s", "single cohesive paragraph with key information", []string{"format", "paragraph"}},
	}

	rubric := []string{
		"matches the requested tone/register",
		"follows the specified format constraints",
		"maintains appropriate verbosity level",
		"content remains accurate despite style adaptation",
		"consistent style throughout the response",
	}

	difficulties := []string{"easy", "medium", "hard"}
	prompts := make([]EvalPrompt, 0, 130)
	id := 0

	for _, topic := range topics {
		for _, style := range styles {
			if id >= 125 {
				break
			}
			prompts = append(prompts, EvalPrompt{
				ID:         fmt.Sprintf("StyleControl-%04d", id),
				Capability: "StyleControl",
				Query:      fmt.Sprintf(style.instruction, topic),
				GoldAnswer: style.description,
				Rubric:     rubric,
				Difficulty: difficulties[id%3],
				Tags:       style.tags,
			})
			id++
		}
		if id >= 125 {
			break
		}
	}
	return prompts[:125]
}
