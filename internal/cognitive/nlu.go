package cognitive

import (
	"regexp"
	"strings"
	"unicode"
)

// NLUResult is defined in action.go — shared struct for NLU output.
// Fields: Intent, Action, Entities, Confidence, Raw.

// NLU is a pure-code Natural Language Understanding engine.
// Zero LLM calls. Microsecond-level. Deterministic.
type NLU struct {
	greetings    []string
	farewells    []string
	affirmatives []string
	negatives    []string
	metaPatterns []string

	// Neural intent classifier — trained replacement for pattern matching.
	// When available and confident, overrides the pattern-based classification.
	Neural *NeuralNLU

	// compiled regexes — built once at init
	urlRe       *regexp.Regexp
	pathRe      *regexp.Regexp
	mathRe      *regexp.Regexp
	quotedRe    *regexp.Regexp
	dateWordRe  *regexp.Regexp
	dateFormalRe *regexp.Regexp

	questionPrefixes []string
	commandVerbs     []string
	searchVerbs      []string
	fileVerbs        []string
	memoryVerbs      []string
	recallVerbs      []string
	planVerbs        []string
	explainVerbs     []string
	computeVerbs     []string

	recommendVerbs []string
	compareVerbs   []string
	compareVsRe    *regexp.Regexp // matches "X vs Y" patterns

	// web-lookup signals: topics that require external/current knowledge
	currentEventWords []string
	webLookupPatterns []*regexp.Regexp

	// Assistant feature patterns
	weatherWords   []string
	convertWords   []string
	reminderWords  []string
	sysinfoWords   []string
	clipboardWords []string
	noteWords      []string
	todoWords      []string
	newsWords      []string
	calendarWords  []string
	emailWords     []string
	screenshotWords  []string
	codeRunWords     []string
	fileFinderWords  []string

	// Batch 2 tools
	volumeWords     []string
	brightnessWords []string
	timerWords      []string
	appWords        []string
	hashWords       []string
	dictWords       []string
	networkWords    []string
	translateWords  []string
	archiveWords    []string
	diskUsageWords  []string
	processWords    []string
	qrcodeWords     []string
	calculatorWords []string
	passwordWords   []string
	bookmarkWords   []string
	journalWords    []string
	habitWords      []string
	expenseWords    []string
	transformWords  []string
	transformRe     []*regexp.Regexp
	creativeRe      []*regexp.Regexp

	// Freeform fallback classifier — catches inputs that pattern NLU misses
	Freeform *FreeformClassifier
}

// NewNLU creates a new deterministic NLU engine with all pattern tables initialized.
func NewNLU() *NLU {
	n := &NLU{
		greetings: []string{
			"hi", "hello", "hey", "hey there", "howdy", "hiya", "yo",
			"good morning", "good afternoon", "good evening", "good night",
			"morning", "evening", "afternoon",
			"what's up", "whats up", "sup", "greetings", "salutations",
			"how are you", "how's it going", "how is it going",
			"how are you doing", "how you doing", "how do you do",
			"how have you been", "how's your day", "how is your day",
			"what's going on", "whats going on", "what's new",
		},
		farewells: []string{
			"bye", "goodbye", "good bye", "see ya", "see you", "later",
			"farewell", "ciao", "adios", "peace", "take care",
			"good night", "gn", "ttyl", "talk later",
			"gotta go", "catch you later", "ok bye", "okay bye",
			"bye bye", "night", "nite",
		},
		affirmatives: []string{
			"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "k",
			"thanks", "thank you", "thx", "ty", "thanks a lot", "thanks so much",
			"thank you so much", "great", "good", "nice",
			"awesome", "cool", "perfect", "exactly", "correct", "right",
			"got it", "understood", "makes sense", "agreed", "absolutely",
			"no", "nope", "nah", "not really", "negative",
		},
		metaPatterns: []string{
			"what can you do", "who are you", "what are you",
			"how do you work", "what do you know",
			"tell me about yourself", "your capabilities",
			"what's your name", "whats your name", "what is your name",
			"who made you", "who created you", "who built you",
			"who programmed you", "who designed you",
			"do you have feelings", "do you have emotions",
			"are you alive", "are you sentient", "are you conscious",
			"are you real", "are you a robot", "are you human",
			"are you an ai", "are you a bot", "are you a machine",
			"can you think", "can you feel", "do you dream",
		},

		urlRe:        regexp.MustCompile(`https?://[^\s<>"{}|\\^` + "`" + `\[\]]+`),
		pathRe:       regexp.MustCompile(`(?:^|[\s,])([~.]?/[a-zA-Z0-9_./-]+|[a-zA-Z0-9_./]*\.[a-zA-Z]{1,10})`),
		mathRe:       regexp.MustCompile(`\b(\d+(?:\.\d+)?)\s*([+\-*/^%])\s*(\d+(?:\.\d+)?)\b`),
		quotedRe:     regexp.MustCompile(`"([^"]+)"`),
		dateWordRe:   regexp.MustCompile(`(?i)\b(today|tomorrow|yesterday|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)|last\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)|this\s+(?:week|month|year|weekend))\b`),
		dateFormalRe: regexp.MustCompile(`(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b`),

		questionPrefixes: []string{
			"what", "who", "where", "when", "why", "how",
			"how come ", "why does ", "why is ",
			"what makes ", "who invented ", "when was ", "where is ",
			"is it true that ",
			"is ", "are ", "was ", "were ", "do ", "does ", "did ",
			"can ", "could ", "would ", "should ", "will ",
			"has ", "have ", "had ",
		},
		commandVerbs: []string{
			"do", "run", "execute", "start", "stop", "restart",
			"install", "deploy", "build", "compile", "test",
			"delete", "remove", "kill", "clean", "reset",
			"set", "configure", "enable", "disable", "toggle",
		},
		searchVerbs: []string{
			"search", "find", "look up", "lookup", "look for",
			"google", "bing", "duckduckgo",
		},
		fileVerbs: []string{
			"read", "open", "edit", "create", "write", "save",
			"cat", "show file", "view file", "delete file", "remove file",
			"list files", "ls", "mkdir",
		},
		memoryVerbs: []string{
			"remember", "memorize", "store", "save that",
			"my name is", "i am a", "i'm a", "i work at", "i work as",
			"i live in", "i like", "i love", "i enjoy", "i prefer",
			"i'm interested in", "i am interested in",
			"i'm building", "i am building",
			"note that", "keep in mind",
		},
		recallVerbs: []string{
			"do you remember", "what's my", "whats my", "what is my",
			"what did i", "recall", "have i told you",
			"what do you know about me",
		},
		planVerbs: []string{
			"plan", "schedule", "remind", "reminder", "set alarm",
			"add to calendar", "create task", "todo", "to-do", "to do",
			"routine", "agenda",
		},
		explainVerbs: []string{
			"explain", "describe", "elaborate", "clarify",
			"what is", "what are", "what does",
			"how does", "how do", "how is",
			"tell me about", "teach me",
			"tell me everything about", "tell me all about",
			"give me an overview of", "give me a full overview of",
			"walk me through", "deep dive into",
			"define", "definition of",
		},
		computeVerbs: []string{
			"calculate", "compute", "solve", "evaluate",
			"what is", // followed by math
			"convert", "how much is", "how many",
		},
		recommendVerbs: []string{
			"suggest", "recommend", "any tips", "advice on",
			"what should i", "what would you suggest",
		},
		compareVerbs: []string{
			"difference between", "compare", "better than", "worse than",
		},
		compareVsRe: regexp.MustCompile(`(?i)\b\w+\s+vs\.?\s+\w+`),
		currentEventWords: []string{
			"score", "scores", "won", "winning", "lost",
			"price", "stock", "market", "trading",
			"live", "breaking", "trending", "viral",
			"election", "results",
		},

		weatherWords: []string{
			"weather", "forecast", "temperature", "how hot", "how cold",
			"is it raining", "will it rain", "is it sunny",
			"degrees outside", "what's it like outside",
		},
		convertWords: []string{
			"miles to", "km to", "pounds to", "kg to", "celsius to", "fahrenheit to",
			"gallons to", "liters to", "inches to", "feet to", "meters to",
			"to miles", "to km", "to kilometers", "to pounds", "to kg",
			"to celsius", "to fahrenheit", "to gallons", "to liters",
			"to inches", "to feet", "to meters", "to centimeters",
			"in miles", "in km", "in kilometers", "in pounds", "in kg",
			"in celsius", "in fahrenheit",
			"how many miles", "how many km", "how many kilometers",
			"how many pounds", "how many kg", "how many kilograms",
			"how many gallons", "how many liters", "how many litres",
			"how many inches", "how many feet", "how many meters", "how many metres",
			"usd to", "eur to", "gbp to", "dollars to", "euros to", "pounds to",
			"mph to", "bytes to", "mb to", "gb to",
		},
		reminderWords: []string{
			"remind me", "set a reminder", "set reminder",
			"alarm for", "wake me",
		},
		sysinfoWords: []string{
			"disk space", "free space", "storage space",
			"how much ram", "memory usage", "free memory",
			"my ip", "ip address", "what is my ip",
			"system info", "system information", "cpu info",
			"uptime", "how long has",
			"what time is it", "what's the time", "whats the time",
			"current time", "time right now", "what time",
		},
		clipboardWords: []string{
			"clipboard", "what did i copy", "paste", "what's in my clipboard",
			"whats in my clipboard", "show clipboard", "read clipboard",
			"copy this", "copy to clipboard",
		},
		noteWords: []string{
			"save a note", "save note", "new note", "create note", "create a note",
			"note about", "note:", "my notes", "show notes", "list notes",
			"show my notes", "delete note", "search notes", "find note",
		},
		todoWords: []string{
			"add task", "add todo", "add a task", "add a todo",
			"new task", "new todo", "create task", "create todo",
			"my tasks", "my todos", "show tasks", "show todos",
			"list tasks", "list todos", "task list", "todo list",
			"complete task", "finish task", "done with", "mark done",
			"delete task", "remove task",
		},
		newsWords: []string{
			"news", "headlines", "what's happening", "whats happening",
			"latest news", "tech news", "world news", "science news",
			"rss", "feed", "show me news",
		},
		calendarWords: []string{
			"my calendar", "my schedule", "my events", "my agenda",
			"what's on my calendar", "whats on my calendar",
			"events today", "events tomorrow", "events this week",
			"any meetings", "upcoming events", "upcoming meetings",
		},
		emailWords: []string{
			"check my email", "check email", "any new email", "new emails",
			"my email", "my inbox", "unread email", "unread messages",
			"check my mail", "check mail",
		},
		screenshotWords: []string{
			"screenshot", "take a screenshot", "capture screen", "screen capture",
			"snap the screen", "grab screen", "print screen",
		},
		fileFinderWords: []string{
			"find files", "find file", "find my files", "locate file", "locate files",
			"find documents", "find images", "find photos", "find videos",
			"find pdfs", "search files", "search my files",
			"where is the file", "where are my files",
			".go files", ".py files", ".js files", ".pdf files", ".txt files",
			".md files", ".json files", ".csv files", ".html files",
			"files in ~/", "files in /",
		},
		codeRunWords: []string{
			"run this code", "run this script", "run the code",
			"run python", "run bash", "run javascript", "run node",
			"execute this", "execute code", "execute python", "execute script",
			"python code:", "bash code:", "javascript code:",
			"```python", "```bash", "```javascript", "```node",
		},
		volumeWords: []string{
			"volume", "sound level", "turn up", "turn down", "louder", "quieter",
			"mute", "unmute", "audio",
		},
		brightnessWords: []string{
			"brightness", "screen brightness", "dim", "brighten", "brighter", "dimmer",
			"backlight",
		},
		timerWords: []string{
			"start a timer", "set a timer", "start timer", "set timer",
			"timer for", "countdown", "stopwatch", "pomodoro",
			"how much time left", "cancel timer", "stop timer",
		},
		appWords: []string{
			"open app", "launch app", "start app", "open application",
			"launch application", "kill app", "close app",
			"running apps", "running applications", "what apps",
			"open firefox", "open chrome", "open terminal",
			"open spotify", "open slack", "open discord",
			"open vscode", "open code",
		},
		hashWords: []string{
			"hash", "md5", "sha256", "sha1", "sha512",
			"base64 encode", "base64 decode", "base64",
			"url encode", "url decode", "urlencode", "urldecode",
			"hex encode", "hex decode", "checksum",
		},
		dictWords: []string{
			"define ", "definition of", "meaning of", "what does the word",
			"synonym", "synonyms", "antonym", "antonyms",
			"thesaurus", "dictionary",
		},
		networkWords: []string{
			"is the server", "is the site", "is it down",
			"dns lookup", "dns resolve", "port check", "check port",
			"am i online", "internet connection", "connectivity",
			"network check", "net check",
		},
		translateWords: []string{
			"translate", "translation", "in spanish", "in french", "in german",
			"in japanese", "in chinese", "in korean", "in italian",
			"in portuguese", "in russian", "in arabic", "in hindi",
			"how do you say", "what is the translation",
		},
		archiveWords: []string{
			"zip ", "unzip", "compress", "decompress", "extract",
			"tar ", "untar", "archive", "create archive",
			"extract archive",
		},
		diskUsageWords: []string{
			"disk usage", "disk space", "storage space", "what's taking space",
			"what is using space", "largest folders", "largest directories",
			"folder size", "directory size", "space used",
		},
		processWords: []string{
			"list processes", "running processes", "top processes",
			"kill process", "stop process", "what's using cpu",
			"what is using cpu", "what's using memory", "what is using memory",
			"process list", "task manager",
		},
		qrcodeWords: []string{
			"qr code", "qrcode", "generate qr", "create qr", "scan qr",
			"read qr", "decode qr", "make qr",
		},
		calculatorWords: []string{
			"calculate", "calculator", "compute", "evaluate", "solve",
			"what is the result of", "math", "expression",
			"square root", "factorial", "percent of", "% of", "% off",
		},
		passwordWords: []string{
			"password", "passphrase",
			"generate pin", "generate a pin", "random pin", "new pin",
			"create a pin", "pin code", "pin number",
		},
		bookmarkWords: []string{
			"bookmark", "save link", "save url", "save this link",
			"my bookmarks", "list bookmarks", "show bookmarks",
			"delete bookmark", "remove bookmark", "search bookmarks",
			"saved links",
		},
		journalWords: []string{
			"journal", "diary", "diary entry", "journal entry",
			"dear diary", "write journal", "my journal", "today's journal",
			"log entry", "daily log", "write in journal", "write in diary",
			"show journal", "journal today", "week journal", "weekly journal",
		},
		habitWords: []string{
			"habit", "habits", "track habit", "habit tracker",
			"create habit", "new habit", "check habit", "mark habit",
			"did i", "habit streak", "habit status", "my habits",
			"daily habit", "habit done", "complete habit",
		},
		expenseWords: []string{
			"expense", "expenses", "spent", "purchase", "bought",
			"cost me", "paid for", "add expense", "log expense",
			"track expense", "my expenses", "spending", "how much spent",
			"expense summary", "monthly expenses", "weekly expenses",
		},
		transformWords: []string{
			"paraphrase", "rephrase", "reword", "rewrite",
			"summarize", "summarise", "summary", "tldr", "tl;dr",
			"formalize", "formalise", "make formal", "make it formal",
			"casualize", "make casual", "make it casual", "make informal",
			"bulletize", "bullet points", "make bullets", "turn into bullets",
			"prosify", "make prose", "turn into prose", "convert to prose",
			"simplify", "make simpler", "make it simpler", "dumb it down",
		},
		transformRe: []*regexp.Regexp{
			regexp.MustCompile(`(?i)^paraphrase\s+(?:this\s*:?\s*)?(.+)`),
			regexp.MustCompile(`(?i)^(?:rephrase|reword|rewrite)\s+(?:this\s*:?\s*)?(.+)`),
			regexp.MustCompile(`(?i)^summarize\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^(?:summarise|summary of|tldr|tl;dr)\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^formalize\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^(?:formalise|make (?:this |it )?formal)\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^(?:casualize|make (?:this |it )?casual|make (?:this |it )?informal)\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^(?:bulletize|bullet ?points?|make bullets|turn into bullets)\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^(?:prosify|make prose|turn into prose|convert to prose)\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^simplify\s*:?\s*(.+)`),
			regexp.MustCompile(`(?i)^(?:make (?:this |it )?simpler|dumb (?:this |it )?down)\s*:?\s*(.+)`),
		},
		creativeRe: []*regexp.Regexp{
			// "write about X" / "write me about X" (bare creative request → essay)
			regexp.MustCompile(`(?i)(?:write|compose|create)\s+(?:me\s+)?(?:something\s+)?about\s+(.*)`),
			// "help me write/create/make X" patterns
			regexp.MustCompile(`(?i)(?:can you |could you |would you |please )?help\s+me\s+(?:write|create|make|draft|compose)\s+(?:a\s+|an\s+|the\s+|my\s+)?(.*)`),
			// Poem patterns
			regexp.MustCompile(`(?i)(?:write|compose|create|make|craft|give me|generate)\s+(?:me\s+)?(?:a\s+)?(?:poem|verse|haiku|quatrain|sonnet|limerick|poetry)\s*(?:about|on|for|of)?\s*(.*)`),
			// Story patterns
			regexp.MustCompile(`(?i)(?:write|compose|create|make|craft|give me|tell|generate)\s+(?:me\s+)?(?:a\s+)?(?:story|tale|fable|narrative|short story|micro.?story)\s*(?:about|on|for|of|featuring)?\s*(.*)`),
			regexp.MustCompile(`(?i)(?:make up|come up with|invent)\s+(?:a\s+)?(?:story|tale|fable|narrative)\s*(?:about|on|for|of)?\s*(.*)`),
			// Joke patterns
			regexp.MustCompile(`(?i)(?:tell|give|make|write|say)\s+(?:me\s+)?(?:a\s+)?(?:joke|funny|pun|gag)\s*(?:about|on|for|of)?\s*(.*)`),
			// Reflect / opinion patterns
			regexp.MustCompile(`(?i)what(?:'s|\s+is)\s+(?:your|the)\s+(?:opinion|take|view|thought|perspective)\s+(?:on|about|of)\s+(.*)`),
			regexp.MustCompile(`(?i)what\s+do\s+you\s+(?:think|feel|believe)\s+(?:about|of)\s+(.*)`),
			regexp.MustCompile(`(?i)what(?:'s|\s+is)\s+the\s+meaning\s+of\s+(.*)`),
			regexp.MustCompile(`(?i)(?:reflect|philosophize|muse|meditate|ponder)\s+(?:on|about|upon)\s+(.*)`),
			regexp.MustCompile(`(?i)(?:share|give me)\s+(?:your\s+)?(?:thoughts?|perspective|views?)\s+(?:on|about)\s+(.*)`),
		},
		webLookupPatterns: []*regexp.Regexp{
			regexp.MustCompile(`(?i)what(?:'s| is) the (?:weather|temperature|forecast)`),
			regexp.MustCompile(`(?i)(?:latest|recent|current|breaking|today'?s?)\s+(?:news|headlines|updates?)`),
			regexp.MustCompile(`(?i)who (?:won|is winning|lost|scored|leads?)`),
			regexp.MustCompile(`(?i)(?:stock|share)\s+price`),
			regexp.MustCompile(`(?i)what(?:'s| is)\s+(?:happening|going on)\s+(?:in|at|with)`),
			regexp.MustCompile(`(?i)how (?:much|many)\s+(?:does|is|are)\s+\w+\s+(?:cost|worth)`),
		},
	}

	// Initialize the freeform fallback classifier with a small embedding space.
	// Uses taxonomy-seeded vectors for basic semantic similarity.
	emb := NewWordEmbeddings(64)
	emb.SeedPoolWords()
	n.Freeform = NewFreeformClassifier(emb)

	return n
}

// InitNeural initializes the neural intent classifier. If a model file exists
// at modelPath, it loads the pre-trained model. Otherwise, it trains a new one
// from auto-generated data and saves it.
func (n *NLU) InitNeural(modelPath string) error {
	nn := NewNeuralNLU(modelPath)
	if err := nn.LoadOrTrain(n); err != nil {
		return err
	}
	n.Neural = nn
	return nil
}

// extractEntitiesForIntent performs intent-specific entity extraction
// when the neural classifier has already determined the intent.
func (n *NLU) extractEntitiesForIntent(lower string, r *NLUResult) {
	switch r.Intent {
	case "dict":
		// Strip dict verbs to get the actual word: "define serendipity" → "serendipity"
		dictTriggers := []string{
			"define the word ", "define ", "definition of ",
			"meaning of ", "what does the word ", "what does ",
			"synonyms for ", "synonyms of ", "antonyms for ", "antonyms of ",
			"thesaurus lookup for ", "dictionary lookup for ",
			"give me the definition of ",
		}
		word := lower
		for _, t := range dictTriggers {
			if strings.Contains(lower, t) {
				word = n.extractTopic(lower, t)
				break
			}
		}
		// Strip trailing fragments: "what does X mean", "define X for me"
		word = strings.TrimSuffix(word, " mean")
		word = strings.TrimSuffix(word, " for me")
		word = strings.TrimSuffix(word, " please")
		word = strings.Trim(word, "'\"")
		r.Entities["topic"] = word

	case "translate":
		// Extract text and target language
		r.Entities["topic"] = n.extractTopicGeneral(lower)

	case "creative":
		// Detect specific creative types and extract topic
		switch {
		case strings.Contains(lower, "haiku"):
			r.Entities["creative_type"] = "haiku"
			r.Entities["poem_form"] = "haiku"
		case strings.Contains(lower, "limerick"):
			r.Entities["creative_type"] = "limerick"
		case strings.Contains(lower, "poem"):
			r.Entities["creative_type"] = "poem"
		case strings.Contains(lower, "joke"):
			r.Entities["creative_type"] = "joke"
		case strings.Contains(lower, "story"):
			r.Entities["creative_type"] = "story"
		case strings.Contains(lower, "fun fact") || strings.Contains(lower, "something interesting") || strings.Contains(lower, "random fact"):
			r.Entities["creative_type"] = "fun_fact"
		case strings.Contains(lower, "meaning of") || strings.Contains(lower, "opinion on") ||
			strings.Contains(lower, "think about") || strings.Contains(lower, "thoughts on") ||
			strings.Contains(lower, "philosophize") || strings.Contains(lower, "reflect on"):
			r.Entities["creative_type"] = "reflect"
		case strings.Contains(lower, "essay") || strings.Contains(lower, "article") ||
			strings.Contains(lower, "paragraph") || strings.Contains(lower, "words about") ||
			strings.Contains(lower, "write about"):
			r.Entities["creative_type"] = "essay"
		case strings.Contains(lower, "help me") || strings.Contains(lower, "help create") || strings.Contains(lower, "help write"):
			r.Entities["creative_type"] = "" // general assistance
		default:
			r.Entities["creative_type"] = "" // general creative
		}
		// Extract topic by stripping creative verbs/phrases
		creativePrefixes := []string{
			"write me a poem about ", "write a poem about ",
			"write me a haiku about ", "write a haiku about ",
			"write me a limerick about ", "write a limerick about ",
			"write me a story about ", "write a story about ",
			"write me a joke about ", "write a joke about ",
			"compose a haiku about ", "compose a poem about ",
			"compose a limerick about ", "compose a story about ",
			"create a limerick about ", "create a poem about ",
			"create a haiku about ", "create a story about ",
			"tell me a joke about ", "tell a joke about ",
			"write me a ", "write a ", "write me ", "write ",
			"compose a ", "compose ", "create a ", "create ",
			"tell me a ", "tell me ", "make me a ", "make a ",
			"help me write a ", "help me write ", "help me create a ",
			"help me create ", "help me make a ", "help me make ",
		}
		topic := lower
		for _, p := range creativePrefixes {
			if strings.HasPrefix(topic, p) {
				topic = topic[len(p):]
				break
			}
		}
		// Also try stripping "about" if it's at the start
		topic = strings.TrimPrefix(topic, "about ")
		topic = strings.TrimRight(topic, "?!.")
		topic = strings.TrimSpace(topic)
		// If the "topic" is just the creative type itself (e.g., "tell me a joke" → topic="joke"),
		// treat it as no topic rather than making jokes about "joke".
		creativeTypeWords := []string{"joke", "poem", "haiku", "limerick", "story", "tale", "essay", "article"}
		for _, ctw := range creativeTypeWords {
			if topic == ctw {
				topic = ""
				break
			}
		}
		r.Entities["topic"] = topic

	case "recommendation":
		// Strip recommendation verbs: "suggest a good book" → "book"
		recPrefixes := []string{
			"suggest a good ", "suggest me a ", "suggest a ", "suggest ",
			"recommend a good ", "recommend me a ", "recommend a ", "recommend ",
			"any tips for ", "any suggestions for ", "any ideas for ",
			"what should i ", "what should I ",
		}
		topic := lower
		for _, p := range recPrefixes {
			if strings.HasPrefix(topic, p) {
				topic = topic[len(p):]
				break
			}
		}
		topic = strings.TrimRight(topic, "?!.")
		r.Entities["topic"] = strings.TrimSpace(topic)

	case "remember":
		r.Entities["topic"] = n.extractTopicGeneral(lower)

	case "recall":
		r.Entities["topic"] = n.extractTopicGeneral(lower)

	case "timer":
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		// Extract timer action
		switch {
		case strings.Contains(lower, "pomodoro"):
			r.Entities["action"] = "pomodoro"
		case strings.Contains(lower, "stop") || strings.Contains(lower, "cancel"):
			r.Entities["action"] = "stop"
		case strings.Contains(lower, "how much time") || strings.Contains(lower, "time left") ||
			strings.Contains(lower, "status"):
			r.Entities["action"] = "status"
		case strings.Contains(lower, "list"):
			r.Entities["action"] = "list"
		default:
			r.Entities["action"] = "start"
		}
		// Extract duration: "5 minutes", "1h30m", "30 seconds"
		if dur := extractDurationString(lower); dur != "" {
			r.Entities["duration"] = dur
		}

	case "app":
		// Extract app action: launch, kill, list
		switch {
		case strings.Contains(lower, "close") || strings.Contains(lower, "kill") ||
			strings.Contains(lower, "stop") || strings.Contains(lower, "quit"):
			r.Entities["action"] = "kill"
		case strings.Contains(lower, "list") || strings.Contains(lower, "running") ||
			strings.Contains(lower, "what apps"):
			r.Entities["action"] = "running"
		default:
			r.Entities["action"] = "launch"
		}
		// Extract app name: "open firefox" → "firefox", "launch spotify" → "spotify"
		appPrefixes := []string{"open ", "launch ", "start ", "run ", "close ", "kill ", "stop ", "quit "}
		name := lower
		for _, p := range appPrefixes {
			if strings.HasPrefix(name, p) {
				name = name[len(p):]
				break
			}
		}
		name = strings.TrimSuffix(name, " for me")
		name = strings.TrimSuffix(name, " please")
		name = strings.TrimRight(name, "?!.")
		r.Entities["name"] = strings.TrimSpace(name)
		r.Entities["topic"] = r.Entities["name"]

	default:
		r.Entities["topic"] = n.extractTopicGeneral(lower)
	}
}

// Understand processes raw input and returns structured NLU output.
// This is PURE CODE — no LLM call, no I/O, no network.
// If Confidence < 0.5, the caller should make ONE LLM call for disambiguation.
func (n *NLU) Understand(input string) *NLUResult {
	result := &NLUResult{
		Raw:      input,
		Entities: make(map[string]string),
	}

	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		result.Intent = "unknown"
		result.Action = "respond"
		result.Confidence = 0.0
		return result
	}

	lower := strings.ToLower(trimmed)

	// Strip "nous" prefix — users often address the assistant by name.
	// e.g. "nous how are you?" → "how are you?"
	lower = stripNousPrefix(lower)
	trimmed = stripNousPrefix(trimmed)

	// Phase 1: Extract entities (always, regardless of intent)
	n.extractEntities(trimmed, lower, result)

	// Phase 2: Classify intent (order matters — most specific first)
	n.classifyIntent(trimmed, lower, result)

	// Phase 3: Map intent + entities to action
	n.mapAction(lower, result)

	return result
}

// extractEntities pulls structured data from the raw input using regex and heuristics.
func (n *NLU) extractEntities(raw, lower string, r *NLUResult) {
	// URLs
	if urls := n.urlRe.FindAllString(raw, -1); len(urls) > 0 {
		r.Entities["url"] = urls[0]
		if len(urls) > 1 {
			r.Entities["urls"] = strings.Join(urls, ",")
		}
	}

	// File paths
	if paths := n.pathRe.FindAllStringSubmatch(raw, -1); len(paths) > 0 {
		p := strings.TrimSpace(paths[0][1])
		r.Entities["path"] = p
	}

	// Math expressions
	if m := n.mathRe.FindStringSubmatch(raw); len(m) == 4 {
		r.Entities["expression"] = m[0]
	}

	// Quoted strings
	if m := n.quotedRe.FindAllStringSubmatch(raw, -1); len(m) > 0 {
		r.Entities["quoted"] = m[0][1]
		if len(m) > 1 {
			quoted := make([]string, len(m))
			for i, q := range m {
				quoted[i] = q[1]
			}
			r.Entities["all_quoted"] = strings.Join(quoted, "|")
		}
	}

	// Dates
	if m := n.dateWordRe.FindString(lower); m != "" {
		r.Entities["date"] = m
	} else if m := n.dateFormalRe.FindString(raw); m != "" {
		r.Entities["date"] = m
	}
}

// classifyIntent determines what the user wants.
func (n *NLU) classifyIntent(raw, lower string, r *NLUResult) {
	// Neural classifier runs FIRST on ALL inputs.
	// It's trained on 10,000+ realistic examples and handles
	// casual/messy input that patterns miss.
	if n.Neural != nil && n.Neural.Classifier.IsTrained() {
		nr := n.Neural.Classify(raw)
		if nr != nil && nr.Confidence > 0.60 {
			r.Intent = nr.Intent
			r.Confidence = nr.Confidence
			n.extractEntitiesForIntent(lower, r)
			return
		}
	}

	// Pattern matching as fallback for low-confidence neural results

	// Strip trailing punctuation for matching
	stripped := strings.TrimRightFunc(lower, func(r rune) bool {
		return unicode.IsPunct(r) || unicode.IsSpace(r)
	})

	// 0. Daily briefing — intercepts morning greetings and explicit requests
	briefingTriggers := []string{
		"good morning", "daily briefing", "my day", "brief me",
		"morning briefing", "daily summary", "start my day",
		"what's on today", "whats on today", "how's my day",
		"hows my day", "my morning", "morning report",
	}
	for _, t := range briefingTriggers {
		if stripped == t || strings.HasPrefix(lower, t+" ") || strings.HasPrefix(lower, t+"!") || strings.HasPrefix(lower, t+",") {
			r.Intent = "daily_briefing"
			r.Confidence = 0.95
			return
		}
	}

	// 1. Exact/prefix match: greetings
	for _, g := range n.greetings {
		if stripped == g || strings.HasPrefix(lower, g+" ") || strings.HasPrefix(lower, g+",") || strings.HasPrefix(lower, g+"!") {
			// Check if it's "good night" which can be farewell
			if strings.Contains(lower, "good night") {
				r.Intent = "farewell"
				r.Confidence = 0.95
				return
			}
			r.Intent = "greeting"
			r.Confidence = 0.95
			return
		}
	}

	// 2. Farewells
	for _, f := range n.farewells {
		if stripped == f || strings.HasPrefix(lower, f+" ") || strings.HasPrefix(lower, f+",") || strings.HasPrefix(lower, f+"!") {
			r.Intent = "farewell"
			r.Confidence = 0.95
			return
		}
	}

	// 3. Affirmations/negations (short responses)
	if len(strings.Fields(lower)) <= 4 {
		for _, a := range n.affirmatives {
			if stripped == a || lower == a+"!" || lower == a+"." {
				r.Intent = "affirmation"
				r.Confidence = 0.90
				return
			}
		}
	}

	// 4. Meta queries
	if stripped == "help" || stripped == "help me" {
		r.Intent = "meta"
		r.Confidence = 0.90
		return
	}
	for _, m := range n.metaPatterns {
		if strings.Contains(lower, m) {
			r.Intent = "meta"
			r.Confidence = 0.90
			return
		}
	}

	// 4b. URL present → fetch or summarize (before verb matching to prevent "open URL" → file_op)
	if _, hasURL := r.Entities["url"]; hasURL {
		if strings.Contains(lower, "summarize") || strings.Contains(lower, "summarise") ||
			strings.Contains(lower, "summary") || strings.Contains(lower, "tldr") ||
			strings.Contains(lower, "tl;dr") {
			r.Intent = "summarize"
			r.Action = "summarize_url"
			r.Confidence = 0.90
			return
		}
		r.Intent = "fetch"
		r.Action = "fetch_url"
		r.Confidence = 0.85
		return
	}

	// 4c. Follow-up / continuation detection
	// Catches explicit continuations AND short implicit follow-ups
	// ("why?", "really?", "and?") that reference the previous turn's topic.
	followUpMarkers := []string{
		// Explicit continuation
		"tell me more", "more about that", "go on", "continue",
		"keep going", "what else", "anything else",
		"dig deeper", "more details", "more about",
		// Explanation requests
		"why is that", "why is this", "how come", "how so",
		"can you explain", "explain that", "elaborate",
		"what do you mean", "in what way",
		// Example / comparison requests
		"for example", "like what", "such as",
		"compared to what", "how does that compare",
	}
	for _, marker := range followUpMarkers {
		if lower == marker || strings.HasPrefix(lower, marker+" ") || strings.HasPrefix(lower, marker+"?") {
			r.Intent = "followup"
			r.Action = "respond"
			r.Confidence = 0.85
			return
		}
	}
	// Short follow-up signals: 1-3 word inputs that aren't greetings/farewells
	if wordCount := len(strings.Fields(lower)); wordCount >= 1 && wordCount <= 3 {
		shortFollowUps := []string{
			"why", "how", "really", "interesting", "huh",
			"and", "so", "meaning", "seriously", "oh",
			"wow", "more", "examples", "details",
		}
		for _, sf := range shortFollowUps {
			if stripped == sf {
				r.Intent = "followup"
				r.Action = "respond"
				r.Confidence = 0.85
				return
			}
		}
	}

	// 5. Recall (before remember, since "do you remember" contains "remember")
	for _, v := range n.recallVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "recall"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 6. Remember/store
	for _, v := range n.memoryVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "remember"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 6a. Todos (before plan, since planVerbs contains "todo")
	for _, w := range n.todoWords {
		if strings.Contains(lower, w) {
			r.Intent = "todo"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}

	// 6b. Reminders (before plan, since planVerbs contains "remind")
	for _, w := range n.reminderWords {
		if strings.Contains(lower, w) {
			r.Intent = "reminder"
			r.Confidence = 0.90
			return
		}
	}

	// 6c. Calendar (before plan, since planVerbs contains "schedule")
	for _, w := range n.calendarWords {
		if strings.Contains(lower, w) {
			r.Intent = "calendar"
			r.Confidence = 0.90
			return
		}
	}

	// 6d. Assistant features (before file/search/command verbs to avoid misrouting)
	for _, w := range n.noteWords {
		if strings.Contains(lower, w) {
			r.Intent = "note"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.weatherWords {
		if strings.Contains(lower, w) {
			r.Intent = "weather"
			r.Confidence = 0.90
			r.Entities["location"] = extractWeatherLocation(raw)
			return
		}
	}
	for _, w := range n.sysinfoWords {
		if strings.Contains(lower, w) {
			r.Intent = "sysinfo"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.clipboardWords {
		if strings.Contains(lower, w) {
			r.Intent = "clipboard"
			r.Confidence = 0.90
			return
		}
	}
	for _, w := range n.emailWords {
		if strings.Contains(lower, w) {
			r.Intent = "email"
			r.Confidence = 0.90
			return
		}
	}
	for _, w := range n.screenshotWords {
		if strings.Contains(lower, w) {
			r.Intent = "screenshot"
			r.Confidence = 0.90
			return
		}
	}
	for _, w := range n.codeRunWords {
		if strings.Contains(lower, w) {
			r.Intent = "run_code"
			r.Confidence = 0.85
			return
		}
	}
	for _, w := range n.newsWords {
		if strings.Contains(lower, w) {
			r.Intent = "news"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.fileFinderWords {
		if strings.Contains(lower, w) {
			r.Intent = "find_files"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}

	// 6e. Desktop/system control tools
	for _, w := range n.volumeWords {
		if strings.Contains(lower, w) {
			r.Intent = "volume"
			r.Confidence = 0.90
			return
		}
	}
	for _, w := range n.brightnessWords {
		if strings.Contains(lower, w) {
			r.Intent = "brightness"
			r.Confidence = 0.90
			return
		}
	}
	for _, w := range n.timerWords {
		if strings.Contains(lower, w) {
			r.Intent = "timer"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.appWords {
		if strings.Contains(lower, w) {
			r.Intent = "app"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.hashWords {
		if strings.Contains(lower, w) {
			r.Intent = "hash"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.dictWords {
		if strings.Contains(lower, w) {
			// Strip the dict verb from the topic: "define serendipity" → "serendipity"
			topic := n.extractTopic(lower, w)
			// Only route to dict if the topic looks like a word/phrase to define,
			// not a philosophical question like "what is the meaning of life?"
			topicWords := strings.Fields(topic)
			if len(topicWords) > 3 {
				continue // too many words — probably a question, not a dict lookup
			}
			// Skip existential/philosophical uses of "meaning of"
			if w == "meaning of" {
				philosophical := []string{"life", "love", "existence", "everything", "it all", "happiness", "death", "freedom", "success"}
				isPhilosophical := false
				for _, p := range philosophical {
					if topic == p {
						isPhilosophical = true
						break
					}
				}
				if isPhilosophical {
					continue // let explain/question handle these
				}
			}
			r.Intent = "dict"
			r.Confidence = 0.85
			r.Entities["topic"] = topic
			return
		}
	}
	// Network: use Contains for multi-word patterns, matchWord for single words
	if matchWord(lower, "ping") {
		r.Intent = "network"
		r.Confidence = 0.85
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return
	}
	for _, w := range n.networkWords {
		if strings.Contains(lower, w) {
			r.Intent = "network"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.translateWords {
		if strings.Contains(lower, w) {
			r.Intent = "translate"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.archiveWords {
		if strings.Contains(lower, w) {
			r.Intent = "archive"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.diskUsageWords {
		if strings.Contains(lower, w) {
			r.Intent = "disk_usage"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.processWords {
		if strings.Contains(lower, w) {
			r.Intent = "process"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.qrcodeWords {
		if strings.Contains(lower, w) {
			r.Intent = "qrcode"
			r.Confidence = 0.85
			return
		}
	}
	for _, w := range n.calculatorWords {
		if strings.Contains(lower, w) {
			r.Intent = "calculate"
			r.Confidence = 0.85
			return
		}
	}
	for _, w := range n.passwordWords {
		if strings.Contains(lower, w) {
			r.Intent = "password"
			r.Confidence = 0.90
			return
		}
	}
	for _, w := range n.bookmarkWords {
		if strings.Contains(lower, w) {
			r.Intent = "bookmark"
			r.Confidence = 0.85
			return
		}
	}
	for _, w := range n.journalWords {
		if strings.Contains(lower, w) {
			r.Intent = "journal"
			r.Confidence = 0.85
			return
		}
	}
	for _, w := range n.habitWords {
		if strings.Contains(lower, w) {
			r.Intent = "habit"
			r.Confidence = 0.85
			return
		}
	}
	for _, w := range n.expenseWords {
		if strings.Contains(lower, w) {
			r.Intent = "expense"
			r.Confidence = 0.85
			return
		}
	}

	// 6e-transform. Text transformation patterns — must check regexes first for text extraction.
	for _, re := range n.transformRe {
		if m := re.FindStringSubmatch(raw); m != nil {
			r.Intent = "transform"
			r.Confidence = 0.90
			text := strings.TrimSpace(m[1])
			r.Entities["text"] = text
			// Determine operation from the pattern
			reLower := strings.ToLower(re.String())
			switch {
			case strings.Contains(reLower, "paraphrase") || strings.Contains(reLower, "rephrase") || strings.Contains(reLower, "reword") || strings.Contains(reLower, "rewrite"):
				r.Entities["operation"] = "paraphrase"
			case strings.Contains(reLower, "summarize") || strings.Contains(reLower, "summarise") || strings.Contains(reLower, "summary") || strings.Contains(reLower, "tldr"):
				r.Entities["operation"] = "summarize"
			case strings.Contains(reLower, "casualize") || strings.Contains(reLower, "casual") || strings.Contains(reLower, "informal"):
				r.Entities["operation"] = "casualize"
			case strings.Contains(reLower, "formalize") || strings.Contains(reLower, "formalise") || strings.Contains(reLower, "formal"):
				r.Entities["operation"] = "formalize"
			case strings.Contains(reLower, "bulletize") || strings.Contains(reLower, "bullet"):
				r.Entities["operation"] = "bulletize"
			case strings.Contains(reLower, "prosify") || strings.Contains(reLower, "prose"):
				r.Entities["operation"] = "prosify"
			case strings.Contains(reLower, "simplify") || strings.Contains(reLower, "simpler") || strings.Contains(reLower, "dumb"):
				r.Entities["operation"] = "simplify"
			default:
				r.Entities["operation"] = "paraphrase"
			}
			return
		}
	}
	// Fallback: simple word match for transform-related words
	for _, w := range n.transformWords {
		if strings.Contains(lower, w) {
			r.Intent = "transform"
			r.Confidence = 0.80
			// Determine operation from the matched word
			switch {
			case strings.Contains(w, "paraphrase") || strings.Contains(w, "rephrase") || strings.Contains(w, "reword") || strings.Contains(w, "rewrite"):
				r.Entities["operation"] = "paraphrase"
			case strings.Contains(w, "summar") || strings.Contains(w, "tldr") || strings.Contains(w, "tl;dr"):
				r.Entities["operation"] = "summarize"
			case strings.Contains(w, "formal"):
				if strings.Contains(w, "informal") {
					r.Entities["operation"] = "casualize"
				} else {
					r.Entities["operation"] = "formalize"
				}
			case strings.Contains(w, "casual"):
				r.Entities["operation"] = "casualize"
			case strings.Contains(w, "bullet"):
				r.Entities["operation"] = "bulletize"
			case strings.Contains(w, "pros"):
				r.Entities["operation"] = "prosify"
			case strings.Contains(w, "simplif") || strings.Contains(w, "simpler") || strings.Contains(w, "dumb"):
				r.Entities["operation"] = "simplify"
			default:
				r.Entities["operation"] = "paraphrase"
			}
			return
		}
	}

	// 6f-creative. Creative requests: poems, stories, jokes, reflections/opinions.
	for _, re := range n.creativeRe {
		if m := re.FindStringSubmatch(lower); m != nil {
			r.Intent = "creative"
			r.Confidence = 0.92
			topic := ""
			if len(m) > 1 {
				topic = strings.TrimSpace(m[1])
			}
			// Strip trailing punctuation from topic
			topic = strings.TrimRight(topic, "?!.")
			topic = strings.TrimSpace(topic)
			r.Entities["topic"] = topic
			// Determine creative sub-type from the matched pattern
			patStr := re.String()
			switch {
			case strings.Contains(patStr, "poem") || strings.Contains(patStr, "verse") || strings.Contains(patStr, "haiku") || strings.Contains(patStr, "quatrain") || strings.Contains(patStr, "poetry"):
				r.Entities["creative_type"] = "poem"
				if strings.Contains(lower, "haiku") {
					r.Entities["poem_form"] = "haiku"
				} else if strings.Contains(lower, "quatrain") {
					r.Entities["poem_form"] = "quatrain"
				}
			case strings.Contains(patStr, "story") || strings.Contains(patStr, "tale") || strings.Contains(patStr, "fable") || strings.Contains(patStr, "narrative"):
				r.Entities["creative_type"] = "story"
			case strings.Contains(patStr, "joke") || strings.Contains(patStr, "funny") || strings.Contains(patStr, "pun"):
				r.Entities["creative_type"] = "joke"
			case strings.Contains(patStr, "opinion") || strings.Contains(patStr, "think") || strings.Contains(patStr, "meaning") || strings.Contains(patStr, "reflect") || strings.Contains(patStr, "thought") || strings.Contains(patStr, "perspective") || strings.Contains(patStr, "feel") || strings.Contains(patStr, "believe"):
				r.Entities["creative_type"] = "reflect"
			case strings.Contains(patStr, "help"):
				// "help me write/create X" — general creative request, not a poem
				r.Entities["creative_type"] = "" // handled by creative handler as general
			default:
				r.Entities["creative_type"] = "poem"
			}
			return
		}
	}

	// 6f-emotional. Emotional/sentiment expressions — "I feel happy today"
	// Must run BEFORE fuzzy matching to prevent "feel" → "feed" (news) false positive.
	if isEmotionalInput(lower) {
		r.Intent = "greeting" // route to respond handler which has empathetic support
		r.Confidence = 0.80
		return
	}

	// 6f-fuzzy. Fuzzy fallback for tool word lists — synonym expansion + typo tolerance.
	// Only fires when none of the exact tool checks above matched.
	if n.fuzzyClassifyTools(lower, r) {
		return
	}

	// 6g. Recommendation patterns
	for _, v := range n.recommendVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "recommendation"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 6f. Comparison patterns ("X vs Y", "difference between", etc.)
	if n.compareVsRe.MatchString(lower) {
		r.Intent = "compare"
		r.Confidence = 0.85
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return
	}
	// "should I [verb] X or Y" → comparison/recommendation
	if strings.Contains(lower, " or ") && (strings.HasPrefix(lower, "should i ") || strings.HasPrefix(lower, "which ") || strings.HasPrefix(lower, "what's better ")) {
		r.Intent = "compare"
		r.Confidence = 0.80
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return
	}
	for _, v := range n.compareVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "compare"
			r.Confidence = 0.85
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 7. Plan/schedule
	for _, v := range n.planVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "plan"
			r.Confidence = 0.80
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 8. File operations (before search, since "read file" should be file_op not search)
	// "write"/"create" are ambiguous: "write a poem" is creative, "write to file" is file_op.
	// Skip file_op when the context looks creative (a/an + non-file noun).
	for _, v := range n.fileVerbs {
		if matchWord(lower, v) {
			if (v == "write" || v == "create") && looksCreativeNotFile(lower) {
				continue // let creative/freeform handle it
			}
			r.Intent = "file_op"
			r.Confidence = 0.85
			return
		}
	}

	// 9. (moved to 4b — URL detection now happens before verb matching)

	// 10. Search verbs (before compute — explicit "search for X" wins over incidental math)
	for _, v := range n.searchVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "search"
			r.Confidence = 0.85
			r.Entities["query"] = n.extractTopic(lower, v)
			return
		}
	}

	// 11. Compute: math expression present or compute verbs with numbers
	if _, hasExpr := r.Entities["expression"]; hasExpr {
		r.Intent = "compute"
		r.Confidence = 0.90
		return
	}
	for _, v := range n.computeVerbs {
		if strings.Contains(lower, v) && containsDigit(lower) {
			// Skip if the digit is part of a proper noun, not a math expression.
			// "world war 2" has a digit but isn't math; "what is 2+2" is.
			// Heuristic: strip the verb prefix and check if what remains looks
			// like math (contains operators or is mostly digits).
			remainder := lower
			if idx := strings.Index(lower, v); idx >= 0 {
				remainder = strings.TrimSpace(lower[idx+len(v):])
			}
			if !looksLikeMath(remainder) {
				continue
			}
			// "convert" with units → conversion, not compute
			if v == "convert" || v == "how much is" || v == "how many" {
				for _, cw := range n.convertWords {
					if strings.Contains(lower, cw) {
						r.Intent = "convert"
						r.Confidence = 0.90
						r.Entities["expr"] = raw
						return
					}
				}
			}
			r.Intent = "compute"
			r.Confidence = 0.80
			return
		}
	}

	// 12a. Convert words (fallback for conversions not caught by compute rule 11)
	for _, w := range n.convertWords {
		if strings.Contains(lower, w) {
			r.Intent = "convert"
			r.Confidence = 0.90
			r.Entities["expr"] = raw
			return
		}
	}

	// 12. Web lookup: current events, prices, scores
	for _, pat := range n.webLookupPatterns {
		if pat.MatchString(lower) {
			r.Intent = "web_lookup"
			r.Confidence = 0.90
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}
	for _, w := range n.currentEventWords {
		if strings.Contains(lower, w) {
			r.Intent = "web_lookup"
			r.Confidence = 0.75
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}

	// 12c. Conversational curiosity — "tell me something interesting/fun/cool"
	// Must run BEFORE explain verbs since "tell me" is an explain trigger.
	for _, pat := range []string{
		"tell me something", "say something", "tell me a fact",
		"give me a fun fact", "share something", "surprise me",
		"tell me a random", "random fact",
	} {
		if strings.HasPrefix(lower, pat) || lower == strings.TrimRight(pat, " ") {
			r.Intent = "creative"
			r.Confidence = 0.80
			r.Entities["creative_type"] = "fun_fact"
			r.Entities["topic"] = n.extractTopic(lower, pat)
			return
		}
	}

	// 13. Explain verbs
	for _, v := range n.explainVerbs {
		if strings.HasPrefix(lower, v+" ") || strings.HasPrefix(lower, v+"\t") {
			r.Intent = "explain"
			r.Confidence = 0.80
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 13b. Personal questions — check before command verbs so "do you know who i am"
	// doesn't match the "do" command verb.
	if n.isPersonalQuestion(lower) {
		r.Intent = "question"
		r.Confidence = 0.75
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return
	}

	// 14. Command verbs
	for _, v := range n.commandVerbs {
		if strings.HasPrefix(lower, v+" ") || strings.HasPrefix(lower, v+"\t") {
			r.Intent = "command"
			r.Confidence = 0.80
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 15. Questions (ends with ? or starts with question word)
	if strings.HasSuffix(strings.TrimSpace(raw), "?") {
		r.Intent = "question"
		r.Confidence = 0.75
		r.Entities["topic"] = n.extractTopicGeneral(lower)
		return
	}
	for _, q := range n.questionPrefixes {
		if strings.HasPrefix(lower, q) {
			r.Intent = "question"
			r.Confidence = 0.70
			r.Entities["topic"] = n.extractTopicGeneral(lower)
			return
		}
	}

	// 16. Contains explain verbs (not prefix-only, looser match)
	for _, v := range n.explainVerbs {
		if strings.Contains(lower, v) {
			r.Intent = "explain"
			r.Confidence = 0.65
			r.Entities["topic"] = n.extractTopic(lower, v)
			return
		}
	}

	// 17. Freeform fallback — catches natural language the pattern NLU missed.
	if n.Freeform != nil {
		if fr := n.Freeform.Classify(raw); fr != nil {
			r.Intent = fr.Intent
			r.Confidence = fr.Confidence
			for k, v := range fr.Entities {
				r.Entities[k] = v
			}
			return
		}
	}

	// 18. Last resort: if it's a short phrase, treat as question; if long, treat as statement
	words := strings.Fields(lower)
	if len(words) <= 3 {
		// Short input — likely a topic query (e.g. "isaac newton", "democracy")
		// Bare noun phrases are clear topic lookups — give enough confidence
		// to pass the 0.5 threshold so the pipeline actually runs.
		r.Intent = "question"
		r.Confidence = 0.55
		r.Entities["topic"] = strings.Join(words, " ")
	} else {
		r.Intent = "question"
		r.Confidence = 0.45
		r.Entities["topic"] = n.extractTopicGeneral(lower)
	}
}

// mapAction maps the classified intent + entities to a concrete action.
func (n *NLU) mapAction(lower string, r *NLUResult) {
	// If action was already set (e.g., fetch_url), keep it
	if r.Action != "" {
		return
	}

	// Safety overrides: pattern-based corrections that catch neural misclassifications.
	// These run before the intent switch and fix common misroutes.
	n.applyActionOverrides(lower, r)

	// Check for multi-step chain patterns before single-action mapping.
	if chainType := n.detectChain(lower); chainType != "" {
		r.Action = "chain"
		r.Entities["chain_type"] = chainType
		if r.Entities["topic"] == "" {
			r.Entities["topic"] = n.extractChainTopic(lower)
		}
		return
	}

	// Check for document generation patterns.
	if n.isDocGeneration(lower) {
		r.Action = "generate_doc"
		if r.Entities["topic"] == "" {
			r.Entities["topic"] = n.extractChainTopic(lower)
		}
		return
	}

	// Date questions: if a date entity is present and the query is ABOUT dates/time,
	// route to compute (the date evaluator handles these without LLM).
	if _, hasDate := r.Entities["date"]; hasDate {
		if r.Intent == "question" || r.Intent == "explain" {
			// Only route to compute if the query is actually asking about dates
			dateQuestions := []string{"what day", "what date", "when is", "when was",
				"how many days", "how long until", "what time", "what year"}
			isDateQuestion := false
			for _, dq := range dateQuestions {
				if strings.Contains(lower, dq) {
					isDateQuestion = true
					break
				}
			}
			if isDateQuestion {
				r.Action = "compute"
				r.Entities["expr"] = r.Raw
				return
			}
		}
	}

	switch r.Intent {
	case "greeting", "farewell", "affirmation":
		r.Action = "respond"

	case "meta":
		r.Action = "respond"

	// Assistant features: direct intent → action mapping
	case "weather":
		r.Action = "weather"
	case "convert":
		r.Action = "convert"
	case "reminder":
		r.Action = "reminder"
	case "sysinfo":
		r.Action = "sysinfo"
	case "clipboard":
		r.Action = "clipboard"
	case "note":
		r.Action = "notes"
	case "todo":
		r.Action = "todos"
	case "news":
		r.Action = "news"
	case "calendar":
		r.Action = "calendar"
	case "email":
		r.Action = "check_email"
	case "screenshot":
		r.Action = "screenshot"
	case "run_code":
		r.Action = "run_code"
	case "find_files":
		r.Action = "find_files"

	case "volume":
		r.Action = "volume"
	case "brightness":
		r.Action = "brightness"
	case "timer":
		r.Action = "timer"
	case "app":
		r.Action = "app"
	case "hash":
		r.Action = "hash"
	case "dict":
		r.Action = "dict"
	case "network":
		r.Action = "network"
	case "translate":
		r.Action = "translate"
	case "archive":
		r.Action = "archive"
	case "disk_usage":
		r.Action = "disk_usage"
	case "process":
		r.Action = "process"
	case "qrcode":
		r.Action = "qrcode"
	case "calculate":
		r.Action = "calculate"
	case "password":
		r.Action = "password"
	case "bookmark":
		r.Action = "bookmark"
	case "journal":
		r.Action = "journal"
	case "habit":
		r.Action = "habit"
	case "expense":
		r.Action = "expense"
	case "daily_briefing":
		r.Action = "daily_briefing"
	case "transform":
		r.Action = "transform"
	case "creative":
		r.Action = "creative"

	case "follow_up":
		r.Action = "respond"

	case "conversation":
		r.Action = "respond"

	case "recommendation":
		r.Action = "lookup_knowledge"
	case "compare":
		r.Action = "compare"

	case "remember":
		r.Action = "lookup_memory" // store to memory

	case "recall":
		r.Action = "lookup_memory"

	case "plan":
		r.Action = "schedule"

	case "file_op":
		r.Action = "file_op"

	case "compute":
		r.Action = "compute"

	case "search":
		r.Action = "web_search"

	case "web_lookup":
		r.Action = "web_search"

	case "explain":
		// Explanation of a concept: try knowledge base first
		r.Action = "lookup_knowledge"

	case "question":
		// Determine if we need web, knowledge, or memory
		if n.needsWebLookup(lower) {
			r.Action = "web_search"
		} else if n.isPersonalQuestion(lower) {
			r.Action = "lookup_memory"
		} else {
			// General question — try knowledge base, caller escalates to web if not found
			r.Action = "lookup_knowledge"
		}

	case "command":
		r.Action = "llm_chat" // commands need LLM to figure out specifics

	default:
		r.Action = "llm_chat"
	}
}

// applyActionOverrides corrects neural misclassifications using pattern matching.
// The neural classifier sometimes routes queries to the wrong intent (e.g.,
// "sqrt of 144" → "conversation", "thanks!" → "conversation"). These overrides
// catch the most common misroutes and fix the intent before mapAction's switch.
func (n *NLU) applyActionOverrides(lower string, r *NLUResult) {
	clean := strings.TrimRight(strings.TrimSpace(lower), "!?.\\ ")

	// Deep explanation phrases should always route to knowledge explanation,
	// even when the neural classifier misroutes them to conversation/affirmation.
	deepExplainPrefixes := []string{
		"tell me everything about ",
		"tell me all about ",
		"give me an overview of ",
		"give me a full overview of ",
		"walk me through ",
		"deep dive into ",
	}
	for _, p := range deepExplainPrefixes {
		if strings.HasPrefix(clean, p) {
			r.Intent = "explain"
			if r.Confidence < 0.88 {
				r.Confidence = 0.88
			}
			topic := strings.TrimSpace(clean[len(p):])
			r.Entities["topic"] = topic
			return
		}
	}

	// Thank you patterns → affirmation
	thankPatterns := []string{
		"thanks", "thank you", "thx", "ty", "cheers",
		"thanks a lot", "thank you so much", "thanks so much",
		"much appreciated", "appreciate it",
	}
	for _, t := range thankPatterns {
		if clean == t {
			r.Intent = "affirmation"
			return
		}
	}

	// Math expressions with function calls: "sqrt of 144", "log of 100"
	mathFuncs := []string{"sqrt", "abs", "sin", "cos", "tan", "log", "ln", "ceil", "floor", "round"}
	for _, fn := range mathFuncs {
		if strings.HasPrefix(clean, fn+" ") {
			r.Intent = "calculate"
			r.Entities["expression"] = clean
			return
		}
	}

	// Natural language math: "15 times 23", "100 divided by 5", "3 plus 4"
	mathWordRe := regexp.MustCompile(`(?i)\d+\s+(?:times|multiplied by|plus|minus|divided by)\s+\d+`)
	if mathWordRe.MatchString(lower) {
		r.Intent = "calculate"
		r.Entities["expression"] = lower
		return
	}

	// Comparison patterns: "compare X and Y", "X vs Y", "difference between X and Y"
	// Neural sometimes misclassifies these when items are code-related.
	compareRe := regexp.MustCompile(`(?i)^(?:compare|contrast|difference between|differences between)\b`)
	vsRe := regexp.MustCompile(`(?i)\b\w+\s+vs\.?\s+\w+`)
	if compareRe.MatchString(clean) || vsRe.MatchString(clean) || strings.Contains(clean, "better than") {
		r.Intent = "compare"
		return
	}

	// Emotional/personal statements: "I feel tired", "I'm sad", etc.
	// Neural sometimes misclassifies these as greetings.
	if isEmotionalInput(lower) {
		r.Intent = "conversation"
		return
	}

	// Planning/learning questions: "how do I learn X", "teach me X"
	if IsPlanningQuestion(lower) {
		r.Intent = "question"
		r.Entities["topic"] = ExtractGoal(lower)
		return
	}
}

// needsWebLookup returns true if the question requires external/current knowledge.
func (n *NLU) needsWebLookup(lower string) bool {
	for _, w := range n.currentEventWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	for _, pat := range n.webLookupPatterns {
		if pat.MatchString(lower) {
			return true
		}
	}
	return false
}

// isPersonalQuestion returns true if the question is about the user's stored info.
func (n *NLU) isPersonalQuestion(lower string) bool {
	personalPrefixes := []string{
		"what's my", "whats my", "what is my",
		"where do i", "what do i", "who am i",
		"what did i", "have i", "am i",
		"do you know who i am", "do you know my",
		"do you remember me", "do you know me",
		"what do you know about me",
	}
	for _, p := range personalPrefixes {
		if strings.HasPrefix(lower, p) || strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// extractDurationString extracts a duration string like "5 minutes", "1h30m", "30 seconds"
// from natural language input and returns it in Go-parseable format.
func extractDurationString(lower string) string {
	// Pattern: N minutes/seconds/hours
	durationRe := regexp.MustCompile(`(\d+)\s*(minutes?|mins?|seconds?|secs?|hours?|hrs?)`)
	if m := durationRe.FindStringSubmatch(lower); len(m) >= 3 {
		num := m[1]
		unit := m[2]
		switch {
		case strings.HasPrefix(unit, "min"):
			return num + "m"
		case strings.HasPrefix(unit, "sec"):
			return num + "s"
		case strings.HasPrefix(unit, "hour"), strings.HasPrefix(unit, "hr"):
			return num + "h"
		}
	}
	// Already Go duration format: "5m", "1h30m"
	goRe := regexp.MustCompile(`\b(\d+[hms](?:\d+[ms])?)\b`)
	if m := goRe.FindString(lower); m != "" {
		return m
	}
	return ""
}

// extractTopic strips the verb/trigger from input and returns the remaining subject.
func (n *NLU) extractTopic(lower, trigger string) string {
	idx := strings.Index(lower, trigger)
	if idx < 0 {
		return n.extractTopicGeneral(lower)
	}
	after := strings.TrimSpace(lower[idx+len(trigger):])
	// Strip common filler words at the start
	after = stripLeadingFillers(after)
	// Trim trailing punctuation
	after = strings.TrimRight(after, "?!.")
	return strings.TrimSpace(after)
}

// extractTopicGeneral strips common question/filler words to get the core topic.
func (n *NLU) extractTopicGeneral(lower string) string {
	// Remove question words and common fillers
	topic := lower
	for _, strip := range []string{
		"what is ", "what's ", "what are ", "what does ",
		"who is ", "who are ", "who was ", "who ",
		"where is ", "where are ",
		"when is ", "when was ", "when did ",
		"why is ", "why are ", "why does ", "why did ",
		"how does ", "how do ", "how is ", "how are ", "how can ",
		"can you ", "could you ", "would you ", "please ",
		"tell me ", "i want to know ",
		"is there ", "are there ",
	} {
		if strings.HasPrefix(topic, strip) {
			topic = topic[len(strip):]
			break
		}
	}
	topic = strings.TrimRight(topic, "?!.")
	topic = strings.TrimSpace(topic)
	// Strip leading articles/fillers: "a black hole" → "black hole"
	topic = stripLeadingFillers(topic)
	return strings.TrimSpace(topic)
}

// stripLeadingFillers removes filler words like "about", "the", "a", "for" from the start.
func stripLeadingFillers(s string) string {
	fillers := []string{"about ", "the ", "a ", "an ", "for ", "me ", "that ", "this "}
	changed := true
	for changed {
		changed = false
		for _, f := range fillers {
			if strings.HasPrefix(s, f) {
				s = s[len(f):]
				changed = true
			}
		}
	}
	return s
}

// matchWord checks if the phrase appears in s as a word boundary match,
// not as a substring of another word. For multi-word phrases, uses Contains.
func matchWord(s, phrase string) bool {
	if strings.Contains(phrase, " ") {
		// Multi-word phrase: exact substring match is fine
		return strings.Contains(s, phrase)
	}
	// Single word: check word boundaries
	idx := 0
	for {
		pos := strings.Index(s[idx:], phrase)
		if pos < 0 {
			return false
		}
		pos += idx
		start := pos
		end := pos + len(phrase)
		leftOK := start == 0 || !isWordChar(rune(s[start-1]))
		rightOK := end >= len(s) || !isWordChar(rune(s[end]))
		if leftOK && rightOK {
			return true
		}
		idx = pos + 1
		if idx >= len(s) {
			return false
		}
	}
}

func isWordChar(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_'
}

// followUpPatterns are phrases that indicate the user is referring to a previous topic.
var followUpPatterns = []string{
	"explain further", "explain more", "tell me more", "go on", "continue",
	"elaborate", "more details", "what else", "and then",
	"keep going", "go ahead",
}

// followUpExactPatterns match single-word or very short follow-ups.
var followUpExactPatterns = []string{
	"why", "how", "really", "seriously", "and",
}

// isFollowUp returns true if the input looks like a follow-up referencing a previous turn.
func isFollowUp(lower string, result *NLUResult) bool {
	stripped := strings.TrimRight(lower, "?!. ")

	// Exact match on short follow-ups
	for _, p := range followUpExactPatterns {
		if stripped == p {
			return true
		}
	}

	// Substring match on follow-up phrases
	for _, p := range followUpPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}

	// "what about X" pattern — follow-up with a new angle
	if strings.HasPrefix(lower, "what about ") || strings.HasPrefix(lower, "how about ") {
		return true
	}

	// Short input (under 4 words) starting with a question word and no clear topic
	words := strings.Fields(lower)
	if len(words) > 0 && len(words) < 4 {
		questionWords := map[string]bool{
			"what": true, "why": true, "how": true, "when": true,
			"where": true, "who": true, "which": true,
		}
		first := strings.TrimRight(words[0], "?!.,")
		if questionWords[first] {
			topic := result.Entities["topic"]
			if topic == "" || topic == stripped {
				return true
			}
		}
	}

	return false
}

// lastUserMessage returns the content of the last user message in the conversation,
// or empty string if there is no prior user message.
func lastUserMessage(conv *Conversation) string {
	if conv == nil {
		return ""
	}
	msgs := conv.Messages()
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return msgs[i].Content
		}
	}
	return ""
}

// extractTopicFromPrior extracts the core topic from a prior user message.
func (n *NLU) extractTopicFromPrior(prior string) string {
	lower := strings.ToLower(strings.TrimSpace(prior))
	return n.extractTopicGeneral(lower)
}

// UnderstandWithContext processes input with conversation history for follow-up resolution.
// When the user says "explain further" or "tell me more", this resolves the topic
// from the previous conversation turn.
func (n *NLU) UnderstandWithContext(input string, conv *Conversation) *NLUResult {
	result := n.Understand(input)

	lower := strings.ToLower(strings.TrimSpace(input))

	if !isFollowUp(lower, result) {
		return result
	}

	prior := lastUserMessage(conv)
	if prior == "" {
		return result
	}

	// Resolve topic from the previous turn
	previousTopic := n.extractTopicFromPrior(prior)
	if previousTopic == "" {
		return result
	}

	// "what about X" / "how about X" — combine the new angle with the previous topic
	if strings.HasPrefix(lower, "what about ") {
		newAngle := strings.TrimPrefix(lower, "what about ")
		newAngle = strings.TrimRight(newAngle, "?!. ")
		result.Entities["topic"] = previousTopic + " — " + newAngle
		result.Entities["previous_topic"] = previousTopic
		result.Entities["new_angle"] = newAngle
	} else if strings.HasPrefix(lower, "how about ") {
		newAngle := strings.TrimPrefix(lower, "how about ")
		newAngle = strings.TrimRight(newAngle, "?!. ")
		result.Entities["topic"] = previousTopic + " — " + newAngle
		result.Entities["previous_topic"] = previousTopic
		result.Entities["new_angle"] = newAngle
	} else {
		// Pure follow-up — carry topic forward
		result.Entities["topic"] = previousTopic
		result.Entities["previous_topic"] = previousTopic
	}

	result.Entities["follow_up"] = "true"

	// Boost confidence — we resolved the referent
	if result.Confidence < 0.7 {
		result.Confidence = 0.7
	}

	// If intent was vague or a bare follow-up, sharpen it to explain
	// (most follow-ups want elaboration). "lookup_knowledge" from the
	// freeform catch-all and "followup"/"follow_up" from the NLU are
	// also vague for context-aware follow-ups.
	if result.Intent == "question" || result.Intent == "unknown" || result.Intent == "explain" || result.Intent == "lookup_knowledge" || result.Intent == "followup" || result.Intent == "follow_up" {
		result.Intent = "explain"
		result.Action = "lookup_knowledge"
	}

	return result
}

// isEmotionalInput detects emotional/sentiment expressions that should get an
// empathetic response rather than tool routing.
func isEmotionalInput(lower string) bool {
	// Decision-making, coaching, and planning language should NOT be treated
	// as emotional input — they are substantive queries.
	decisionWords := []string{
		"decide", "torn between", "choosing between", "should i", "help me decide",
		"career", "business", "start my own", "change careers", "switching to",
		"improve", "get better", "learn to", "want to change", "planning",
		"strategy", "how do i", "what should", "figure out",
		"compare", "difference", "versus", "pros and cons",
		"stuck", "don't know where to start", "need guidance",
	}
	for _, d := range decisionWords {
		if strings.Contains(lower, d) {
			return false
		}
	}
	patterns := []string{
		"i feel ", "i'm feeling ", "im feeling ",
		"i'm happy", "im happy", "i'm sad", "im sad",
		"i'm angry", "im angry", "i'm tired", "im tired",
		"i'm excited", "im excited", "i'm stressed", "im stressed",
		"i'm bored", "im bored", "i'm lonely", "im lonely",
		"i'm anxious", "im anxious", "i'm worried", "im worried",
		"i'm frustrated", "im frustrated", "i'm overwhelmed", "im overwhelmed",
		"i'm exhausted", "im exhausted", "i'm grateful", "im grateful",
		"having a bad day", "having a good day", "having a great day",
		"rough day", "tough day", "hard day", "best day", "worst day",
	}
	for _, p := range patterns {
		if strings.HasPrefix(lower, p) || strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// looksCreativeNotFile returns true if "write/create X" is creative, not file I/O.
func looksCreativeNotFile(lower string) bool {
	creativeObjects := []string{
		"a poem", "a story", "a joke", "an email", "a letter",
		"a song", "a haiku", "a limerick", "a report", "a summary",
		"a reflection", "an essay", "a script", "a speech",
		"a list", "a shopping list", "a grocery list", "a todo list",
		"a to-do list", "a checklist", "a recipe", "a menu",
		"a plan", "a schedule", "a draft", "a message",
		"me a", "about", "something",
		"help me write", "help me create", "help me make",
	}
	for _, obj := range creativeObjects {
		if strings.Contains(lower, obj) {
			return true
		}
	}
	return false
}

// stripNousPrefix removes "nous" addressing from the input.
// Handles both prefix ("nous, how are you") and suffix ("hi nous!").
func stripNousPrefix(s string) string {
	lower := strings.ToLower(s)
	// Prefix: "nous, ..." / "nous! ..." / "nous ..."
	for _, prefix := range []string{"nous, ", "nous! ", "nous ", "nous,"} {
		if strings.HasPrefix(lower, prefix) {
			return strings.TrimSpace(s[len(prefix):])
		}
	}
	// Suffix: "hi nous!" / "hello nous" / "hey, nous"
	stripped := strings.TrimRight(s, "!?. ")
	lowerStripped := strings.ToLower(stripped)
	for _, suffix := range []string{" nous", ", nous"} {
		if strings.HasSuffix(lowerStripped, suffix) {
			return strings.TrimSpace(stripped[:len(stripped)-len(suffix)])
		}
	}
	return s
}

// containsDigit returns true if the string contains at least one digit.
func containsDigit(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}

// looksLikeMath returns true if s looks like a math expression rather than
// a proper noun that happens to contain digits.
// "2+2", "15 * 3", "sqrt(16)" → true
// "world war 2", "apollo 13", "henry viii" → false
func looksLikeMath(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	// Contains math operators → definitely math
	for _, op := range []string{"+", "-", "*", "/", "^", "=", "(", ")", "sqrt", "log", "sin", "cos"} {
		if strings.Contains(s, op) {
			return true
		}
	}
	// Starts with a digit → likely math ("2 plus 3", "100 divided by 5")
	if s[0] >= '0' && s[0] <= '9' {
		return true
	}
	// If more than half the non-space chars are digits/operators, it's math
	digits, total := 0, 0
	for _, r := range s {
		if r == ' ' {
			continue
		}
		total++
		if (r >= '0' && r <= '9') || r == '.' || r == ',' {
			digits++
		}
	}
	if total > 0 && float64(digits)/float64(total) > 0.5 {
		return true
	}
	return false
}

// -----------------------------------------------------------------------
// Chain detection — identifies multi-step intent patterns.
// -----------------------------------------------------------------------

// Chain detection regexes compiled once.
var (
	chainSearchAndSaveRe = regexp.MustCompile(
		`(?i)(?:search|find|look up|lookup)\s+(?:for\s+)?(.+?)\s+and\s+(?:save|write|store)\s+(?:it\s+)?(?:to\s+)?(?:a\s+)?(?:file)?`)
	chainSearchAndExplainRe = regexp.MustCompile(
		`(?i)(?:look up|lookup|search|find)\s+(?:for\s+)?(.+?)\s+and\s+(?:explain|summarize|describe)\s+(?:it)?`)
	chainResearchRe = regexp.MustCompile(
		`(?i)^(?:research|investigate|deep dive into|explore)\s+(.+)`)
	chainSummarizeFromWebRe = regexp.MustCompile(
		`(?i)(?:summarize|summarise)\s+(.+?)\s+from\s+(?:the\s+)?(?:web|internet|online)`)
)

// detectChain returns the chain_type if the input matches a multi-step pattern,
// or empty string if no chain is detected.
func (n *NLU) detectChain(lower string) string {
	// "search X and save it" / "find X and write to file"
	if chainSearchAndSaveRe.MatchString(lower) {
		return "search_and_save"
	}

	// "look up X and explain it" / "search X and summarize it"
	if chainSearchAndExplainRe.MatchString(lower) {
		return "search_and_explain"
	}

	// "research X" / "investigate X" / "deep dive into X"
	if chainResearchRe.MatchString(lower) {
		return "research_and_write"
	}

	// "summarize X from the web"
	if chainSummarizeFromWebRe.MatchString(lower) {
		return "search_and_explain"
	}

	return ""
}

// isDocGeneration returns true if the input asks for document creation.
func (n *NLU) isDocGeneration(lower string) bool {
	docPatterns := []string{
		"create a document about ",
		"create a report about ",
		"create a report on ",
		"create a document on ",
		"write a document about ",
		"write a report about ",
		"write a report on ",
		"write a document on ",
		"generate a document about ",
		"generate a report about ",
		"generate a report on ",
		"make a document about ",
		"make a report about ",
		"draft a document about ",
		"draft a report about ",
	}
	for _, p := range docPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// extractChainTopic extracts the subject/topic from a chain-type query.
func (n *NLU) extractChainTopic(lower string) string {
	// Try each chain regex to extract the topic capture group.
	if m := chainSearchAndSaveRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}
	if m := chainSearchAndExplainRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}
	if m := chainResearchRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}
	if m := chainSummarizeFromWebRe.FindStringSubmatch(lower); len(m) >= 2 {
		return strings.TrimSpace(m[1])
	}

	// Document generation topic extraction.
	docPrefixes := []string{
		"create a document about ", "create a report about ",
		"create a report on ", "create a document on ",
		"write a document about ", "write a report about ",
		"write a report on ", "write a document on ",
		"generate a document about ", "generate a report about ",
		"generate a report on ", "make a document about ",
		"make a report about ", "draft a document about ",
		"draft a report about ",
	}
	for _, p := range docPrefixes {
		if idx := strings.Index(lower, p); idx >= 0 {
			topic := lower[idx+len(p):]
			topic = strings.TrimRight(topic, "?!. ")
			return strings.TrimSpace(topic)
		}
	}

	return n.extractTopicGeneral(lower)
}
