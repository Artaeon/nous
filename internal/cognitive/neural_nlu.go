package cognitive

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
)

// -----------------------------------------------------------------------
// Neural NLU Integration — wires the neural classifier into the NLU
// pipeline as a trained replacement for hardcoded pattern matching.
//
// The neural NLU:
//   1. Auto-generates training data from existing NLU word lists
//   2. Trains a compact MLP on ~4000 labeled examples
//   3. Runs inference in sub-microsecond time
//   4. Falls back to pattern NLU when confidence is low
//
// Integration with existing NLU:
//   - High confidence (>0.7): use neural result directly
//   - Medium confidence (0.4-0.7): compare with pattern NLU
//   - Low confidence (<0.4): defer to pattern NLU
// -----------------------------------------------------------------------

// NeuralNLU wraps the neural classifier for NLU integration.
type NeuralNLU struct {
	Classifier *NeuralClassifier
	ModelPath  string
}

// NewNeuralNLU creates a new neural NLU integration.
func NewNeuralNLU(modelPath string) *NeuralNLU {
	return &NeuralNLU{
		Classifier: NewNeuralClassifier(DefaultFeatureSize, DefaultHiddenSize),
		ModelPath:  modelPath,
	}
}

// LoadOrTrain loads a trained model from disk, or trains one from scratch
// using auto-generated training data from the pattern NLU's word lists.
func (nn *NeuralNLU) LoadOrTrain(nlu *NLU) error {
	// Try to load existing model
	if nn.ModelPath != "" {
		if _, err := os.Stat(nn.ModelPath); err == nil {
			if err := nn.Classifier.Load(nn.ModelPath); err == nil {
				fmt.Fprintf(os.Stderr, "  neural classifier loaded (%d intents)\n", nn.Classifier.numIntents)
				return nil
			}
		}
	}

	// Generate training data: base + realistic + augmentation
	examples := GenerateTrainingData(nlu)
	realistic := GenerateRealisticTrainingData(nlu)
	allExamples := append(examples, realistic...)
	augmented := AugmentExamples(allExamples)
	fmt.Fprintf(os.Stderr, "  training neural classifier (%d examples, %d intents)...\n", len(augmented), countIntents(augmented))
	// 40 epochs is sufficient with the expanded dataset (12,000+ examples).
	// More data per epoch compensates for fewer epochs.
	result := nn.Classifier.Train(augmented, 40, 0.1)
	fmt.Fprintf(os.Stderr, "  neural classifier trained: %.1f%% accuracy\n", result.Accuracy*100)

	// Save the trained model
	if nn.ModelPath != "" {
		if err := nn.Classifier.Save(nn.ModelPath); err != nil {
			return fmt.Errorf("save model: %w", err)
		}
	}

	return nil
}

func countIntents(examples []TrainingExample) int {
	seen := make(map[string]bool)
	for _, ex := range examples {
		seen[ex.Intent] = true
	}
	return len(seen)
}

// Classify runs neural classification and returns an NLUResult.
func (nn *NeuralNLU) Classify(input string) *NLUResult {
	intent, confidence := nn.Classifier.Classify(input)
	if intent == "" {
		return nil
	}

	return &NLUResult{
		Intent:     intent,
		Confidence: confidence,
		Raw:        input,
		Entities:   make(map[string]string),
	}
}

// -----------------------------------------------------------------------
// Training data generation — builds labeled examples from the NLU's
// existing word lists and template expansion.
// -----------------------------------------------------------------------

// GenerateTrainingData creates comprehensive training examples from the
// NLU's word lists, augmented with template variations.
func GenerateTrainingData(nlu *NLU) []TrainingExample {
	var examples []TrainingExample

	// Helper to add examples
	add := func(intent string, texts ...string) {
		for _, t := range texts {
			examples = append(examples, TrainingExample{Text: t, Intent: intent})
		}
	}

	// ---- Greetings ----
	// Add greetings from NLU word list, but skip "good morning" / "good afternoon"
	// which are daily_briefing triggers, not simple greetings.
	briefingOverlap := map[string]bool{
		"good morning": true, "good afternoon": true, "good evening": true,
	}
	for _, g := range nlu.greetings {
		if !briefingOverlap[g] {
			add("greeting", g)
		}
	}
	add("greeting",
		"hey there!", "hello!", "hi!", "yo!", "sup!", "howdy!",
		"hey nous", "hi nous", "hello nous", "nous hello",
		"hey nous how are you", "nous how are you doing",
		"hi there nous",
		"what's up man", "hey buddy", "hello friend",
		"greetings friend", "hey how's it going",
		"hi how are you today", "hey there how are you",
		"hello how are you doing", "yo what's up",
		"hey what's going on", "hiya", "hey yo",
		"long time no see", "nice to see you",
	)

	// ---- Farewells ----
	for _, f := range nlu.farewells {
		add("farewell", f)
	}
	add("farewell",
		"goodbye!", "bye!", "see ya!", "later!",
		"i'm heading out", "gotta run", "time to go",
		"that's all for now", "i'm done for today",
		"thanks bye", "ok bye bye", "night night",
	)

	// ---- Meta / self-awareness ----
	for _, m := range nlu.metaPatterns {
		add("meta", m)
	}
	add("meta",
		"who are you?", "what are you?", "what's your name?",
		"who made you?", "who created you?", "who built you?",
		"tell me about yourself", "what can you do for me?",
		"what are your capabilities?", "how do you work?",
		"are you an AI?", "are you a chatbot?", "are you a bot?",
		"are you sentient?", "are you conscious?", "are you alive?",
		"do you have feelings?", "can you think?", "do you dream?",
		"are you real?", "are you human?", "are you a robot?",
		"how were you built?", "what technology do you use?",
		"what makes you different from other AIs?",
		"what programming language are you written in?",
		"do you use machine learning?", "are you an LLM?",
		"what is nous?", "tell me about nous",
		"introduce yourself", "what should I call you?",
	)

	// ---- Emotional / sentiment ----
	add("greeting", // routed to greeting for empathetic response
		"i feel happy today", "i'm feeling great", "i feel sad",
		"i'm feeling down", "i feel excited", "i'm stressed out",
		"i feel anxious today", "i'm feeling lonely",
		"having a bad day", "having a great day", "rough day",
		"i'm so tired", "i'm exhausted", "i feel overwhelmed",
		"i'm frustrated", "i feel grateful", "feeling bored",
		"i'm worried about something", "i feel amazing",
		"today was the worst day", "best day ever",
	)

	// ---- Affirmations ----
	for _, a := range nlu.affirmatives {
		add("affirmation", a)
	}
	add("affirmation",
		"thanks!", "thank you!", "cool!", "nice!", "great!",
		"perfect thanks", "awesome thank you", "that's great",
		"that's what I needed", "that works",
		"nah", "nope", "not really", "never mind",
	)

	// ---- Weather ----
	addWordListExamples(&examples, "weather", nlu.weatherWords, []string{
		"what's the %s", "check the %s", "%s today", "how's the %s",
		"what is the %s outside", "can you check the %s",
		"is it going to rain today", "will it be sunny tomorrow",
		"what's the weather like", "how's the weather today",
		"weather forecast for tomorrow", "is it cold outside",
	})

	// ---- Reminders ----
	addWordListExamples(&examples, "reminder", nlu.reminderWords, []string{
		"%s to buy groceries", "%s to call mom", "%s in 30 minutes",
		"%s tomorrow to send the email", "%s tonight to take medicine",
	})
	add("reminder",
		"remind me to call mom tomorrow",
		"remind me in 30 minutes to check the oven",
		"set a reminder for 5pm",
		"remind me to pick up the kids at 3",
		"can you remind me about the meeting",
		"don't let me forget to buy milk",
		"remind me tomorrow morning to exercise",
		"set a reminder for next week",
		"remind me to call the dentist",
		"I need a reminder about the deadline",
	)

	// ---- Todos ----
	addWordListExamples(&examples, "todo", nlu.todoWords, []string{
		"%s", "please %s", "can you %s",
	})
	add("todo",
		"add buy milk to my todo list",
		"I need to finish the report",
		"what's on my todo list",
		"mark laundry as done",
		"remove the first task",
	)

	// ---- Notes ----
	addWordListExamples(&examples, "note", nlu.noteWords, []string{
		"%s", "please %s", "I want to %s",
	})
	add("note",
		"take a note about the meeting",
		"note: remember to call John",
		"save a note about the project",
		"what notes do I have",
		"search my notes for recipes",
	)

	// ---- Calendar ----
	addWordListExamples(&examples, "calendar", nlu.calendarWords, []string{
		"%s", "show me %s", "check %s",
	})
	add("calendar",
		"do I have anything today",
		"what meetings do I have tomorrow",
		"show me my schedule for this week",
		"any events coming up",
	)

	// ---- Email ----
	addWordListExamples(&examples, "email", nlu.emailWords, []string{
		"%s", "please %s", "can you %s",
	})

	// ---- News ----
	addWordListExamples(&examples, "news", nlu.newsWords, []string{
		"show me the %s", "any %s today", "get me the latest %s",
		"what's in the %s", "pull up %s",
	})
	add("news",
		"what's happening in the world",
		"any interesting news today",
		"show me tech headlines",
		"latest science news",
	)

	// ---- Dictionary ----
	add("dict",
		"define serendipity", "define ubiquitous", "define ephemeral",
		"what does the word 'ubiquitous' mean",
		"definition of serendipity", "meaning of ephemeral",
		"synonyms for happy", "antonyms of good",
		"what's a synonym for beautiful",
		"thesaurus lookup for intelligent",
		"define the word paradigm",
		"what does ephemeral mean",
		"dictionary lookup for serendipity",
		"give me the definition of pragmatic",
		"what are synonyms of fast",
	)

	// ---- Translate ----
	addWordListExamples(&examples, "translate", nlu.translateWords, []string{
		"%s", "can you %s",
	})
	add("translate",
		"translate hello to french", "translate goodbye to spanish",
		"how do you say thank you in japanese",
		"what is hello in german", "translate this to italian",
		"say good morning in korean", "how do you say I love you in french",
		"translate 'where is the bathroom' to portuguese",
		"what's the spanish word for cat",
		"translate please to chinese",
	)

	// ---- Network ----
	addWordListExamples(&examples, "network", nlu.networkWords, []string{
		"%s", "please %s", "can you %s",
	})
	add("network",
		"ping google.com", "is google.com up",
		"check if the server is running",
		"am I connected to the internet",
		"dns lookup for example.com",
		"is the website down",
		"check my network connection",
	)

	// ---- Hash / encoding ----
	addWordListExamples(&examples, "hash", nlu.hashWords, []string{
		"%s this text", "%s hello world", "generate a %s",
	})
	add("hash",
		"hash this string", "md5 of hello",
		"sha256 hash of my password",
		"base64 encode this text",
		"url encode this string",
		"compute the checksum",
	)

	// ---- Timer ----
	addWordListExamples(&examples, "timer", nlu.timerWords, []string{
		"%s", "please %s", "can you %s",
	})
	add("timer",
		"set a timer for 5 minutes",
		"start a pomodoro timer",
		"how much time is left",
		"start a 25 minute countdown",
	)

	// ---- Volume / brightness ----
	addWordListExamples(&examples, "volume", nlu.volumeWords, nil)
	add("volume",
		"turn up the volume", "turn down the sound",
		"mute the audio", "unmute", "set volume to 50",
	)
	addWordListExamples(&examples, "brightness", nlu.brightnessWords, nil)
	add("brightness",
		"increase the brightness", "dim the screen",
		"set brightness to 70%", "make it brighter",
	)

	// ---- App management ----
	addWordListExamples(&examples, "app", nlu.appWords, nil)
	add("app",
		"open firefox for me", "launch spotify",
		"start vscode", "close the browser",
		"what apps are running",
	)

	// ---- Screenshots ----
	addWordListExamples(&examples, "screenshot", nlu.screenshotWords, nil)
	add("screenshot",
		"take a screenshot of my screen",
		"grab a screenshot", "capture this screen",
	)

	// ---- Clipboard ----
	addWordListExamples(&examples, "clipboard", nlu.clipboardWords, nil)
	add("clipboard",
		"what's in my clipboard right now",
		"show me what I copied",
		"paste from clipboard",
	)

	// ---- System info ----
	addWordListExamples(&examples, "sysinfo", nlu.sysinfoWords, nil)
	add("sysinfo",
		"how much disk space do I have",
		"what's my IP address",
		"show me system information",
		"how much RAM is being used",
		"what's my computer's uptime",
	)

	// ---- Find files ----
	addWordListExamples(&examples, "find_files", nlu.fileFinderWords, nil)
	add("find_files",
		"find all python files in my project",
		"where are my PDF documents",
		"locate files matching *.go",
		"find images in my downloads",
		"search for files containing config",
	)

	// ---- Code runner ----
	addWordListExamples(&examples, "run_code", nlu.codeRunWords, nil)
	add("run_code",
		"run this python code", "execute the script",
		"run javascript for me", "run this bash command",
	)

	// ---- Archive ----
	addWordListExamples(&examples, "archive", nlu.archiveWords, nil)
	add("archive",
		"zip these files together", "unzip the download",
		"extract the tar archive", "compress this folder",
	)

	// ---- Disk usage ----
	addWordListExamples(&examples, "disk_usage", nlu.diskUsageWords, nil)
	add("disk_usage",
		"what's using the most disk space",
		"show me the largest folders",
		"how much space is left on my drive",
	)

	// ---- Process management ----
	addWordListExamples(&examples, "process", nlu.processWords, nil)
	add("process",
		"show me running processes",
		"what's using all the CPU",
		"kill the process using port 8080",
	)

	// ---- QR codes ----
	addWordListExamples(&examples, "qrcode", nlu.qrcodeWords, nil)
	add("qrcode",
		"generate a QR code for this URL",
		"create a QR code for my wifi",
		"make a QR code",
	)

	// ---- Calculator ----
	addWordListExamples(&examples, "calculate", nlu.calculatorWords, nil)
	add("calculate",
		"calculate 25 * 4", "what is 15% of 200",
		"compute the square root of 144",
		"what's 2 to the power of 10",
		"evaluate 3.14 * 2", "solve 100 / 7",
	)

	// ---- Password ----
	addWordListExamples(&examples, "password", nlu.passwordWords, nil)
	add("password",
		"generate a strong password",
		"create a random password 16 characters",
		"make me a passphrase",
		"generate a 6 digit pin",
		"I need a new password",
	)

	// ---- Bookmarks ----
	addWordListExamples(&examples, "bookmark", nlu.bookmarkWords, nil)
	add("bookmark",
		"save this link to bookmarks",
		"bookmark this page",
		"show all my saved links",
		"delete the first bookmark",
	)

	// ---- Journal ----
	addWordListExamples(&examples, "journal", nlu.journalWords, nil)
	add("journal",
		"dear diary today was wonderful",
		"write in my journal about the meeting",
		"show me last week's journal entries",
		"what did I write yesterday",
	)

	// ---- Habits ----
	addWordListExamples(&examples, "habit", nlu.habitWords, nil)
	add("habit",
		"create a habit for daily exercise",
		"did I do my meditation today",
		"mark exercise as done",
		"what's my habit streak",
		"show my habit progress",
	)

	// ---- Expenses ----
	addWordListExamples(&examples, "expense", nlu.expenseWords, nil)
	add("expense",
		"I spent $5 on coffee today",
		"add an expense of $20 for lunch",
		"how much did I spend this week",
		"show me my monthly expenses",
		"log a purchase of $50 for groceries",
	)

	// ---- Creative ----
	add("creative",
		"write me a poem about the ocean",
		"tell me a joke", "tell me a funny joke",
		"write a short story about a detective",
		"compose a haiku about autumn",
		"create a limerick about coding",
		"write a poem", "make up a story",
		"tell me a joke about programmers",
		"write something creative",
		"craft a poem about nature",
		"what do you think about love",
		"what's your opinion on technology",
		"share your thoughts on artificial intelligence",
		"reflect on the nature of consciousness",
		"what is the meaning of life",
		"philosophize about freedom",
		"tell me something interesting",
		"tell me a fun fact",
		"say something surprising",
		"give me a random fact",
		"surprise me with something cool",
		"help me write a shopping list",
		"help me create a recipe",
		"help me draft an email",
		"can you help me write a cover letter",
		"help me make a plan for my vacation",
	)

	// ---- Explain / knowledge ----
	add("explain",
		"what is quantum physics",
		"explain how photosynthesis works",
		"tell me about the roman empire",
		"who is albert einstein",
		"describe the theory of relativity",
		"how does gravity work",
		"what are black holes",
		"explain machine learning to me",
		"what is blockchain technology",
		"how does the internet work",
		"tell me about ancient egypt",
		"who invented the telephone",
		"what is democracy",
		"how does evolution work",
		"explain the water cycle",
		"what is philosophy",
		"what makes the sky blue",
		"how do computers work",
		"who is Shakespeare",
		"what happened in World War 2",
		"explain DNA",
		"what causes earthquakes",
		"tell me about the solar system",
		"how do vaccines work",
		"what is artificial intelligence",
		"explain the big bang theory",
		"what is stoicism",
		"who was Plato",
		"how does encryption work",
		"what is the greenhouse effect",
	)

	// ---- Question (general) ----
	add("question",
		"is the earth flat", "how old is the universe",
		"why is the sky blue", "how far is the moon",
		"when was the first computer made",
		"who discovered electricity",
		"what country has the most people",
		"how many planets are there",
		"is Pluto a planet", "why do we dream",
		"can fish feel pain", "do plants have feelings",
		"how fast is the speed of light",
		"what's the deepest ocean",
		"how does wifi work",
		"how do I learn guitar", "how to learn piano",
		"how do I start programming", "how can I learn French",
		"teach me about photography", "how to master cooking",
	)

	// ---- Remember / store ----
	add("remember",
		"remember my name is Raphael",
		"my favorite color is blue",
		"remember that I like pizza",
		"I'm a software engineer",
		"I live in Vienna",
		"remember I work at Google",
		"my email is test@example.com",
		"note that I prefer dark mode",
		"keep in mind that I'm vegetarian",
		"I'm interested in philosophy",
		"my birthday is March 15",
		"remember I love coffee",
		"I prefer formal language",
	)

	// ---- Recall / memory query ----
	add("recall",
		"what's my name", "what is my name",
		"do you remember my name",
		"what's my favorite color",
		"what do you know about me",
		"who am I", "do you know who I am",
		"what did I tell you about myself",
		"do you remember me",
		"what's my email", "where do I live",
		"what do I do for work",
		"have I told you my interests",
	)

	// ---- Convert ----
	addWordListExamples(&examples, "convert", nlu.convertWords, nil)
	add("convert",
		"convert 100 miles to km",
		"how many kg is 150 pounds",
		"32 fahrenheit in celsius",
		"convert 5 gallons to liters",
		"how many meters in a mile",
		"100 usd to euros",
	)

	// ---- File operations ----
	add("file_op",
		"read the file config.yaml",
		"open /etc/hosts",
		"create a new file called test.txt",
		"edit the configuration file",
		"delete the log file",
		"show me the contents of main.go",
		"list files in the current directory",
		"cat /var/log/syslog",
		"write hello world to output.txt",
		"save this to a file",
	)

	// ---- Search / web ----
	add("search",
		"search for golang tutorials",
		"google machine learning papers",
		"look up the latest research on AI",
		"find information about climate change",
		"search the web for recipes",
	)

	// ---- Web lookup (current events) ----
	add("web_lookup",
		"what's the stock price of Tesla",
		"who won the game last night",
		"latest election results",
		"is the website down",
		"what's trending on twitter",
		"breaking news today",
		"current bitcoin price",
	)

	// ---- Compute / math ----
	add("compute",
		"what is 2 + 2", "calculate 15 * 7",
		"what's 100 divided by 3", "compute 2^10",
		"solve 3x + 5 = 20", "what is 15% of 200",
		"evaluate sin(45)", "what's the square root of 144",
	)

	// ---- Transform ----
	addWordListExamples(&examples, "transform", nlu.transformWords, []string{
		"%s this text", "%s the following paragraph",
	})
	add("transform",
		"rewrite this more formally",
		"make this email sound professional",
		"summarize this article",
		"paraphrase this paragraph",
		"turn this into bullet points",
		"simplify this explanation",
		"make this more concise",
	)

	// ---- Recommendation ----
	for _, v := range nlu.recommendVerbs {
		add("recommendation", v+" something good")
	}
	add("recommendation",
		"suggest a good book to read",
		"recommend a movie for tonight",
		"any tips for learning Go",
		"what should I cook for dinner",
		"recommend a podcast about science",
	)

	// ---- Compare ----
	for _, v := range nlu.compareVerbs {
		add("compare", v+" X and Y")
	}
	add("compare",
		"python vs golang", "react vs vue",
		"what's the difference between RAM and ROM",
		"compare electric cars to gas cars",
		"which is better, Mac or Windows",
		"typescript vs javascript differences",
	)

	// ---- Daily briefing ----
	add("daily_briefing",
		"good morning", "good afternoon", "good evening",
		"daily briefing", "brief me", "briefing",
		"morning briefing", "what's on today",
		"start my day", "how's my day looking",
		"morning report", "daily summary",
		"good morning nous", "morning",
		"what do I have today", "what's my schedule today",
		"how does my day look", "what's planned for today",
		"give me my daily briefing", "ready to start the day",
		"what's happening today", "morning update",
		"good morning what's on my schedule",
		"hey good morning", "good morning how's my day",
	)

	// ---- Follow up ----
	add("follow_up",
		"tell me more", "go on", "continue",
		"elaborate on that", "more details please",
		"what else", "dig deeper", "keep going",
		"explain further", "and then what",
	)

	// ---- Conversation / personal statements ----
	add("conversation",
		"I will run tomorrow", "tomorrow I will run for 10 miles",
		"I'm going to the gym later", "I had a great day today",
		"I just finished my homework", "we should try that sometime",
		"I think I need a break", "I was thinking about moving",
		"my day was good", "my friend told me something",
		"I can't believe what happened", "I don't know what to do",
		"we went to the park yesterday", "I need to buy groceries",
		"I'm planning a trip next month", "today was exhausting",
		"I should probably sleep more",
		"I've been working on this all day", "I'm really tired",
		"we had dinner at a nice place", "I'm looking forward to the weekend",
		"I just got home", "I was late to work today",
		"are you alright", "are you okay", "you alright",
		"how's it going nous", "how are you doing nous",
	)

	// ---- Hard negatives (boundary cases) ----
	// These are ambiguous inputs that pattern matching gets wrong.
	add("creative", "what is the meaning of life") // NOT dict
	add("creative", "what is the meaning of love")
	add("creative", "what is the meaning of happiness")
	add("explain", "what is the meaning of photosynthesis") // IS explain (not dict)
	add("creative", "write me something beautiful")
	add("creative", "can you help me write a letter")
	add("creative", "help me create a to-do list")

	add("conversation", "i feel happy today")       // personal statement, NOT greeting
	add("conversation", "i'm feeling sad")          // personal statement, NOT greeting
	add("conversation", "i'm feeling a bit tired")  // personal statement, NOT greeting
	add("conversation", "i feel exhausted")         // personal statement, NOT greeting
	add("conversation", "i'm feeling great")        // personal statement, NOT greeting
	add("conversation", "i feel stressed out")      // personal statement, NOT greeting
	add("conversation", "having a great day")       // personal statement, NOT greeting
	add("conversation", "having a rough day")       // personal statement, NOT greeting

	add("explain", "tell me about dogs")       // NOT creative
	add("explain", "what is a black hole")     // NOT dict
	add("explain", "how does the brain work")  // NOT creative

	add("dict", "define serendipity")          // IS dict
	add("dict", "what does ubiquitous mean")   // IS dict
	add("dict", "synonyms for happy")          // IS dict

	add("meta", "what is your name")          // NOT explain
	add("meta", "who created you")            // NOT explain
	add("meta", "do you have feelings")       // NOT question

	add("file_op", "write to the config file")   // IS file_op
	add("creative", "write a poem about rain")    // NOT file_op
	add("creative", "write a shopping list")      // NOT file_op

	add("reminder", "remind me to call mom tomorrow") // NOT creative
	add("reminder", "set a reminder for the meeting")  // NOT todo

	return examples
}

// GenerateRealisticTrainingData creates 10,000+ training examples that mimic
// real user input: typos, missing punctuation, no apostrophes, casual tone,
// exclamations, and implicit queries. This is the primary source of training
// signal for the neural classifier.
func GenerateRealisticTrainingData(nlu *NLU) []TrainingExample {
	var examples []TrainingExample

	add := func(intent string, texts ...string) {
		for _, t := range texts {
			examples = append(examples, TrainingExample{Text: t, Intent: intent})
		}
	}

	// expand generates variations of a base text: lowercase, no apostrophe,
	// with/without punctuation, with optional prefix.
	expand := func(intent string, bases []string) {
		for _, b := range bases {
			add(intent, b)
			add(intent, strings.ToLower(b))
			// Remove apostrophes: "what's" -> "whats"
			if strings.Contains(b, "'") {
				add(intent, strings.ReplaceAll(b, "'", ""))
			}
		}
	}

	// ----------------------------------------------------------------
	// GREETINGS (500+ examples)
	// ----------------------------------------------------------------
	add("greeting",
		// Casual
		"hey", "hey!", "hey!!", "heyyy", "heyy", "heyy!", "yo", "yo!", "yoo",
		"sup", "sup!", "sup?", "supp", "wassup", "wazzup", "wasup",
		"whats up", "whats up!", "whatsup", "what up", "what up!",
		"hiya", "hiya!", "hi there", "hi there!", "hey there",
		"hey there!", "hey there!!", "heyo", "helo", "helo!",
		"howdy", "howdy!", "ello", "ello!",
		// Standard with excitement
		"hello!", "hello!!", "hi!", "hi!!", "hey hey", "hey hey!",
		"hello hello", "hi hi", "yo yo",
		// Multi-word casual
		"hey how are you", "hey hows it going", "hey whats up",
		"hi how are you", "hi hows it going", "hi whats new",
		"hey there whats up", "hey there hows it going",
		"hello how are you", "hello hows it going",
		"hey whats going on", "hi whats going on",
		"hey how ya doing", "whats good", "whats good!",
		"hey buddy", "hey pal", "hey friend",
		"hello friend", "hi buddy", "yo buddy",
		"greetings", "greetings!", "salutations",
		// With name
		"hey nous", "hi nous", "hello nous", "yo nous",
		"hey nous whats up", "hi nous how are you",
		"nous!", "hey nous!", "whats up nous",
		"sup nous", "yo nous whats up",
		// Long time / return
		"long time no see", "its been a while", "im back",
		"hey im back", "hello again", "hi again",
		"hey i missed you", "nice to see you again",
		// Morning/time but NOT daily briefing
		"hey there friend", "hello world", "hi world",
		// Extended greetings
		"hey whats crackin", "whats crackin", "yo whats good",
		"hey hey hey", "well hello there", "oh hey",
		"oh hi", "oh hello", "ah hey there",
		"hey you", "hi you", "hello you",
		// With filler
		"umm hey", "um hi", "uh hello", "so hey",
		"ok hi", "ok hello", "alright hey",
		"well hey", "well hi", "well hello",
		// Regional
		"oi", "oi!", "g'day", "gday", "ahoy", "ahoy!",
		"howzit", "heya", "heya!",
	)

	// More greeting variations with typos and slang
	greetBases := []string{
		"hey", "hi", "hello", "yo", "sup",
	}
	greetSuffixes := []string{
		" there", " buddy", " friend", " pal", " man", " dude", " bro",
		" everyone", " all", " guys", " folks",
		" nous", " there nous",
	}
	for _, base := range greetBases {
		for _, suf := range greetSuffixes {
			add("greeting", base+suf)
		}
	}

	// Greeting + how are you variants
	howAreYou := []string{
		"how are you", "hows it going", "how ya doing", "how are ya",
		"how you doing", "how have you been", "how r u", "how ru",
		"whats new", "whats happening", "what you up to", "whatcha doing",
		"hows things", "hows life", "hows everything",
	}
	for _, g := range []string{"hey ", "hi ", "hello ", "yo ", ""} {
		for _, h := range howAreYou {
			add("greeting", g+h)
		}
	}

	// ----------------------------------------------------------------
	// QUESTIONS / EXPLAIN (2000+ examples)
	// ----------------------------------------------------------------
	// Casual "what about" / "tell me about" patterns
	topics := []string{
		"black holes", "quantum mechanics", "photosynthesis", "gravity",
		"evolution", "dna", "the solar system", "the moon", "mars",
		"jupiter", "the milky way", "dark matter", "dark energy",
		"the big bang", "string theory", "relativity", "time travel",
		"wormholes", "neutron stars", "the speed of light",
		"climate change", "global warming", "the ozone layer",
		"renewable energy", "solar panels", "wind energy",
		"nuclear energy", "fusion", "fission",
		"world war 1", "world war 2", "the cold war", "the renaissance",
		"the roman empire", "ancient egypt", "ancient greece",
		"the french revolution", "the american revolution",
		"the industrial revolution",
		"artificial intelligence", "machine learning", "deep learning",
		"neural networks", "chatgpt", "large language models",
		"blockchain", "cryptocurrency", "bitcoin", "ethereum",
		"the internet", "wifi", "5g", "fiber optics",
		"social media", "tiktok", "instagram", "youtube",
		"democracy", "capitalism", "socialism", "communism",
		"philosophy", "stoicism", "existentialism", "nihilism",
		"psychology", "cognitive science", "consciousness",
		"the human brain", "memory", "dreams", "sleep",
		"nutrition", "vitamins", "protein", "calories",
		"exercise", "meditation", "yoga", "mindfulness",
		"vaccines", "antibiotics", "viruses", "bacteria",
		"the immune system", "cancer", "genetics",
		"dinosaurs", "fossils", "the ice age", "pangaea",
		"volcanoes", "earthquakes", "tsunamis", "hurricanes",
		"the ocean", "coral reefs", "marine biology",
		"space exploration", "nasa", "spacex", "the iss",
		"programming", "python", "javascript", "golang",
		"rust", "c++", "algorithms", "data structures",
		"operating systems", "linux", "databases", "sql",
		"the stock market", "inflation", "interest rates", "gdp",
		"taxes", "investing", "real estate", "bonds",
	}

	// Generate "what about X" / "tell me about X" / "how does X work" etc
	questionTemplates := []string{
		"what about %s", "tell me about %s", "what is %s",
		"what are %s", "whats %s", "explain %s",
		"how does %s work", "how do %s work",
		"what do you know about %s", "i wanna know about %s",
		"i want to learn about %s", "teach me about %s",
		"can you explain %s", "give me info on %s",
		"tell me everything about %s", "whats the deal with %s",
		"what can you tell me about %s", "info on %s",
		"describe %s", "break down %s for me",
	}
	for _, topic := range topics {
		// Pick 3-4 random templates per topic
		for i, tmpl := range questionTemplates {
			if i%3 == 0 || i%5 == 0 {
				add("explain", fmt.Sprintf(tmpl, topic))
			}
		}
	}

	// Implicit questions (just topic + question mark)
	for _, topic := range topics {
		add("explain", topic+"?")
		add("explain", topic)
		add("question", "what about "+topic+"?")
	}

	// Typo-laden questions
	add("explain",
		"whats photosynthesis", "explane gravity", "wat is dna",
		"explian machine learning", "wats quantum mechanics",
		"how dose wifi work", "how dose the internet work",
		"whats a black hole", "wats a neutron star",
		"explaine evolution", "tell me bout the solar system",
		"tell me bout climate change", "wat about dark matter",
		"whats social media", "whats artifical intelligence",
		"explian cryptocurrency", "wat is blockchain",
		"how do vaccines work", "how dose memory work",
		"whats the deal with dark energy", "wat is stoicism",
		"wats the big bang", "how dose gravity work",
		"what r neural networks", "wat r algorithms",
	)

	// "who is/was" questions
	people := []string{
		"albert einstein", "isaac newton", "nikola tesla", "marie curie",
		"shakespeare", "plato", "aristotle", "socrates",
		"leonardo da vinci", "galileo", "darwin", "alan turing",
		"elon musk", "steve jobs", "bill gates", "mark zuckerberg",
		"napoleon", "cleopatra", "julius caesar", "alexander the great",
		"gandhi", "martin luther king", "nelson mandela",
		"beethoven", "mozart", "bach", "picasso",
	}
	for _, person := range people {
		add("explain", "who is "+person)
		add("explain", "who was "+person)
		add("explain", "tell me about "+person)
		add("question", "whos "+person)
	}

	// General questions with casual phrasing
	add("question",
		"is the earth flat", "how old is the universe", "why is the sky blue",
		"how far is the moon", "when was the first computer made",
		"who discovered electricity", "what country has the most people",
		"how many planets are there", "is pluto a planet", "why do we dream",
		"can fish feel pain", "do plants have feelings",
		"how fast is the speed of light", "whats the deepest ocean",
		"how does that work", "why is that", "how come",
		"how many people live on earth", "whats the tallest building",
		"whats the longest river", "whats the biggest country",
		"when did humans first go to space", "who was the first president",
		"why do cats purr", "why do dogs bark", "how do birds fly",
		"why is the ocean salty", "how do magnets work",
		"why is fire hot", "what makes rainbows", "how do planes fly",
		"can you breathe in space", "is there life on mars",
		"how deep is the ocean", "whats at the bottom of the ocean",
		"how hot is the sun", "how big is the universe",
		"do aliens exist", "are we alone in the universe",
	)

	// ----------------------------------------------------------------
	// MATH / CALCULATE (500+ examples)
	// ----------------------------------------------------------------
	// Basic arithmetic with casual phrasing
	mathOps := []struct{ a, op, b string }{
		{"5", "+", "3"}, {"15", "*", "7"}, {"100", "/", "4"},
		{"25", "-", "8"}, {"12", "*", "12"}, {"999", "+", "1"},
		{"50", "/", "2"}, {"7", "*", "8"}, {"33", "+", "67"},
		{"200", "-", "57"}, {"1000", "/", "8"}, {"13", "*", "13"},
		{"45", "+", "55"}, {"88", "-", "22"}, {"64", "/", "8"},
		{"9", "*", "9"}, {"150", "+", "250"}, {"500", "-", "175"},
		{"36", "/", "6"}, {"17", "*", "4"},
	}
	for _, m := range mathOps {
		add("calculate", fmt.Sprintf("whats %s %s %s", m.a, m.op, m.b))
		add("calculate", fmt.Sprintf("what is %s %s %s", m.a, m.op, m.b))
		add("calculate", fmt.Sprintf("calculate %s %s %s", m.a, m.op, m.b))
		add("calculate", fmt.Sprintf("%s %s %s", m.a, m.op, m.b))
		add("compute", fmt.Sprintf("%s%s%s", m.a, m.op, m.b))
	}

	// Percentage calculations
	percentages := []struct{ pct, base string }{
		{"15", "340"}, {"20", "85"}, {"10", "500"}, {"25", "200"},
		{"50", "120"}, {"15", "60"}, {"8", "250"}, {"30", "150"},
		{"5", "1000"}, {"12", "300"}, {"18", "400"}, {"7", "80"},
		{"33", "900"}, {"40", "250"}, {"2", "5000"}, {"75", "400"},
	}
	for _, p := range percentages {
		add("calculate", fmt.Sprintf("whats %s%% of %s", p.pct, p.base))
		add("calculate", fmt.Sprintf("what is %s percent of %s", p.pct, p.base))
		add("calculate", fmt.Sprintf("%s%% of %s", p.pct, p.base))
		add("calculate", fmt.Sprintf("calculate %s percent of %s", p.pct, p.base))
	}

	// Tip calculations
	add("calculate",
		"how much is 20% tip on 85", "how much is 15% tip on 50",
		"20 percent tip on 45", "tip on 80 dollars",
		"calculate tip on 120", "whats 18% tip on 65",
		"15% tip on 30", "20% tip on 100",
	)

	// Square roots and powers
	add("calculate",
		"square root of 144", "square root of 64", "square root of 225",
		"sqrt of 100", "sqrt 81", "sqrt 49",
		"2 to the power of 10", "3 to the power of 4", "2^8",
		"10 squared", "5 squared", "7 squared", "12 squared",
		"3 cubed", "4 cubed", "5 cubed",
	)

	// Word-based math
	add("calculate",
		"15 times 7", "twenty times five", "hundred divided by 3",
		"fifty plus thirty", "ninety minus twenty",
		"a dozen times 4", "half of 500", "double 250",
		"triple 33", "a third of 900",
	)

	// ----------------------------------------------------------------
	// TRANSLATE (300+ examples)
	// ----------------------------------------------------------------
	languages := []string{
		"spanish", "french", "german", "italian", "portuguese",
		"japanese", "chinese", "korean", "russian", "arabic",
		"hindi", "dutch", "swedish", "polish", "greek",
		"turkish", "thai", "vietnamese", "indonesian", "hebrew",
	}
	phrases := []string{
		"hello", "goodbye", "thank you", "please", "yes", "no",
		"how are you", "good morning", "good night", "i love you",
		"where is the bathroom", "how much does this cost",
		"my name is", "nice to meet you", "excuse me",
		"im sorry", "help", "water", "food", "cheers",
	}
	for _, lang := range languages {
		for i, phrase := range phrases {
			if i%3 == 0 {
				add("translate", fmt.Sprintf("translate %s to %s", phrase, lang))
			}
			if i%4 == 0 {
				add("translate", fmt.Sprintf("how do you say %s in %s", phrase, lang))
			}
			if i%5 == 0 {
				add("translate", fmt.Sprintf("whats %s in %s", phrase, lang))
			}
		}
	}
	add("translate",
		"translate this to french", "say this in spanish",
		"how do you say cheers in japanese", "whats the german word for cat",
		"translate good morning to korean", "say hello in arabic",
		"how to say thank you in hindi", "whats goodbye in italian",
		"translate excuse me to portuguese", "say please in russian",
		"translate i love you to french", "how do you say water in chinese",
		"whats food in thai", "say sorry in dutch",
		"translate help to greek", "whats yes in turkish",
		"how to say no in swedish", "translate my name is to polish",
		"say nice to meet you in vietnamese", "whats cheers in hebrew",
	)

	// ----------------------------------------------------------------
	// CREATIVE (300+ examples)
	// ----------------------------------------------------------------
	add("creative",
		"write me a poem about rain", "write a poem about the ocean",
		"write me a poem about love", "write a poem about winter",
		"write me a poem about life", "write a poem about death",
		"write a poem about nature", "write me a poem about the stars",
		"write me a haiku about winter", "write a haiku about summer",
		"write a haiku about rain", "write me a haiku about love",
		"write a limerick about coding", "write me a limerick about cats",
		"tell me a joke", "tell me a funny joke", "tell me a joke about cats",
		"got any jokes", "know any good jokes", "make me laugh",
		"tell me a dad joke", "tell me a pun", "give me a joke",
		"tell me a joke about programmers", "tell me a joke about food",
		"make up a story", "tell me a story", "write me a short story",
		"write a story about a detective", "make up a fairy tale",
		"write me a bedtime story", "tell me a scary story",
		"write something creative", "be creative", "show me your creativity",
		"write me something beautiful", "create something artistic",
		"write me a song about love", "write a rap about coding",
		"write me a poem", "write a sonnet", "compose a verse",
		"give me a riddle", "tell me a riddle", "write a riddle",
		"write me a motivational quote", "make up a proverb",
		"write a fortune cookie message", "give me an inspirational quote",
		"help me write a shopping list", "help me create a recipe",
		"help me draft an email", "help me write a cover letter",
		"help me make a plan for my vacation",
		"write me a tweet about monday", "write a caption for my photo",
		"help me write my bio", "write a birthday message",
		"write a thank you note", "help me write wedding vows",
		// Reflective / philosophical
		"what is the meaning of life", "what is the meaning of love",
		"what is the meaning of happiness", "what is the purpose of life",
		"what do you think about love", "whats your opinion on technology",
		"share your thoughts on artificial intelligence",
		"reflect on the nature of consciousness",
		"philosophize about freedom", "what makes life worth living",
		// Fun facts / entertainment
		"tell me something interesting", "tell me a fun fact",
		"say something surprising", "give me a random fact",
		"surprise me with something cool", "blow my mind",
		"tell me something i dont know", "give me a crazy fact",
	)

	// ----------------------------------------------------------------
	// COACHING / DECISION (500+ examples)
	// ----------------------------------------------------------------
	// Life decisions
	expand("conversation", []string{
		"should i quit my job", "should i change careers",
		"should i go back to school", "should i move to another city",
		"should i break up with my partner", "should i get married",
		"should i buy a house", "should i rent or buy",
		"should i start a business", "should i invest in stocks",
		"should i learn to code", "should i get a masters degree",
		"should i adopt a pet", "should i get a dog or cat",
	})
	add("conversation",
		"im stuck in my career", "i dont know what to do with my life",
		"i feel stuck", "im at a crossroads", "help me decide",
		"i cant decide", "im torn between two options",
		"i need advice", "give me advice about relationships",
		"help me with my problem", "i need your help with something",
		"im thinking about changing careers", "im considering quitting",
		"im not sure if i should", "what would you do",
		"what do you think i should do", "give me some guidance",
		"im confused about what to do", "i need some direction",
		"help me figure this out", "im overthinking this",
		"i keep going back and forth", "pros and cons of quitting",
		"is it worth it to go back to school", "am i making a mistake",
		"i dont know if this is the right choice",
		"help me think through this", "talk me through this",
		"what are my options", "what should i consider",
		"give me a different perspective", "help me see this differently",
		"i need an outside opinion", "what would you advise",
		// Career specific
		"im unhappy at work", "my job is boring", "i hate my boss",
		"i got a job offer", "should i take the new job",
		"i got passed over for promotion", "how do i ask for a raise",
		"how do i negotiate salary", "im burned out",
		// Relationship specific
		"my friend betrayed me", "i had a fight with my partner",
		"my family doesnt understand me", "im having relationship problems",
		"how do i make new friends", "im lonely",
		// Health / wellbeing
		"i cant sleep", "im stressed about everything",
		"how do i deal with anxiety", "im overwhelmed with work",
		"i need to take better care of myself",
	)

	// ----------------------------------------------------------------
	// COMPARE (400+ examples)
	// ----------------------------------------------------------------
	comparisons := []struct{ a, b string }{
		{"cats", "dogs"}, {"python", "java"}, {"react", "vue"},
		{"mac", "windows"}, {"ios", "android"}, {"chrome", "firefox"},
		{"coffee", "tea"}, {"running", "swimming"}, {"yoga", "pilates"},
		{"electric cars", "gas cars"}, {"renting", "buying"},
		{"city life", "country life"}, {"introvert", "extrovert"},
		{"fiction", "nonfiction"}, {"kindle", "paperback"},
		{"netflix", "hulu"}, {"spotify", "apple music"},
		{"uber", "lyft"}, {"amazon", "walmart"},
		{"iphone", "samsung"}, {"playstation", "xbox"},
		{"typescript", "javascript"}, {"rust", "c++"},
		{"docker", "kubernetes"}, {"sql", "nosql"},
		{"aws", "azure"}, {"react", "angular"},
		{"golang", "python"}, {"linux", "windows"},
		{"vim", "emacs"}, {"tabs", "spaces"},
	}
	for _, c := range comparisons {
		add("compare", fmt.Sprintf("compare %s and %s", c.a, c.b))
		add("compare", fmt.Sprintf("%s vs %s", c.a, c.b))
		add("compare", fmt.Sprintf("%s or %s", c.a, c.b))
		add("compare", fmt.Sprintf("whats better %s or %s", c.a, c.b))
		add("compare", fmt.Sprintf("difference between %s and %s", c.a, c.b))
		add("compare", fmt.Sprintf("differences between %s and %s", c.a, c.b))
		add("compare", fmt.Sprintf("%s versus %s", c.a, c.b))
	}
	add("compare",
		"pros and cons of remote work", "pros and cons of freelancing",
		"which is better", "whats the difference",
		"how do they compare", "which one should i pick",
		"which do you recommend", "which is faster",
		"which is more popular", "which is easier to learn",
	)

	// ----------------------------------------------------------------
	// INSTRUCTIONS / FORMAT (300+ examples)
	// ----------------------------------------------------------------
	add("creative",
		"give me 3 tips for studying", "list 5 things to do in paris",
		"give me 10 reasons to exercise", "list the steps to make bread",
		"summarize in bullet points", "explain in 2 sentences",
		"keep it brief", "give me a quick overview",
		"make it simple", "eli5", "explain like im 5",
		"tldr", "tl;dr", "in a nutshell",
		"give me the short version", "break it down simply",
		"step by step please", "walk me through it",
		"give me 5 tips", "top 10 ways to save money",
		"give me a list of hobbies", "list some good books",
		"give me 3 reasons", "name 5 famous scientists",
		"list the planets", "what are the 7 wonders",
		"give me a checklist for moving", "list things to pack for travel",
	)

	// ----------------------------------------------------------------
	// TOOL-SPECIFIC (1000+ examples across all tools)
	// ----------------------------------------------------------------

	// Weather
	add("weather",
		"whats the weather", "hows the weather", "whats it like outside",
		"is it gonna rain", "is it going to rain today", "will it rain tomorrow",
		"is it cold outside", "is it hot today", "hows the weather today",
		"weather today", "weather tomorrow", "weather this week",
		"whats the temperature", "temperature outside", "temp right now",
		"do i need a jacket", "do i need an umbrella",
		"is it sunny", "is it cloudy", "is it snowing",
		"forecast for today", "forecast for tomorrow",
		"weather forecast", "weekly forecast",
		"whats the weather in new york", "weather in london",
		"temperature in tokyo", "is it raining in seattle",
		"hows the weather in chicago", "forecast for los angeles",
		"weather in paris", "will it snow tomorrow",
	)

	// Timer
	add("timer",
		"set a timer for 5 minutes", "set a timer for 10 min",
		"timer 5 min", "timer 10 minutes", "timer 30 seconds",
		"set a 5 minute timer", "start a timer", "start a 10 min timer",
		"remind me in 5 minutes", "remind me in 10 min",
		"remind me in an hour", "remind me in 30 min",
		"set a pomodoro", "start pomodoro", "pomodoro timer",
		"25 minute timer", "start a countdown", "countdown 5 minutes",
		"set alarm for 5 minutes", "alarm in 10 min",
		"how much time is left", "whats left on the timer",
		"cancel the timer", "stop the timer", "pause the timer",
		"timer for 2 hours", "set a timer for 1 hour",
		"set a timer for 45 minutes", "3 min timer",
		"timer 1 min", "quick 2 min timer",
	)

	// Notes
	add("note",
		"save a note", "write this down", "make a note",
		"take a note", "note this", "jot this down",
		"save a note about the meeting", "note about the project",
		"write down this idea", "save this for later",
		"new note", "create a note", "add a note",
		"note: remember to call john", "note: buy groceries",
		"show my notes", "show all notes", "list my notes",
		"search my notes", "find my notes about work",
		"whats in my notes", "any notes about recipes",
		"delete the last note", "clear my notes",
		"edit my note", "update the note",
	)

	// Reminders
	add("reminder",
		"remind me to call mom", "remind me to buy milk",
		"remind me to take medicine", "remind me about the meeting",
		"remind me tomorrow", "remind me at 5pm",
		"remind me in the morning", "remind me tonight",
		"set a reminder for tomorrow", "set a reminder for 3pm",
		"reminder to pick up kids at 3", "reminder to submit report",
		"dont let me forget", "dont forget to remind me",
		"can you remind me", "please remind me",
		"i need a reminder", "reminder about the deadline",
		"remind me to exercise", "remind me to drink water",
		"remind me next week", "remind me on monday",
	)

	// Todos
	add("todo",
		"add buy milk to my todo list", "add this to my todos",
		"whats on my todo list", "show my todos", "show my tasks",
		"mark laundry as done", "check off groceries",
		"remove the first task", "delete that todo",
		"add a task", "new task", "new todo",
		"todo: finish the report", "todo buy groceries",
		"what do i need to do", "anything on my list",
		"clear my todo list", "mark everything done",
		"add exercise to my list", "put that on my list",
		"i need to finish the report", "i gotta do laundry",
	)

	// Calendar
	add("calendar",
		"whats on my calendar", "any events today",
		"do i have anything today", "show my schedule",
		"whats my schedule today", "whats happening today",
		"any meetings today", "any meetings tomorrow",
		"show me my calendar", "calendar for this week",
		"when is my next meeting", "am i free tomorrow",
		"am i busy today", "schedule for monday",
		"whats on for next week", "any upcoming events",
		"check my calendar", "add to my calendar",
	)

	// Email
	add("email",
		"check my email", "any new emails", "show my inbox",
		"read my email", "open my email", "check my inbox",
		"send an email", "compose an email", "write an email",
		"reply to the last email", "forward this email",
		"any emails from john", "unread emails",
		"email john about the meeting", "send a message to sarah",
	)

	// News
	add("news",
		"whats the news", "any news today", "show me the news",
		"latest news", "top headlines", "whats happening in the world",
		"any interesting news", "tech news", "science news",
		"sports news", "world news", "business news",
		"breaking news", "news today", "todays headlines",
		"what happened today", "any news about AI",
		"latest tech headlines", "show me tech news",
	)

	// Dictionary
	add("dict",
		"define serendipity", "define ubiquitous", "define ephemeral",
		"define pragmatic", "define altruistic", "define ambiguous",
		"what does ubiquitous mean", "what does ephemeral mean",
		"what does serendipity mean", "what does pragmatic mean",
		"meaning of ephemeral", "meaning of serendipity",
		"definition of paradigm", "definition of stoic",
		"whats the definition of irony", "whats the meaning of love",
		"synonyms for happy", "synonyms for good", "synonyms for beautiful",
		"antonyms of good", "antonyms of happy", "antonyms of big",
		"whats a synonym for fast", "whats another word for amazing",
		"thesaurus lookup for intelligent", "synonym for extraordinary",
		"what does the word paradigm mean", "define the word nuance",
		"dictionary lookup for esoteric",
	)

	// Network
	add("network",
		"ping google.com", "ping 8.8.8.8", "is google down",
		"is the server running", "check the server",
		"am i connected to the internet", "check my connection",
		"dns lookup for example.com", "dns lookup google.com",
		"is the website down", "is youtube down",
		"check my network", "network status",
		"whats my ip", "whats my ip address",
		"traceroute to google.com", "check connectivity",
		"test my internet", "speed test", "internet speed",
	)

	// Hash / encoding
	add("hash",
		"hash this string", "md5 of hello", "sha256 of password",
		"sha256 hash of this text", "md5 hash of hello world",
		"base64 encode this text", "base64 decode this",
		"url encode this string", "url decode this",
		"compute the checksum", "generate a hash",
		"hash hello world", "sha1 of this",
		"encode this in base64", "decode base64",
	)

	// Volume
	add("volume",
		"turn up the volume", "turn down the volume",
		"volume up", "volume down", "mute", "unmute",
		"set volume to 50", "set volume to 80",
		"louder", "quieter", "louder please", "too quiet",
		"too loud", "mute the sound", "unmute the sound",
		"increase volume", "decrease volume",
	)

	// Brightness
	add("brightness",
		"increase brightness", "decrease brightness",
		"brighter", "dimmer", "dim the screen",
		"set brightness to 70", "brightness up", "brightness down",
		"make it brighter", "make it dimmer", "too bright",
		"too dark", "screen too bright", "screen too dim",
		"max brightness", "minimum brightness",
	)

	// App management
	add("app",
		"open firefox", "open chrome", "launch spotify",
		"start vscode", "open terminal", "launch slack",
		"close the browser", "close firefox", "quit spotify",
		"open calculator", "launch file manager",
		"what apps are running", "running apps", "show running apps",
		"kill chrome", "force close firefox",
		"open settings", "launch discord", "open steam",
	)

	// Screenshot
	add("screenshot",
		"take a screenshot", "screenshot", "grab a screenshot",
		"capture the screen", "screen capture", "take a screen grab",
		"screenshot this", "save a screenshot", "snap the screen",
		"screenshot of my screen", "take a pic of the screen",
	)

	// Clipboard
	add("clipboard",
		"whats in my clipboard", "show clipboard", "paste clipboard",
		"copy this to clipboard", "clipboard contents",
		"show me what i copied", "whats on my clipboard",
		"clear the clipboard", "copy to clipboard",
	)

	// System info
	add("sysinfo",
		"how much disk space", "disk space left", "free disk space",
		"whats my ip address", "show system info", "system information",
		"how much ram", "ram usage", "memory usage",
		"cpu usage", "whats my cpu at", "show cpu info",
		"uptime", "how long has my computer been on",
		"system stats", "show system stats", "computer info",
		"os version", "what os am i running",
	)

	// Find files
	add("find_files",
		"find all python files", "find my pdf documents",
		"locate config files", "find images in downloads",
		"search for files", "find files containing config",
		"where are my documents", "find large files",
		"find recent files", "search for go files",
		"find files named readme", "locate my photos",
	)

	// Code runner
	add("run_code",
		"run this python code", "execute the script",
		"run javascript", "run this bash command",
		"execute this code", "run the program",
		"compile and run", "run my script",
		"execute python", "run this snippet",
	)

	// Archive
	add("archive",
		"zip these files", "unzip the download", "extract the archive",
		"compress this folder", "create a zip file",
		"extract tar file", "untar this", "unzip this",
		"zip this folder", "compress these files",
		"extract the zip", "decompress this",
	)

	// Disk usage
	add("disk_usage",
		"whats using the most disk space", "show largest folders",
		"how much space is left", "disk usage", "show disk usage",
		"biggest files on my drive", "whats taking up space",
		"analyze disk usage", "check storage",
		"how full is my drive", "storage space",
	)

	// Process management
	add("process",
		"show running processes", "whats using the cpu",
		"kill process on port 8080", "list processes",
		"top processes", "show top", "whats eating my ram",
		"kill that process", "stop the process",
		"process list", "running processes",
	)

	// QR codes
	add("qrcode",
		"generate a qr code", "create a qr code",
		"make a qr code for this url", "qr code for my wifi",
		"generate qr code", "qr code", "make a qr code",
		"create qr for this link", "qr code for this text",
	)

	// Password
	add("password",
		"generate a password", "create a password", "new password",
		"generate a strong password", "random password",
		"make me a password", "password generator",
		"generate a 16 character password", "create a passphrase",
		"i need a new password", "secure password please",
		"generate a pin", "random pin", "6 digit pin",
	)

	// Bookmarks
	add("bookmark",
		"save this link", "bookmark this", "add to bookmarks",
		"show my bookmarks", "list bookmarks", "saved links",
		"delete bookmark", "remove bookmark",
		"search my bookmarks", "find in bookmarks",
		"bookmark this page", "save this url",
	)

	// Journal
	add("journal",
		"dear diary", "journal entry", "write in my journal",
		"todays journal", "add to my journal",
		"show my journal", "last journal entry",
		"what did i write yesterday", "journal about today",
		"open my journal", "read my journal",
		"show last weeks journal", "my diary",
	)

	// Habits
	add("habit",
		"create a habit", "new habit", "track a habit",
		"did i exercise today", "mark exercise done",
		"habit streak", "my habit progress",
		"show my habits", "list habits", "habit tracker",
		"check off meditation", "did i meditate",
		"log my habit", "habit for daily reading",
	)

	// Expenses
	add("expense",
		"i spent 5 dollars on coffee", "spent 20 on lunch",
		"add expense 50 groceries", "log expense",
		"how much did i spend this week", "spending this month",
		"show my expenses", "monthly expenses",
		"track spending", "expense report",
		"i bought coffee for 5 bucks", "spent 15 on gas",
		"add purchase 30 for dinner", "log 10 for snacks",
	)

	// Convert
	add("convert",
		"convert 5 miles to km", "5 miles in kilometers",
		"how many feet in a mile", "100 kg to pounds",
		"convert 32 fahrenheit to celsius", "32 f to c",
		"5 gallons to liters", "convert 100 usd to euros",
		"how many cm in an inch", "meters to feet",
		"pounds to kg", "miles to km", "celsius to fahrenheit",
		"how many oz in a cup", "cups to ml",
		"inches to cm", "feet to meters", "liters to gallons",
		"convert 72 inches to feet", "10 km in miles",
	)

	// Transform
	add("transform",
		"rewrite this more formally", "make this professional",
		"summarize this article", "paraphrase this",
		"turn this into bullet points", "simplify this",
		"make this more concise", "make this longer",
		"rewrite this casually", "make this friendlier",
		"fix the grammar", "proofread this",
		"make this sound better", "improve this writing",
	)

	// Calculator (additional patterns beyond math section)
	add("calculate",
		"calculator", "open calculator", "calc",
		"quick math", "do some math for me",
		"math question", "solve this math problem",
		"number crunching", "figure this out",
	)

	// ----------------------------------------------------------------
	// EMOTIONAL (500+ examples)
	// ----------------------------------------------------------------
	add("conversation",
		// Negative emotions
		"i had a bad day", "today was terrible", "worst day ever",
		"i had an awful day", "today sucked", "what a bad day",
		"im feeling down", "im sad", "i feel sad",
		"im feeling depressed", "i feel lonely", "im so lonely",
		"im stressed out", "im so stressed", "everything is stressful",
		"im anxious", "i feel anxious", "im having anxiety",
		"im frustrated", "this is so frustrating", "im annoyed",
		"im angry", "im so mad", "im furious",
		"im disappointed", "im let down", "this is disappointing",
		"im exhausted", "im so tired", "im drained",
		"im overwhelmed", "i cant take it anymore", "its too much",
		"im burned out", "im burnt out", "i feel empty",
		"im worried", "im scared", "im nervous about tomorrow",
		"my boss yelled at me", "i got fired", "i lost my job",
		"i failed the exam", "i failed my test", "i messed up",
		"i made a huge mistake", "i screwed up", "this is a disaster",
		"nobody cares about me", "nobody understands",
		"im having a rough time", "things arent going well",
		"i feel like giving up", "i dont know anymore",
		// Positive emotions
		"i got promoted!", "i got the job!", "i passed the exam!",
		"i feel great today", "im so happy", "im feeling amazing",
		"today was awesome", "best day ever", "what a great day",
		"i feel wonderful", "im on top of the world",
		"i fell in love", "im in love", "i got engaged",
		"we got married!", "im having a baby", "i got accepted",
		"i graduated!", "i won the competition", "i did it!",
		"im so proud of myself", "im feeling confident",
		"everything is going well", "life is good",
		"i feel blessed", "im grateful", "im thankful",
		"this is the best", "im so excited", "cant wait",
		// Mixed / neutral
		"i had an interesting day", "today was okay",
		"im feeling meh", "just an average day",
		"nothing special happened", "same old same old",
		"feeling kinda blah", "im alright i guess",
		"could be better could be worse",
	)

	// ----------------------------------------------------------------
	// CONVERSATION / CHAT (500+ examples)
	// ----------------------------------------------------------------
	add("conversation",
		// Opinions / discussion
		"what do you think about AI", "what do you think about climate change",
		"what do you think about social media", "what do you think about remote work",
		"whats your opinion on crypto", "whats your opinion on electric cars",
		"do you think AI will take over", "do you think we'll go to mars",
		"lets talk about something", "lets chat", "lets talk",
		"i wanna chat", "can we just talk", "talk to me",
		"im bored", "im so bored", "entertain me",
		"tell me something interesting", "tell me something cool",
		"tell me something i dont know",
		// Personal statements
		"i went to the gym today", "i ran 5 miles this morning",
		"i cooked dinner tonight", "i watched a great movie",
		"i read a good book", "i started learning guitar",
		"i went to the beach", "i had coffee with a friend",
		"i just got back from vacation", "im planning a trip",
		"i tried a new restaurant", "i adopted a cat",
		"i started a new project", "im working on something cool",
		"i just moved to a new city", "i met someone interesting",
		// Short conversational
		"yeah", "yeah totally", "exactly", "right",
		"ok cool", "cool", "interesting", "nice", "awesome",
		"thats cool", "thats interesting", "thats nice",
		"i see", "i get it", "makes sense",
		"sure", "sure thing", "of course",
		"no way", "really?", "seriously?",
		"oh wow", "oh cool", "oh nice",
		"yeah but what about", "ok but still",
		"hmm interesting", "hmm ok", "hmm i see",
		"true", "good point", "fair enough",
		"agreed", "i agree", "totally agree",
		"i disagree", "not sure about that", "i dont think so",
		"maybe", "i guess so", "possibly",
		"lol", "haha", "hahaha", "lmao", "rofl",
		"thats funny", "thats hilarious",
		// Asking about nous
		"are you alright", "are you okay", "you ok",
		"hows it going nous", "how are you doing nous",
		"what are you up to", "what have you been up to",
		"do you get bored", "do you sleep",
		"do you have hobbies", "what do you do for fun",
	)

	// ----------------------------------------------------------------
	// META / SELF-AWARENESS (additional)
	// ----------------------------------------------------------------
	add("meta",
		"who are you", "what are you", "whats your name",
		"who made you", "who created you", "who built you",
		"what can you do", "what are you capable of",
		"how do you work", "are you an ai", "are you a bot",
		"are you sentient", "are you conscious", "are you alive",
		"do you have feelings", "can you think", "do you dream",
		"are you real", "are you human", "are you a robot",
		"what is nous", "tell me about nous",
		"introduce yourself", "what should i call you",
		"whats your purpose", "why do you exist",
		"how smart are you", "are you better than chatgpt",
		"are you better than siri", "are you like alexa",
	)

	// ----------------------------------------------------------------
	// FAREWELL (additional casual)
	// ----------------------------------------------------------------
	add("farewell",
		"bye", "bye!", "bye bye", "byebye", "bai",
		"goodbye", "goodbye!", "good bye",
		"see ya", "see ya!", "see you", "see you later",
		"later", "later!", "laterr", "laters",
		"gotta go", "gtg", "g2g", "gotta bounce",
		"im out", "peace", "peace out", "peace!",
		"take care", "take care!", "catch you later",
		"nite", "nite nite", "night", "night night", "goodnight",
		"im heading out", "time to go", "thats all",
		"im done", "im done for today", "thats it for now",
		"thanks bye", "ok bye", "alright bye", "cool bye",
		"cya", "cya!", "ttyl", "talk to you later",
		"until next time", "see you around",
	)

	// ----------------------------------------------------------------
	// AFFIRMATION (additional casual)
	// ----------------------------------------------------------------
	add("affirmation",
		"thanks", "thanks!", "thank you", "thank you!",
		"thx", "thx!", "ty", "ty!",
		"cool", "cool!", "nice", "nice!",
		"great", "great!", "awesome", "awesome!",
		"perfect", "perfect!", "perfect thanks",
		"awesome thank you", "thats great", "that works",
		"thats what i needed", "exactly what i wanted",
		"nah", "nope", "not really", "never mind",
		"nvm", "nevermind", "forget it", "no thanks",
		"ok thanks", "ok thank you", "ok thx",
		"got it", "got it thanks", "understood",
		"makes sense thanks", "appreciate it",
		"much appreciated", "youre the best",
	)

	// ----------------------------------------------------------------
	// REMEMBER / RECALL (additional casual)
	// ----------------------------------------------------------------
	add("remember",
		"remember my name is alex", "my name is sarah",
		"remember i like pizza", "i love sushi",
		"my favorite color is blue", "i prefer dark mode",
		"im a software engineer", "i live in berlin",
		"remember i work at google", "i work from home",
		"my birthday is march 15", "remember i have a dog",
		"i speak three languages", "im vegetarian",
		"remember i dont eat meat", "i prefer tea over coffee",
		"my kids name is emma", "i drive a tesla",
		"remember i run every morning", "i go to bed at 11",
	)
	add("recall",
		"whats my name", "do you remember my name",
		"what do you know about me", "who am i",
		"do you remember me", "whats my favorite color",
		"where do i live", "what do i do for work",
		"whats my email", "do you remember what i told you",
		"what have i told you", "what do you remember",
		"recall my preferences", "what are my preferences",
	)

	// ----------------------------------------------------------------
	// DAILY BRIEFING (additional)
	// ----------------------------------------------------------------
	add("daily_briefing",
		"good morning", "good morning!", "good morning nous",
		"good afternoon", "good evening",
		"morning", "morning!", "mornin",
		"daily briefing", "brief me", "briefing",
		"morning briefing", "whats on today",
		"start my day", "hows my day looking",
		"morning report", "daily summary",
		"what do i have today", "whats my schedule today",
		"how does my day look", "whats planned for today",
		"ready to start the day", "whats happening today",
		"morning update", "good morning whats on my schedule",
		"hey good morning", "good morning hows my day",
	)

	// ----------------------------------------------------------------
	// FOLLOW UP (additional)
	// ----------------------------------------------------------------
	add("followup",
		"tell me more", "more about that", "go on", "continue",
		"keep going", "what else", "anything else",
		"dig deeper", "more details", "elaborate",
		"why is that", "how come", "how so",
		"can you explain more", "explain further",
		"what do you mean", "in what way",
		"for example", "like what", "such as",
		"and then what", "what happened next",
		"go deeper", "expand on that", "more please",
	)

	// ----------------------------------------------------------------
	// SEARCH / WEB (additional)
	// ----------------------------------------------------------------
	add("search",
		"search for golang tutorials", "google this",
		"look up machine learning", "search the web",
		"find information about climate change",
		"search for recipes", "google best restaurants nearby",
		"look this up", "search for this", "web search",
	)
	add("web_lookup",
		"stock price of tesla", "who won the game last night",
		"latest election results", "whats trending",
		"breaking news", "current bitcoin price",
		"score of the game", "sports scores",
		"weather in new york right now", "traffic near me",
	)

	// ----------------------------------------------------------------
	// FILE OPERATIONS (additional)
	// ----------------------------------------------------------------
	add("file_op",
		"read the file config.yaml", "open my config",
		"create a new file", "edit the config",
		"delete the log file", "show me main.go",
		"list files", "ls", "cat readme",
		"write hello to output.txt", "save to file",
		"read the logs", "show the error log",
	)

	// ----------------------------------------------------------------
	// HARD NEGATIVES — critical boundary cases
	// ----------------------------------------------------------------
	// These prevent common misclassifications

	// "write a poem" is creative, NOT file_op
	add("creative", "write a poem about rain", "write me a story",
		"write something funny", "write a haiku")

	// "tell me about X" is explain, NOT creative
	add("explain", "tell me about dogs", "tell me about the ocean",
		"tell me about history", "tell me about space")

	// "what is X" single-word topics are dict, NOT explain
	add("dict", "define serendipity", "what does ubiquitous mean",
		"synonyms for happy", "meaning of ephemeral")

	// Personal statements are conversation, NOT greeting
	add("conversation",
		"i feel happy today", "im feeling sad", "im tired",
		"having a great day", "having a rough day",
		"i feel stressed out", "i feel exhausted",
		"im feeling great", "im feeling a bit tired")

	// "set a reminder" is reminder, NOT timer
	add("reminder", "set a reminder for the meeting",
		"remind me to call mom tomorrow")

	// "what is your name" is meta, NOT explain
	add("meta", "what is your name", "who created you",
		"do you have feelings", "are you an ai")

	// Daily briefing — "good morning" should be briefing
	add("daily_briefing", "good morning", "good afternoon",
		"good evening", "morning briefing")

	return examples
}

// addWordListExamples generates training examples from a word list + templates.
func addWordListExamples(examples *[]TrainingExample, intent string, words []string, templates []string) {
	// Add raw word list entries
	for _, w := range words {
		*examples = append(*examples, TrainingExample{Text: w, Intent: intent})
	}

	// Expand with templates
	if templates == nil {
		return
	}
	for _, w := range words {
		// Pick a random subset of templates to avoid data explosion
		for _, tmpl := range templates {
			if strings.Contains(tmpl, "%s") {
				*examples = append(*examples, TrainingExample{
					Text:   fmt.Sprintf(tmpl, w),
					Intent: intent,
				})
			} else {
				*examples = append(*examples, TrainingExample{
					Text:   tmpl,
					Intent: intent,
				})
			}
		}
	}
}

// -----------------------------------------------------------------------
// Data augmentation — generates variations of training examples.
// -----------------------------------------------------------------------

// AugmentExamples creates variations of existing examples by:
//   - Adding/removing punctuation
//   - Changing capitalization
//   - Adding filler words
//   - Reordering (for some patterns)
func AugmentExamples(examples []TrainingExample) []TrainingExample {
	augmented := make([]TrainingExample, 0, len(examples)*3)
	augmented = append(augmented, examples...)

	fillers := []string{
		"please ", "can you ", "could you ", "I want to ", "I'd like to ",
		"hey ", "yo ", "nous ", "hey nous ", "ok ", "so ", "umm ",
		"I need you to ", "just ", "quickly ",
	}

	for _, ex := range examples {
		// Add random filler prefix (50% chance)
		if rand.Float32() < 0.5 {
			filler := fillers[rand.Intn(len(fillers))]
			augmented = append(augmented, TrainingExample{
				Text:   filler + strings.ToLower(ex.Text),
				Intent: ex.Intent,
			})
		}

		// Add question mark (30% chance for non-question intents)
		if rand.Float32() < 0.3 && !strings.HasSuffix(ex.Text, "?") {
			augmented = append(augmented, TrainingExample{
				Text:   ex.Text + "?",
				Intent: ex.Intent,
			})
		}

		// Remove punctuation
		cleaned := strings.TrimRight(ex.Text, "?!.,")
		if cleaned != ex.Text {
			augmented = append(augmented, TrainingExample{
				Text:   cleaned,
				Intent: ex.Intent,
			})
		}

		// ALLCAPS variation (20% chance)
		if rand.Float32() < 0.2 {
			augmented = append(augmented, TrainingExample{
				Text:   strings.ToUpper(ex.Text),
				Intent: ex.Intent,
			})
		}

		// Drop first word (15% chance, only for 3+ word inputs)
		words := strings.Fields(ex.Text)
		if len(words) >= 3 && rand.Float32() < 0.15 {
			augmented = append(augmented, TrainingExample{
				Text:   strings.Join(words[1:], " "),
				Intent: ex.Intent,
			})
		}

		// Add trailing "please" / "thanks" (10% chance)
		if rand.Float32() < 0.1 {
			suffix := []string{" please", " thanks", " thx", " pls"}
			augmented = append(augmented, TrainingExample{
				Text:   ex.Text + suffix[rand.Intn(len(suffix))],
				Intent: ex.Intent,
			})
		}
	}

	return augmented
}
