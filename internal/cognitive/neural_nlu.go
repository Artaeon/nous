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

	// Generate training data, augment, and train
	examples := GenerateTrainingData(nlu)
	augmented := AugmentExamples(examples)
	fmt.Fprintf(os.Stderr, "  training neural classifier (%d examples, %d intents)...\n", len(augmented), countIntents(augmented))
	result := nn.Classifier.Train(augmented, 80, 0.1)
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
