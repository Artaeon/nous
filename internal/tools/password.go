package tools

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"strings"
)

// RegisterPasswordTools registers the password generator tool.
func RegisterPasswordTools(r *Registry) {
	r.Register(Tool{
		Name:        "password",
		Description: "Generate a secure random password, passphrase, or PIN. Args: type (optional: password/passphrase/pin, default password), length (optional: chars for password default 16, words for passphrase default 4, digits for pin default 6).",
		Execute: func(args map[string]string) (string, error) {
			return toolPassword(args)
		},
	})
}

func toolPassword(args map[string]string) (string, error) {
	genType := args["type"]
	if genType == "" {
		genType = "password"
	}

	lengthStr := args["length"]

	switch genType {
	case "password":
		length := 16
		if lengthStr != "" {
			fmt.Sscanf(lengthStr, "%d", &length)
		}
		if length < 4 {
			length = 4
		}
		if length > 128 {
			length = 128
		}
		return generatePassword(length)

	case "passphrase":
		wordCount := 4
		if lengthStr != "" {
			fmt.Sscanf(lengthStr, "%d", &wordCount)
		}
		if wordCount < 2 {
			wordCount = 2
		}
		if wordCount > 12 {
			wordCount = 12
		}
		return generatePassphrase(wordCount)

	case "pin":
		length := 6
		if lengthStr != "" {
			fmt.Sscanf(lengthStr, "%d", &length)
		}
		if length < 4 {
			length = 4
		}
		if length > 20 {
			length = 20
		}
		return generatePIN(length)

	default:
		return "", fmt.Errorf("unknown type %q — use password, passphrase, or pin", genType)
	}
}

func generatePassword(length int) (string, error) {
	const (
		upper   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		lower   = "abcdefghijklmnopqrstuvwxyz"
		digits  = "0123456789"
		symbols = "!@#$%^&*()-_=+[]{}|;:,.<>?"
	)
	allChars := upper + lower + digits + symbols

	// Guarantee at least one of each class
	result := make([]byte, length)
	classes := []string{upper, lower, digits, symbols}
	for i, class := range classes {
		idx, err := cryptoRandInt(len(class))
		if err != nil {
			return "", err
		}
		result[i] = class[idx]
	}

	// Fill remainder
	for i := len(classes); i < length; i++ {
		idx, err := cryptoRandInt(len(allChars))
		if err != nil {
			return "", err
		}
		result[i] = allChars[idx]
	}

	// Shuffle using Fisher-Yates
	for i := length - 1; i > 0; i-- {
		j, err := cryptoRandInt(i + 1)
		if err != nil {
			return "", err
		}
		result[i], result[j] = result[j], result[i]
	}

	return string(result), nil
}

func generatePassphrase(wordCount int) (string, error) {
	words := make([]string, wordCount)
	for i := 0; i < wordCount; i++ {
		idx, err := cryptoRandInt(len(passphraseWordList))
		if err != nil {
			return "", err
		}
		words[i] = passphraseWordList[idx]
	}
	return strings.Join(words, "-"), nil
}

func generatePIN(length int) (string, error) {
	digits := make([]byte, length)
	for i := 0; i < length; i++ {
		d, err := cryptoRandInt(10)
		if err != nil {
			return "", err
		}
		digits[i] = byte('0') + byte(d)
	}
	return string(digits), nil
}

func cryptoRandInt(max int) (int, error) {
	n, err := rand.Int(rand.Reader, big.NewInt(int64(max)))
	if err != nil {
		return 0, fmt.Errorf("crypto/rand: %w", err)
	}
	return int(n.Int64()), nil
}

// passphraseWordList contains common English words for passphrase generation.
var passphraseWordList = []string{
	"abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
	"accept", "access", "account", "accuse", "achieve", "acid", "across", "action",
	"actor", "actual", "adapt", "address", "adjust", "admit", "adult", "advance",
	"advice", "afford", "again", "agent", "agree", "ahead", "alarm", "album",
	"alert", "alien", "almost", "alone", "alpha", "already", "also", "alter",
	"always", "amount", "anchor", "ancient", "angel", "anger", "angle", "animal",
	"annual", "answer", "anvil", "apart", "apple", "april", "arctic", "arena",
	"argue", "armor", "army", "arrow", "artist", "asthma", "atlas", "atom",
	"august", "autumn", "avocado", "avoid", "awake", "awesome", "awful", "axis",
	"badge", "balance", "bamboo", "banana", "banner", "barely", "barrel", "basket",
	"battle", "beach", "beauty", "become", "before", "begin", "behave", "believe",
	"below", "bench", "benefit", "beyond", "bicycle", "bird", "bitter", "blanket",
	"blast", "blind", "blood", "blossom", "board", "body", "bomb", "bone",
	"bonus", "border", "bottle", "bottom", "bounce", "brain", "brand", "brave",
	"bread", "bridge", "bright", "bring", "broken", "bronze", "brother", "brush",
	"bubble", "budget", "buffalo", "build", "bullet", "bundle", "burden", "burger",
	"butter", "cabin", "cable", "cactus", "camera", "camp", "canal", "candle",
	"candy", "canvas", "canyon", "captain", "carbon", "cargo", "carpet", "carry",
	"castle", "catalog", "catch", "cattle", "cause", "cedar", "celery", "cement",
	"census", "central", "cereal", "certain", "chair", "chalk", "champion", "change",
	"chapter", "charge", "chart", "chase", "cheap", "cheese", "cherry", "chicken",
	"chief", "chimney", "choice", "chunk", "circle", "citizen", "civil", "claim",
	"clap", "clarify", "clean", "clever", "click", "client", "cliff", "climb",
	"clock", "close", "cloth", "cloud", "cluster", "coach", "coconut", "code",
	"coffee", "collect", "color", "column", "combine", "come", "comfort", "comic",
	"common", "company", "concert", "conduct", "confirm", "congress", "connect", "consider",
	"control", "convert", "cookie", "copper", "coral", "corner", "correct", "cotton",
	"couch", "country", "couple", "course", "cousin", "cover", "craft", "crane",
	"crash", "crater", "crazy", "cream", "credit", "creek", "crew", "cricket",
	"crime", "crisp", "critic", "cross", "crouch", "crowd", "crucial", "cruel",
	"cruise", "crumble", "crush", "crystal", "cube", "culture", "curtain", "curve",
	"cushion", "custom", "cycle", "damage", "dance", "danger", "daring", "dash",
	"daughter", "dawn", "debate", "decade", "december", "decide", "decline", "decorate",
	"decrease", "deer", "defense", "define", "defy", "degree", "delay", "deliver",
	"demand", "denial", "dentist", "depart", "depend", "deposit", "depth", "deputy",
	"derive", "describe", "desert", "design", "despair", "destroy", "detail", "detect",
	"develop", "device", "devote", "diagram", "diamond", "diary", "diesel", "differ",
	"digital", "dignity", "dilemma", "dinner", "dinosaur", "direct", "discover", "disease",
	"dish", "display", "distance", "divert", "dizzy", "doctor", "document", "dolphin",
	"domain", "donate", "donkey", "donor", "door", "double", "dove", "dragon",
	"drama", "drastic", "dream", "drift", "drink", "drive", "drum", "duck",
	"dune", "during", "dust", "dwarf", "dynamic", "eager", "eagle", "early",
	"earth", "easily", "echo", "ecology", "economy", "edge", "editor", "educate",
	"effort", "either", "elbow", "elder", "electric", "elegant", "element", "elephant",
	"elevator", "elite", "else", "embark", "embody", "embrace", "emerge", "emotion",
	"emperor", "employ", "empower", "empty", "enable", "enact", "endorse", "enemy",
	"energy", "enforce", "engage", "engine", "enhance", "enjoy", "enlist", "enough",
	"enrich", "enroll", "ensure", "enter", "entire", "entry", "envelope", "episode",
	"equip", "erosion", "error", "escape", "essay", "essence", "estate", "eternal",
	"evening", "evidence", "evil", "evolve", "exact", "example", "excess", "exchange",
	"excite", "exclude", "excuse", "execute", "exercise", "exhaust", "exile", "exist",
	"expand", "expect", "expire", "explain", "expose", "express", "extend", "extra",
	"fabric", "face", "faculty", "fade", "faint", "faith", "fall", "false",
	"family", "famous", "fancy", "fantasy", "fashion", "father", "fatigue", "fault",
	"favorite", "feature", "february", "federal", "fence", "festival", "fetch", "fever",
	"fiction", "field", "figure", "file", "film", "filter", "final", "finance",
	"finger", "finish", "fire", "firm", "first", "fiscal", "fitness", "flag",
	"flame", "flash", "flavor", "flight", "flip", "float", "flock", "floor",
	"flower", "fluid", "flush", "focus", "folder", "follow", "food", "force",
	"forest", "forget", "fork", "fortune", "forum", "fossil", "foster", "found",
	"fox", "fragile", "frame", "frequent", "fresh", "friend", "fringe", "frog",
	"front", "frost", "frozen", "fruit", "fuel", "fun", "funny", "furnace",
	"fury", "future", "gadget", "galaxy", "gallery", "game", "garbage", "garden",
	"garlic", "garment", "gather", "gauge", "gaze", "general", "genius", "genre",
	"gentle", "genuine", "gesture", "ghost", "giant", "gift", "giggle", "ginger",
	"giraffe", "glad", "glance", "glass", "glimpse", "globe", "gloom", "glory",
	"glove", "glow", "glue", "goat", "goddess", "gold", "good", "goose",
	"gorilla", "gospel", "gossip", "govern", "grace", "grain", "grammar", "grape",
	"grass", "gravity", "great", "green", "grid", "grief", "grit", "grocery",
	"group", "grow", "grunt", "guard", "guess", "guide", "guitar", "gun",
	"habit", "half", "hammer", "hamster", "hand", "harbor", "hard", "harvest",
	"hawk", "hazard", "heart", "heavy", "hedge", "height", "hello", "helmet",
	"hero", "hidden", "highway", "hill", "hint", "history", "hobby", "hockey",
	"hold", "hollow", "home", "honey", "hood", "hope", "horizon", "horn",
	"horror", "horse", "hospital", "host", "hotel", "hour", "hover", "hub",
	"huge", "human", "humble", "humor", "hundred", "hunger", "hunt", "hurdle",
	"hurry", "hybrid", "ice", "icon", "idea", "identify", "idle", "ignore",
	"image", "imitate", "immense", "immune", "impact", "impose", "improve", "impulse",
	"inch", "include", "income", "increase", "index", "indicate", "indoor", "industry",
	"infant", "inflict", "inform", "inhale", "initial", "inject", "inner", "innocent",
	"input", "inquiry", "insane", "insect", "inside", "inspire", "install", "intact",
	"interest", "invest", "invite", "involve", "iron", "island", "isolate", "issue",
	"ivory", "jacket", "jaguar", "January", "jazz", "jealous", "jelly", "jewel",
	"journey", "judge", "juice", "jump", "jungle", "junior", "junk", "just",
	"kangaroo", "keen", "keep", "kernel", "key", "kick", "kidney", "kind",
	"kingdom", "kiss", "kitchen", "kite", "kitten", "knee", "knife", "knock",
	"know", "label", "labor", "ladder", "lady", "lake", "lamp", "language",
	"laptop", "large", "later", "laugh", "laundry", "lawn", "layer", "leader",
	"leaf", "learn", "leave", "lecture", "left", "legend", "leisure", "lemon",
	"length", "lens", "leopard", "lesson", "letter", "level", "liberty", "library",
	"license", "life", "light", "limb", "limit", "link", "lion", "liquid",
	"list", "little", "live", "lizard", "load", "loan", "lobster", "local",
	"lock", "logic", "lonely", "long", "loop", "lottery", "loud", "lounge",
	"loyal", "lucky", "lumber", "lunar", "lunch", "luxury", "lyrics", "machine",
	"magic", "magnet", "maid", "mail", "major", "make", "mammal", "manage",
	"mandate", "mango", "mansion", "manual", "maple", "marble", "march", "margin",
	"marine", "market", "marriage", "mask", "mass", "master", "match", "material",
	"matrix", "matter", "maximum", "meadow", "mean", "measure", "media", "melody",
	"member", "memory", "mention", "mentor", "mercy", "merge", "merit", "mesh",
	"metal", "method", "middle", "midnight", "milk", "million", "mimic", "mind",
	"minimum", "minor", "minute", "miracle", "mirror", "misery", "miss", "mistake",
	"mixture", "mobile", "model", "modify", "moment", "monitor", "monkey", "monster",
	"month", "moral", "morning", "mosquito", "mother", "motion", "motor", "mountain",
	"mouse", "move", "movie", "much", "muffin", "mule", "multiply", "muscle",
	"museum", "mushroom", "music", "must", "mutual", "myself", "mystery", "myth",
	"naive", "name", "napkin", "narrow", "nation", "nature", "near", "neck",
	"negative", "neglect", "neither", "nephew", "nerve", "nest", "network", "neutral",
	"never", "news", "nice", "night", "noble", "noise", "normal", "north",
	"notable", "nothing", "notice", "novel", "nuclear", "number", "nurse", "nut",
	"oak", "obey", "object", "oblige", "obscure", "observe", "obtain", "obvious",
	"occur", "ocean", "october", "odor", "offer", "office", "often", "olive",
	"olympic", "omit", "once", "onion", "online", "open", "opera", "opinion",
	"oppose", "option", "orange", "orbit", "orchard", "order", "ordinary", "organ",
	"orient", "origin", "orphan", "ostrich", "other", "outdoor", "outer", "output",
	"outside", "oval", "oven", "over", "owner", "oxygen", "oyster", "ozone",
	"pact", "paddle", "page", "palace", "palm", "panda", "panel", "panic",
	"panther", "paper", "parade", "parent", "park", "parrot", "party", "pass",
	"patch", "path", "patient", "patrol", "pattern", "pause", "pave", "peace",
	"peanut", "pear", "peasant", "pelican", "pencil", "penguin", "pepper", "perfect",
	"permit", "person", "pet", "phone", "photo", "phrase", "physical", "piano",
	"picnic", "picture", "piece", "pig", "pigeon", "pill", "pilot", "pink",
	"pioneer", "pipe", "pistol", "pitch", "pizza", "place", "planet", "plastic",
	"plate", "play", "please", "pledge", "pluck", "plug", "plunge", "poem",
	"poet", "point", "polar", "pole", "police", "pond", "pony", "pool",
	"popular", "pork", "position", "possible", "potato", "pottery", "poverty", "powder",
	"power", "practice", "praise", "predict", "prefer", "prepare", "present", "pretty",
	"prevent", "price", "pride", "primary", "print", "priority", "prison", "private",
	"prize", "problem", "process", "produce", "profit", "program", "project", "promote",
	"proof", "property", "prosper", "protect", "proud", "provide", "public", "pudding",
	"pulse", "pumpkin", "punch", "pupil", "puppy", "puzzle", "pyramid", "quality",
	"quantum", "quarter", "question", "quick", "quit", "quiz", "quote", "rabbit",
	"raccoon", "race", "rack", "radar", "radio", "rail", "rain", "raise",
	"rally", "ramp", "ranch", "random", "range", "rapid", "rare", "rate",
	"rather", "raven", "razor", "ready", "real", "reason", "rebel", "rebuild",
	"recall", "receive", "recipe", "record", "recycle", "reduce", "reform", "region",
	"regret", "regular", "reject", "relax", "release", "relief", "rely", "remain",
	"remember", "remind", "remove", "render", "renew", "rent", "reopen", "repair",
	"repeat", "replace", "report", "require", "rescue", "resemble", "resist", "resource",
	"response", "result", "retire", "retreat", "return", "reunion", "reveal", "review",
	"reward", "rhythm", "ribbon", "rice", "rich", "ride", "rifle", "right",
	"rigid", "ring", "riot", "ripple", "risk", "ritual", "rival", "river",
	"road", "roast", "robot", "robust", "rocket", "romance", "roof", "rookie",
	"room", "rose", "rotate", "rough", "round", "route", "royal", "rubber",
	"rude", "rug", "rule", "runway", "rural", "saddle", "sadness", "safe",
	"sail", "salad", "salmon", "salon", "salt", "salute", "same", "sample",
	"sand", "satisfy", "satoshi", "sauce", "sausage", "save", "scale", "scan",
	"scatter", "scene", "scheme", "school", "science", "scissors", "scorpion", "scout",
	"scrap", "screen", "script", "scrub", "search", "season", "seat", "second",
	"secret", "section", "security", "seed", "select", "sell", "senior", "sense",
	"sentence", "series", "service", "session", "settle", "setup", "seven", "shadow",
	"shaft", "shallow", "share", "shed", "shell", "sheriff", "shield", "shift",
	"shine", "ship", "shiver", "shock", "shoe", "shoot", "shop", "short",
	"shoulder", "shove", "shrimp", "shrug", "shuffle", "sibling", "side", "siege",
	"sight", "sign", "silent", "silk", "silly", "silver", "similar", "simple",
	"since", "siren", "sister", "situate", "size", "skate", "sketch", "skin",
	"skirt", "skull", "slab", "slam", "sleep", "slender", "slice", "slide",
	"slight", "slim", "slogan", "slow", "small", "smart", "smile", "smoke",
	"smooth", "snack", "snake", "snap", "snow", "soap", "soccer", "social",
	"sock", "soda", "soft", "solar", "soldier", "solid", "solution", "someone",
	"song", "soon", "sort", "soul", "sound", "soup", "source", "south",
	"space", "spare", "spatial", "spawn", "speak", "special", "speed", "spell",
	"spend", "sphere", "spice", "spider", "spike", "spirit", "split", "sponsor",
	"spoon", "sport", "spot", "spray", "spread", "spring", "square", "squeeze",
	"squirrel", "stable", "stadium", "staff", "stage", "stairs", "stamp", "stand",
	"start", "state", "stay", "steak", "steel", "stem", "step", "stereo",
	"stick", "still", "sting", "stock", "stomach", "stone", "stool", "story",
	"stove", "strategy", "street", "strike", "strong", "struggle", "student", "stuff",
	"stumble", "style", "subject", "submit", "subway", "success", "such", "sudden",
	"suffer", "sugar", "suggest", "suit", "summer", "sun", "sunny", "sunset",
	"super", "supply", "supreme", "sure", "surface", "surge", "surprise", "surround",
	"survey", "suspect", "sustain", "swallow", "swamp", "swap", "swarm", "sweet",
	"swim", "swing", "switch", "sword", "symbol", "symptom", "syrup", "system",
	"table", "tackle", "tag", "tail", "talent", "talk", "tank", "tape",
	"target", "task", "taste", "tattoo", "taxi", "teach", "team", "tell",
	"ten", "tenant", "tennis", "tent", "term", "test", "text", "thank",
	"that", "theme", "then", "theory", "there", "they", "thing", "this",
	"thought", "three", "thrive", "throw", "thumb", "thunder", "ticket", "tide",
	"tiger", "tilt", "timber", "time", "tiny", "tip", "tired", "tissue",
	"title", "toast", "tobacco", "today", "toddler", "toe", "together", "toilet",
	"token", "tomato", "tomorrow", "tone", "tongue", "tonight", "tool", "tooth",
	"top", "topic", "topple", "torch", "tornado", "tortoise", "toss", "total",
	"tourist", "toward", "tower", "town", "toy", "track", "trade", "traffic",
	"train", "transfer", "trap", "trash", "travel", "tray", "treat", "tree",
	"trend", "trial", "tribe", "trick", "trigger", "trim", "trip", "trophy",
	"trouble", "truck", "true", "trumpet", "trust", "truth", "try", "tube",
	"tuna", "tunnel", "turkey", "turn", "turtle", "twelve", "twenty", "twice",
	"twin", "twist", "type", "typical", "ugly", "umbrella", "unable", "unaware",
	"uncle", "uncover", "under", "undo", "unfair", "unfold", "unhappy", "uniform",
	"unique", "unit", "universe", "unknown", "unlock", "until", "unusual", "unveil",
	"update", "upgrade", "uphold", "upon", "upper", "upset", "urban", "usage",
	"useful", "useless", "usual", "utility", "vacant", "vacuum", "vague", "valid",
	"valley", "valve", "van", "vanish", "vapor", "various", "vast", "vault",
	"vehicle", "velvet", "vendor", "venture", "venue", "verb", "verify", "version",
	"very", "vessel", "veteran", "viable", "vibrant", "vicious", "victory", "video",
	"view", "village", "vintage", "violin", "virtual", "virus", "visa", "visit",
	"visual", "vital", "vivid", "vocal", "voice", "void", "volcano", "voyage",
	"wage", "wagon", "wait", "walk", "wall", "walnut", "wander", "want",
	"warfare", "warm", "warrior", "wash", "waste", "water", "wave", "way",
	"wealth", "weapon", "wear", "weasel", "weather", "web", "wedding", "weekend",
	"weird", "welcome", "west", "whale", "wheat", "wheel", "when", "where",
	"whip", "whisper", "wide", "width", "wife", "wild", "will", "win",
	"window", "wine", "wing", "wink", "winner", "winter", "wire", "wisdom",
	"wise", "wish", "witness", "wolf", "woman", "wonder", "wood", "wool",
	"word", "work", "world", "worry", "worth", "wrap", "wreck", "wrestle",
	"wrist", "write", "wrong", "yard", "year", "yellow", "young", "youth",
	"zebra", "zero", "zone", "zoo",
}
