package cognitive

import (
	"fmt"
	"math/rand"
	"regexp"
	"strings"
	"time"
)

// -----------------------------------------------------------------------
// Thinking Engine — the cognitive loop that makes Nous "think."
//
// This is what bridges the gap between a tool-router and an LLM.
// Instead of just looking up facts, the thinking engine:
//   1. Understands WHAT the user wants (task classification)
//   2. Figures out HOW to produce it (frame selection + parameter extraction)
//   3. GATHERS relevant material (graph, memory, web, context)
//   4. REASONS about it (decomposition, analogy, evaluation)
//   5. COMPOSES structured output (frame filling + text generation)
//
// The result: Nous can write emails, brainstorm, explain, compare,
// advise, teach, create — all without an LLM.
// -----------------------------------------------------------------------

// ThinkTask classifies what the user wants done.
type ThinkTask int

const (
	TaskCompose    ThinkTask = iota // write an email, letter, message
	TaskBrainstorm                 // generate ideas
	TaskAnalyze                    // break down, examine
	TaskTeach                      // explain for learning
	TaskAdvise                     // give recommendations
	TaskCompare                    // compare options
	TaskSummarize                  // condense information
	TaskCreate                     // creative writing (poem, story)
	TaskPlan                       // create a plan or strategy
	TaskDebate                     // argue a position
	TaskReflect                    // think through a problem
	TaskConverse                   // general conversation
)

func taskName(t ThinkTask) string {
	switch t {
	case TaskCompose:
		return "compose"
	case TaskBrainstorm:
		return "brainstorm"
	case TaskAnalyze:
		return "analyze"
	case TaskTeach:
		return "teach"
	case TaskAdvise:
		return "advise"
	case TaskCompare:
		return "compare"
	case TaskSummarize:
		return "summarize"
	case TaskCreate:
		return "create"
	case TaskPlan:
		return "plan"
	case TaskDebate:
		return "debate"
	case TaskReflect:
		return "reflect"
	case TaskConverse:
		return "converse"
	default:
		return "unknown"
	}
}

// TaskParams holds extracted parameters for a thinking task.
type TaskParams struct {
	Topic     string       // main topic or subject
	Purpose   string       // "asking for vacation", "about the project"
	Recipient string       // "my boss", "the team"
	Format    OutputFormat  // email, list, prose
	Tone      Tone     // formal, casual, etc.
	ItemA     string       // for comparisons
	ItemB     string       // for comparisons
	Keywords  []string     // extracted content words
	Audience  string       // "a 10-year-old", "beginners"
}

// ThoughtResult is the output of the thinking process.
type ThoughtResult struct {
	Text       string
	Task       ThinkTask
	Frame      string
	Trace      string // debugging: what the engine did
}

// ThinkContext provides conversation context to the thinking engine.
type ThinkContext struct {
	UserName     string
	RecentTopics []string
	History      []string
}

// ThinkingEngine is the cognitive loop that ties everything together.
type ThinkingEngine struct {
	graph      *CognitiveGraph
	composer   *Composer
	generative *GenerativeEngine
	discourse  *DiscoursePlanner
	rng        *rand.Rand

	// User preferences (learned over time)
	Preferences map[string]string
}

// NewThinkingEngine creates the cognitive loop.
func NewThinkingEngine(graph *CognitiveGraph, composer *Composer) *ThinkingEngine {
	te := &ThinkingEngine{
		graph:       graph,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
		Preferences: make(map[string]string),
	}
	if composer != nil {
		te.composer = composer
		te.generative = composer.Generative
		te.discourse = composer.Discourse
	}
	return te
}

// -----------------------------------------------------------------------
// Task Classification — what does the user want?
// -----------------------------------------------------------------------

var composeSignals = []string{
	"write me", "write an email", "write a letter", "write a message",
	"draft an email", "draft a letter", "draft a message",
	"compose an email", "compose a letter", "compose a message",
	"send an email", "send a message", "write to",
	"help me write", "can you write", "create an email",
	"write a casual email", "write a formal email",
	"email to my", "email to the",
}

var brainstormSignals = []string{
	"brainstorm", "ideas for", "give me ideas", "suggest ideas",
	"think of ideas", "come up with", "creative ideas",
	"what are some ideas", "help me think of", "list some ideas",
	"generate ideas", "ideas about", "name ideas", "names for",
}

var analyzeSignals = []string{
	"analyze", "break down", "examine", "dissect",
	"what's behind", "look into", "deep dive",
	"investigate", "explore why", "understand why",
}

var thinkTeachSignals = []string{
	"teach me", "explain to me", "help me understand",
	"explain like", "explain as if", "can you explain",
	"how does", "how do", "walk me through",
	"show me how", "help me learn", "introduction to",
	"what is the concept of", "what is ", "what are ",
	"tell me about", "tell me everything about", "tell me all about",
	"give me an overview of", "overview of", "deep dive into",
	"explain ",
}

var adviseSignals = []string{
	"what should i do", "advice", "advise me", "help me decide",
	"what do you recommend", "what would you suggest", "how should i",
	"i need help with", "i'm struggling with", "i don't know what to do",
	"any suggestions", "what do you think i should",
	"feeling overwhelmed", "feeling stuck", "feeling lost",
	"help deciding", "need help", "should i",
}

var compareSignals = []string{
	"compare", "versus", " vs ", "difference between",
	"which is better", "pros and cons", "advantages and disadvantages",
	"should i choose", "which one", "or should i",
	"what's the difference",
}

var summarizeSignals = []string{
	"summarize", "summary of", "sum up", "tldr", "tl;dr",
	"key points of", "main takeaways", "in a nutshell",
	"give me the gist", "brief overview",
}

var createSignals = []string{
	"write a poem", "write a story", "write a song",
	"create a poem", "create a story", "make up a story",
	"write something creative", "write me a haiku",
	"write a limerick", "compose a poem",
	"creative writing", "short story about",
	"create a haiku", "write me a poem",
	"create a limerick", "write verse",
}

var planSignals = []string{
	"help me plan", "create a plan", "plan for",
	"strategy for", "roadmap for", "how to approach",
	"action plan", "what's the best approach",
	"help me organize", "plan out",
}

var debateSignals = []string{
	"argue for", "argue against", "make the case",
	"defend the position", "debate", "why is it true that",
	"convince me", "prove that", "make an argument",
}

// classifyTask determines what kind of task the user wants.
func (te *ThinkingEngine) classifyTask(query string) ThinkTask {
	lower := strings.ToLower(query)

	if matchesAnySignal(lower, composeSignals) {
		return TaskCompose
	}
	if matchesAnySignal(lower, createSignals) {
		return TaskCreate
	}
	if matchesAnySignal(lower, brainstormSignals) {
		return TaskBrainstorm
	}
	if matchesAnySignal(lower, compareSignals) {
		return TaskCompare
	}
	if matchesAnySignal(lower, thinkTeachSignals) {
		return TaskTeach
	}
	if matchesAnySignal(lower, adviseSignals) {
		return TaskAdvise
	}
	if matchesAnySignal(lower, analyzeSignals) {
		return TaskAnalyze
	}
	if matchesAnySignal(lower, summarizeSignals) {
		return TaskSummarize
	}
	if matchesAnySignal(lower, planSignals) {
		return TaskPlan
	}
	if matchesAnySignal(lower, debateSignals) {
		return TaskDebate
	}

	return TaskConverse
}

func matchesAnySignal(text string, signals []string) bool {
	for _, s := range signals {
		if strings.Contains(text, s) {
			return true
		}
	}
	return false
}

// -----------------------------------------------------------------------
// Parameter Extraction — what are the details?
// -----------------------------------------------------------------------

var recipientRe = regexp.MustCompile(`(?i)(?:to|for)\s+(?:my\s+)?(\w+)(?:\s+(?:about|asking|regarding|on))`)
var aboutRe = regexp.MustCompile(`(?i)(?:about|regarding|for|on)\s+(.+?)(?:\s*[.!?]?\s*$)`)
var askingRe = regexp.MustCompile(`(?i)asking\s+(?:for|about|if)\s+(.+?)(?:\s*[.!?]?\s*$)`)
var audienceRe = regexp.MustCompile(`(?i)(?:to|for|like)\s+(?:a\s+)?([\w-]+(?:\s+[\w-]+)?)`)
var vsRe = regexp.MustCompile(`(?i)(\w[\w\s]*?)\s+(?:vs\.?|versus|or|compared to)\s+(\w[\w\s]*?)(?:\s*[.!?]?\s*$)`)

func (te *ThinkingEngine) extractParams(query string, task ThinkTask) *TaskParams {
	params := &TaskParams{
		Format: InferFormat(query),
	}

	lower := strings.ToLower(query)

	// Extract recipient
	if m := recipientRe.FindStringSubmatch(lower); len(m) > 1 {
		recipient := strings.TrimSpace(m[1])
		// Don't capture common false positives
		if recipient != "me" && recipient != "that" && recipient != "this" &&
			recipient != "it" && recipient != "ideas" && recipient != "some" {
			params.Recipient = recipient
		}
	}

	// Infer tone from recipient and explicit hints
	params.Tone = InferTone(query, params.Recipient)

	// Extract purpose (asking for X)
	if m := askingRe.FindStringSubmatch(lower); len(m) > 1 {
		params.Purpose = strings.TrimSpace(m[1])
	}

	// Extract topic (about X)
	if params.Purpose == "" {
		if m := aboutRe.FindStringSubmatch(lower); len(m) > 1 {
			params.Topic = cleanTopic(m[1])
		}
	}

	// For comparisons, extract both items
	if task == TaskCompare {
		if m := vsRe.FindStringSubmatch(query); len(m) > 2 {
			params.ItemA = strings.TrimSpace(m[1])
			params.ItemB = strings.TrimSpace(m[2])
		}
	}

	// Extract audience for teaching
	if task == TaskTeach {
		if m := audienceRe.FindStringSubmatch(lower); len(m) > 1 {
			aud := strings.TrimSpace(m[1])
			if strings.Contains(aud, "year") || strings.Contains(aud, "beginner") ||
				strings.Contains(aud, "child") || strings.Contains(aud, "expert") {
				params.Audience = aud
			}
		}
	}

	// Fallback topic: extract content words from the query
	if params.Topic == "" && params.Purpose == "" {
		params.Topic = extractTopicFromQuery(lower)
	}

	// For teach/analyze/summarize tasks, try harder to extract a clean topic
	if task == TaskTeach || task == TaskAnalyze || task == TaskSummarize {
		cleaned := extractTeachTopic(lower)
		if cleaned != "" {
			params.Topic = cleaned
		}
	}

	// For advice tasks, extract the core problem
	if task == TaskAdvise {
		cleaned := extractAdviceTopic(lower)
		if cleaned != "" {
			params.Topic = cleaned
		}
	}

	// Extract keywords for content generation
	params.Keywords = extractContentWords(lower)

	return params
}

// cleanTopic strips common prefix junk from extracted topics.
func cleanTopic(topic string) string {
	topic = strings.TrimSpace(topic)
	for _, prefix := range []string{"the ", "a ", "an ", "some ", "my "} {
		topic = strings.TrimPrefix(topic, prefix)
	}
	return topic
}

// extractTopicFromQuery pulls the main content words from a query.
func extractTopicFromQuery(lower string) string {
	// Strip common signal words
	strip := []string{
		"write me", "write an", "write a", "help me", "can you",
		"please", "i want", "i need", "could you", "would you",
		"tell me about", "explain", "teach me about", "brainstorm",
		"ideas for", "create a", "compose a", "draft a",
		"email", "letter", "message", "poem", "story",
	}
	result := lower
	for _, s := range strip {
		result = strings.ReplaceAll(result, s, "")
	}
	result = strings.TrimSpace(result)
	result = strings.Trim(result, "?!.,")
	return strings.TrimSpace(result)
}

// extractTeachTopic extracts a clean topic from teaching/explanation queries.
func extractTeachTopic(lower string) string {
	// Direct patterns: "teach me about X", "explain X to me", "help me understand X"
	patterns := []struct {
		prefix string
		suffix string
	}{
		{"teach me about ", ""},
		{"teach me ", ""},
		{"explain ", " to me"},
		{"explain ", " like"},
		{"explain ", " as if"},
		{"explain ", ""},
		{"help me understand ", ""},
		{"tell me about ", ""},
		{"walk me through ", ""},
		{"show me how ", " works"},
		{"how does ", " work"},
		{"how do ", " work"},
		{"summarize ", ""},
		{"summarize the key principles of ", ""},
		{"give me the gist of ", ""},
		{"introduction to ", ""},
	}

	for _, p := range patterns {
		if strings.HasPrefix(lower, p.prefix) {
			topic := strings.TrimPrefix(lower, p.prefix)
			if p.suffix != "" {
				topic = strings.TrimSuffix(topic, p.suffix)
			}
			topic = strings.Trim(topic, "?!. ")
			topic = strings.TrimPrefix(topic, "the ")
			topic = strings.TrimPrefix(topic, "basics of ")
			// Strip trailing style modifiers: "simply", "briefly", "in detail"
			topic = stripTopicModifiers(topic)
			if topic != "" {
				return topic
			}
		}
	}
	return ""
}

// stripTopicModifiers removes trailing style modifiers from topic strings.
// "quantum physics simply" → "quantum physics"
// "gravity in simple terms" → "gravity"
func stripTopicModifiers(topic string) string {
	suffixes := []string{
		" simply", " briefly", " concisely", " easily",
		" in simple terms", " in plain english", " in detail",
		" in a nutshell", " for beginners", " for dummies",
		" like i'm 5", " like i'm five", " eli5",
		" for me", " to me",
	}
	lower := strings.ToLower(topic)
	for _, s := range suffixes {
		if strings.HasSuffix(lower, s) {
			return strings.TrimSpace(topic[:len(topic)-len(s)])
		}
	}
	return topic
}

// extractAdviceTopic extracts the core problem from advice queries.
func extractAdviceTopic(lower string) string {
	// Strip common advice scaffolding
	prefixes := []string{
		"what should i do about ", "what should i do with ",
		"i need help with ", "i need help deciding ",
		"help me with ", "help me decide about ",
		"any suggestions for ", "i'm struggling with ",
		"i'm feeling ", "i feel ", "feeling ",
		"how should i handle ", "how should i deal with ",
		"what do you recommend for ", "advice on ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			topic := strings.TrimPrefix(lower, p)
			// Also strip trailing question parts
			if idx := strings.Index(topic, ", what"); idx > 0 {
				topic = topic[:idx]
			}
			if idx := strings.Index(topic, "? "); idx > 0 {
				topic = topic[:idx]
			}
			return strings.Trim(topic, "?!. ")
		}
	}
	// Try extracting after "about"
	if idx := strings.Index(lower, " about "); idx > 0 {
		topic := lower[idx+7:]
		topic = strings.Trim(topic, "?!. ")
		return topic
	}
	return ""
}

// extractContentWords pulls meaningful words from a query.
func extractContentWords(lower string) []string {
	stopWords := map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true,
		"to": true, "for": true, "of": true, "in": true, "on": true,
		"at": true, "by": true, "with": true, "from": true,
		"and": true, "or": true, "but": true, "not": true,
		"i": true, "me": true, "my": true, "you": true, "your": true,
		"it": true, "its": true, "this": true, "that": true,
		"can": true, "could": true, "would": true, "should": true,
		"do": true, "does": true, "did": true, "will": true,
		"have": true, "has": true, "had": true,
		"what": true, "how": true, "why": true, "which": true,
		"write": true, "help": true, "please": true, "tell": true,
		"some": true, "about": true, "give": true, "make": true,
		// Task signal words — don't treat these as content
		"brainstorm": true, "ideas": true, "explain": true, "teach": true,
		"compare": true, "summarize": true, "create": true, "compose": true,
		"draft": true, "plan": true, "analyze": true, "debate": true,
		"suggest": true, "recommend": true, "think": true, "come": true,
		"up": true, "email": true, "letter": true, "message": true,
		"poem": true, "story": true, "haiku": true,
	}

	var words []string
	for _, w := range strings.Fields(lower) {
		w = strings.Trim(w, "?!.,;:")
		if len(w) > 1 && !stopWords[w] {
			words = append(words, w)
		}
	}
	return words
}

// -----------------------------------------------------------------------
// The Cognitive Loop — Think()
// -----------------------------------------------------------------------

// Think processes a query through the full cognitive loop.
// This is the main entry point for the thinking engine.
func (te *ThinkingEngine) Think(query string, ctx *ThinkContext) *ThoughtResult {
	// 1. Classify the task
	task := te.classifyTask(query)

	// 2. Extract parameters
	params := te.extractParams(query, task)

	// 3. Select the output frame
	frame := SelectFrame(task, params.Format)

	// 4. Generate content for each section
	var sections []string
	var trace strings.Builder
	trace.WriteString(fmt.Sprintf("Task: %s | Frame: %s | Topic: %q | Tone: %d\n",
		taskName(task), frame.Name, params.Topic, params.Tone))

	for _, sec := range frame.Sections {
		content := te.generateSectionContent(query, task, params, sec, ctx)
		if content != "" {
			sections = append(sections, content)
			trace.WriteString(fmt.Sprintf("  [%s] %s → %d chars\n",
				sec.Role, sec.Goal, len(content)))
		} else if sec.Required && task != TaskCreate {
			// Required section couldn't be filled — generate fallback.
			// Skip for creative tasks (poems/stories produce all content in one section).
			fb := te.fallbackContent(params, sec)
			if fb != "" {
				sections = append(sections, fb)
			}
		}
	}

	// 5. Assemble the baseline response
	text := te.assemble(sections, frame)

	// 6. Non-LLM semantic reranking for knowledge-heavy tasks.
	if te.shouldUsePlanRerank(task) {
		plan := te.BuildContentPlan(query, task, params)
		candidates := te.BuildPlanCandidates(query, task, plan, frame, text)
		if len(candidates) > 0 {
			text = te.SelectBestCandidate(plan, candidates)
			trace.WriteString(fmt.Sprintf("Plan rerank: %d candidates selected\n", len(candidates)))
		}
	}

	return &ThoughtResult{
		Text:  text,
		Task:  task,
		Frame: frame.Name,
		Trace: trace.String(),
	}
}

func (te *ThinkingEngine) shouldUsePlanRerank(task ThinkTask) bool {
	switch task {
	case TaskTeach, TaskAnalyze, TaskSummarize, TaskDebate, TaskCompare, TaskConverse:
		return true
	default:
		return false
	}
}

// CanHandle returns true if this query is something the thinking engine
// should handle (vs falling through to simple Composer responses).
func (te *ThinkingEngine) CanHandle(query string) bool {
	task := te.classifyTask(query)
	return task != TaskConverse
}

// assemble combines sections into final output based on the frame.
func (te *ThinkingEngine) assemble(sections []string, frame *Frame) string {
	if len(sections) == 0 {
		return ""
	}

	switch frame.Name {
	case "email":
		// Email: sections separated by blank lines
		return strings.Join(sections, "\n\n")
	case "brainstorm":
		return strings.Join(sections, "\n\n")
	case "comparison":
		return strings.Join(sections, "\n\n")
	case "tutorial":
		return strings.Join(sections, "\n\n")
	case "poem":
		return strings.Join(sections, "\n")
	default:
		return strings.Join(sections, " ")
	}
}

// -----------------------------------------------------------------------
// Section Content Generation — the heart of the thinking engine
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateSectionContent(query string, task ThinkTask, params *TaskParams, sec FrameSection, ctx *ThinkContext) string {
	switch task {
	case TaskCompose:
		return te.generateComposeSection(params, sec, ctx)
	case TaskBrainstorm:
		return te.generateBrainstormSection(params, sec)
	case TaskTeach, TaskAnalyze:
		return te.generateExplainSection(query, params, sec)
	case TaskAdvise:
		return te.generateAdviceSection(params, sec)
	case TaskCompare:
		return te.generateCompareSection(params, sec)
	case TaskSummarize:
		return te.generateSummarySection(query, params, sec)
	case TaskCreate:
		return te.generateCreativeSection(query, params, sec)
	case TaskPlan:
		return te.generatePlanSection(params, sec)
	case TaskDebate:
		return te.generateDebateSection(query, params, sec)
	default:
		return te.generateConversational(query, params, sec)
	}
}

// -----------------------------------------------------------------------
// Email Composition
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateComposeSection(params *TaskParams, sec FrameSection, ctx *ThinkContext) string {
	switch sec.Role {
	case "greeting":
		recipient := params.Recipient
		if recipient == "" {
			recipient = "[Recipient]"
		} else {
			recipient = capitalizeFirst(recipient)
		}
		return fmt.Sprintf(pickFromTone(greetingsByTone, params.Tone, te.rng), recipient)

	case "opening":
		opener := pickFromTone(emailOpenersByTone, params.Tone, te.rng)
		purpose := params.Purpose
		if purpose == "" {
			purpose = params.Topic
		}
		if purpose != "" {
			purposeTemplate := te.pick(purposeTemplates)
			return opener + " " + fmt.Sprintf(purposeTemplate, purpose)
		}
		return opener

	case "body":
		// Generate body from purpose/topic
		topic := params.Purpose
		if topic == "" {
			topic = params.Topic
		}
		if topic == "" {
			return "I would appreciate the opportunity to discuss this matter further."
		}

		// Try to pull relevant knowledge
		body := te.generateTopicContent(topic, params.Tone, 2)
		if body != "" {
			return body
		}

		// Template-based body generation
		lines := []string{
			fmt.Sprintf("Regarding %s, I wanted to share my thoughts.", topic),
		}
		if params.Tone == ToneFormal {
			lines = append(lines, "I believe this is an important matter that deserves careful consideration.")
		} else {
			lines = append(lines, "I think this is worth talking about.")
		}
		return strings.Join(lines, " ")

	case "closing":
		return pickFromTone(emailClosersByTone, params.Tone, te.rng)

	case "signoff":
		signoff := pickFromTone(signoffsByTone, params.Tone, te.rng)
		name := ""
		if ctx != nil && ctx.UserName != "" {
			name = ctx.UserName
		}
		if name != "" {
			return signoff + "\n" + name
		}
		return signoff
	}
	return ""
}

// -----------------------------------------------------------------------
// Brainstorming
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateBrainstormSection(params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}
	if topic == "" {
		topic = "this topic"
	}

	switch sec.Role {
	case "context":
		return fmt.Sprintf("Here are some ideas for %s:", topic)

	case "ideas":
		ideas := te.brainstormIdeas(topic, params.Keywords)
		if len(ideas) == 0 {
			return te.pick([]string{
				fmt.Sprintf("Consider different angles on %s.", topic),
				fmt.Sprintf("Think about what makes %s unique and build from there.", topic),
			})
		}
		var lines []string
		for i, idea := range ideas {
			lines = append(lines, fmt.Sprintf("%d. %s", i+1, idea))
		}
		return strings.Join(lines, "\n")

	case "synthesis":
		return te.pick([]string{
			"The most promising ideas are the ones that combine multiple angles.",
			"Consider which of these fits your specific situation best.",
			"Start with the approach that excites you most — momentum matters.",
			"Mix and match from these ideas to create something unique.",
		})
	}
	return ""
}

// brainstormIdeas generates a list of ideas for a topic.
func (te *ThinkingEngine) brainstormIdeas(topic string, keywords []string) []string {
	var ideas []string
	seen := make(map[string]bool)

	addIdea := func(idea string) {
		lower := strings.ToLower(idea)
		if !seen[lower] {
			seen[lower] = true
			ideas = append(ideas, idea)
		}
	}

	// 1. Knowledge graph exploration — if we know about the topic
	if te.graph != nil {
		topicFacts := te.gatherTopicFacts(topic)
		for _, f := range topicFacts {
			switch f.Relation {
			case RelUsedFor:
				addIdea(fmt.Sprintf("Explore the connection to %s", f.Object))
			case RelHas:
				addIdea(fmt.Sprintf("Build on the %s aspect", f.Object))
			case RelRelatedTo:
				addIdea(fmt.Sprintf("Look into how %s connects to %s", topic, f.Object))
			case RelIsA:
				addIdea(fmt.Sprintf("Consider other %s for inspiration", f.Object))
			}
		}
	}

	// 2. Keyword-derived ideas using templates
	ideaTemplates := []string{
		"A %s-focused approach with a fresh perspective",
		"Combine %s with something unexpected",
		"The minimalist take: strip %s down to essentials",
		"The ambitious version: go all-in on %s",
		"A collaborative angle involving %s",
		"The practical solution: focus on what works for %s",
		"A creative twist on traditional %s",
		"The modern approach to %s",
		"Something hands-on and interactive around %s",
		"A unique spin that challenges assumptions about %s",
	}

	// Generate ideas from keywords
	for _, kw := range keywords {
		if len(ideas) >= 10 {
			break
		}
		tmpl := ideaTemplates[te.rng.Intn(len(ideaTemplates))]
		addIdea(fmt.Sprintf(tmpl, kw))
	}

	// 3. If still few ideas, use generic creative prompts
	if len(ideas) < 5 {
		genericIdeas := []string{
			fmt.Sprintf("Start small and iterate: begin with a simple version of %s", topic),
			fmt.Sprintf("Look at how others approach %s and find gaps", topic),
			fmt.Sprintf("Flip the problem: what's the opposite of %s?", topic),
			fmt.Sprintf("Combine two unrelated things with %s for novelty", topic),
			fmt.Sprintf("Ask: who else needs something related to %s?", topic),
			fmt.Sprintf("Consider the long-term vision for %s", topic),
			fmt.Sprintf("What would %s look like in 5 years?", topic),
		}
		for _, idea := range genericIdeas {
			if len(ideas) >= 8 {
				break
			}
			addIdea(idea)
		}
	}

	// Cap at 10 ideas
	if len(ideas) > 10 {
		ideas = ideas[:10]
	}

	return ideas
}

// -----------------------------------------------------------------------
// Explanation / Teaching
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateExplainSection(query string, params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}
	if topic == "" {
		topic = "this topic"
	}

	switch sec.Role {
	// Explanation frame roles
	case "hook":
		// Skip generic hooks — go straight to substance. If we have
		// real knowledge, the definition section speaks for itself.
		return ""

	case "definition":
		content := te.generateTopicContent(topic, ToneNeutral, 4)
		if content != "" {
			return content
		}
		// Try individual keywords if compound topic failed
		// (e.g., "quantum mechanics" → try "quantum", "mechanics")
		for _, kw := range params.Keywords {
			if len(kw) > 3 {
				content = te.generateTopicContent(kw, ToneNeutral, 4)
				if content != "" {
					return content
				}
			}
		}
		// If we found related facts via keywords, they were used above.
		// Final resort: generate an honest admission of limited knowledge.
		essence := te.inferEssence(topic, params.Keywords)
		if essence != "" {
			return fmt.Sprintf("%s is fundamentally about %s.", capitalizeFirst(topic), essence)
		}
		return fmt.Sprintf("I don't have detailed knowledge about %s yet, but I can learn about it if you point me to a source.", topic)

	case "mechanism":
		content := te.generateTopicContent(topic, ToneNeutral, 2)
		if content != "" {
			return content
		}
		return ""

	case "example":
		if te.graph != nil {
			facts := te.gatherTopicFacts(topic)
			for _, f := range facts {
				if f.Relation == RelUsedFor {
					return fmt.Sprintf("For example, %s is used in %s.", topic, f.Object)
				}
			}
		}
		return ""

	case "significance":
		// Only include if we have something specific to say
		return ""

	// Tutorial frame roles
	case "goal":
		// Skip generic hooks — the content speaks for itself
		return ""

	case "prerequisites":
		return ""

	case "steps":
		// Main content: pull knowledge about the topic
		content := te.generateTopicContent(topic, ToneNeutral, 6)
		if content != "" {
			return content
		}
		// Try individual keywords for compound topics
		for _, kw := range params.Keywords {
			if len(kw) > 3 {
				content = te.generateTopicContent(kw, ToneNeutral, 6)
				if content != "" {
					return content
				}
			}
		}
		return fmt.Sprintf("I don't have detailed knowledge about %s yet. Try asking me to look it up, or point me to a source I can learn from.", topic)

	case "tips":
		return ""

	case "next_steps":
		return ""
	}
	return ""
}

// inferEssence generates a brief essence of a topic from keywords.
func (te *ThinkingEngine) inferEssence(topic string, keywords []string) string {
	// Check graph for is_a relation
	if te.graph != nil {
		facts := te.gatherTopicFacts(topic)
		for _, f := range facts {
			if f.Relation == RelIsA {
				return f.Object
			}
		}
	}
	// Fallback: use the topic itself
	return "understanding its fundamental nature and applications"
}

// -----------------------------------------------------------------------
// Advice
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateAdviceSection(params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}

	switch sec.Role {
	case "empathy":
		return te.pick(adviceOpeners)

	case "analysis":
		return te.pick([]string{
			fmt.Sprintf("When it comes to %s, there are a few things worth considering.", topic),
			fmt.Sprintf("Looking at %s from different angles helps clarify the situation.", topic),
			fmt.Sprintf("The key to handling %s is breaking it into manageable pieces.", topic),
		})

	case "suggestions":
		suggestions := te.generateSuggestions(topic, params.Keywords)
		var lines []string
		for i, s := range suggestions {
			lines = append(lines, fmt.Sprintf("%d. %s", i+1, s))
		}
		return strings.Join(lines, "\n")

	case "encouragement":
		return te.pick(encouragements)
	}
	return ""
}

func (te *ThinkingEngine) generateSuggestions(topic string, keywords []string) []string {
	// Clean the topic — don't insert raw queries like "i'm bored, what should i do"
	// into templates. Extract the core subject.
	clean := extractTopicFromQuery(topic)
	if clean == "" || len(clean) > 40 {
		clean = "this"
	}

	suggestions := []string{
		fmt.Sprintf("Start by clarifying what matters most to you about %s.", clean),
		fmt.Sprintf("Break %s into smaller, concrete steps you can tackle one at a time.", clean),
		fmt.Sprintf("Talk to someone who has experience with %s — perspective helps.", clean),
		fmt.Sprintf("Set a specific, achievable goal related to %s for this week.", clean),
		fmt.Sprintf("Give yourself permission to experiment with %s without pressure.", clean),
	}
	return suggestions
}

// -----------------------------------------------------------------------
// Comparison
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateCompareSection(params *TaskParams, sec FrameSection) string {
	a := params.ItemA
	b := params.ItemB
	if a == "" {
		a = "the first option"
	}
	if b == "" {
		b = "the second option"
	}

	switch sec.Role {
	case "intro":
		return fmt.Sprintf("Let's compare %s and %s to see how they stack up.", a, b)

	case "item_a":
		content := te.generateTopicContent(a, ToneNeutral, 2)
		if content != "" {
			return content
		}
		return fmt.Sprintf("%s has its own strengths and characteristics worth understanding.", capitalizeFirst(a))

	case "item_b":
		content := te.generateTopicContent(b, ToneNeutral, 2)
		if content != "" {
			return content
		}
		return fmt.Sprintf("%s brings a different set of qualities to the table.", capitalizeFirst(b))

	case "differences":
		// Try graph-based comparison
		if te.graph != nil {
			factsA := te.gatherTopicFacts(a)
			factsB := te.gatherTopicFacts(b)
			if len(factsA) > 0 && len(factsB) > 0 {
				return te.composeDifferences(a, b, factsA, factsB)
			}
		}
		return fmt.Sprintf("The main difference comes down to their core purpose and approach. %s and %s each solve different problems in different ways.",
			capitalizeFirst(a), capitalizeFirst(b))

	case "verdict":
		return te.pick([]string{
			fmt.Sprintf("Neither %s nor %s is universally better — it depends on your needs.", a, b),
			fmt.Sprintf("The best choice between %s and %s depends on what you're optimizing for.", a, b),
			"Consider what matters most to you, and the right choice will become clear.",
		})
	}
	return ""
}

func (te *ThinkingEngine) composeDifferences(a, b string, factsA, factsB []edgeFact) string {
	var diffs []string

	propsA := make(map[string]string)
	propsB := make(map[string]string)
	for _, f := range factsA {
		propsA[string(f.Relation)] = f.Object
	}
	for _, f := range factsB {
		propsB[string(f.Relation)] = f.Object
	}

	// Find properties unique to each
	for rel, val := range propsA {
		if _, ok := propsB[rel]; !ok {
			diffs = append(diffs, fmt.Sprintf("%s is known for %s", capitalizeFirst(a), val))
		}
	}
	for rel, val := range propsB {
		if _, ok := propsA[rel]; !ok {
			diffs = append(diffs, fmt.Sprintf("%s stands out with %s", capitalizeFirst(b), val))
		}
	}

	if len(diffs) > 4 {
		diffs = diffs[:4]
	}

	if len(diffs) == 0 {
		return fmt.Sprintf("Both %s and %s have their own unique qualities.", a, b)
	}

	return strings.Join(diffs, ". ") + "."
}

// -----------------------------------------------------------------------
// Summary
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateSummarySection(query string, params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}

	switch sec.Role {
	case "overview":
		return fmt.Sprintf("Here's a summary of the key points about %s.", topic)

	case "key_points":
		content := te.generateTopicContent(topic, ToneNeutral, 4)
		if content != "" {
			return content
		}
		return fmt.Sprintf("The essential aspects of %s deserve focused attention.", topic)

	case "conclusion":
		return fmt.Sprintf("In short, %s is worth understanding in depth.", topic)
	}
	return ""
}

// -----------------------------------------------------------------------
// Creative Writing (Poems, Stories)
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateCreativeSection(query string, params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}
	if topic == "" {
		topic = "life"
	}

	lower := strings.ToLower(query)

	// Poem detection — generate full poem in the first section, skip rest
	if strings.Contains(lower, "poem") || strings.Contains(lower, "haiku") ||
		strings.Contains(lower, "verse") || strings.Contains(lower, "limerick") {
		if sec.Role == "setup" {
			return te.generatePoem(topic, lower)
		}
		return "" // poem is complete in the setup section
	}

	// Story
	switch sec.Role {
	case "setup":
		return te.pick([]string{
			fmt.Sprintf("There was a time when %s changed everything.", topic),
			fmt.Sprintf("It began, as most things do, with a question about %s.", topic),
			fmt.Sprintf("Nobody expected %s to matter as much as it did.", topic),
		})
	case "development":
		return te.pick([]string{
			fmt.Sprintf("As the days passed, the reality of %s became impossible to ignore. What had seemed simple revealed layers of complexity. Those who paid attention found themselves drawn deeper.", topic),
			fmt.Sprintf("The thing about %s is that it never stays still. Each new discovery opened doors that no one knew existed. And with each door came a choice.", topic),
		})
	case "resolution":
		return te.pick([]string{
			fmt.Sprintf("In the end, %s taught something that no book could capture: that understanding comes not from answers, but from better questions.", topic),
			fmt.Sprintf("Looking back, %s wasn't just a chapter — it was a turning point. And those who lived through it were never quite the same.", topic),
		})
	}
	return ""
}

// generatePoem creates verse from grammar rules and word pools.
func (te *ThinkingEngine) generatePoem(topic string, query string) string {
	if strings.Contains(query, "haiku") {
		return te.generateHaiku(topic)
	}
	return te.generateFreeVerse(topic)
}

// generateHaiku creates a 5-7-5 haiku.
func (te *ThinkingEngine) generateHaiku(topic string) string {
	// Haiku pool organized by theme
	adjectives := []string{"silent", "gentle", "ancient", "bright", "deep",
		"still", "wild", "soft", "vast", "true", "warm", "cold"}
	nouns := []string{"light", "wind", "dream", "voice", "wave",
		"stone", "leaf", "rain", "dawn", "dusk", "path", "shade"}
	verbs := []string{"flows", "shines", "drifts", "turns", "falls",
		"rests", "grows", "fades", "calls", "waits", "holds", "bends"}
	naturals := []string{"mountain", "river", "moonlight", "starlight",
		"ocean", "forest", "meadow", "sunset", "morning", "silence"}

	adj1 := te.pick(adjectives)
	adj2 := te.pick(adjectives)
	noun1 := te.pick(nouns)
	noun2 := te.pick(nouns)
	verb := te.pick(verbs)
	natural := te.pick(naturals)

	line1 := fmt.Sprintf("%s %s, %s", capitalizeFirst(adj1), topic, adj2)
	line2 := fmt.Sprintf("%s %s through the %s", capitalizeFirst(noun1), verb, natural)
	line3 := fmt.Sprintf("%s becomes %s", capitalizeFirst(topic), noun2)

	return line1 + "\n" + line2 + "\n" + line3
}

// generateFreeVerse creates a free verse poem.
func (te *ThinkingEngine) generateFreeVerse(topic string) string {
	stanzaTemplates := [][]string{
		{
			"%s is not what you think.",
			"It is the space between knowing and being,",
			"the breath before the word is spoken.",
		},
		{
			"When I think of %s,",
			"I think of rivers that carve stone,",
			"patient, relentless, certain.",
		},
		{
			"There is a silence in %s",
			"that speaks louder than any voice —",
			"a truth that doesn't need defending.",
		},
		{
			"Like water finding its level,",
			"%s settles into what it was always meant to be:",
			"simple, necessary, whole.",
		},
		{
			"They say %s is complicated.",
			"But the complicated things are often just",
			"simple things we haven't looked at long enough.",
		},
	}

	// Pick 2-3 unique stanzas
	indices := te.rng.Perm(len(stanzaTemplates))
	numStanzas := 2 + te.rng.Intn(2) // 2-3
	if numStanzas > len(indices) {
		numStanzas = len(indices)
	}

	var stanzas []string
	for i := 0; i < numStanzas; i++ {
		tmpl := stanzaTemplates[indices[i]]
		var lines []string
		for _, line := range tmpl {
			if strings.Contains(line, "%s") {
				lines = append(lines, fmt.Sprintf(line, topic))
			} else {
				lines = append(lines, line)
			}
		}
		stanzas = append(stanzas, strings.Join(lines, "\n"))
	}

	return strings.Join(stanzas, "\n\n")
}

// -----------------------------------------------------------------------
// Planning
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generatePlanSection(params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}

	switch sec.Role {
	case "objective":
		return fmt.Sprintf("Here's a plan for %s.", topic)

	case "phases":
		phases := []string{
			fmt.Sprintf("Phase 1 — Research: Understand the landscape of %s. Look at what exists, what works, and what doesn't.", topic),
			fmt.Sprintf("Phase 2 — Define: Clarify your specific goals and constraints for %s. What does success look like?", topic),
			fmt.Sprintf("Phase 3 — Build: Start with the simplest viable approach to %s. Iterate from there.", topic),
			fmt.Sprintf("Phase 4 — Test: Validate your approach to %s against real conditions. Adjust based on feedback.", topic),
			fmt.Sprintf("Phase 5 — Refine: Polish and optimize your %s based on what you learned.", topic),
		}
		return strings.Join(phases, "\n")

	case "considerations":
		return te.pick([]string{
			"Keep scope manageable — it's better to do less well than more poorly.",
			"Build in checkpoints to reassess progress and adjust course.",
			"Don't aim for perfection in the first iteration.",
		})

	case "timeline":
		return "Adjust the timeline based on your available time and the complexity of each phase."
	}
	return ""
}

// -----------------------------------------------------------------------
// Debate / Argument
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateDebateSection(query string, params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}

	switch sec.Role {
	case "thesis":
		return fmt.Sprintf("The case for %s rests on several strong foundations.", topic)

	case "evidence":
		content := te.generateTopicContent(topic, ToneAcademic, 3)
		if content != "" {
			return "Consider the evidence. " + content
		}
		return fmt.Sprintf("The strength of %s lies in its practical value, its proven track record, and its alignment with real-world needs.", topic)

	case "counterpoint":
		return fmt.Sprintf("Of course, no position is without challenges. Critics of %s raise valid concerns that deserve honest consideration. But the weight of evidence still favors the core argument.", topic)

	case "conclusion":
		return fmt.Sprintf("On balance, %s holds up under scrutiny. The key is to engage with it honestly, weigh the evidence, and let the facts guide the conclusion.", topic)
	}
	return ""
}

// -----------------------------------------------------------------------
// Conversational Fallback
// -----------------------------------------------------------------------

func (te *ThinkingEngine) generateConversational(query string, params *TaskParams, sec FrameSection) string {
	topic := params.Topic
	if topic == "" {
		topic = strings.Join(params.Keywords, " ")
	}

	// Try knowledge graph first
	content := te.generateTopicContent(topic, ToneNeutral, 3)
	if content != "" {
		return content
	}

	return te.pick([]string{
		fmt.Sprintf("That's an interesting area. %s has a lot of dimensions worth exploring.", topic),
		fmt.Sprintf("When it comes to %s, context really matters. What angle are you most interested in?", topic),
		fmt.Sprintf("I can see why %s is on your mind. Let me share what I know.", topic),
	})
}

// -----------------------------------------------------------------------
// Content Generation Helpers
// -----------------------------------------------------------------------

// generateTopicContent pulls knowledge from the graph and generates prose.
// Uses clean fact realization rather than the discourse planner, which
// produces grammatically awkward output from its template system.
func (te *ThinkingEngine) generateTopicContent(topic string, tone Tone, maxFacts int) string {
	if te.composer == nil || te.graph == nil {
		return ""
	}

	facts, _ := te.composer.gatherFacts(topic)
	if len(facts) == 0 {
		return ""
	}
	if len(facts) > maxFacts {
		facts = facts[:maxFacts]
	}

	// Build a clean response: description (if available) + fact sentences.
	var parts []string

	// Lead with Wikipedia description
	if desc := te.graph.LookupDescription(topic); len(desc) >= 40 {
		parts = append(parts, desc)
	}

	// Add facts as clean sentences
	factText := te.composer.structuredRealization(facts)
	if factText != "" {
		parts = append(parts, factText)
	}

	if len(parts) > 0 {
		return strings.Join(parts, "\n\n")
	}

	// Last resort: basic fact list
	return te.composer.realizeFacts(facts)
}

// gatherTopicFacts gets facts about a topic from the graph.
func (te *ThinkingEngine) gatherTopicFacts(topic string) []edgeFact {
	if te.composer == nil || te.graph == nil {
		return nil
	}
	facts, _ := te.composer.gatherFacts(topic)
	if len(facts) > 0 {
		return facts
	}

	// Fallback for short labels (e.g. "go", "ai") that may be filtered by
	// keyword extraction in gatherFacts.
	lower := strings.ToLower(strings.TrimSpace(topic))
	if lower == "" {
		return nil
	}

	te.graph.mu.RLock()
	defer te.graph.mu.RUnlock()

	ids := te.graph.byLabel[lower]
	if len(ids) == 0 {
		return nil
	}

	seen := make(map[string]bool)
	for _, id := range ids {
		node := te.graph.nodes[id]
		if node == nil {
			continue
		}
		for _, edge := range te.graph.outEdges[id] {
			if edge.Relation == RelDescribedAs {
				continue
			}
			to := te.graph.nodes[edge.To]
			if to == nil {
				continue
			}
			key := node.Label + "|" + string(edge.Relation) + "|" + to.Label
			if seen[key] {
				continue
			}
			seen[key] = true
			facts = append(facts, edgeFact{
				Subject:  node.Label,
				Relation: edge.Relation,
				Object:   to.Label,
				Inferred: edge.Inferred,
			})
		}
		if len(facts) > 0 {
			break
		}
	}

	return facts
}

// fallbackContent generates minimal content for required sections.
// Only produces content for opening sections — body/closing filler sounds robotic.
func (te *ThinkingEngine) fallbackContent(params *TaskParams, sec FrameSection) string {
	return ""
}

func (te *ThinkingEngine) pick(options []string) string {
	if len(options) == 0 {
		return ""
	}
	return options[te.rng.Intn(len(options))]
}

// -----------------------------------------------------------------------
// Analogy Engine — creative connections via embeddings
// -----------------------------------------------------------------------

// GenerateAnalogy creates "X is like Y because..." connections.
func (te *ThinkingEngine) GenerateAnalogy(topic string) string {
	if te.graph == nil {
		return ""
	}

	topicFacts := te.gatherTopicFacts(topic)
	if len(topicFacts) == 0 {
		return ""
	}

	// Get the topic's category
	var topicCategory string
	for _, f := range topicFacts {
		if f.Relation == RelIsA {
			topicCategory = f.Object
			break
		}
	}
	if topicCategory == "" {
		return ""
	}

	// Find other things in the same category
	te.graph.mu.RLock()
	defer te.graph.mu.RUnlock()

	for _, node := range te.graph.nodes {
		if strings.ToLower(node.Label) == strings.ToLower(topic) {
			continue
		}
		// Check if this node is also in the same category
		nid := nodeID(node.Label)
		for _, edge := range te.graph.outEdges[nid] {
			to := te.graph.nodes[edge.To]
			if to == nil {
				continue
			}
			if edge.Relation == RelIsA && strings.ToLower(to.Label) == strings.ToLower(topicCategory) {
				// Found an analog! Find shared properties
				analogFacts := te.gatherTopicFactsUnlocked(node.Label)
				shared := findSharedObjects(topicFacts, analogFacts)
				if len(shared) > 0 {
					return fmt.Sprintf("%s is like %s — both share %s.",
						capitalizeFirst(topic), node.Label, shared[0])
				}
				return fmt.Sprintf("%s is like %s in that both are %s.",
					capitalizeFirst(topic), node.Label, topicCategory)
			}
		}
	}

	return ""
}

// gatherTopicFactsUnlocked gathers facts without acquiring the lock.
func (te *ThinkingEngine) gatherTopicFactsUnlocked(topic string) []edgeFact {
	id := nodeID(topic)
	node := te.graph.nodes[id]
	if node == nil {
		return nil
	}

	var facts []edgeFact
	for _, edge := range te.graph.outEdges[id] {
		to := te.graph.nodes[edge.To]
		if to == nil {
			continue
		}
		facts = append(facts, edgeFact{
			Subject:  node.Label,
			Relation: edge.Relation,
			Object:   to.Label,
		})
	}
	return facts
}

func findSharedObjects(factsA, factsB []edgeFact) []string {
	objsA := make(map[string]bool)
	for _, f := range factsA {
		if f.Relation != RelIsA { // skip category itself
			objsA[strings.ToLower(f.Object)] = true
		}
	}
	var shared []string
	for _, f := range factsB {
		if f.Relation != RelIsA && objsA[strings.ToLower(f.Object)] {
			shared = append(shared, f.Object)
		}
	}
	return shared
}

// -----------------------------------------------------------------------
// Concept Blending — creative idea generation
// -----------------------------------------------------------------------

// ConceptBlend merges attributes from two concepts to spark ideas.
func (te *ThinkingEngine) ConceptBlend(conceptA, conceptB string) []string {
	factsA := te.gatherTopicFacts(conceptA)
	factsB := te.gatherTopicFacts(conceptB)

	if len(factsA) == 0 || len(factsB) == 0 {
		return nil
	}

	var blends []string
	for _, a := range factsA {
		if a.Relation == RelHas || a.Relation == RelUsedFor {
			blends = append(blends,
				fmt.Sprintf("What if %s had %s's %s?", conceptB, conceptA, a.Object))
		}
	}
	for _, b := range factsB {
		if b.Relation == RelHas || b.Relation == RelUsedFor {
			blends = append(blends,
				fmt.Sprintf("What if %s applied %s's %s?", conceptA, conceptB, b.Object))
		}
	}

	if len(blends) > 5 {
		blends = blends[:5]
	}
	return blends
}

// -----------------------------------------------------------------------
// Style Learning — fine-tuning Nous to the user
// -----------------------------------------------------------------------

// LearnPreference stores a user preference.
func (te *ThinkingEngine) LearnPreference(key, value string) {
	te.Preferences[key] = value
}

// GetPreference retrieves a user preference.
func (te *ThinkingEngine) GetPreference(key string) string {
	return te.Preferences[key]
}
