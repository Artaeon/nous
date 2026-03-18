package cognitive

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

// ActionResult holds the output of a deterministic action.
type ActionResult struct {
	Data           string            // raw data/facts gathered
	Structured     map[string]string // key-value structured data
	Source         string            // where the data came from (memory, web, file, computed)
	NeedsLLM       bool             // whether an LLM call is needed to format the response
	DirectResponse string           // if non-empty, send this directly (no LLM needed)
}

// NLUResult holds the output of natural language understanding.
// Defined here as the canonical type; nlu.go may also define it
// (whichever is compiled first wins — they must be identical).
// If the NLU agent has already created this type, delete this block.
type NLUResult struct {
	Intent     string
	Action     string
	Entities   map[string]string
	Confidence float64
	Raw        string
}

// ActionRouter executes deterministic actions based on NLU results.
// It NEVER calls the LLM. It gathers facts, computes results, searches,
// and returns raw data for the response layer to format.
type ActionRouter struct {
	Tools       *tools.Registry
	WorkingMem  *memory.WorkingMemory
	LongTermMem *memory.LongTermMemory
	EpisodicMem *memory.EpisodicMemory
	Knowledge   *KnowledgeVec
	Crystals    *ResponseCrystalStore
	Growth      *PersonalGrowth
	VCtx        *VirtualContext
	Researcher  *InlineResearcher
}

// NewActionRouter creates a router with nil subsystems.
// Wire up the fields after creation as each subsystem initialises.
func NewActionRouter() *ActionRouter {
	return &ActionRouter{}
}

// ActionChain represents a sequence of actions to execute in order.
// Each step's output feeds into the next step's input.
type ActionChain struct {
	Steps   []ChainStep
	Results []ActionResult
}

// ChainStep is one step in an action chain.
type ChainStep struct {
	Action    string            // action name (web_search, fetch_url, file_op, etc.)
	Entities  map[string]string // entities for this step
	DependsOn int              // index of previous step whose output feeds in (-1 for none)
}

// Execute runs the appropriate action for an NLU result.
// This is PURE CODE — no LLM calls. Returns raw data/facts.
func (ar *ActionRouter) Execute(nlu *NLUResult, conv *Conversation) *ActionResult {
	switch nlu.Action {
	case "respond":
		return ar.handleRespond(nlu)
	case "web_search":
		return ar.handleWebSearch(nlu)
	case "fetch_url":
		return ar.handleFetchURL(nlu)
	case "file_op":
		return ar.handleFileOp(nlu)
	case "compute":
		return ar.handleCompute(nlu)
	case "lookup_memory":
		return ar.handleLookupMemory(nlu)
	case "lookup_knowledge":
		return ar.handleLookupKnowledge(nlu)
	case "lookup_web":
		return ar.handleLookupWeb(nlu)
	case "schedule":
		return ar.handleSchedule(nlu)
	case "llm_chat":
		return ar.handleLLMChat(nlu, conv)
	case "research":
		return ar.handleResearch(nlu)
	case "chain":
		return ar.handleChain(nlu, conv)
	case "generate_doc":
		return ar.handleGenerateDoc(nlu, conv)
	default:
		// Unknown action — let the LLM handle it.
		return &ActionResult{
			Data:     nlu.Raw,
			Source:   "fallback",
			NeedsLLM: true,
		}
	}
}

// -----------------------------------------------------------------------
// Action handlers — each is pure code, no LLM.
// -----------------------------------------------------------------------

// handleRespond returns a canned response for greetings, farewells, etc.
func (ar *ActionRouter) handleRespond(nlu *NLUResult) *ActionResult {
	if quick := tryQuickResponse(nlu.Raw); quick != "" {
		return &ActionResult{
			DirectResponse: quick,
			Source:         "canned",
		}
	}
	// No canned match — let LLM handle the simple response.
	return &ActionResult{
		Data:     nlu.Raw,
		Source:   "canned",
		NeedsLLM: true,
	}
}

// handleWebSearch executes a web search via the tools registry.
func (ar *ActionRouter) handleWebSearch(nlu *NLUResult) *ActionResult {
	query := nlu.Entities["query"]
	if query == "" {
		query = nlu.Raw
	}
	if ar.Tools == nil {
		return &ActionResult{Data: "web search unavailable", Source: "web", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("websearch")
	if err != nil {
		return &ActionResult{Data: "web search tool not found", Source: "web", NeedsLLM: true}
	}
	result, err := tool.Execute(map[string]string{"query": query})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("search error: %v", err), Source: "web", NeedsLLM: true}
	}
	return &ActionResult{Data: result, Source: "web", NeedsLLM: true}
}

// handleFetchURL downloads content from a URL.
func (ar *ActionRouter) handleFetchURL(nlu *NLUResult) *ActionResult {
	url := nlu.Entities["url"]
	if url == "" {
		return &ActionResult{Data: "no URL provided", Source: "web", NeedsLLM: true}
	}
	if ar.Tools == nil {
		return &ActionResult{Data: "fetch tool unavailable", Source: "web", NeedsLLM: true}
	}
	tool, err := ar.Tools.Get("fetch")
	if err != nil {
		return &ActionResult{Data: "fetch tool not found", Source: "web", NeedsLLM: true}
	}
	result, err := tool.Execute(map[string]string{"url": url})
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("fetch error: %v", err), Source: "web", NeedsLLM: true}
	}
	return &ActionResult{Data: result, Source: "web", NeedsLLM: true}
}

// handleFileOp executes file operations (read, write, edit, grep, glob, ls).
func (ar *ActionRouter) handleFileOp(nlu *NLUResult) *ActionResult {
	op := nlu.Entities["op"]
	if op == "" {
		op = "read" // default
	}
	if ar.Tools == nil {
		return &ActionResult{Data: "file tools unavailable", Source: "file", NeedsLLM: true}
	}

	// Map operation names to tool names.
	toolName := op
	switch op {
	case "read", "write", "edit", "grep", "glob", "ls":
		// direct mapping
	case "list":
		toolName = "ls"
	case "search", "find":
		toolName = "grep"
	case "create":
		toolName = "write"
	default:
		toolName = "read"
	}

	tool, err := ar.Tools.Get(toolName)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("tool %q not found", toolName), Source: "file", NeedsLLM: true}
	}

	// Build args from entities.
	args := make(map[string]string)
	for k, v := range nlu.Entities {
		if k != "op" { // skip meta-key
			args[k] = v
		}
	}

	result, err := tool.Execute(args)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("file op error: %v", err), Source: "file", NeedsLLM: true}
	}
	return &ActionResult{Data: result, Source: "file", NeedsLLM: true}
}

// handleCompute evaluates math expressions and date calculations.
func (ar *ActionRouter) handleCompute(nlu *NLUResult) *ActionResult {
	expr := nlu.Entities["expr"]
	if expr == "" {
		expr = nlu.Raw
	}

	// Try date computation first.
	if dateResult, ok := evaluateDate(expr); ok {
		return &ActionResult{
			DirectResponse: dateResult,
			Source:         "computed",
			Structured:     map[string]string{"result": dateResult},
		}
	}

	// Try math evaluation.
	result, err := evaluateMath(expr)
	if err != nil {
		return &ActionResult{Data: fmt.Sprintf("cannot compute: %v", err), Source: "computed", NeedsLLM: true}
	}
	return &ActionResult{
		DirectResponse: result,
		Source:         "computed",
		Structured:     map[string]string{"result": result},
	}
}

// handleLookupMemory searches across all memory systems.
func (ar *ActionRouter) handleLookupMemory(nlu *NLUResult) *ActionResult {
	query := nlu.Entities["query"]
	if query == "" {
		query = nlu.Raw
	}

	var parts []string

	// Working memory — recent context.
	if ar.WorkingMem != nil {
		slots := ar.WorkingMem.MostRelevant(5)
		for _, s := range slots {
			parts = append(parts, fmt.Sprintf("[working] %s: %v", s.Key, s.Value))
		}
	}

	// Long-term memory — persistent facts.
	if ar.LongTermMem != nil {
		// Try direct key lookup first.
		if val, ok := ar.LongTermMem.Retrieve(query); ok {
			parts = append(parts, fmt.Sprintf("[longterm] %s: %s", query, val))
		}
		// Also try category search if an entity specifies category.
		if cat := nlu.Entities["category"]; cat != "" {
			entries := ar.LongTermMem.Search(cat)
			for _, e := range entries {
				parts = append(parts, fmt.Sprintf("[longterm:%s] %s: %s", e.Category, e.Key, e.Value))
			}
		}
	}

	// Episodic memory — past interactions.
	if ar.EpisodicMem != nil {
		episodes := ar.EpisodicMem.SearchKeyword(query, 3)
		for _, ep := range episodes {
			parts = append(parts, fmt.Sprintf("[episode %s] Q: %s A: %s", ep.Timestamp.Format("2006-01-02"), ep.Input, ep.Output))
		}
	}

	if len(parts) == 0 {
		return &ActionResult{Data: "no relevant memories found", Source: "memory", NeedsLLM: true}
	}
	return &ActionResult{Data: strings.Join(parts, "\n"), Source: "memory", NeedsLLM: true}
}

// handleLookupKnowledge searches the knowledge vector store.
func (ar *ActionRouter) handleLookupKnowledge(nlu *NLUResult) *ActionResult {
	query := nlu.Entities["topic"]
	if query == "" {
		query = nlu.Entities["query"]
	}
	if query == "" {
		query = nlu.Raw
	}

	var parts []string

	// Knowledge vector search.
	if ar.Knowledge != nil {
		results, err := ar.Knowledge.Search(query, 5)
		if err == nil {
			for _, r := range results {
				parts = append(parts, fmt.Sprintf("[%s (%.2f)] %s", r.Source, r.Score, r.Text))
			}
		}
	}

	// Weave virtual context for additional grounding.
	if ar.VCtx != nil {
		assembly := ar.VCtx.Weave(query)
		if woven := assembly.FormatForPrompt(); woven != "" {
			parts = append(parts, woven)
		}
	}

	if len(parts) == 0 {
		return &ActionResult{Data: "no relevant knowledge found", Source: "knowledge", NeedsLLM: true}
	}
	return &ActionResult{Data: strings.Join(parts, "\n"), Source: "knowledge", NeedsLLM: true}
}

// handleLookupWeb tries knowledge first, falls back to web search.
func (ar *ActionRouter) handleLookupWeb(nlu *NLUResult) *ActionResult {
	// Try knowledge base first.
	if ar.Knowledge != nil {
		query := nlu.Entities["query"]
		if query == "" {
			query = nlu.Raw
		}
		results, err := ar.Knowledge.Search(query, 3)
		if err == nil && len(results) > 0 && results[0].Score > 0.5 {
			var parts []string
			for _, r := range results {
				parts = append(parts, fmt.Sprintf("[%s] %s", r.Source, r.Text))
			}
			return &ActionResult{Data: strings.Join(parts, "\n"), Source: "knowledge", NeedsLLM: true}
		}
	}

	// Fall back to web search.
	return ar.handleWebSearch(nlu)
}

// handleSchedule parses scheduling entities into structured data.
func (ar *ActionRouter) handleSchedule(nlu *NLUResult) *ActionResult {
	structured := make(map[string]string)
	for k, v := range nlu.Entities {
		structured[k] = v
	}

	// Parse relative time if present.
	if when := nlu.Entities["when"]; when != "" {
		if t, ok := parseRelativeTime(when); ok {
			structured["parsed_time"] = t.Format(time.RFC3339)
		}
	}

	return &ActionResult{
		Data:       fmt.Sprintf("Task scheduled: %s", nlu.Entities["task"]),
		Structured: structured,
		Source:     "schedule",
		NeedsLLM:  true,
	}
}

// handleResearch delegates to the InlineResearcher for deep topic research.
func (ar *ActionRouter) handleResearch(nlu *NLUResult) *ActionResult {
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	if topic == "" {
		topic = nlu.Raw
	}

	// Use the dedicated InlineResearcher if available.
	if ar.Researcher != nil {
		return ar.Researcher.Research(topic)
	}

	// Fallback: construct an InlineResearcher from the router's tools.
	ir := &InlineResearcher{Tools: ar.Tools}
	return ir.Research(topic)
}

// ExecuteChain runs a sequence of actions, piping outputs forward.
// Each step executes in order. If a step depends on a previous step,
// the previous step's output is injected as context into the current step's entities.
func (ar *ActionRouter) ExecuteChain(chain *ActionChain, nlu *NLUResult, conv *Conversation) *ActionResult {
	chain.Results = make([]ActionResult, len(chain.Steps))

	var allData []string
	var lastSource string

	for i, step := range chain.Steps {
		// Build a synthetic NLUResult for this step.
		stepNLU := &NLUResult{
			Intent:   nlu.Intent,
			Action:   step.Action,
			Entities: make(map[string]string),
			Raw:      nlu.Raw,
		}
		for k, v := range step.Entities {
			stepNLU.Entities[k] = v
		}

		// If this step depends on a previous step, inject that step's output.
		if step.DependsOn >= 0 && step.DependsOn < i {
			prev := chain.Results[step.DependsOn]
			stepNLU.Entities["_chain_input"] = prev.Data
			// For file write steps, use previous output as content.
			if step.Action == "file_op" && stepNLU.Entities["op"] == "write" {
				stepNLU.Entities["content"] = prev.Data
			}
		}

		// Execute the step using normal single-action dispatch (no recursion).
		result := ar.executeSingleAction(stepNLU, conv)
		chain.Results[i] = *result

		if result.Data != "" {
			allData = append(allData, fmt.Sprintf("[%s] %s", result.Source, result.Data))
		}
		if result.DirectResponse != "" {
			allData = append(allData, result.DirectResponse)
		}
		lastSource = result.Source
	}

	// Combine all step outputs into a single result.
	combined := strings.Join(allData, "\n\n---\n\n")
	return &ActionResult{
		Data:     combined,
		Source:   "chain:" + lastSource,
		NeedsLLM: true,
	}
}

// executeSingleAction dispatches a single action without chain/generate_doc handling.
// This avoids infinite recursion if a chain step is itself "chain".
func (ar *ActionRouter) executeSingleAction(nlu *NLUResult, conv *Conversation) *ActionResult {
	switch nlu.Action {
	case "respond":
		return ar.handleRespond(nlu)
	case "web_search":
		return ar.handleWebSearch(nlu)
	case "fetch_url":
		return ar.handleFetchURL(nlu)
	case "file_op":
		return ar.handleFileOp(nlu)
	case "compute":
		return ar.handleCompute(nlu)
	case "lookup_memory":
		return ar.handleLookupMemory(nlu)
	case "lookup_knowledge":
		return ar.handleLookupKnowledge(nlu)
	case "lookup_web":
		return ar.handleLookupWeb(nlu)
	case "schedule":
		return ar.handleSchedule(nlu)
	case "llm_chat":
		return ar.handleLLMChat(nlu, conv)
	default:
		return &ActionResult{
			Data:     nlu.Raw,
			Source:   "fallback",
			NeedsLLM: true,
		}
	}
}

// handleChain builds and executes a chain based on the chain_type entity.
func (ar *ActionRouter) handleChain(nlu *NLUResult, conv *Conversation) *ActionResult {
	chainType := nlu.Entities["chain_type"]
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	if topic == "" {
		topic = nlu.Raw
	}

	var chain *ActionChain
	switch chainType {
	case "research_and_write":
		chain = researchChain(topic)
	case "search_and_save":
		filepath := nlu.Entities["path"]
		if filepath == "" {
			filepath = strings.ReplaceAll(strings.ToLower(topic), " ", "_") + ".txt"
		}
		chain = searchAndSaveChain(topic, filepath)
	case "search_and_explain":
		chain = searchAndExplainChain(topic)
	default:
		chain = researchChain(topic)
	}

	return ar.ExecuteChain(chain, nlu, conv)
}

// handleGenerateDoc extracts a topic, runs web search + knowledge lookup,
// combines results, and returns with NeedsLLM=true for document formatting.
func (ar *ActionRouter) handleGenerateDoc(nlu *NLUResult, conv *Conversation) *ActionResult {
	topic := nlu.Entities["topic"]
	if topic == "" {
		topic = nlu.Entities["query"]
	}
	if topic == "" {
		topic = nlu.Raw
	}

	chain := researchChain(topic)
	result := ar.ExecuteChain(chain, nlu, conv)

	result.Structured = map[string]string{
		"format": "document",
		"topic":  topic,
	}
	result.Data = fmt.Sprintf("[Document Request: %s]\n\n%s", topic, result.Data)
	result.NeedsLLM = true
	return result
}

// -----------------------------------------------------------------------
// Chain templates — pre-built pipelines for common multi-step tasks.
// -----------------------------------------------------------------------

// researchChain builds a chain for "research X" or "create a document about X".
// Steps: web search -> knowledge lookup -> combine (NeedsLLM=true for synthesis).
func researchChain(topic string) *ActionChain {
	return &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
			{
				Action:    "lookup_knowledge",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
		},
	}
}

// searchAndSaveChain builds a chain for "search X and save to file".
// Steps: web search -> file write with search results.
func searchAndSaveChain(topic, filepath string) *ActionChain {
	return &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
			{
				Action:    "file_op",
				Entities:  map[string]string{"op": "write", "path": filepath},
				DependsOn: 0,
			},
		},
	}
}

// searchAndExplainChain builds a chain for "look up X and explain it".
// Steps: web search -> knowledge lookup -> combine for LLM explanation.
func searchAndExplainChain(topic string) *ActionChain {
	return &ActionChain{
		Steps: []ChainStep{
			{
				Action:    "web_search",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
			{
				Action:    "lookup_knowledge",
				Entities:  map[string]string{"query": topic},
				DependsOn: -1,
			},
		},
	}
}

// handleLLMChat passes through to the LLM for conversational responses.
func (ar *ActionRouter) handleLLMChat(nlu *NLUResult, conv *Conversation) *ActionResult {
	// Gather any relevant context from memory to enrich the LLM call.
	var context []string
	if ar.WorkingMem != nil {
		slots := ar.WorkingMem.MostRelevant(3)
		for _, s := range slots {
			context = append(context, fmt.Sprintf("%s: %v", s.Key, s.Value))
		}
	}

	// Weave virtual context for richer grounding.
	if ar.VCtx != nil {
		query := nlu.Entities["topic"]
		if query == "" {
			query = nlu.Raw
		}
		assembly := ar.VCtx.Weave(query)
		if woven := assembly.FormatForPrompt(); woven != "" {
			context = append(context, woven)
		}
	}

	data := nlu.Raw
	if len(context) > 0 {
		data = nlu.Raw + "\n\n[Context]\n" + strings.Join(context, "\n")
	}

	return &ActionResult{
		Data:     data,
		Source:   "conversation",
		NeedsLLM: true,
	}
}

// -----------------------------------------------------------------------
// Math evaluator — handles basic arithmetic without any LLM.
// -----------------------------------------------------------------------

// evaluateMath evaluates simple math expressions.
// Supports: +, -, *, /, ^, %, parentheses, sqrt, abs.
// Examples: "2+2" -> "4", "sqrt(16)" -> "4", "15% of 200" -> "30"
func evaluateMath(expr string) (string, error) {
	// Clean input.
	expr = strings.TrimSpace(expr)
	expr = stripMathProse(expr)
	if expr == "" {
		return "", fmt.Errorf("empty expression")
	}

	// Handle "X% of Y" pattern.
	if m := percentOfRe.FindStringSubmatch(expr); len(m) == 3 {
		pct, err1 := strconv.ParseFloat(m[1], 64)
		val, err2 := strconv.ParseFloat(m[2], 64)
		if err1 == nil && err2 == nil {
			return formatNumber(pct / 100 * val), nil
		}
	}

	val, rest, err := parseExpr(expr)
	if err != nil {
		return "", err
	}
	rest = strings.TrimSpace(rest)
	if rest != "" {
		return "", fmt.Errorf("unexpected trailing text: %q", rest)
	}
	return formatNumber(val), nil
}

var percentOfRe = regexp.MustCompile(`(?i)^([\d.]+)\s*%\s*of\s+([\d.]+)$`)

// stripMathProse removes common phrasing around math expressions.
var mathProseRe = regexp.MustCompile(`(?i)^(?:what(?:'s| is)\s+|calculate\s+|compute\s+|eval(?:uate)?\s+)`)

func stripMathProse(s string) string {
	s = mathProseRe.ReplaceAllString(s, "")
	s = strings.TrimRight(s, "? ")
	// Replace unicode operators.
	s = strings.ReplaceAll(s, "\u00d7", "*") // ×
	s = strings.ReplaceAll(s, "\u00f7", "/") // ÷
	s = strings.NewReplacer("x", "*").Replace(s) // only if between digits
	return s
}

// Recursive descent parser for arithmetic expressions.
// Grammar:
//   expr   = term (('+' | '-') term)*
//   term   = power (('*' | '/' | '%') power)*
//   power  = unary ('^' unary)*
//   unary  = '-'? atom
//   atom   = number | func '(' expr ')' | '(' expr ')'

func parseExpr(s string) (float64, string, error) {
	val, rest, err := parseTerm(s)
	if err != nil {
		return 0, s, err
	}
	for {
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		if len(rest) == 0 {
			break
		}
		op := rest[0]
		if op != '+' && op != '-' {
			break
		}
		rest = rest[1:]
		right, r, err := parseTerm(rest)
		if err != nil {
			return 0, r, err
		}
		rest = r
		if op == '+' {
			val += right
		} else {
			val -= right
		}
	}
	return val, rest, nil
}

func parseTerm(s string) (float64, string, error) {
	val, rest, err := parsePower(s)
	if err != nil {
		return 0, s, err
	}
	for {
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		if len(rest) == 0 {
			break
		}
		op := rest[0]
		if op != '*' && op != '/' && op != '%' {
			break
		}
		rest = rest[1:]
		right, r, err := parsePower(rest)
		if err != nil {
			return 0, r, err
		}
		rest = r
		switch op {
		case '*':
			val *= right
		case '/':
			if right == 0 {
				return 0, rest, fmt.Errorf("division by zero")
			}
			val /= right
		case '%':
			if right == 0 {
				return 0, rest, fmt.Errorf("modulo by zero")
			}
			val = math.Mod(val, right)
		}
	}
	return val, rest, nil
}

func parsePower(s string) (float64, string, error) {
	val, rest, err := parseUnary(s)
	if err != nil {
		return 0, s, err
	}
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	if len(rest) > 0 && rest[0] == '^' {
		rest = rest[1:]
		right, r, err := parseUnary(rest)
		if err != nil {
			return 0, r, err
		}
		val = math.Pow(val, right)
		rest = r
	}
	return val, rest, nil
}

func parseUnary(s string) (float64, string, error) {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	if len(s) > 0 && s[0] == '-' {
		val, rest, err := parseAtom(s[1:])
		if err != nil {
			return 0, rest, err
		}
		return -val, rest, nil
	}
	return parseAtom(s)
}

// mathFuncs supported by the evaluator.
var mathFuncs = map[string]func(float64) float64{
	"sqrt": math.Sqrt,
	"abs":  math.Abs,
	"sin":  math.Sin,
	"cos":  math.Cos,
	"tan":  math.Tan,
	"log":  math.Log10,
	"ln":   math.Log,
	"ceil": math.Ceil,
	"floor": math.Floor,
	"round": math.Round,
}

func parseAtom(s string) (float64, string, error) {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	if len(s) == 0 {
		return 0, s, fmt.Errorf("unexpected end of expression")
	}

	// Check for named constants.
	if strings.HasPrefix(strings.ToLower(s), "pi") {
		rest := s[2:]
		if len(rest) == 0 || !isAlpha(rest[0]) {
			return math.Pi, rest, nil
		}
	}
	if strings.HasPrefix(strings.ToLower(s), "e") {
		rest := s[1:]
		if len(rest) == 0 || (!isAlpha(rest[0]) && rest[0] != '.') {
			return math.E, rest, nil
		}
	}

	// Check for function calls: func(expr)
	for name, fn := range mathFuncs {
		if strings.HasPrefix(strings.ToLower(s), name) {
			after := s[len(name):]
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			if len(after) > 0 && after[0] == '(' {
				inner, rest, err := parseExpr(after[1:])
				if err != nil {
					return 0, rest, err
				}
				rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
				if len(rest) == 0 || rest[0] != ')' {
					return 0, rest, fmt.Errorf("missing closing parenthesis for %s", name)
				}
				return fn(inner), rest[1:], nil
			}
		}
	}

	// Parenthesised sub-expression.
	if s[0] == '(' {
		val, rest, err := parseExpr(s[1:])
		if err != nil {
			return 0, rest, err
		}
		rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
		if len(rest) == 0 || rest[0] != ')' {
			return 0, rest, fmt.Errorf("missing closing parenthesis")
		}
		return val, rest[1:], nil
	}

	// Number literal.
	return parseNumber(s)
}

func parseNumber(s string) (float64, string, error) {
	s = strings.TrimLeftFunc(s, unicode.IsSpace)
	i := 0
	for i < len(s) && (s[i] >= '0' && s[i] <= '9' || s[i] == '.') {
		i++
	}
	if i == 0 {
		return 0, s, fmt.Errorf("expected number, got %q", truncStr(s, 20))
	}
	val, err := strconv.ParseFloat(s[:i], 64)
	if err != nil {
		return 0, s, fmt.Errorf("invalid number %q", s[:i])
	}
	return val, s[i:], nil
}

func isAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// formatNumber formats a float, stripping trailing zeroes for integers.
func formatNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	s := strconv.FormatFloat(v, 'f', 10, 64)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

func truncStr(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// -----------------------------------------------------------------------
// Date evaluator — handles relative date questions.
// -----------------------------------------------------------------------

var (
	tomorrowRe   = regexp.MustCompile(`(?i)\btomorrow\b`)
	yesterdayRe   = regexp.MustCompile(`(?i)\byesterday\b`)
	todayRe       = regexp.MustCompile(`(?i)\btoday\b|\bwhat day is it\b|\bwhat is today\b`)
	daysUntilRe   = regexp.MustCompile(`(?i)(?:days?\s+(?:until|to|till|before))\s+(.+)`)
	daysSinceRe   = regexp.MustCompile(`(?i)(?:days?\s+(?:since|from|after))\s+(.+)`)
	whatDayRe     = regexp.MustCompile(`(?i)what\s+(?:day|date)\s+(?:is|was|will be)\s+(.+)`)
)

func evaluateDate(expr string) (string, bool) {
	now := time.Now()

	if todayRe.MatchString(expr) {
		return now.Format("Monday, January 2, 2006"), true
	}
	if tomorrowRe.MatchString(expr) {
		t := now.AddDate(0, 0, 1)
		return t.Format("Monday, January 2, 2006"), true
	}
	if yesterdayRe.MatchString(expr) {
		t := now.AddDate(0, 0, -1)
		return t.Format("Monday, January 2, 2006"), true
	}

	if m := daysUntilRe.FindStringSubmatch(expr); len(m) == 2 {
		if target, ok := parseDate(strings.TrimSpace(m[1])); ok {
			days := int(target.Sub(now).Hours() / 24)
			if days < 0 {
				return fmt.Sprintf("%d days ago", -days), true
			}
			return fmt.Sprintf("%d days", days), true
		}
	}
	if m := daysSinceRe.FindStringSubmatch(expr); len(m) == 2 {
		if target, ok := parseDate(strings.TrimSpace(m[1])); ok {
			days := int(now.Sub(target).Hours() / 24)
			if days < 0 {
				return fmt.Sprintf("%d days in the future", -days), true
			}
			return fmt.Sprintf("%d days", days), true
		}
	}

	if m := whatDayRe.FindStringSubmatch(expr); len(m) == 2 {
		if target, ok := parseDate(strings.TrimSpace(m[1])); ok {
			return target.Format("Monday, January 2, 2006"), true
		}
	}

	return "", false
}

// parseDate tries several common date formats.
func parseDate(s string) (time.Time, bool) {
	s = strings.TrimRight(s, "?. ")
	formats := []string{
		"2006-01-02",
		"January 2, 2006",
		"Jan 2, 2006",
		"January 2 2006",
		"Jan 2 2006",
		"01/02/2006",
		"02-01-2006",
		"2 January 2006",
		"2 Jan 2006",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}

// parseRelativeTime parses relative time expressions like "in 2 hours", "tomorrow at 3pm".
func parseRelativeTime(s string) (time.Time, bool) {
	now := time.Now()
	s = strings.ToLower(strings.TrimSpace(s))

	if s == "tomorrow" {
		return now.AddDate(0, 0, 1), true
	}
	if s == "today" || s == "now" {
		return now, true
	}

	// "in N hours/minutes/days"
	re := regexp.MustCompile(`in\s+(\d+)\s+(hour|minute|min|day|week)s?`)
	if m := re.FindStringSubmatch(s); len(m) == 3 {
		n, _ := strconv.Atoi(m[1])
		switch m[2] {
		case "hour":
			return now.Add(time.Duration(n) * time.Hour), true
		case "minute", "min":
			return now.Add(time.Duration(n) * time.Minute), true
		case "day":
			return now.AddDate(0, 0, n), true
		case "week":
			return now.AddDate(0, 0, n*7), true
		}
	}

	return time.Time{}, false
}

// -----------------------------------------------------------------------
// ResponseFormatter — the ONLY place where LLM is called for response.
// -----------------------------------------------------------------------

// ResponseFormatter takes raw action data and formats it into natural language.
// This is the ONLY place where the LLM is called for response generation.
type ResponseFormatter struct {
	LLM *ollama.Client
}

const responseSystemPrompt = `You are Nous. Present the following information naturally and concisely. Do not add information beyond what is provided. Be direct.`

// Format turns raw data into a natural language response.
// If ActionResult.DirectResponse is set, returns it immediately (no LLM).
// Otherwise, makes ONE LLM call to format the data naturally.
func (rf *ResponseFormatter) Format(query string, result *ActionResult, conv *Conversation) (string, error) {
	// Direct response — zero LLM calls.
	if result.DirectResponse != "" {
		return result.DirectResponse, nil
	}

	if rf.LLM == nil {
		// No LLM available — return raw data.
		return result.Data, nil
	}

	msgs := []ollama.Message{
		{Role: "system", Content: responseSystemPrompt},
		{Role: "user", Content: fmt.Sprintf("Question: %s\nData: %s\nAnswer:", query, result.Data)},
	}

	resp, err := rf.LLM.Chat(msgs, &ollama.ModelOptions{
		Temperature: 0.5,
		NumPredict:  150,
		NumCtx:      1024,
	})
	if err != nil {
		return result.Data, err
	}

	return strings.TrimSpace(resp.Message.Content), nil
}

// FormatStream is the streaming version of Format.
func (rf *ResponseFormatter) FormatStream(query string, result *ActionResult, conv *Conversation, onToken func(string, bool)) (string, error) {
	// Direct response — zero LLM calls.
	if result.DirectResponse != "" {
		onToken(result.DirectResponse, true)
		return result.DirectResponse, nil
	}

	if rf.LLM == nil {
		onToken(result.Data, true)
		return result.Data, nil
	}

	msgs := []ollama.Message{
		{Role: "system", Content: responseSystemPrompt},
		{Role: "user", Content: fmt.Sprintf("Question: %s\nData: %s\nAnswer:", query, result.Data)},
	}

	var fullAnswer strings.Builder
	_, err := rf.LLM.ChatStream(msgs, &ollama.ModelOptions{
		Temperature: 0.5,
		NumPredict:  150,
		NumCtx:      1024,
	}, func(token string, done bool) {
		if !done {
			fullAnswer.WriteString(token)
		}
		onToken(token, done)
	})
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(fullAnswer.String()), nil
}
