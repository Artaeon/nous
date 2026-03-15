package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/compress"
	"github.com/artaeon/nous/internal/index"
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

const maxToolIterations = 4

// Reasoner performs autonomous chain-of-thought inference with tool use.
// It listens for percepts, reasons about them, and can autonomously chain
// multiple tool calls to accomplish complex tasks — like an autonomous agent.
//
// The Cognitive Grounding system prevents hallucinations by:
// 1. Progressive tool disclosure — only showing relevant tools per intent
// 2. Structured THINK/ACT/OBSERVE protocol — forcing reasoning structure
// 3. Smart result truncation — keeping tool results within token budget
// 4. Context budget tracking — compressing when context fills up
// 5. Synchronous reflection gate — catching errors before the next LLM call
type Reasoner struct {
	Base
	Tools            *tools.Registry
	Conv             *Conversation
	WorkingMem       *memory.WorkingMemory
	LongTermMem      *memory.LongTermMemory
	ProjectMem       *memory.ProjectMemory
	EpisodicMem      *memory.EpisodicMemory
	Compressor       *compress.Compressor
	CodeIndex        *index.CodebaseIndex
	Budget           *ContextBudget
	Gate             *ReflectionGate
	Recipes          *RecipeBook
	Predictor        *Predictor
	Learner          *Learner
	Router           *ModelRouter
	Intent           *IntentCompiler
	Crystals         *CrystalBook
	Speculator       *SpeculativeExecutor
	MicroInfer       *MicroInference
	Neuroplastic     *NeuroplasticDescriptions
	Grammar          *GrammarDecoder
	Distiller        *SelfDistiller
	EmbedGround      *EmbedGrounder
	Exo              *Exocortex
	Firewall         *CognitiveFirewall
	Phantom          *PhantomReasoner
	PromptDist       *PromptDistiller
	Cortex           *NeuralCortex
	Knowledge        *KnowledgeVec
	Compiler         *ModelCompiler
	VCtx             *VirtualContext
	Growth           *PersonalGrowth
	Ensemble         *ToolEnsemble
	Feedback         *FeedbackLoop
	OnToken          func(token string, done bool)
	OnStatus         func(status string)
	Confirm          ConfirmFunc
	AssistantContext func(input string, recent string) string
	AssistantAnswer  func(input string, recent string) (string, bool)

	// Active tool subset for current reasoning cycle
	activeTools []tools.Tool
	activeCats  map[ToolCategory]bool
	currentInput string
}

// CurrentProject holds the scanned project info for the system prompt.
var CurrentProject *ProjectInfo

func NewReasoner(board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry) *Reasoner {
	return &Reasoner{
		Base:       Base{Board: board, LLM: llm},
		Tools:      toolReg,
		Conv:       NewConversation(40),
		Budget:     DefaultBudget(),
		Gate:       &ReflectionGate{},
		activeCats: make(map[ToolCategory]bool),
	}
}

func (r *Reasoner) Name() string { return "reasoner" }

func (r *Reasoner) Run(ctx context.Context) error {
	events := r.Board.Subscribe("percept")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev := <-events:
			percept, ok := ev.Payload.(blackboard.Percept)
			if !ok {
				continue
			}
			if err := r.reason(ctx, percept); err != nil {
				r.Board.Set("reasoner_error", err.Error())
			}
		}
	}
}

func (r *Reasoner) reason(ctx context.Context, percept blackboard.Percept) error {
	// Reset reflection gate for this reasoning cycle
	r.Gate.Reset()
	r.currentInput = percept.Raw
	defer func() { r.currentInput = "" }()
	recentConversation := recentConversationContext(r.Conv.Messages(), 6)

	if r.AssistantAnswer != nil {
		if answer, ok := r.AssistantAnswer(percept.Raw, recentConversation); ok {
			r.Conv.User(percept.Raw)
			r.Conv.Assistant(answer)
			r.publishAnswer(answer)
			r.finishReasoning(percept, NewPipeline(percept.Raw), answer)
			return nil
		}
	}

	if answer, ok, err := r.tryResearchAndWrite(percept.Raw); ok {
		if err != nil {
			return err
		}
		r.Conv.User(percept.Raw)
		r.Conv.Assistant(answer)
		r.publishAnswer(answer)
		r.finishReasoning(percept, NewPipeline(percept.Raw), answer)
		return nil
	}

	// 0. Cognitive Exocortex — deterministic tool resolution AND response synthesis.
	// The LLM is not called at all for tool-based queries. Intent compiler resolves
	// the tool call, tools execute, and the response is synthesized from facts.
	// Zero hallucination possible — the response IS the data, formatted for humans.
	if r.Intent != nil {
		if actions := r.Intent.Compile(percept.Raw); len(actions) > 0 && actions[0].Confidence >= 0.8 {
			pipe := NewPipeline(percept.Raw)
			synth := NewResponseSynthesizer()
			r.emitStatus(fmt.Sprintf("  %s↳ exo-bypass: %s(%s)%s", ColorDim, actions[0].Tool, actions[0].Source, ColorReset))

			// Execute compiled actions and collect results for synthesis
			var steps []synthStep
			for _, action := range actions {
				tc := toolCall{Name: action.Tool, Args: action.Args}
				start := time.Now()
				result, toolErr := r.executeTool(tc)
				duration := time.Since(start)
				r.emitStatus(ToolStatus(tc.Name, formatArgs(tc.Args), duration))

				if toolErr == nil {
					result = SmartTruncate(tc.Name, result)
				}
				result, _ = ValidateToolResult(tc.Name, result, toolErr)
				pipe.AddStep(tc.Name, result)

				steps = append(steps, synthStep{
					Tool:   action.Tool,
					Args:   action.Args,
					Result: result,
					Err:    toolErr,
				})

				// Track neuroplastic stats
				if r.Neuroplastic != nil {
					r.Neuroplastic.RecordAttempt(action.Tool)
					if toolErr == nil && strings.TrimSpace(result) != "" {
						r.Neuroplastic.RecordSuccess(action.Tool)
					}
				}
			}

			// Tier 1: Synthesize response WITHOUT LLM — zero hallucination
			if synth.CanSynthesize(actions[0].Tool) {
				answer := synth.SynthesizeMulti(percept.Raw, steps)
				if answer != "" {
					r.Conv.User(percept.Raw)
					r.Conv.Assistant(answer)
					r.publishAnswer(answer)
					r.finishReasoning(percept, pipe, answer)
					return nil
				}
			}

			// Tier 2 fallback: Phantom Reasoning + Neural Scaffolding
			// Pre-compute the entire reasoning chain so the LLM only writes the conclusion.
			if r.Phantom != nil {
				chain := r.Phantom.BuildChain(percept.Raw, steps)
				if chain.CanBypass {
					// Phantom bypass: chain contains a complete answer
					r.Conv.User(percept.Raw)
					r.Conv.Assistant(chain.DirectAnswer)
					r.publishAnswer(chain.DirectAnswer)
					r.finishReasoning(percept, pipe, chain.DirectAnswer)
					return nil
				}
				// Seed the LLM with phantom chain
				memoryFacts := r.recallKeyFacts(percept.Raw)
				answer, ok, err := r.finalAnswerFromEvidence(percept.Raw, "", memoryFacts, pipe,
					chain.FullContext+"\n[Complete the reasoning above with a concise final answer.]",
				)
				if err != nil {
					return err
				}
				if ok {
					answer = r.firewallCheck(answer, pipe)
					r.publishAnswer(answer)
					r.finishReasoning(percept, pipe, answer)
					return nil
				}
				// Phantom evidence collected but LLM couldn't synthesize —
				// publish the raw evidence so the user isn't left with silence.
				if raw := pipe.LastResult(); raw != "" {
					fallback := "Here's what I found:\n\n" + raw
					r.Conv.User(percept.Raw)
					r.Conv.Assistant(fallback)
					r.publishAnswer(fallback)
					r.finishReasoning(percept, pipe, fallback)
					return nil
				}
			} else {
				// Fallback: original scaffold path
				scaffold := NewNeuralScaffold()
				sp := scaffold.BuildFromMultipleResults(percept.Raw, steps)

				memoryFacts := r.recallKeyFacts(percept.Raw)
				answer, ok, err := r.finalAnswerFromEvidence(percept.Raw, "", memoryFacts, pipe,
					sp.ResponseSeed+"\n\n[Continue from the text above. Add detail from the evidence. Be direct.]",
				)
				if err != nil {
					return err
				}
				if ok {
					answer = scaffold.ValidateResponse(answer, sp.ResponseSeed, actions[0].Tool,
						pipe.LastResult())
					answer = r.firewallCheck(answer, pipe)
					r.publishAnswer(answer)
					r.finishReasoning(percept, pipe, answer)
					return nil
				}
				// Scaffold evidence collected but LLM couldn't synthesize —
				// publish the raw evidence so the user isn't left with silence.
				if raw := pipe.LastResult(); raw != "" {
					fallback := "Here's what I found:\n\n" + raw
					r.Conv.User(percept.Raw)
					r.Conv.Assistant(fallback)
					r.publishAnswer(fallback)
					r.finishReasoning(percept, pipe, fallback)
					return nil
				}
			}
			// Fall through to normal reasoning
		}
	}

	// 1. Progressive Tool Disclosure — select relevant tools based on intent
	r.activeTools = SelectToolsForIntent(percept.Intent, percept.Entities, percept.Raw, r.Tools.List())
	r.activeCats = make(map[ToolCategory]bool)
	for _, t := range r.activeTools {
		if cat, ok := ToolCategoryMap[t.Name]; ok {
			r.activeCats[cat] = true
		}
	}

	// Create pipeline for this reasoning cycle.
	pipe := NewPipeline(percept.Raw)

	// Configure thought distillation with the fast model (if multi-model routing available)
	if r.Router != nil {
		distiller := r.Router.ClientFor(TaskCompression)
		if distiller != nil {
			pipe.SetDistiller(distiller)
		}
	}

	// Recall key facts from memory for system context (not user message).
	// Only inject if truly relevant — avoid dumping memory into the prompt.
	memoryFacts := r.recallKeyFacts(percept.Raw)
	assistantFacts := ""
	if r.AssistantContext != nil {
		assistantFacts = r.AssistantContext(percept.Raw, recentConversation)
	}

	// Pre-ground code questions: if the query looks code-related and the codebase
	// index has a matching file, auto-read it BEFORE the first LLM call. This ensures
	// the model has real source code in context instead of hallucinating.
	groundingHint := r.preGroundCodeQuery(percept.Raw, pipe)
	if answer, ok := deterministicCodeAnswer(percept.Raw, pipe); ok {
		r.Conv.User(percept.Raw)
		r.Conv.Assistant(answer)
		r.publishAnswer(answer)
		r.finishReasoning(percept, pipe, answer)
		return nil
	}
	if groundingHint != "" {
		if answer, ok, err := r.finalAnswerFromEvidence(percept.Raw, assistantFacts, memoryFacts, pipe,
			"[System: You already have the relevant grounded code excerpt. Answer directly from that evidence. Do not call tools. If the excerpt is insufficient, say what is missing.]",
		); err != nil {
			return err
		} else if ok {
			r.publishAnswer(answer)
			r.finishReasoning(percept, pipe, answer)
			return nil
		} else if raw := pipe.LastResult(); raw != "" {
			// Grounding evidence collected but LLM couldn't synthesize —
			// publish the raw evidence so the user isn't left with silence.
			fallback := "Here's what I found:\n\n" + raw
			r.publishAnswer(fallback)
			r.finishReasoning(percept, pipe, fallback)
			return nil
		}
	}

	// 2. Autonomous tool loop with fresh-context pipeline
	// nativeConv tracks the accumulated conversation when using native tool calling.
	// When the model uses native tool calls, we keep building on the same conversation
	// so the model sees its own tool_calls and the tool results. When using fallback
	// JSON-in-text parsing, we rebuild fresh context from the pipeline each iteration.
	var nativeConv *Conversation

	for i := 0; i < maxToolIterations; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if nativeConv != nil {
			// Native API continuation: reuse the accumulated conversation
			r.Conv = nativeConv
		} else {
			// Build FRESH conversation each iteration — this is the key innovation.
			conv := NewConversation(10)

			// System prompt: tools + instructions only (no memory, no self-knowledge)
			sysPrompt := r.compactSystemPrompt()
			if assistantFacts != "" {
				sysPrompt += "\n" + assistantFacts
			}
			if memoryFacts != "" {
				sysPrompt += "\n" + memoryFacts
			}
			if recentConversation != "" {
				sysPrompt += "\n[Recent conversation]\n" + recentConversation
			}
			conv.System(sysPrompt)

			// User message: ONLY the raw query + pipeline context
			userMsg := percept.Raw
			if groundingHint != "" && i == 0 {
				userMsg += "\n\n" + groundingHint
			}
			if pipeCtx := pipe.BuildContext(); pipeCtx != "" {
				userMsg += "\n\nPrevious steps:\n" + pipeCtx
			}
			conv.User(userMsg)

			// If we have a raw result from the last step, include it as a tool observation
			if pipe.StepCount() > 0 && pipe.LastResult() != "" {
				lastStep := pipe.steps[len(pipe.steps)-1]
				conv.ToolResult(lastStep.ToolName, SmartTruncate(lastStep.ToolName, lastStep.RawResult))
			}

			r.Conv = conv
		}

		// Call LLM with native tool definitions
		llmRes, err := r.callLLMNative()
		if err != nil {
			return fmt.Errorf("reasoner iteration %d: %w", i, err)
		}

		fullResponse := llmRes.Content

		// Convert native tool calls to our internal format
		var toolCalls []toolCall
		usedNativeAPI := len(llmRes.ToolCalls) > 0

		if usedNativeAPI {
			// Native tool calling — model returned structured tool calls
			r.Conv.AssistantToolCalls(fullResponse, llmRes.ToolCalls)
			for _, ntc := range llmRes.ToolCalls {
				tc := toolCall{Name: ntc.Function.Name, Args: make(map[string]string)}
				for k, v := range ntc.Function.Arguments {
					tc.Args[k] = fmt.Sprintf("%v", v)
				}
				toolCalls = append(toolCalls, tc)
			}
		} else {
			// No native tool calls — either fallback text parsing or final answer
			r.Conv.Assistant(fullResponse)
			parsed := r.parseResponse(fullResponse)
			toolCalls = parsed.ToolCalls

			if len(toolCalls) == 0 {
				// Intent Compiler recovery: parse the model's natural language
				// response to extract tool calls it failed to express as JSON.
				if r.Intent != nil {
					if actions := r.Intent.CompileResponse(fullResponse, percept.Raw); len(actions) > 0 {
						r.emitStatus(fmt.Sprintf("  %s↳ intent-recovered: %s%s", ColorDim, actions[0].Tool, ColorReset))
						for _, a := range actions {
							toolCalls = append(toolCalls, toolCall{Name: a.Tool, Args: a.Args})
						}
					}
				}
			}
			if len(toolCalls) == 0 && r.Grammar != nil {
				// Grammar-constrained resolution: uses per-tool JSON Schema
				// to force the model to output valid tool calls.
				resolved := r.grammarToolResolve(percept.Raw)
				if len(resolved) > 0 {
					toolCalls = resolved
				}
			}
			if len(toolCalls) == 0 {
				// Check if the response looks like a failed tool call attempt
				// (contains JSON-like patterns but couldn't be parsed)
				if looksLikeFailedToolCall(fullResponse) {
					retried := r.structuredToolRetry(percept.Raw)
					if len(retried) > 0 {
						toolCalls = retried
					}
				}
			}

			if len(toolCalls) == 0 {
				// No tool calls at all — this is the final answer
				finalAnswer := fullResponse
				if parsed.Answer != "" {
					finalAnswer = parsed.Answer
				}
				finalAnswer = r.firewallCheck(finalAnswer, pipe)
				r.publishAnswer(finalAnswer)
				if parsed.Think != "" {
					r.Board.Set("last_thought", parsed.Think)
				}
				r.finishReasoning(percept, pipe, finalAnswer)
				return nil
			}
			// Using fallback text-parsed tool calls; reset native conv tracking
			nativeConv = nil
		}

		// Execute tool calls
		for _, tc := range toolCalls {
			// Handle request_tools meta-tool
			if tc.Name == "request_tools" {
				cat := ToolCategory(tc.Args["category"])
				newTools := ExpandCategory(cat, r.Tools.List(), r.activeTools)
				if len(newTools) > 0 {
					r.activeTools = append(r.activeTools, newTools...)
					r.activeCats[cat] = true
					names := CategoryNames(cat, r.Tools.List())
					r.emitStatus(fmt.Sprintf("  %s+ %s%s", ColorDim, strings.Join(names, ", "), ColorReset))
				}
				continue
			}

			start := time.Now()
			result, toolErr := r.executeTool(tc)
			duration := time.Since(start)

			// Emit tool status with duration
			r.emitStatus(ToolStatus(tc.Name, formatArgs(tc.Args), duration))

			// 3a. Smart truncation (tool-specific)
			if toolErr == nil {
				result = SmartTruncate(tc.Name, result)
			}

			// 3b. Result validation
			result, hint := ValidateToolResult(tc.Name, result, toolErr)

			// Emit action_recorded for Reflector stream
			r.Board.RecordAction(blackboard.ActionRecord{
				StepID:    fmt.Sprintf("reason-%d-%d", i, len(pipe.steps)),
				Tool:      tc.Name,
				Input:     formatArgs(tc.Args),
				Output:    result,
				Success:   toolErr == nil,
				Duration:  duration,
				Timestamp: time.Now(),
			})

			// 5. Synchronous reflection gate
			gateCheck := r.Gate.Check(tc.Name, result, toolErr)

			// Inject hints into result for pipeline compression
			if hint != "" {
				result = result + "\nHint: " + hint
			}
			if gateCheck.Hint != "" {
				r.emitStatus(fmt.Sprintf("  %s⚠ %s%s", ColorYellow, gateCheck.Hint, ColorReset))
				result = result + "\n[System: " + gateCheck.Hint + "]"
			}

			// Check if Reflector posted feedback about this action
			if reflection, ok := r.Board.Get("reflection"); ok {
				if msg, isStr := reflection.(string); isStr && msg != "" {
					r.emitStatus(fmt.Sprintf("  %s⚠ %s%s", ColorYellow, msg, ColorReset))
					result = result + "\n[Reflection: " + msg + "]"
					r.Board.Delete("reflection")
				}
			}

			// Send tool result back via native API
			if usedNativeAPI {
				r.Conv.NativeToolResult(tc.Name, result)
			}

			// Compress and add to pipeline (used by both paths)
			pipe.AddStep(tc.Name, result)

			// Force stop if gate says so
			if gateCheck.ForceStop {
				r.emitStatus(fmt.Sprintf("  %s⚠ forcing final answer%s", ColorYellow, ColorReset))
				// Phantom reasoning: pre-compute chain from pipeline evidence
				forceInstruction := "[System: You MUST give your final answer now. No more tool calls. Use the evidence you already collected and state any remaining uncertainty plainly.]"
				if r.Phantom != nil {
					chain := r.Phantom.BuildChainFromPipeline(percept.Raw, pipe)
					if chain.CanBypass {
						r.publishAnswer(chain.DirectAnswer)
						r.finishReasoning(percept, pipe, chain.DirectAnswer)
						return nil
					}
					if chain.FullContext != "" {
						forceInstruction = chain.FullContext + "\n" + forceInstruction
					}
				}
				answer, ok, err := r.finalAnswerFromEvidence(percept.Raw, assistantFacts, memoryFacts, pipe,
					forceInstruction,
				)
				if err != nil {
					return err
				}
				if ok {
					answer = r.firewallCheck(answer, pipe)
					r.publishAnswer(answer)
					r.finishReasoning(percept, pipe, answer)
					return nil
				}
				r.publishAnswer("(unable to finalize from collected evidence)")
				return nil
			}
		}

		// Track native conversation for continuation
		if usedNativeAPI {
			nativeConv = r.Conv
		} else {
			nativeConv = nil
		}
	}

	r.emitStatus(fmt.Sprintf("  %s⚠ reached maximum tool iterations; finalizing from evidence%s", ColorYellow, ColorReset))
	// Phantom reasoning: pre-compute chain from all pipeline evidence
	maxIterInstruction := "[System: You have reached the tool limit. Give the best final answer you can from the evidence collected so far. Do not call tools.]"
	if r.Phantom != nil {
		chain := r.Phantom.BuildChainFromPipeline(percept.Raw, pipe)
		if chain.CanBypass {
			r.publishAnswer(chain.DirectAnswer)
			r.finishReasoning(percept, pipe, chain.DirectAnswer)
			return nil
		}
		if chain.FullContext != "" {
			maxIterInstruction = chain.FullContext + "\n" + maxIterInstruction
		}
	}
	if answer, ok, err := r.finalAnswerFromEvidence(percept.Raw, assistantFacts, memoryFacts, pipe,
		maxIterInstruction,
	); err == nil && ok {
		answer = r.firewallCheck(answer, pipe)
		r.publishAnswer(answer)
		r.finishReasoning(percept, pipe, answer)
		return nil
	} else if err != nil {
		return err
	}
	r.publishAnswer("(reached maximum tool iterations)")
	return nil
}

func (r *Reasoner) finalAnswerFromEvidence(userQuery, assistantFacts, memoryFacts string, pipe *Pipeline, instruction string) (string, bool, error) {
	finalConv := NewConversation(10)
	var sys strings.Builder
	// Use distilled final-answer prompt if available
	if r.PromptDist != nil {
		lang := ""
		if CurrentProject != nil {
			lang = CurrentProject.Language
		}
		sys.WriteString(r.PromptDist.BuildFinalAnswerPrompt(lang))
		sys.WriteString("\n")
	} else {
		sys.WriteString("You are Nous in final-answer mode. Give a direct plain-text answer using only the evidence already collected. Do not call tools. Do not emit JSON. If evidence is incomplete, say so clearly.\n")
	}
	if assistantFacts != "" {
		sys.WriteString("\n")
		sys.WriteString(assistantFacts)
	}
	if memoryFacts != "" {
		sys.WriteString("\n")
		sys.WriteString(memoryFacts)
	}
	finalConv.System(strings.TrimSpace(sys.String()))

	userMsg := userQuery
	if pipeCtx := pipe.BuildContext(); pipeCtx != "" {
		userMsg += "\n\nEvidence summary:\n" + pipeCtx
	}
	if instruction != "" {
		userMsg += "\n\n" + instruction
	}
	finalConv.User(userMsg)
	if pipe.StepCount() > 0 && pipe.LastResult() != "" {
		lastStep := pipe.steps[len(pipe.steps)-1]
		finalConv.ToolResult(lastStep.ToolName, SmartTruncate(lastStep.ToolName, lastStep.RawResult))
	}

	r.Conv = finalConv
	resp, err := r.callLLM()
	if err != nil {
		return "", false, err
	}

	parsed := r.parseResponse(resp)
	if len(parsed.ToolCalls) > 0 {
		return "", false, nil
	}

	answer := strings.TrimSpace(parsed.Answer)
	if answer == "" {
		answer = strings.TrimSpace(resp)
	}
	if answer == "" || looksLikeFailedToolCall(answer) {
		return "", false, nil
	}

	finalConv.Assistant(answer)
	return answer, true, nil
}

// firewallCheck validates and corrects an LLM response using the cognitive firewall.
func (r *Reasoner) firewallCheck(answer string, pipe *Pipeline) string {
	if r.Firewall == nil {
		return answer
	}

	ctx := &FirewallContext{
		Query:    r.currentInput,
		Response: answer,
	}

	// Populate tool results from pipeline
	if pipe != nil {
		for _, s := range pipe.steps {
			ctx.ToolResults = append(ctx.ToolResults, FirewallToolResult{
				Tool:   s.ToolName,
				Result: s.RawResult,
			})
		}
	}

	// Detect language from project
	if CurrentProject != nil {
		ctx.Language = CurrentProject.Language
	}

	corrected, _ := r.Firewall.Validate(ctx)
	return corrected
}

func (r *Reasoner) tryResearchAndWrite(input string) (string, bool, error) {
	lower := strings.ToLower(strings.TrimSpace(input))
	if !strings.Contains(lower, "research") || !strings.Contains(lower, "create") || !strings.Contains(lower, "file") {
		return "", false, nil
	}
	if !strings.Contains(lower, "online") && !strings.Contains(lower, "web") {
		return "", false, nil
	}
	path := inferRequestedPath(input, "")
	if path == "" {
		return "", false, nil
	}
	topic := inferResearchTopic(input, path)
	if topic == "" {
		return "", false, nil
	}
	url := inferURL(input)
	if url == "" {
		url = wikipediaSummaryURL(topic)
	}

	fetchRes, err := r.executeTool(toolCall{Name: "fetch", Args: map[string]string{"url": url}})
	if err != nil {
		return "", true, err
	}
	content := buildResearchMarkdown(topic, url, fetchRes)
	if _, err := r.executeTool(toolCall{Name: "write", Args: map[string]string{"path": path, "content": content}}); err != nil {
		return "", true, err
	}
	return fmt.Sprintf("Done — I researched %s online and created %q.", topic, path), true, nil
}

func inferResearchTopic(input, path string) string {
	lower := strings.ToLower(input)
	for _, marker := range []string{"about ", "on ", "research "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			rest := input[idx+len(marker):]
			for _, stopper := range []string{" online", " and create", " and write", " into ", " in the current folder", ",", "."} {
				if end := strings.Index(strings.ToLower(rest), stopper); end >= 0 {
					rest = rest[:end]
					break
				}
			}
			rest = strings.TrimSpace(rest)
			if rest != "" && !strings.Contains(rest, ".md") {
				words := strings.Fields(rest)
				if len(words) > 5 {
					words = words[:5]
				}
				return strings.Join(words, " ")
			}
		}
	}
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	base = strings.ReplaceAll(base, "-", " ")
	base = strings.ReplaceAll(base, "_", " ")
	return strings.TrimSpace(base)
}

func buildResearchMarkdown(topic, url, fetched string) string {
	trimmed := strings.TrimSpace(fetched)
	if strings.HasPrefix(trimmed, "{") {
		var payload struct {
			Title   string `json:"title"`
			Extract string `json:"extract"`
		}
		if err := json.Unmarshal([]byte(trimmed), &payload); err == nil && strings.TrimSpace(payload.Extract) != "" {
			bullets := make([]string, 0, 3)
			for _, part := range strings.Split(payload.Extract, ". ") {
				part = strings.TrimSpace(part)
				if part == "" {
					continue
				}
				bullets = append(bullets, "- "+strings.TrimSuffix(part, ".")+".")
				if len(bullets) >= 3 {
					break
				}
			}
			heading := strings.Title(topic)
			if strings.TrimSpace(payload.Title) != "" {
				heading = payload.Title
			}
			return fmt.Sprintf("# %s\n\n## Summary\n%s\n\n## Sources\n- %s\n", heading, strings.Join(bullets, "\n"), url)
		}
	}

	lines := strings.FieldsFunc(strings.TrimSpace(fetched), func(r rune) bool {
		return r == '\n' || r == '\r'
	})
	text := strings.Join(lines, " ")
	text = strings.Join(strings.Fields(text), " ")
	parts := strings.Split(text, ". ")
	bullets := make([]string, 0, 3)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		bullets = append(bullets, "- "+strings.TrimSuffix(part, ".")+".")
		if len(bullets) >= 3 {
			break
		}
	}
	if len(bullets) == 0 {
		bullets = []string{"- " + topic + " is a topic researched from online sources."}
	}
	return fmt.Sprintf("# %s\n\n## Summary\n%s\n\n## Sources\n- %s\n", strings.Title(topic), strings.Join(bullets, "\n"), url)
}

func wikipediaSummaryURL(topic string) string {
	return "https://en.wikipedia.org/api/rest_v1/page/summary/" + strings.ReplaceAll(slugifyWords(topic), "-", "_")
}

// llmResult holds the response from an LLM call, including any native tool calls.
type llmResult struct {
	Content   string
	ToolCalls []ollama.ToolCall
}

// callLLM calls the model with streaming or batch mode, using native tool calling.
func (r *Reasoner) callLLM() (string, error) {
	res, err := r.callLLMNative()
	if err != nil {
		return "", err
	}
	return res.Content, err
}

func (r *Reasoner) publishAnswer(answer string) {
	r.Board.Set("last_answer", answer)

	if key, ok := r.Board.Get("answer_key"); ok {
		if s, ok := key.(string); ok && strings.TrimSpace(s) != "" {
			r.Board.Set(s, answer)
		}
	}
}

// callLLMNative calls the model and returns both content and native tool calls.
func (r *Reasoner) callLLMNative() (*llmResult, error) {
	nativeTools := r.buildNativeTools()
	if r.OnToken != nil {
		return r.streamCallNative(nativeTools)
	}
	return r.batchCallNative(nativeTools)
}

// compressOldTurns compresses the oldest conversation turns to free context space.
func (r *Reasoner) compressOldTurns() {
	msgs := r.Conv.Messages()
	if len(msgs) < 5 {
		return
	}

	// Compress using LLM if compressor is available
	if r.Compressor != nil {
		// Gather oldest 4 non-system messages
		var toCompress []string
		count := 0
		for _, m := range msgs[1:] {
			if count >= 4 {
				break
			}
			toCompress = append(toCompress, m.Role+": "+m.Content)
			count++
		}
		combined := strings.Join(toCompress, "\n")
		atom, err := r.Compressor.Compress(combined, "")
		if err == nil && atom != nil {
			r.Conv.CompressOldest(4, atom.Content)
			r.emitStatus(fmt.Sprintf("  %scompressed context%s", ColorDim, ColorReset))
			return
		}
	}

	// Fallback: rule-based compression (no LLM call)
	var summary []string
	count := 0
	for _, m := range msgs[1:] {
		if count >= 4 {
			break
		}
		// Keep first line of each message as summary
		first := strings.SplitN(m.Content, "\n", 2)[0]
		if len(first) > 100 {
			first = first[:100] + "..."
		}
		summary = append(summary, m.Role+": "+first)
		count++
	}
	r.Conv.CompressOldest(4, strings.Join(summary, "\n"))
	r.emitStatus(fmt.Sprintf("  %scompressed context%s", ColorDim, ColorReset))
}

func recentConversationContext(msgs []ollama.Message, limit int) string {
	if len(msgs) == 0 || limit <= 0 {
		return ""
	}

	lines := make([]string, 0, limit)
	for i := len(msgs) - 1; i >= 0 && len(lines) < limit; i-- {
		msg := msgs[i]
		if msg.Role != "user" && msg.Role != "assistant" {
			continue
		}
		content := strings.TrimSpace(msg.Content)
		if content == "" || strings.HasPrefix(content, "OBSERVE [") || strings.HasPrefix(content, "[Earlier context]") {
			continue
		}
		first := strings.SplitN(content, "\n", 2)[0]
		if len(first) > 180 {
			first = first[:180] + "..."
		}
		role := "User"
		if msg.Role == "assistant" {
			role = "Assistant"
		}
		lines = append(lines, fmt.Sprintf("%s: %s", role, first))
	}

	for i, j := 0, len(lines)-1; i < j; i, j = i+1, j-1 {
		lines[i], lines[j] = lines[j], lines[i]
	}
	return strings.Join(lines, "\n")
}

func (r *Reasoner) toolPrompt() string {
	var sb strings.Builder
	sb.WriteString("Available tools:\n")
	for _, t := range r.Tools.List() {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
	}
	return sb.String()
}

// toolParamSchema defines the parameters for each tool for the native tool calling API.
// Extracted from the Description strings in builtin.go.
var toolParamSchema = map[string]ollama.ToolFunctionParams{
	"read": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":   {Type: "string", Description: "File path to read"},
		"offset": {Type: "string", Description: "Line number to start from"},
		"limit":  {Type: "string", Description: "Max lines to read"},
	}, Required: []string{"path"}},
	"write": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":    {Type: "string", Description: "File path to create/overwrite"},
		"content": {Type: "string", Description: "File content to write"},
	}, Required: []string{"path", "content"}},
	"edit": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path": {Type: "string", Description: "File path to edit"},
		"old":  {Type: "string", Description: "Exact text to find"},
		"new":  {Type: "string", Description: "Replacement text"},
		"line": {Type: "string", Description: "Line number for context matching"},
	}, Required: []string{"path", "old", "new"}},
	"glob": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"pattern": {Type: "string", Description: "Glob pattern (e.g. '**/*.go')"},
		"path":    {Type: "string", Description: "Base directory"},
	}, Required: []string{"pattern"}},
	"grep": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"pattern": {Type: "string", Description: "Regex search pattern"},
		"path":    {Type: "string", Description: "Directory to search in"},
		"glob":    {Type: "string", Description: "File filter (e.g. '*.go')"},
	}, Required: []string{"pattern"}},
	"ls": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path": {Type: "string", Description: "Directory path"},
	}},
	"shell": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"command": {Type: "string", Description: "Shell command to execute"},
	}, Required: []string{"command"}},
	"mkdir": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path": {Type: "string", Description: "Directory path to create"},
	}, Required: []string{"path"}},
	"tree": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":  {Type: "string", Description: "Root directory"},
		"depth": {Type: "string", Description: "Max depth (default 3)"},
	}},
	"fetch": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"url": {Type: "string", Description: "URL to fetch"},
	}, Required: []string{"url"}},
	"run": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"command": {Type: "string", Description: "Command to execute"},
		"stdin":   {Type: "string", Description: "Standard input"},
	}, Required: []string{"command"}},
	"sysinfo": {Type: "object", Properties: map[string]ollama.ToolProperty{}},
	"find_replace": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":        {Type: "string", Description: "File path"},
		"pattern":     {Type: "string", Description: "Regex pattern to find"},
		"replacement": {Type: "string", Description: "Replacement string"},
		"all":         {Type: "string", Description: "Replace all occurrences ('true')"},
	}, Required: []string{"path", "pattern", "replacement"}},
	"git": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"command": {Type: "string", Description: "Git subcommand (e.g. 'status', 'diff', 'log --oneline -10')"},
	}, Required: []string{"command"}},
	"patch": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":   {Type: "string", Description: "File path"},
		"before": {Type: "string", Description: "Exact multi-line text to find"},
		"after":  {Type: "string", Description: "Replacement text"},
	}, Required: []string{"path", "before", "after"}},
	"replace_all": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"old":  {Type: "string", Description: "String to find"},
		"new":  {Type: "string", Description: "Replacement string"},
		"glob": {Type: "string", Description: "File pattern (e.g. '*.go')"},
	}, Required: []string{"old", "new"}},
	"diff": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"path":   {Type: "string", Description: "File path"},
		"staged": {Type: "string", Description: "Show staged changes ('true')"},
	}},
	"clipboard": {Type: "object", Properties: map[string]ollama.ToolProperty{
		"action":  {Type: "string", Description: "Action to perform", Enum: []string{"read", "write"}},
		"content": {Type: "string", Description: "Content to write to clipboard"},
	}, Required: []string{"action"}},
}

// buildNativeTools converts the tool registry into Ollama native tool definitions.
func (r *Reasoner) buildNativeTools() []ollama.Tool {
	toolList := r.activeTools
	if len(toolList) == 0 {
		toolList = r.Tools.List()
	}

	var nativeTools []ollama.Tool
	for _, t := range toolList {
		params, ok := toolParamSchema[t.Name]
		if !ok {
			// Fallback: empty params for unknown tools
			params = ollama.ToolFunctionParams{
				Type:       "object",
				Properties: map[string]ollama.ToolProperty{},
			}
		}
		nativeTools = append(nativeTools, ollama.Tool{
			Type: "function",
			Function: ollama.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return nativeTools
}

// WorkDir is set by main to inform the system prompt.
var WorkDir string

// compactSystemPrompt returns a focused system prompt optimized for small models.
// Uses the THINK/ACT/OBSERVE protocol and progressive tool disclosure to
// minimize token usage while maximizing reasoning quality.
func (r *Reasoner) compactSystemPrompt() string {
	// Adaptive Prompt Distillation: if distiller available, classify and build minimal prompt
	if r.PromptDist != nil && r.currentInput != "" {
		class := r.PromptDist.Classify(r.currentInput)
		toolList := r.toolPrompt()
		if len(r.activeTools) > 0 {
			toolList = ToolPromptForSubset(r.activeTools)
		}
		lang := ""
		if CurrentProject != nil {
			lang = CurrentProject.Language
		}
		return r.PromptDist.BuildSystemPrompt(class, toolList, lang)
	}

	// Fallback: original monolithic prompt
	wd := WorkDir
	if wd == "" {
		wd = "."
	}

	// Use progressive tool disclosure — only show selected tools
	toolList := r.toolPrompt()
	if len(r.activeTools) > 0 {
		toolList = ToolPromptForSubset(r.activeTools)
	}

	// Build minimal system prompt — every token counts for 1.5B models.
	var sb strings.Builder
	sb.WriteString("You are Nous, a personal AI assistant running fully on the user's machine. Be warm, helpful, concise.\n")
	sb.WriteString("You have vast knowledge and tools. You grow and learn with every interaction.\n")
	sb.WriteString("NEVER repeat instructions. NEVER invent facts — search your knowledge first.\n\n")
	sb.WriteString("RULES:\n")
	sb.WriteString("- Answer from knowledge and memory. If unsure, use tools to find out.\n")
	sb.WriteString("- read/show → read tool | create/write → write tool | find/search → grep/glob tool\n")
	sb.WriteString("- Paths are relative to working directory below\n\n")
	sb.WriteString("Tool call format:\n")
	sb.WriteString(`{"tool": "NAME", "args": {"key": "value"}}`)
	sb.WriteString("\n\n")
	sb.WriteString(toolList)
	sb.WriteString("\nwd: ")
	sb.WriteString(wd)
	sb.WriteString("\n")

	// Inject project context — capped to 200 chars to save tokens
	if CurrentProject != nil {
		ctx := CurrentProject.ContextString()
		if len(ctx) > 200 {
			ctx = ctx[:200]
		}
		sb.WriteString("\n")
		sb.WriteString(ctx)
	}

	return sb.String()
}

type toolCallRaw struct {
	Name string                 `json:"tool"`
	Args map[string]interface{} `json:"args"`
}

type toolCall struct {
	Name string
	Args map[string]string
}

func (r toolCallRaw) normalize() toolCall {
	tc := toolCall{Name: r.Name, Args: make(map[string]string)}
	for k, v := range r.Args {
		tc.Args[k] = fmt.Sprintf("%v", v)
	}
	return tc
}

type parsedResponse struct {
	Think     string
	Answer    string
	ToolCalls []toolCall
}

func (r *Reasoner) parseResponse(content string) parsedResponse {
	var parsed parsedResponse

	// Extract tool calls from ```tool blocks
	parts := strings.Split(content, "```")
	var nonToolParts []string

	for i, part := range parts {
		if i%2 == 1 { // Inside a code block
			trimmed := strings.TrimSpace(part)
			// Check if it starts with "tool" marker
			if strings.HasPrefix(trimmed, "tool\n") || strings.HasPrefix(trimmed, "tool\r\n") {
				jsonStr := strings.TrimPrefix(trimmed, "tool\n")
				jsonStr = strings.TrimPrefix(jsonStr, "tool\r\n")
				jsonStr = strings.TrimSpace(jsonStr)

				var raw toolCallRaw
				if err := json.Unmarshal([]byte(jsonStr), &raw); err == nil && raw.Name != "" {
					parsed.ToolCalls = append(parsed.ToolCalls, raw.normalize())
					continue
				}
			}
			// Also try parsing bare JSON blocks
			if strings.HasPrefix(trimmed, "json\n") || strings.HasPrefix(trimmed, "json\r\n") {
				jsonStr := strings.TrimPrefix(trimmed, "json\n")
				jsonStr = strings.TrimPrefix(jsonStr, "json\r\n")
				jsonStr = strings.TrimSpace(jsonStr)

				var raw toolCallRaw
				if err := json.Unmarshal([]byte(jsonStr), &raw); err == nil && raw.Name != "" {
					parsed.ToolCalls = append(parsed.ToolCalls, raw.normalize())
					continue
				}
			}
		}
		nonToolParts = append(nonToolParts, part)
	}

	// Also try to find inline JSON tool calls (for models that don't use code blocks)
	if len(parsed.ToolCalls) == 0 {
		parsed.ToolCalls = r.findInlineToolCalls(content)
	}

	// Extract THINK/ACT/ANSWER from non-tool text
	text := strings.Join(nonToolParts, "")
	var thinkParts []string
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		upper := strings.ToUpper(trimmed)

		if strings.HasPrefix(upper, "THINK:") {
			thinkParts = append(thinkParts, strings.TrimSpace(trimmed[6:]))
		} else if strings.HasPrefix(upper, "ACT:") {
			// Parse ACT: line as inline tool call JSON
			actJSON := strings.TrimSpace(trimmed[4:])
			var raw toolCallRaw
			if err := json.Unmarshal([]byte(actJSON), &raw); err == nil && raw.Name != "" {
				parsed.ToolCalls = append(parsed.ToolCalls, raw.normalize())
			}
		} else if strings.HasPrefix(upper, "ANSWER:") {
			parsed.Answer = strings.TrimSpace(trimmed[7:])
		}
	}
	if len(thinkParts) > 0 {
		parsed.Think = strings.Join(thinkParts, " ")
	}

	// If no explicit ANSWER but also no tool calls, the whole text is the answer
	if parsed.Answer == "" && len(parsed.ToolCalls) == 0 {
		parsed.Answer = strings.TrimSpace(text)
	}

	return parsed
}

func (r *Reasoner) findInlineToolCalls(content string) []toolCall {
	var calls []toolCall

	// Look for {"tool": "..."} patterns in the text
	for i := 0; i < len(content); i++ {
		if content[i] != '{' {
			continue
		}

		// Find matching closing brace
		depth := 0
		for j := i; j < len(content); j++ {
			if content[j] == '{' {
				depth++
			} else if content[j] == '}' {
				depth--
				if depth == 0 {
					candidate := content[i : j+1]
					var raw toolCallRaw
					if err := json.Unmarshal([]byte(candidate), &raw); err == nil && raw.Name != "" {
						calls = append(calls, raw.normalize())
					}
					i = j
					break
				}
			}
		}
	}

	return calls
}

func (r *Reasoner) executeTool(tc toolCall) (string, error) {
	tool, err := r.Tools.Get(tc.Name)
	if err != nil {
		// Fuzzy match: try to find the closest tool name
		if corrected := r.fuzzyMatchTool(tc.Name); corrected != "" {
			r.emitStatus(fmt.Sprintf("↳ corrected %s → %s", tc.Name, corrected))
			tc.Name = corrected
			tool, err = r.Tools.Get(corrected)
		}
		if err != nil {
			return "", fmt.Errorf("unknown tool %q (available: %s)", tc.Name, r.availableToolNames())
		}
	}

	if tc.Args == nil {
		tc.Args = make(map[string]string)
	}

	// Auto-correct common argument name mistakes from small models
	tc.Args = correctArgNames(tc.Name, tc.Args)
	inferMissingToolArgs(tc.Name, tc.Args, r.currentInput)

	// Validate required arguments are present
	if missing := validateRequiredArgs(tc.Name, tc.Args); missing != "" {
		return "", fmt.Errorf("missing required argument %q for tool %q", missing, tc.Name)
	}

	// Check predictive cache first (read-only tools only)
	if r.Predictor != nil {
		if cached, ok := r.Predictor.Lookup(tc.Name, tc.Args); ok {
			r.emitStatus("↳ cached")
			return cached, nil
		}
	}

	// Check if this is a dangerous action requiring confirmation
	if r.Confirm != nil {
		if reason, dangerous := IsDangerous(tc.Name); dangerous {
			action := fmt.Sprintf("%s — %s", reason, formatArgs(tc.Args))
			// Generate diff preview for file-modifying tools
			detail := r.buildDiffPreview(tc)
			if !r.Confirm(action, detail) {
				return "Action denied by user.", nil
			}
		}
	}

	result, execErr := tool.Execute(tc.Args)

	// Feed result to predictor for speculative pre-computation
	if r.Predictor != nil && execErr == nil {
		r.Predictor.Predict(tc.Name, tc.Args, result)
	}

	return result, execErr
}

func inferMissingToolArgs(toolName string, args map[string]string, input string) {
	switch toolName {
	case "write":
		if strings.TrimSpace(args["path"]) == "" {
			if inferred := inferRequestedPath(input, args["content"]); inferred != "" {
				args["path"] = inferred
			}
		}
		normalizeRequestedPath(args, input)
	case "fetch":
		if strings.TrimSpace(args["url"]) == "" {
			if inferred := inferURL(input); inferred != "" {
				args["url"] = inferred
			}
		}
	}
}

func normalizeRequestedPath(args map[string]string, input string) {
	path := strings.TrimSpace(args["path"])
	if path == "" {
		return
	}
	if strings.HasPrefix(path, "/") && strings.Count(path, "/") == 1 {
		lower := strings.ToLower(input)
		if strings.Contains(lower, "current folder") || strings.Contains(lower, "this folder") {
			args["path"] = strings.TrimPrefix(path, "/")
		}
	}
}

var explicitFilePattern = regexp.MustCompile(`(?i)(?:named|called)\s+["']?([a-z0-9_./-]+\.[a-z0-9]+)["']?`)
var quotedFilePattern = regexp.MustCompile(`["']([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)["']`)
var fileReferencePattern = regexp.MustCompile(`[a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+`)
var urlPattern = regexp.MustCompile(`https?://[^\s"']+`)

func inferRequestedPath(input string, content string) string {
	if match := explicitFilePattern.FindStringSubmatch(input); len(match) == 2 {
		return match[1]
	}
	if match := quotedFilePattern.FindStringSubmatch(input); len(match) == 2 {
		return match[1]
	}
	lower := strings.ToLower(input)
	ext := ".txt"
	if strings.Contains(lower, "markdown") || strings.Contains(lower, ".md") {
		ext = ".md"
	}
	if strings.Contains(lower, "file about ") {
		topic := afterPhrase(lower, "file about ")
		if slug := slugifyWords(topic); slug != "" {
			return slug + ext
		}
	}
	if strings.Contains(lower, "about ") {
		topic := afterPhrase(lower, "about ")
		if slug := slugifyWords(topic); slug != "" {
			return slug + ext
		}
	}
	if strings.HasPrefix(strings.TrimSpace(content), "#") {
		heading := strings.TrimSpace(strings.TrimLeft(strings.SplitN(content, "\n", 2)[0], "#"))
		if slug := slugifyWords(strings.ToLower(heading)); slug != "" {
			return slug + ext
		}
	}
	return ""
}

func inferURL(input string) string {
	if match := urlPattern.FindString(input); match != "" {
		return match
	}
	return ""
}

func afterPhrase(text string, phrase string) string {
	idx := strings.Index(text, phrase)
	if idx < 0 {
		return ""
	}
	rest := text[idx+len(phrase):]
	for _, stopper := range []string{" in ", " with ", " containing ", " and ", ".", ","} {
		if end := strings.Index(rest, stopper); end >= 0 {
			rest = rest[:end]
			break
		}
	}
	return strings.TrimSpace(rest)
}

func slugifyWords(text string) string {
	words := strings.FieldsFunc(strings.ToLower(strings.TrimSpace(text)), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	if len(words) == 0 {
		return ""
	}
	if len(words) > 4 {
		words = words[:4]
	}
	return strings.Join(words, "-")
}

// buildDiffPreview generates a colored diff preview for file-modifying tools.
// Returns an empty string if a preview cannot be generated.
func (r *Reasoner) buildDiffPreview(tc toolCall) string {
	wd := WorkDir
	if wd == "" {
		wd = "."
	}

	resolvePath := func(path string) string {
		if filepath.IsAbs(path) {
			return path
		}
		return filepath.Join(wd, path)
	}

	switch tc.Name {
	case "write":
		path := tc.Args["path"]
		content := tc.Args["content"]
		if path == "" || content == "" {
			return ""
		}
		resolved := resolvePath(path)
		oldData, err := os.ReadFile(resolved)
		if err != nil {
			// New file — show write preview
			return FormatWritePreview(path, content)
		}
		return FormatEditPreview(path, string(oldData), content)

	case "edit":
		path := tc.Args["path"]
		oldText := tc.Args["old"]
		newText := tc.Args["new"]
		if path == "" || oldText == "" {
			return ""
		}
		resolved := resolvePath(path)
		data, err := os.ReadFile(resolved)
		if err != nil {
			return ""
		}
		oldContent := string(data)
		if !strings.Contains(oldContent, oldText) {
			return ""
		}
		newContent := strings.Replace(oldContent, oldText, newText, 1)
		return FormatEditPreview(path, oldContent, newContent)

	case "patch":
		path := tc.Args["path"]
		before := tc.Args["before"]
		after := tc.Args["after"]
		if path == "" || before == "" {
			return ""
		}
		resolved := resolvePath(path)
		data, err := os.ReadFile(resolved)
		if err != nil {
			return ""
		}
		oldContent := string(data)
		if !strings.Contains(oldContent, before) {
			return ""
		}
		newContent := strings.Replace(oldContent, before, after, 1)
		return FormatEditPreview(path, oldContent, newContent)

	case "find_replace":
		// find_replace uses regex, so we can't easily preview without importing regexp.
		// Return empty — the confirmation still shows the action detail.
		return ""
	}

	return ""
}

func formatArgs(args map[string]string) string {
	if path, ok := args["path"]; ok {
		return path
	}
	if cmd, ok := args["command"]; ok {
		return cmd
	}
	parts := make([]string, 0, len(args))
	for k, v := range args {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, " ")
}

// modelOpts returns tuned ModelOptions for small model reliability.
// RepeatPenalty prevents the repetition loops that plague 1.5B models.
// NumPredict is kept modest to encourage tool calls over long prose.
func (r *Reasoner) modelOpts() *ollama.ModelOptions {
	return &ollama.ModelOptions{
		Temperature:   0.5,
		NumPredict:    512,
		RepeatPenalty: 1.15,
		RepeatLastN:   64,
	}
}

func (r *Reasoner) streamCallNative(nativeTools []ollama.Tool) (*llmResult, error) {
	var full strings.Builder
	var allToolCalls []ollama.ToolCall
	opts := r.modelOpts()
	_, err := r.LLM.ChatStreamWithTools(r.Conv.Messages(), nativeTools, opts,
		func(token string, toolCalls []ollama.ToolCall, done bool) {
			if token != "" {
				full.WriteString(token)
				if r.OnToken != nil {
					r.OnToken(token, done)
				}
			}
			if len(toolCalls) > 0 {
				allToolCalls = append(allToolCalls, toolCalls...)
			}
		})
	if err != nil {
		return nil, err
	}
	return &llmResult{Content: full.String(), ToolCalls: allToolCalls}, nil
}

func (r *Reasoner) batchCallNative(nativeTools []ollama.Tool) (*llmResult, error) {
	resp, err := r.LLM.ChatWithTools(r.Conv.Messages(), nativeTools, r.modelOpts())
	if err != nil {
		return nil, err
	}
	return &llmResult{Content: resp.Message.Content, ToolCalls: resp.Message.ToolCalls}, nil
}

// Legacy methods kept for compatibility with non-tool-calling code paths.
func (r *Reasoner) streamCall() (string, error) {
	res, err := r.streamCallNative(nil)
	if err != nil {
		return "", err
	}
	return res.Content, nil
}

func (r *Reasoner) batchCall() (string, error) {
	res, err := r.batchCallNative(nil)
	if err != nil {
		return "", err
	}
	return res.Content, nil
}

func (r *Reasoner) emitStatus(msg string) {
	if r.OnStatus != nil {
		r.OnStatus(msg)
	}
}

// recallKeyFacts retrieves essential facts for the system prompt using
// semantic similarity when embeddings are available, falling back to
// recency-based retrieval otherwise.
func (r *Reasoner) recallKeyFacts(input string) string {
	var facts []string

	if r.WorkingMem != nil {
		// Try semantic retrieval first — find slots relevant to this query
		var items []memory.Slot
		if r.WorkingMem.Size() > 0 {
			// Attempt semantic search via embedding
			if r.LLM != nil {
				vec, err := r.LLM.Embed(input)
				if err == nil && len(vec) > 0 {
					items = r.WorkingMem.SemanticSearch(vec, 5)
				}
			}
			// Fallback to recency-based if semantic search didn't produce results
			if len(items) == 0 {
				items = r.WorkingMem.MostRelevant(5)
			}
		}

		var memLines []string
		for _, item := range items {
			if strings.HasPrefix(item.Key, "user_identity") {
				facts = append(facts, fmt.Sprintf("User info: %v", item.Value))
			} else {
				memLines = append(memLines, fmt.Sprintf("- %s: %v", item.Key, item.Value))
			}
		}
		if len(memLines) > 0 {
			facts = append(facts, "[Working memory]\n"+strings.Join(memLines, "\n"))
		}
	}

	// Recall from episodic memory (semantic search across past interactions)
	if r.EpisodicMem != nil && r.EpisodicMem.Size() > 0 {
		episodes := r.EpisodicMem.Search(input, 2)
		if len(episodes) > 0 {
			var epLines []string
			for _, ep := range episodes {
				summary := ep.Input
				if len(summary) > 60 {
					summary = summary[:60] + "..."
				}
				answer := ep.Output
				if len(answer) > 80 {
					answer = answer[:80] + "..."
				}
				epLines = append(epLines, fmt.Sprintf("- Q: %s → A: %s", summary, answer))
			}
			facts = append(facts, "[Past interactions]\n"+strings.Join(epLines, "\n"))
		}
	}

	// Get relevant codebase context — provide 5 symbols so the model knows where to look
	if r.CodeIndex != nil && r.CodeIndex.Size() > 0 {
		indexCtx := r.CodeIndex.RelevantContext(input, 5)
		if indexCtx != "" {
			facts = append(facts, indexCtx)
		}
	}

	// Virtual Context Engine — weave relevant context from ALL sources
	// (knowledge, personal growth, episodic memory) into the prompt
	if r.VCtx != nil {
		assembly := r.VCtx.Weave(input)
		if prompt := assembly.FormatForPrompt(); prompt != "" {
			facts = append(facts, prompt)
		}
	} else if r.Knowledge != nil {
		// Fallback: direct knowledge lookup if virtual context not available
		if kCtx := r.knowledgeLookup(input); kCtx != "" {
			facts = append(facts, kCtx)
		}
	}

	if len(facts) == 0 {
		return ""
	}
	return strings.Join(facts, "\n")
}

// recallMemories retrieves relevant context from working and long-term memory.
func (r *Reasoner) recallMemories(input string) string {
	var parts []string

	// Recall from working memory (most relevant items)
	if r.WorkingMem != nil {
		items := r.WorkingMem.MostRelevant(5)
		if len(items) > 0 {
			var memLines []string
			for _, item := range items {
				memLines = append(memLines, fmt.Sprintf("- %s → %v", item.Key, item.Value))
			}
			parts = append(parts, "[Previous context — use this to answer]\n"+strings.Join(memLines, "\n"))
		}
	}

	// Recall from long-term memory (keyword match)
	if r.LongTermMem != nil {
		words := strings.Fields(strings.ToLower(input))
		seen := make(map[string]bool)
		var ltmLines []string
		for _, word := range words {
			if len(word) < 3 {
				continue
			}
			entries := r.LongTermMem.Search(word)
			for _, e := range entries {
				if !seen[e.Key] {
					seen[e.Key] = true
					ltmLines = append(ltmLines, fmt.Sprintf("- %s: %s", e.Key, e.Value))
				}
			}
			if len(ltmLines) >= 5 {
				break
			}
		}
		if len(ltmLines) > 0 {
			parts = append(parts, "[Long-term Memory]\n"+strings.Join(ltmLines, "\n"))
		}
	}

	// Recall from codebase index (structural context)
	if r.CodeIndex != nil && r.CodeIndex.Size() > 0 {
		indexCtx := r.CodeIndex.RelevantContext(input, 5)
		if indexCtx != "" {
			parts = append(parts, indexCtx)
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// consultPriorKnowledge checks learned patterns and recipes for relevant prior experience.
// This is the key feedback loop: what the Learner extracted gets injected back into reasoning.
func (r *Reasoner) consultPriorKnowledge(intent, rawInput string) string {
	var parts []string

	// Check learned behavioral patterns from the Learner
	if r.Learner != nil {
		patterns := r.Learner.FindRelevantPatterns(intent)
		if len(patterns) > 0 {
			var patternLines []string
			for _, p := range patterns {
				if len(patternLines) >= 3 {
					break
				}
				chain := ""
				for i, tool := range p.ToolChain {
					if i > 0 {
						chain += " → "
					}
					chain += tool
				}
				patternLines = append(patternLines, fmt.Sprintf("- %s (confidence: %.0f%%, used %d times)", chain, p.Confidence*100, p.Uses))
			}
			if len(patternLines) > 0 {
				parts = append(parts, "[Learned Patterns]\nFor similar tasks, these tool sequences worked before:\n"+strings.Join(patternLines, "\n"))
			}
		}
	}

	// Check recipe book for matching multi-step sequences
	if r.Recipes != nil {
		recipes := r.Recipes.Match(intent, rawInput)
		if len(recipes) > 0 {
			best := recipes[0]
			var stepNames []string
			for _, s := range best.Steps {
				stepNames = append(stepNames, s.Tool)
			}
			chain := strings.Join(stepNames, " → ")
			parts = append(parts, fmt.Sprintf("[Recipe: %s]\nTried approach: %s (used %d times, %.0f%% success)", best.Name, chain, best.Uses, best.Confidence()*100))
		}
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, "\n\n")
}

// fuzzyMatchTool finds the closest tool name using simple substring/alias matching.
// Handles common LLM mistakes like "find_replace" → "find_replace", "search" → "grep", etc.
func (r *Reasoner) fuzzyMatchTool(name string) string {
	lower := strings.ToLower(name)

	// Common aliases that 1.5B models generate
	aliases := map[string]string{
		"search":  "grep",
		"find":    "grep",
		"cat":     "read",
		"view":    "read",
		"open":    "read",
		"list":    "ls",
		"dir":     "ls",
		"create":  "write",
		"touch":   "write",
		"exec":    "shell",
		"execute": "shell",
		"cmd":     "shell",
		"command": "shell",
		"bash":    "shell",
		"mv":      "shell",
		"cp":      "shell",
		"rm":      "shell",
		"replace": "find_replace",
		"sed":     "find_replace",
		"status":  "git",
		"commit":  "git",
		"log":     "git",
		"browse":  "fetch",
		"curl":    "fetch",
		"wget":    "fetch",
		"info":    "sysinfo",
		"system":  "sysinfo",
	}

	if corrected, ok := aliases[lower]; ok {
		return corrected
	}

	// Substring match: if the name contains a tool name or vice versa
	for _, t := range r.Tools.List() {
		if strings.Contains(lower, t.Name) || strings.Contains(t.Name, lower) {
			return t.Name
		}
	}

	return ""
}

// availableToolNames returns a comma-separated list of tool names for error messages.
func (r *Reasoner) availableToolNames() string {
	var names []string
	for _, t := range r.Tools.List() {
		names = append(names, t.Name)
	}
	return strings.Join(names, ", ")
}

// correctArgNames fixes common argument name mistakes from small models.
// E.g., "file" → "path", "query" → "pattern", "text" → "content".
func correctArgNames(toolName string, args map[string]string) map[string]string {
	// Common arg name aliases that 1.5B models produce
	argAliases := map[string]string{
		"file":      "path",
		"filename":  "path",
		"filepath":  "path",
		"file_path": "path",
		"name":      "path",
		"query":     "pattern",
		"search":    "pattern",
		"text":      "content",
		"data":      "content",
		"body":      "content",
		"old_text":  "old",
		"new_text":  "new",
		"old_str":   "old",
		"new_str":   "new",
		"original":  "old",
		"updated":   "new",
		"cmd":       "command",
		"dir":       "path",
		"directory": "path",
	}

	// Known arg names per tool — don't correct if the arg name is already valid
	knownArgs := map[string]map[string]bool{
		"read":         {"path": true, "offset": true, "limit": true},
		"write":        {"path": true, "content": true},
		"edit":         {"path": true, "old": true, "new": true, "line": true},
		"grep":         {"pattern": true, "path": true, "glob": true},
		"glob":         {"pattern": true, "path": true},
		"ls":           {"path": true},
		"tree":         {"path": true},
		"mkdir":        {"path": true},
		"shell":        {"command": true},
		"run":          {"command": true},
		"git":          {"command": true},
		"diff":         {"path": true},
		"fetch":        {"url": true},
		"find_replace": {"path": true, "old": true, "new": true},
		"replace_all":  {"path": true, "old": true, "new": true},
		"patch":        {"path": true, "patch": true},
		"sysinfo":      {},
		"clipboard":    {"action": true, "content": true},
	}

	known, hasSchema := knownArgs[toolName]
	if !hasSchema {
		return args
	}

	corrected := make(map[string]string, len(args))
	for k, v := range args {
		if known[k] {
			// Already a valid arg name
			corrected[k] = v
			continue
		}
		// Try to correct
		if replacement, ok := argAliases[strings.ToLower(k)]; ok && known[replacement] {
			if _, exists := corrected[replacement]; !exists {
				corrected[replacement] = v
				continue
			}
		}
		// Keep as-is if no correction found
		corrected[k] = v
	}
	return corrected
}

// validateRequiredArgs checks that required arguments are present for a tool.
// Returns the name of the first missing required arg, or "" if all present.
func validateRequiredArgs(toolName string, args map[string]string) string {
	required := map[string][]string{
		"read":         {"path"},
		"write":        {"path", "content"},
		"edit":         {"path", "old", "new"},
		"grep":         {"pattern"},
		"glob":         {"pattern"},
		"shell":        {"command"},
		"run":          {"command"},
		"git":          {"command"},
		"fetch":        {"url"},
		"find_replace": {"path", "old", "new"},
		"replace_all":  {"path", "old", "new"},
		"patch":        {"path", "patch"},
	}

	reqs, ok := required[toolName]
	if !ok {
		return ""
	}

	for _, req := range reqs {
		if v, exists := args[req]; !exists || strings.TrimSpace(v) == "" {
			return req
		}
	}
	return ""
}

// isCodeQuery returns true if the input looks like a question about code,
// files, functions, implementations, or the repository structure.
func isCodeQuery(input string) bool {
	lower := strings.ToLower(input)
	codeSignals := []string{
		"function", "func ", "method", "struct", "type ",
		"implement", "definition", "defined", "signature",
		"package", "import", "module",
		".go", ".py", ".js", ".ts",
		"read file", "show file", "open file",
		"read the", "show me the", "show the",
		"grep", "search for", "find in", "look for", "where is",
		"what does", "how does", "what is the", "explain",
		"code", "source", "codebase",
		"semantic", "ranking", "memory", "decay",
		"variable", "constant", "return", "parameter",
	}
	for _, sig := range codeSignals {
		if strings.Contains(lower, sig) {
			return true
		}
	}
	return false
}

// preGroundCodeQuery auto-reads the most relevant symbol's definition before
// the first LLM call. Reads a window centered on the symbol's line number so
// even deep-in-file symbols (e.g., line 1180 of a 1400-line file) are captured.
// Returns a grounding hint that should be injected into the user message.
func (r *Reasoner) preGroundCodeQuery(query string, pipe *Pipeline) string {
	if !isCodeQuery(query) {
		return ""
	}
	readTool, err := r.Tools.Get("read")
	if err != nil {
		return ""
	}

	if path := explicitCodePathFromQuery(query); path != "" {
		const fileWindow = 120
		result, readErr := readTool.Execute(map[string]string{
			"path":   path,
			"offset": "0",
			"limit":  fmt.Sprintf("%d", fileWindow),
		})
		if readErr == nil {
			label := fmt.Sprintf("file %s", path)
			r.emitStatus(fmt.Sprintf("  %s↳ pre-read %s%s", ColorDim, label, ColorReset))
			pipe.AddStep("read", result)
			return fmt.Sprintf("[System: I already read %s for you — lines 1-%d. Answer ONLY from the code shown above. Do NOT make additional tool calls. If the answer is not in the code shown, say so.]", path, fileWindow)
		}
	}

	if r.CodeIndex == nil || r.CodeIndex.Size() == 0 {
		return ""
	}

	sym := r.CodeIndex.BestSymbolForQuery(query)
	if sym == nil {
		return ""
	}

	// Execute read tool with offset/limit centered on the symbol

	// Read a 60-line window centered on the symbol definition
	const windowSize = 60
	startLine := sym.Line - windowSize/2
	if startLine < 0 {
		startLine = 0
	}

	args := map[string]string{
		"path":   sym.File,
		"offset": fmt.Sprintf("%d", startLine),
		"limit":  fmt.Sprintf("%d", windowSize),
	}

	result, err := readTool.Execute(args)
	if err != nil {
		return ""
	}

	label := fmt.Sprintf("%s %s [%s:%d]", sym.Kind, sym.Name, sym.File, sym.Line)
	r.emitStatus(fmt.Sprintf("  %s↳ pre-read %s%s", ColorDim, label, ColorReset))
	pipe.AddStep("read", result)

	// Return a grounding constraint for the user message
	return fmt.Sprintf("[System: I already read %s for you — lines %d-%d of %s. "+
		"Answer ONLY from the code shown above. Do NOT make additional tool calls. "+
		"If the answer is not in the code shown, say so.]",
		label, startLine, startLine+windowSize, sym.File)
}

func explicitCodePathFromQuery(query string) string {
	if match := quotedFilePattern.FindStringSubmatch(query); len(match) == 2 {
		return match[1]
	}
	if match := fileReferencePattern.FindString(query); match != "" {
		path := strings.TrimSpace(match)
		path = strings.Trim(path, `"'()[]{}.,:`)
		if strings.Contains(path, "/") || strings.HasPrefix(path, ".") {
			return path
		}
	}
	return ""
}

var packagePattern = regexp.MustCompile(`(?m)^package\s+([A-Za-z_][A-Za-z0-9_]*)`)
var queryFuncPattern = regexp.MustCompile(`(?i)(?:func(?:tion)?\s+)([A-Za-z_][A-Za-z0-9_]*)`)
var readLinePrefixPattern = regexp.MustCompile(`(?m)^\s*\d+\s+\|\s*`)

func deterministicCodeAnswer(query string, pipe *Pipeline) (string, bool) {
	if pipe.StepCount() == 0 || pipe.LastResult() == "" {
		return "", false
	}
	path := explicitCodePathFromQuery(query)
	if path == "" {
		return "", false
	}

	code := stripReadLinePrefixes(pipe.LastResult())
	lower := strings.ToLower(query)
	packageName := ""
	if match := packagePattern.FindStringSubmatch(code); len(match) == 2 {
		packageName = match[1]
	}

	funcMention := ""
	if match := queryFuncPattern.FindStringSubmatch(query); len(match) == 2 {
		funcMention = match[1]
	}
	if funcMention == "" && strings.Contains(lower, "func main") {
		funcMention = "main"
	}

	hasFunc := false
	if funcMention != "" {
		hasFunc = regexp.MustCompile(`\bfunc\s+` + regexp.QuoteMeta(funcMention) + `\s*\(`).MatchString(code)
	}

	if strings.Contains(lower, "package name") && funcMention != "" {
		if packageName == "" {
			return "", false
		}
		if hasFunc {
			return fmt.Sprintf("%s is in package %s, and it defines func %s().", path, packageName, funcMention), true
		}
		return fmt.Sprintf("%s is in package %s, and it does not define func %s() in the code shown.", path, packageName, funcMention), true
	}
	if strings.Contains(lower, "package name") && packageName != "" {
		return fmt.Sprintf("%s is in package %s.", path, packageName), true
	}
	if funcMention != "" {
		if hasFunc {
			return fmt.Sprintf("%s defines func %s().", path, funcMention), true
		}
		return fmt.Sprintf("%s does not define func %s() in the code shown.", path, funcMention), true
	}

	return "", false
}

func stripReadLinePrefixes(code string) string {
	return readLinePrefixPattern.ReplaceAllString(code, "")
}

// looksLikeFailedToolCall checks if a response contains JSON-like patterns
// that suggest the model tried to call a tool but produced malformed output.
func looksLikeFailedToolCall(response string) bool {
	r := strings.ToLower(response)
	// Look for JSON-like fragments that suggest tool call intent
	hasToolHint := strings.Contains(r, `"tool"`) || strings.Contains(r, `"name"`) ||
		strings.Contains(r, `"action"`) || strings.Contains(r, `"function"`)
	hasBrace := strings.Contains(response, "{") && strings.Contains(response, "}")
	return hasToolHint && hasBrace
}

// grammarToolResolve uses schema-constrained decoding to classify the query,
// select a tool, and extract arguments — each step enforced by a JSON Schema.
// More reliable than structuredToolRetry because it uses exact per-tool schemas
// instead of generic format:"json".
func (r *Reasoner) grammarToolResolve(userQuery string) []toolCall {
	if r.Grammar == nil {
		return nil
	}

	toolNames := r.availableToolNames()
	if toolNames == "" {
		return nil
	}

	available := strings.Split(toolNames, ", ")
	result, err := r.Grammar.Resolve(userQuery, available)
	if err != nil {
		return nil
	}

	if result.Tool == "" || result.Tool == "chat" || result.Tool == "explain" {
		return nil
	}

	r.emitStatus(fmt.Sprintf("  %s↳ grammar-resolved: %s%s", ColorDim, result.Tool, ColorReset))

	args := make(map[string]string)
	for k, v := range result.Args {
		args[k] = v
	}

	return []toolCall{{Name: result.Tool, Args: args}}
}

// structuredToolRetry uses format:"json" to force a clean tool call when
// fallback parsing failed. Returns parsed tool calls or nil.
func (r *Reasoner) structuredToolRetry(userQuery string) []toolCall {
	if r.LLM == nil {
		return nil
	}

	prompt := fmt.Sprintf(`You must respond with a JSON object. Choose one:

To use a tool: {"tool": "TOOL_NAME", "args": {"key": "value"}}
To give a final answer: {"answer": "your answer text"}

Available tools: %s

User request: %s`, r.availableToolNames(), userQuery)

	resp, err := r.LLM.ChatJSON([]ollama.Message{
		{Role: "user", Content: prompt},
	}, &ollama.ModelOptions{
		Temperature: 0.1,
		NumPredict:  256,
	})
	if err != nil {
		return nil
	}

	content := strings.TrimSpace(resp.Message.Content)
	if content == "" {
		return nil
	}

	var raw toolCallRaw
	if err := json.Unmarshal([]byte(content), &raw); err == nil && raw.Name != "" {
		return []toolCall{raw.normalize()}
	}

	return nil
}

// finishReasoning records recipes, emits goal completion for the Learner, and stores to memory.
func (r *Reasoner) finishReasoning(percept blackboard.Percept, pipe *Pipeline, finalAnswer string) {
	// Record successful tool sequence as a recipe
	if r.Recipes != nil && pipe.StepCount() >= 2 {
		r.Recipes.Record(pipe, percept.Intent, percept.Raw)
	}

	// Teach the learner from successful tool sequences (direct, no goal overhead)
	if r.Learner != nil && pipe.StepCount() >= 1 {
		var toolNames []string
		for _, s := range pipe.steps {
			if s.ToolName != "" {
				toolNames = append(toolNames, s.ToolName)
			}
		}
		if len(toolNames) > 0 {
			go r.Learner.LearnFromTools(percept.Raw, toolNames)
		}
	}

	// Train neural cortex on successful tool usage
	if r.Cortex != nil && pipe.StepCount() >= 1 {
		go r.trainCortex(percept.Raw, pipe)
	}

	// Personal growth — track user interests and interaction patterns
	if r.Growth != nil {
		go r.Growth.RecordInteraction(percept.Raw)
	}

	// Cross-memory feedback: propagate success across all subsystems
	if r.Feedback != nil && pipe.StepCount() >= 1 {
		var toolNames []string
		for _, s := range pipe.steps {
			if s.ToolName != "" {
				toolNames = append(toolNames, s.ToolName)
			}
		}
		if len(toolNames) > 0 {
			go r.Feedback.OnToolSuccess(percept.Raw, toolNames[0], toolNames)
		}
	}

	r.storeToMemory(percept.Raw, finalAnswer)
}

// trainCortex trains the neural cortex from a successful pipeline execution.
// It uses a simple bag-of-words input vector and trains on the first tool used.
func (r *Reasoner) trainCortex(query string, pipe *Pipeline) {
	if r.Cortex == nil || pipe.StepCount() == 0 {
		return
	}
	firstTool := pipe.steps[0].ToolName
	if firstTool == "" {
		return
	}

	// Build input vector from query (simple character frequency encoding)
	input := CortexInputFromQuery(query, r.Cortex.InputSize)
	r.Cortex.Train(input, firstTool)
}

// CortexInputFromQuery creates a numeric input vector from a query string.
// Uses character frequency encoding normalized to unit length.
func CortexInputFromQuery(query string, size int) []float64 {
	vec := make([]float64, size)
	lower := strings.ToLower(query)
	for _, c := range lower {
		idx := int(c) % size
		vec[idx] += 1.0
	}
	// Normalize
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	if norm > 0 {
		norm = 1.0 / norm // fast inverse (no sqrt needed for training signal)
		for i := range vec {
			vec[i] *= norm
		}
	}
	return vec
}

// predictTool uses the neural cortex to predict which tool to use for a query.
func (r *Reasoner) predictTool(query string) *CortexPrediction {
	if r.Cortex == nil {
		return nil
	}
	input := CortexInputFromQuery(query, r.Cortex.InputSize)
	pred := r.Cortex.Predict(input)
	if pred.Confidence < 0.3 {
		return nil
	}
	return &pred
}

// knowledgeLookup searches the knowledge store for relevant context.
func (r *Reasoner) knowledgeLookup(query string) string {
	if r.Knowledge == nil {
		return ""
	}
	results, err := r.Knowledge.Search(query, 3)
	if err != nil || len(results) == 0 {
		return ""
	}
	return FormatKnowledgeContext(results)
}

// storeToMemory saves the current interaction context to working memory.
func (r *Reasoner) storeToMemory(input, answer string) {
	if r.WorkingMem == nil {
		return
	}

	// Store as a conversational exchange so the model understands context
	key := input
	if len(key) > 60 {
		key = key[:60] + "..."
	}

	value := answer
	if len(value) > 150 {
		value = value[:150] + "..."
	}

	// Store the exchange with high strength for recent recall
	r.WorkingMem.Store("user said: "+key, "I replied: "+value, 0.9)

	// Extract and store key facts (names, preferences, etc.)
	lower := strings.ToLower(input)
	if strings.Contains(lower, "my name") || strings.Contains(lower, "i'm ") || strings.Contains(lower, "i am ") {
		// Store the user's self-introduction as a persistent fact
		r.WorkingMem.Store("user_identity", input, 1.0)
	}
}
