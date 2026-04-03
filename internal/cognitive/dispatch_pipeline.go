package cognitive

import (
	"sort"
	"strings"
)

// -----------------------------------------------------------------------
// Priority-Tagged Dispatch Pipeline — declarative routing architecture.
//
// Replaces the sequential 15-layer intercept chain in Execute() with
// a priority-based, bypass-aware pipeline where each processing layer
// declares its priority and the input declares which channels it should
// flow through.
//
// Key innovations:
//
//   1. Multi-intent tagging — tag the input with ALL detected intents,
//      not just the highest-confidence one.
//
//   2. Bypass rules — stages declare bypass conditions. Empathy is
//      bypassed when simulate/persona/task tags are present.
//
//   3. Phase separation — pre-dispatch → routing → dispatch → post.
//
//   4. Annotation over interception — stages annotate instead of block.
//      Empathy acknowledgment composes WITH the simulation result.
//
// Adding a new intent type = registering a DispatchStage, not weaving
// new if-blocks into a 400-line function.
// -----------------------------------------------------------------------

// DispatchPhase determines when a stage runs.
type DispatchPhase int

const (
	DPPreDispatch  DispatchPhase = iota // always runs first
	DPRouting                           // runs with bypass rules
	DPDispatch                          // main action dispatch
	DPPostDispatch                      // always runs on result
)

// DispatchStage is one processing layer.
type DispatchStage struct {
	Name     string
	Priority int           // lower = earlier within phase
	Phase    DispatchPhase
	CanBlock bool          // can produce a final response?
	Filter   func(ctx *DispatchContext) bool
	Process  func(ctx *DispatchContext) *ActionResult
	Annotate func(ctx *DispatchContext, result *ActionResult)
}

// DispatchContext carries state through the pipeline.
type DispatchContext struct {
	NLU         *NLUResult
	Conv        *Conversation
	Tags        map[string]bool   // all detected intent signals
	Annotations []string          // text to compose into response
	Bypassed    map[string]bool   // stages to skip
	Router      *ActionRouter
}

// DispatchPipeline is the ordered set of processing stages.
type DispatchPipeline struct {
	stages []DispatchStage
	sorted bool
}

// NewDispatchPipeline creates an empty pipeline.
func NewDispatchPipeline() *DispatchPipeline {
	return &DispatchPipeline{}
}

// Register adds a stage.
func (dp *DispatchPipeline) Register(stage DispatchStage) {
	dp.stages = append(dp.stages, stage)
	dp.sorted = false
}

// Execute runs all stages respecting phases, priorities, and bypass rules.
func (dp *DispatchPipeline) Execute(nlu *NLUResult, conv *Conversation, router *ActionRouter) *ActionResult {
	if !dp.sorted {
		dp.sortStages()
	}

	ctx := &DispatchContext{
		NLU:      nlu,
		Conv:     conv,
		Tags:     TagIntents(nlu),
		Bypassed: make(map[string]bool),
		Router:   router,
	}

	// Apply automatic bypass rules from tags.
	ApplyBypassRules(ctx)

	var result *ActionResult

	for _, stage := range dp.stages {
		if ctx.Bypassed[stage.Name] {
			continue
		}
		if stage.Filter != nil && !stage.Filter(ctx) {
			continue
		}

		// Blocking stages can produce the final response.
		if stage.CanBlock && stage.Process != nil && result == nil {
			stageResult := stage.Process(ctx)
			if stageResult != nil && stageResult.DirectResponse != "" {
				result = stageResult
				continue // still run post-dispatch stages
			}
		} else if stage.Process != nil && result == nil {
			stage.Process(ctx)
		}

		// Annotations enrich the result.
		if stage.Annotate != nil && result != nil {
			stage.Annotate(ctx, result)
		}
	}

	// Compose annotations into the final response.
	if result != nil && len(ctx.Annotations) > 0 {
		annotation := strings.Join(ctx.Annotations, " ")
		if !strings.Contains(result.DirectResponse, annotation) {
			result.DirectResponse += "\n\n" + annotation
		}
	}

	return result
}

// -----------------------------------------------------------------------
// Multi-intent tagging — detect ALL intent signals in the input.
// -----------------------------------------------------------------------

// TagIntents detects all intent signals in the input, not just the primary one.
func TagIntents(nlu *NLUResult) map[string]bool {
	tags := make(map[string]bool)

	if nlu.Intent != "" {
		tags[nlu.Intent] = true
	}

	lower := strings.ToLower(nlu.Raw)

	if IsSimulationQuery(lower) {
		tags["simulate"] = true
	}
	if isP, _ := IsExpertPersonaQuery(lower); isP {
		tags["persona"] = true
	}
	if isExplicitTaskPrompt(nlu.Raw) {
		tags["task"] = true
	}
	if isEmotionalStatement(nlu.Raw) {
		tags["emotional"] = true
	}

	knowledgeIntents := map[string]bool{
		"explain": true, "question": true, "compare": true,
	}
	if knowledgeIntents[nlu.Intent] {
		tags["knowledge"] = true
	}

	return tags
}

// ApplyBypassRules determines which stages to skip based on intent tags.
func ApplyBypassRules(ctx *DispatchContext) {
	// Simulation, persona, and explicit task queries bypass emotional intercepts.
	if ctx.Tags["simulate"] || ctx.Tags["persona"] || ctx.Tags["task"] {
		ctx.Bypassed["socratic"] = true
		ctx.Bypassed["empathy"] = true
		ctx.Bypassed["subtext_empathy"] = true
	}

	// Code intents bypass all routing intercepts.
	if ctx.Tags["code"] || ctx.Tags["codegen"] {
		ctx.Bypassed["socratic"] = true
		ctx.Bypassed["empathy"] = true
		ctx.Bypassed["subtext_empathy"] = true
	}
}

// -----------------------------------------------------------------------
// Default stage registration
// -----------------------------------------------------------------------

// RegisterDefaultStages adds stages mirroring the existing Execute() logic.
func (dp *DispatchPipeline) RegisterDefaultStages(router *ActionRouter) {
	// Pre-dispatch: safety guard.
	dp.Register(DispatchStage{
		Name: "safety", Priority: 0, Phase: DPPreDispatch, CanBlock: true,
		Process: func(ctx *DispatchContext) *ActionResult {
			if isHarmfulRequest(ctx.NLU.Raw) {
				return &ActionResult{
					DirectResponse: "I can't help with that. I'm designed to be helpful, but I need to avoid assisting with activities that could harm others. Is there something else I can help you with?",
					Source:         "safety",
				}
			}
			return nil
		},
	})

	// Pre-dispatch: reference resolution.
	dp.Register(DispatchStage{
		Name: "reference_resolution", Priority: 10, Phase: DPPreDispatch,
		Process: func(ctx *DispatchContext) *ActionResult {
			ctx.Router.resolveConversationalReferences(ctx.NLU)
			return nil
		},
	})

	// Pre-dispatch: entity extraction.
	dp.Register(DispatchStage{
		Name: "entity_extraction", Priority: 20, Phase: DPPreDispatch,
		Process: func(ctx *DispatchContext) *ActionResult {
			if ctx.Router.EntityExtract != nil {
				if ctx.NLU.Entities == nil {
					ctx.NLU.Entities = make(map[string]string)
				}
				ctx.Router.EntityExtract.ExtractForIntent(ctx.NLU.Raw, ctx.NLU.Intent, ctx.NLU.Entities)
			}
			return nil
		},
	})

	// Routing: Socratic engine (bypassed for simulate/persona/task).
	dp.Register(DispatchStage{
		Name: "socratic", Priority: 30, Phase: DPRouting, CanBlock: true,
		Filter: func(ctx *DispatchContext) bool {
			return ctx.Router.Socratic != nil &&
				!ctx.Tags["task"] &&
				ctx.NLU.Intent != "recommendation" &&
				ctx.NLU.Intent != "question"
		},
		Process: func(ctx *DispatchContext) *ActionResult {
			eligible := ctx.NLU.Action == "respond" || ctx.NLU.Action == "llm_chat" ||
				ctx.NLU.Action == "" || ctx.NLU.Action == "lookup_knowledge"
			if !eligible {
				return nil
			}
			mode := ctx.Router.Socratic.DetectMode(ctx.NLU.Raw, ctx.Router.ConvState)
			if mode == SocraticNone {
				return nil
			}
			resp := ctx.Router.Socratic.Generate(ctx.NLU.Raw, mode, ctx.Router.ConvState)
			if resp != nil && len(resp.Questions) > 0 {
				var parts []string
				if resp.Framing != "" {
					parts = append(parts, resp.Framing)
				}
				for _, q := range resp.Questions {
					parts = append(parts, q.Text)
				}
				return &ActionResult{
					DirectResponse: strings.Join(parts, "\n\n"),
					Source:         "socratic:" + mode.String(),
				}
			}
			return nil
		},
	})

	// Routing: Empathy (bypassed for simulate/persona/task).
	dp.Register(DispatchStage{
		Name: "empathy", Priority: 40, Phase: DPRouting, CanBlock: true,
		Filter: func(ctx *DispatchContext) bool {
			return ctx.Router.Composer != nil && isEmotionalStatement(ctx.NLU.Raw)
		},
		Process: func(ctx *DispatchContext) *ActionResult {
			cctx := ctx.Router.BuildComposeContextWithSubtext(ctx.NLU.Raw, ctx.NLU)
			resp := ctx.Router.Composer.Compose(ctx.NLU.Raw, RespEmpathetic, cctx)
			if resp != nil && resp.Text != "" {
				return &ActionResult{DirectResponse: resp.Text, Source: "empathy"}
			}
			return nil
		},
	})

	// Dispatch: main action switch.
	dp.Register(DispatchStage{
		Name: "dispatch", Priority: 100, Phase: DPDispatch, CanBlock: true,
		Process: func(ctx *DispatchContext) *ActionResult {
			return ctx.Router.dispatch(ctx.NLU, ctx.Conv)
		},
	})
}

// -----------------------------------------------------------------------
// Sorting
// -----------------------------------------------------------------------

func (dp *DispatchPipeline) sortStages() {
	sort.Slice(dp.stages, func(i, j int) bool {
		if dp.stages[i].Phase != dp.stages[j].Phase {
			return dp.stages[i].Phase < dp.stages[j].Phase
		}
		return dp.stages[i].Priority < dp.stages[j].Priority
	})
	dp.sorted = true
}
