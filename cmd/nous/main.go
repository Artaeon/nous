package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"sort"
	"strings"
	"syscall"
	"time"
	"unicode"

	"github.com/artaeon/nous/internal/agent"
	"github.com/artaeon/nous/internal/micromodel"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/assistant"
	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/cognitive"
	"github.com/artaeon/nous/internal/compress"
	"github.com/artaeon/nous/internal/federation"
	"github.com/artaeon/nous/internal/hands"
	"github.com/artaeon/nous/internal/index"
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/sentinel"
	"github.com/artaeon/nous/internal/server"
	"github.com/artaeon/nous/internal/channels"
	"github.com/artaeon/nous/internal/tools"
	"github.com/artaeon/nous/internal/training"
)

// pkgLoader is the global package loader, accessible from handleCommand.
var pkgLoader *cognitive.PackageLoader

// channelHandler builds a MessageHandler that routes channel messages through
// the same cognitive pipeline as the REPL: classify → extract facts → respond → record.
func channelHandler(
	nlu *cognitive.NLU,
	actions *cognitive.ActionRouter,
	conv *cognitive.Conversation,
	board *blackboard.Blackboard,
	wm *memory.WorkingMemory,
	ltm *memory.LongTermMemory,
	episodic *memory.EpisodicMemory,
	collector *training.Collector,
) channels.MessageHandler {
	return func(channel, chatID, userID, text string) (string, error) {
		text = strings.TrimSpace(text)
		if text == "" {
			return "", nil
		}

		// Extract personal facts
		fe := &cognitive.FactExtractor{LTM: ltm, WorkingMem: wm}
		fe.Extract(text)

		start := time.Now()
		var answer string

		// Try cognitive compiler first (compiled patterns, ~0ms)
		if compiled, ok := actions.TryCompiled(text); ok {
			answer = compiled
		}

		nluResult := nlu.Understand(text)
		if answer == "" && nluResult.Confidence >= 0.5 {
			actionResult := actions.Execute(nluResult, conv)
			if actionResult.DirectResponse != "" {
				answer = actionResult.DirectResponse
			}
		}

		if answer == "" && actions.Composer != nil {
			respType := actions.ClassifyForComposer(text)
			ctx := actions.BuildComposeContext()
			resp := actions.Composer.Compose(text, respType, ctx)
			if resp != nil && resp.Text != "" {
				answer = resp.Text
			}
		}

		if answer == "" {
			answer = "I'm not sure how to answer that — could you rephrase?"
		}

		duration := time.Since(start)

		// Learn from this response for future compilation
		go actions.LearnResponse(text, answer, nluResult)

		// Record in episodic memory
		go episodic.Record(memory.Episode{
			Timestamp: time.Now(),
			Input:     text,
			Output:    answer,
			Success:   true,
			Duration:  duration.Milliseconds(),
		})

		// Collect training data
		msgs := conv.Messages()
		sysPrompt := ""
		if len(msgs) > 0 {
			sysPrompt = msgs[0].Content
		}
		quality := scoreInteractionQuality(answer, duration, board)
		go collector.Collect(sysPrompt, text, answer, nil, quality)

		return answer, nil
	}
}

const version = "1.0.0"

func main() {
	// Flags
	model := flag.String("model", "nous-cognitive", "Engine identifier")
	memoryPath := flag.String("memory", defaultMemoryPath(), "Path for persistent memory storage")
	allowShell := flag.Bool("allow-shell", false, "Enable shell command execution")
	trustMode := flag.Bool("trust", false, "Skip confirmation prompts for file operations")
	sessionID := flag.String("resume", "", "Resume a previous session by ID")
	serveMode := flag.Bool("serve", false, "Run as HTTP server instead of REPL")
	listenHost := flag.String("listen", "127.0.0.1", "HTTP listen host (with --serve)")
	publicMode := flag.Bool("public", false, "Bind the HTTP server to 0.0.0.0 (with --serve)")
	servePort := flag.String("port", "3333", "HTTP server port (with --serve)")
	apiKey := flag.String("api-key", "", "API key for HTTP server authentication (with --serve)")
	showVersion := flag.Bool("version", false, "Show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("nous %s\n", version)
		os.Exit(0)
	}

	// Get working directory
	workDir, _ := os.Getwd()
	cognitive.WorkDir = workDir

	// Project auto-scan
	projectInfo := cognitive.ScanProject(workDir)
	cognitive.CurrentProject = projectInfo

	// Build codebase index (Go AST symbols + file metadata for all languages)
	nousDir := filepath.Join(workDir, ".nous")
	codeIndex := index.NewCodebaseIndex(nousDir)
	_ = codeIndex.Build(workDir)

	// Filesystem Sentinel — ambient file watching with inotify
	var fileWatcher *sentinel.Watcher
	fileWatcher, err := sentinel.NewWatcher(workDir, 500*time.Millisecond, func(events []sentinel.FileEvent) {
		// Auto-update codebase index on Go file changes
		if projectInfo.Language == "Go" {
			changed := sentinel.ChangedGoFiles(events)
			if len(changed) > 0 {
				codeIndex.IncrementalUpdate(workDir, changed)
			}
		}
	})
	if err != nil {
		// sentinel disabled, not critical
	} else {
		go fileWatcher.Run()
	}

	// Tool Choreography — learned multi-step recipes
	recipeBook := cognitive.NewRecipeBook(nousDir)

	// Initialize undo stack and tool registry
	undoStack := memory.NewUndoStack(100)
	toolReg := tools.NewRegistry()
	tools.RegisterBuiltins(toolReg, workDir, *allowShell, undoStack)
	tools.RegisterRealTimeData(toolReg)

	// NLU + ActionRouter: deterministic intent → action pipeline (shared with HTTP API)
	nlu := cognitive.NewNLU()

	// Initialize neural intent classifier (loads saved model or trains from scratch)
	neuralModelPath := filepath.Join(filepath.Dir(*memoryPath), "nous_neural.bin")
	if err := nlu.InitNeural(neuralModelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: neural classifier init failed: %v (falling back to pattern matching)\n", err)
	}

	actions := cognitive.NewActionRouter()
	actions.Tools = toolReg
	actions.EntityExtract = cognitive.NewSmartEntityExtractor()

	// Predictive Pre-computation
	predictor := cognitive.NewPredictor(toolReg)

	// Initialize core systems
	board := blackboard.New()

	// Memory systems
	wm := memory.NewWorkingMemory(64)
	// Built-in embeddings — wired after Composer is initialized below
	ltm := memory.NewLongTermMemory(*memoryPath)

	// Ensure user name is stored for the Composer's personal greetings
	if _, ok := ltm.Retrieve("user_name"); !ok {
		if u, err := os.UserHomeDir(); err == nil {
			name := filepath.Base(u)
			// Capitalize first letter
			if len(name) > 0 {
				name = strings.ToUpper(name[:1]) + name[1:]
			}
			ltm.Store("user_name", name, "identity")
		}
	}

	// Project-level memory (stored in the project's .nous/ directory)
	projMem := memory.NewProjectMemory(workDir)

	// Episodic memory — remembers every interaction forever (semantic search via embeddings)
	episodic := memory.NewEpisodicMemory(nousDir, nil)
	assistantStore := assistant.NewStore(*memoryPath)

	// Wire memory into action router
	actions.WorkingMem = wm
	actions.LongTermMem = ltm
	actions.EpisodicMem = episodic
	actions.Reminders = cognitive.NewReminderManager()
	actions.Tracker = cognitive.NewConversationTrackerPersistent(filepath.Join(nousDir, "facts.json"))
	actions.CogGraph = cognitive.NewCognitiveGraph(filepath.Join(nousDir, "cognitive_graph.json"))
	actions.Patterns = cognitive.NewPatternDetector()
	actions.Semantic = cognitive.NewSemanticEngine()
	actions.Reasoner = cognitive.NewReasoningEngine(actions.CogGraph, actions.Semantic)
	actions.Causal = cognitive.NewCausalEngine()
	actions.Composer = cognitive.NewComposer(actions.CogGraph, actions.Semantic, actions.Causal, actions.Patterns)
	actions.Inference = cognitive.NewInferenceEngine(actions.CogGraph)
	actions.Analogy = cognitive.NewAnalogyEngine(actions.CogGraph, actions.Semantic)
	actions.Thinker = cognitive.NewThinkingEngine(actions.CogGraph, actions.Composer)
	actions.Reasoner.Analogy = actions.Analogy
	actions.CausalReasoner = cognitive.NewGraphCausalReasoner(actions.CogGraph)
	actions.GoalPlanner = cognitive.NewGoalPlanner(actions.CogGraph, actions.Semantic)
	actions.Pipeline = cognitive.NewReasoningPipeline(
		actions.CogGraph, actions.Inference, actions.Reasoner,
		actions.Thinker, actions.CausalReasoner, actions.Composer,
		actions.Semantic, actions.Analogy,
	)

	actions.Creative = cognitive.NewCreativeEngine(actions.CogGraph, actions.Composer)
	actions.Dialogue = cognitive.NewDialogueManager()
	actions.Transformer = cognitive.NewTextTransformer(actions.Composer.Generative.Embeddings())

	// Wire built-in embeddings for semantic memory (reuse Composer's embeddings)
	embedFn := cognitive.MakeEmbedFunc(actions.Composer.Generative.Embeddings())
	wm.SetEmbedFunc(embedFn)

	// Load knowledge packages — instant world knowledge and vocabulary expansion
	packDir := filepath.Join(workDir, "packages")
	pkgLoader = cognitive.NewPackageLoader(actions.CogGraph, actions.Composer.Generative, actions.Composer, packDir)
	actions.Packages = pkgLoader
	if _, err := os.Stat(packDir); err == nil {
		results, err := pkgLoader.LoadAll()
		if err == nil && len(results) > 0 {
			totalFacts, totalVocab := 0, 0
			for _, r := range results {
				totalFacts += r.FactsLoaded
				totalVocab += r.VocabLoaded
			}
			fmt.Printf("  loaded %d knowledge packages (%d facts, %d vocab)\n", len(results), totalFacts, totalVocab)
		}
	}

	// Load mined Wikipedia sentence templates if available
	tmplPath := filepath.Join(packDir, "templates", "wiki-templates.json")
	if err := cognitive.LoadMinedTemplates(tmplPath); err == nil {
		fmt.Println("  loaded Wikipedia sentence templates")
	}

	// Load GRU text generation model if available
	textgenPath := filepath.Join(filepath.Dir(*memoryPath), "nous-textgen.bin")
	if err := actions.Composer.LoadTextGen(textgenPath); err == nil {
		fmt.Println("  loaded GRU text generation model")
	}

	// Load sentence corpus for Layer 2 retrieval-based generation
	corpusPath := filepath.Join(packDir, "wiki", "sentence_corpus.json")
	sc := cognitive.NewSentenceCorpus()
	if err := sc.Load(corpusPath); err == nil && sc.Size() > 0 {
		actions.Composer.SentenceCorpus = sc
		fmt.Printf("  loaded sentence corpus (%d exemplars)\n", sc.Size())
	}

	// Load discourse corpus for Layer 2b
	discPath := filepath.Join(packDir, "wiki", "discourse_corpus.json")
	dc := cognitive.NewDiscourseCorpus()
	if err := dc.Load(discPath); err == nil && dc.Size() > 0 {
		actions.Composer.DiscourseCorpus = dc
		if actions.Creative != nil {
			actions.Creative.DiscourseCorpus = dc
		}
		fmt.Printf("  loaded discourse corpus (%d sentences)\n", dc.Size())
	}

	// Common sense knowledge layer — everyday associations
	csGraph := cognitive.NewCommonSenseGraph()
	actions.CommonSense = csGraph
	fmt.Printf("  loaded common sense graph (%d associations)\n", len(csGraph.Lookup("dinner"))+len(csGraph.Lookup("bored"))+len(csGraph.Lookup("book")))

	// Conversation learner — improves from every interaction
	convLearner := cognitive.NewConversationLearner(filepath.Join(nousDir, "conversation_patterns.json"))
	convLearner.SeedDefaults()
	if dc != nil && dc.Size() > 0 {
		convLearner.SeedFromCorpus(dc)
	}
	actions.ConvLearner = convLearner
	fmt.Printf("  loaded conversation learner (%d patterns)\n", convLearner.PatternCount())

	// --- Phase 1-4 Cognitive Improvements ---

	// Subtext Engine — reads between the lines (emotional context, implied needs)
	subtextEngine := cognitive.NewSubtextEngine(episodic)
	actions.Subtext = subtextEngine

	// Memory Trigger Engine — involuntary episodic recall
	memTrigger := cognitive.NewMemoryTriggerEngine(episodic, embedFn)
	actions.MemTrigger = memTrigger

	// Absorption Engine — learns expression patterns from text
	absEngine := cognitive.NewAbsorptionEngine(filepath.Join(nousDir, "absorption_patterns.json"))
	if err := absEngine.Load(); err == nil && len(absEngine.Stats().ByFunction) > 0 {
		fmt.Printf("  loaded absorption engine (%d patterns)\n", absEngine.Stats().Total)
	}
	actions.Absorption = absEngine
	actions.Composer.Absorption = absEngine

	// Associative Spark Engine — surfaces unexpected knowledge connections
	sparkEngine := cognitive.NewSparkEngine(actions.CogGraph)
	actions.Sparks = sparkEngine

	// Inner Council — multiple perspectives deliberate before responding
	council := cognitive.NewInnerCouncil(actions.CogGraph, episodic)
	actions.Council = council

	// Opinion Engine — forms and accumulates opinions from evidence
	opinionEngine := cognitive.NewOpinionEngine(filepath.Join(nousDir, "opinions.json"))
	if err := opinionEngine.Load(); err == nil {
		tops := opinionEngine.TopOpinions(3)
		if len(tops) > 0 {
			fmt.Printf("  loaded opinion engine (%d opinions formed)\n", len(tops))
		}
	}
	actions.Opinions = opinionEngine

	// Phase 1-4 subsystems — conversation state, follow-up resolution,
	// user preferences, deep retrieval, query rewriting, filler policy.
	actions.ConvState = cognitive.NewConversationState()
	actions.FollowUp = cognitive.NewFollowUpResolver()
	actions.Preferences = cognitive.NewPreferenceModel()
	actions.QueryRewrite = cognitive.NewQueryRewriter()
	actions.Filler = cognitive.NewFillerDetector()
	// TwoTierRetriever — needs KnowledgeVec and CogGraph
	if actions.Knowledge != nil || actions.CogGraph != nil {
		actions.Retriever = cognitive.NewTwoTierRetriever(actions.Knowledge, actions.CogGraph)
	}

	// NLG engines
	actions.NLG = cognitive.NewNLGEngine()
	actions.Format = cognitive.NewFormatCompliance()
	// Build a SentenceCorpus from extracted facts for the hybrid generator.
	// Each extracted fact with a source sentence becomes a retrieval exemplar.
	hybridCorpus := actions.Composer.SentenceCorpus
	if hybridCorpus == nil || hybridCorpus.Size() == 0 {
		// No pre-built corpus — mine one from the knowledge text
		hybridCorpus = cognitive.NewSentenceCorpus()
		miner := cognitive.NewWikiFactExtractor()
		if allFacts, err := miner.ExtractFromDirectory(filepath.Join(workDir, "knowledge")); err == nil {
			for _, f := range allFacts {
				if f.Source != "" && len(f.Source) > 20 && f.Relation != cognitive.RelDescribedAs {
					hybridCorpus.Add(cognitive.SentenceExemplar{
						Sentence: f.Source,
						Subject:  f.Subject,
						Object:   f.Object,
						Relation: f.Relation,
					})
				}
			}
			if hybridCorpus.Size() > 0 {
				fmt.Fprintf(os.Stderr, "  built sentence corpus (%d exemplars from knowledge text)\n", hybridCorpus.Size())
				// NOTE: Do NOT give this to the Composer — cross-domain entity
				// swapping produces garbage ("Quantum mechanics was a Pre-Socratic
				// philosopher from Elea"). The mined corpus is only for the hybrid
				// generator's direct-text path (which is currently disabled).
			}
		}
	}
	corpusNLG := cognitive.NewCorpusNLG()
	if err := corpusNLG.IngestCorpus(filepath.Join(workDir, "knowledge")); err == nil {
		actions.CorpusNLG = corpusNLG
	}

	// Fact extraction — 5x the knowledge graph from raw text
	factExtractor := cognitive.NewWikiFactExtractor()
	knowledgePath := filepath.Join(workDir, "knowledge")
	if extractedFacts, err := factExtractor.ExtractFromDirectory(knowledgePath); err == nil && len(extractedFacts) > 0 {
		added := factExtractor.IngestIntoGraph(actions.CogGraph, extractedFacts)
		if added > 0 {
			fmt.Fprintf(os.Stderr, "  extracted %d facts from knowledge text (%d new)\n", len(extractedFacts), added)
		}
	}
	actions.FactExtract = factExtractor

	// Deep Reasoner — multi-step structured reasoning for "why" and "how" questions
	actions.DeepReason = cognitive.NewDeepReasoner(actions.CogGraph, knowledgePath)

	// GRU text generation training is available but deferred to background.
	// With 4000+ facts, training takes ~60s blocking startup. The GRU can
	// be trained via the /train command in the REPL instead.
	// TODO: Run training in background goroutine after server starts.

	// Create hybrid generator AFTER fact extraction and GRU training
	// so it can use the trained TextGen model
	var hybridTextGen *cognitive.TextGenModel
	if actions.Composer != nil {
		hybridTextGen = actions.Composer.TextGen
	}
	hybridFluency := cognitive.NewFluencyScorer()
	hybridFluency.LoadCorpus(knowledgePath)
	actions.HybridGen = cognitive.NewHybridGenerator(hybridCorpus, hybridTextGen, hybridFluency, knowledgePath)
	actions.Fluency = hybridFluency

	// Fluency scorer — bigram-based sentence quality scoring
	fluencyScorer := cognitive.NewFluencyScorer()
	fluencyScorer.LoadCorpus(knowledgePath)
	actions.Fluency = fluencyScorer

	// Dialogue policy — learns which strategies work best
	actions.Policy = cognitive.NewDialoguePolicy()

	// Context window — prevents repetition across turns
	actions.CtxWindow = cognitive.NewContextWindow(5)

	// Self-teach — mines knowledge files at query time for unknown topics
	if actions.CogGraph != nil {
		actions.SelfTeacher = cognitive.NewSelfTeach(knowledgePath, actions.CogGraph)
	}

	// Innovation systems
	actions.Socratic = cognitive.NewSocraticEngine()
	actions.Crystallizer = cognitive.NewInsightCrystallizer()
	actions.Transparency = cognitive.NewCognitiveTransparency()
	actions.SelfModel = cognitive.NewSelfModel()
	if actions.CogGraph != nil {
		actions.Synthesizer = cognitive.NewKnowledgeSynthesizer(actions.CogGraph, actions.Analogy)
	}

	// Extractive text summarizer — heuristic sentence scoring, no ML
	actions.Summarizer = cognitive.NewSummarizer()

	// Document generator — structured long-form documents from knowledge graph
	actions.DocGen = cognitive.NewDocumentGenerator(actions.CogGraph, knowledgePath)

	// Prose composer — clause-level retrieval-augmented generation
	actions.ProseComposer = cognitive.NewProseComposer(knowledgePath)

	// Code generator — template-based code generation
	actions.CodeGen = cognitive.NewCodeGenerator()

	// Micro language model — load Mamba (preferred) or transformer
	mambaModelPath := filepath.Join(nousDir, "mamba.bin")
	microModelPath := filepath.Join(nousDir, "micromodel.bin")
	if _, err := os.Stat(mambaModelPath); err == nil {
		if bridge, err := micromodel.LoadBridge(mambaModelPath); err == nil {
			actions.MicroModel = bridge
			if bridge.IsMamba() {
				fmt.Printf("  %s✓%s Mamba language model loaded (%d params, O(1)/token)\n",
					cognitive.ColorGreen, cognitive.ColorReset, bridge.Mamba.ParamCount())
			}
		}
	} else if _, err := os.Stat(microModelPath); err == nil {
		if bridge, err := micromodel.LoadBridge(microModelPath); err == nil {
			actions.MicroModel = bridge
			fmt.Printf("  loaded micro language model (transformer)\n")
		}
	} else {
		fmt.Printf("  %s⚡%s no Mamba model found — train one for neural generation:\n",
			cognitive.ColorYellow, cognitive.ColorReset)
		fmt.Printf("       nous-train mamba -knowledge knowledge/\n")
	}

	// Cognitive compiler — compiles responses into deterministic handlers.
	// Every response generation feeds back into the compiler, making Nous
	// faster with every interaction. The more you use it, the more queries
	// resolve in O(1) via compiled patterns instead of neural inference.
	compilerPath := filepath.Join(nousDir, "compiler.json")
	actions.CogCompiler = cognitive.NewCognitiveCompiler(compilerPath)
	if stats := actions.CogCompiler.Stats(); stats.TotalHandlers > 0 {
		fmt.Printf("  %s✓%s cognitive compiler: %d handlers (%.0f%% hit rate)\n",
			cognitive.ColorGreen, cognitive.ColorReset, stats.TotalHandlers, stats.HitRate*100)
	}

	// Conversational Learning Engine — Nous learns from every interaction
	learningEngine := cognitive.NewLearningEngine(actions.CogGraph, actions.Composer, nousDir)

	// Decay stale knowledge on startup
	learningEngine.DecayKnowledge()

	// Consolidate on startup — the AI "dreams", discovering new connections
	if actions.CogGraph.NodeCount() > 0 {
		consolidator := cognitive.NewConsolidator(actions.CogGraph)
		insights := consolidator.Consolidate()
		if len(insights) > 0 {
			actions.CogGraph.Save()
		}

		// Run abstraction discovery
		ae := cognitive.NewAbstractionEngine(actions.CogGraph)
		abstractions := ae.Discover()
		_ = abstractions // abstractions are available for reasoning
	}

	// PersonalResp is wired after Growth is initialized (see below)

	// Training data collector — gathers successful interactions for fine-tuning
	collector := training.NewCollector(nousDir)

	// Auto-tuner — monitors training data and triggers Modelfile-based tuning
	autoTuner := training.NewAutoTuner(collector, *model).
		WithCallback(func(msg string) {
			fmt.Printf("  %s%s%s\n", cognitive.ColorDim, msg, cognitive.ColorReset)
		})

	// Session management
	sessionStore := cognitive.NewSessionStore(*memoryPath)
	currentSession := &cognitive.Session{
		ID:        cognitive.GenerateSessionID(),
		Name:      "session-" + time.Now().Format("2006-01-02-15-04"),
		Model:     *model,
		CreatedAt: time.Now(),
		Metadata:  map[string]string{"workdir": workDir},
	}

	// Resume previous session if requested
	if *sessionID != "" {
		loaded, err := sessionStore.Load(*sessionID)
		if err != nil {
			fmt.Printf("  %swarning: could not resume session %s: %v%s\n", cognitive.ColorYellow, *sessionID, err, cognitive.ColorReset)
		} else {
			currentSession = loaded
		}
	}

	// Create cognitive streams
	perceiver := cognitive.NewPerceiver(board, nil)
	learner := cognitive.NewLearner(board, nil, *memoryPath)
	reasoner := cognitive.NewReasoner(board, nil, toolReg)
	reasoner.WorkingMem = wm
	reasoner.LongTermMem = ltm
	reasoner.ProjectMem = projMem
	reasoner.Compressor = compress.NewCompressor(nil)
	reasoner.CodeIndex = codeIndex
	reasoner.Recipes = recipeBook
	reasoner.Predictor = predictor
	reasoner.Learner = learner
	reasoner.EpisodicMem = episodic

	// --- Cognitive Prosthetics: 8 innovations for small-model enhancement ---

	// 1. Intent-to-Action Compiler — deterministic NLP-to-tool-call translation
	intentCompiler := cognitive.NewIntentCompiler(workDir)
	reasoner.Intent = intentCompiler

	// 2. Grammar-Constrained Decoding — dynamic JSON schemas for structured output
	reasoner.Grammar = cognitive.NewGrammarDecoder(nil)

	// 3. Decomposed Micro-Inference — break complex LLM calls into micro-steps
	reasoner.MicroInfer = cognitive.NewMicroInference(nil)

	// 4. Speculative Multi-Tool Execution — fire plausible tools in parallel
	reasoner.Speculator = cognitive.NewSpeculativeExecutor(toolReg, intentCompiler)

	// 5. Thought Crystallization — JIT compile reasoning chains into rules
	reasoner.Crystals = cognitive.NewCrystalBook(filepath.Join(nousDir, "crystals.json"))
	if seeded := reasoner.Crystals.SeedDevWorkflows(); seeded > 0 {
		fmt.Fprintf(os.Stderr, "  seeded %d dev workflow crystals\n", seeded)
	}

	// 6. Self-Distillation Loop — learn from the model's own failures
	reasoner.Distiller = cognitive.NewSelfDistiller(filepath.Join(nousDir, "distill.json"))

	// 7. Embedding-Driven Grounding — semantic secondary brain
	embedGrounder := cognitive.NewEmbedGrounder(func(text string) ([]float64, error) {
		return actions.Composer.Generative.Embeddings().SentenceEmbed(text)
	})
	reasoner.EmbedGround = embedGrounder

	// Index tools for semantic matching
	for _, tool := range toolReg.List() {
		embedGrounder.IndexTool(tool.Name, tool.Description)
	}

	// 8. Neuroplastic Tool Descriptions — evolve descriptions based on success rates
	reasoner.Neuroplastic = cognitive.NewNeuroplasticDescriptions(*model, filepath.Join(nousDir, "neuroplastic.json"))
	for _, tool := range toolReg.List() {
		reasoner.Neuroplastic.RegisterDefault(tool.Name, tool.Description)
	}

	// 9. Cognitive Exocortex — the external brain that bypasses the LLM
	reasoner.Exo = cognitive.NewExocortex(
		intentCompiler, toolReg, reasoner.Distiller,
		reasoner.Neuroplastic, embedGrounder, reasoner.Crystals,
	)

	// 10. Cognitive Firewall — post-LLM validation and correction
	reasoner.Firewall = cognitive.NewCognitiveFirewall(reasoner.Distiller)

	// 11. Phantom Reasoning — pre-computed chains, model only writes conclusion
	reasoner.Phantom = cognitive.NewPhantomReasoner()

	// 12. Adaptive Prompt Distillation — JIT-compiled minimal prompts per query type
	reasoner.PromptDist = cognitive.NewPromptDistiller()

	// 13. Synthetic Neural Cortex — pure Go neural network learns tool prediction
	toolLabels := make([]string, 0, len(toolReg.List()))
	for _, t := range toolReg.List() {
		toolLabels = append(toolLabels, t.Name)
	}
	reasoner.Cortex = cognitive.NewNeuralCortex(
		64, 32, toolLabels,
		filepath.Join(nousDir, "cortex.json"),
	)

	// 14. Knowledge Vector Store — unlimited knowledge via vector search
	// Knowledge embedding uses nomic-embed-text (fast, purpose-built for embeddings)
	knowledgeEmbedFn := func(text string) ([]float64, error) {
		return actions.Composer.Generative.Embeddings().SentenceEmbed(text)
	}
	reasoner.Knowledge = cognitive.NewKnowledgeVec(
		knowledgeEmbedFn,
		filepath.Join(nousDir, "knowledge.json"),
	)

	// Auto-ingest bundled knowledge on first run (when knowledge store is empty).
	// Runs in background so startup isn't blocked by embedding calls.
	if reasoner.Knowledge.Size() == 0 {
		knowledgeDirs := []string{
			filepath.Join(workDir, "knowledge"),  // repo checkout
			"/app/knowledge",                      // Docker container
		}
		for _, kdir := range knowledgeDirs {
			files, _ := filepath.Glob(filepath.Join(kdir, "*.txt"))
			if len(files) > 0 {
				fmt.Printf("  %singesting %d knowledge files in background...%s\n",
					cognitive.ColorDim, len(files), cognitive.ColorReset)
				go func(fileList []string) {
					total := 0
					for _, f := range fileList {
						n, err := reasoner.Knowledge.Ingest(f)
						if err == nil {
							total += n
						}
					}
					if total > 0 {
						fmt.Printf("\n  %s✓%s Knowledge ready: %d chunks loaded\n",
							cognitive.ColorGreen, cognitive.ColorReset, total)
					}
				}(files)
				break
			}
		}
	}

	// 14b. Response Crystal Store — semantic cache that learns from every LLM response.
	// The more you use Nous, the more queries are served instantly (0ms) from cache.
	// Uses EmbedCache to avoid 5-10s embedding calls on repeated queries.
	embedCache := cognitive.NewEmbedCache(256, 30*time.Minute)
	responseCrystals := cognitive.NewResponseCrystalStore(knowledgeEmbedFn, nousDir)
	responseCrystals.SetEmbedCache(embedCache)
	if responseCrystals.Size() > 0 {
		size, hits := responseCrystals.Stats()
		fmt.Printf("  %s✓%s %d response crystals loaded (%d total hits)\n",
			cognitive.ColorGreen, cognitive.ColorReset, size, hits)
	}

	// 15. Model Compiler — compile experience into optimized Modelfiles
	reasoner.Compiler = cognitive.NewModelCompiler(
		*model, reasoner.Distiller, reasoner.Crystals,
	)
	reasoner.Compiler.SetKnowledge(reasoner.Knowledge)
	reasoner.Compiler.SetCortex(reasoner.Cortex)

	// 16. Personal Growth — learns user's interests, preferences, and patterns
	reasoner.Growth = cognitive.NewPersonalGrowth(
		filepath.Join(nousDir, "growth.json"),
	)

	// Wire PersonalResponseGenerator into ActionRouter and ResponseFormatter
	personalResp := &cognitive.PersonalResponseGenerator{
		Growth:  reasoner.Growth,
		Tracker: actions.Tracker,
	}
	actions.PersonalResp = personalResp
	actions.Growth = reasoner.Growth

	// 17. Virtual Context Engine — makes 4k tokens feel like 200k+
	reasoner.VCtx = cognitive.NewVirtualContext(1500)
	reasoner.VCtx.AddSource(cognitive.KnowledgeSource(reasoner.Knowledge))
	reasoner.VCtx.AddSource(cognitive.GrowthSource(reasoner.Growth))
	reasoner.VCtx.AddSource(cognitive.EpisodicSource(func(query string, limit int) []string {
		episodes := episodic.SearchKeyword(query, limit)
		var results []string
		for _, ep := range episodes {
			results = append(results, fmt.Sprintf("Q: %s\nA: %s", ep.Input, ep.Output))
		}
		return results
	}))

	// 18. Intent-Cortex Ensemble — two prediction systems voting together
	reasoner.Ensemble = cognitive.NewToolEnsemble(reasoner.Intent, reasoner.Cortex)

	// 19. Cross-Memory Feedback Loop — wires all subsystems together
	reasoner.Feedback = cognitive.NewFeedbackLoop(
		reasoner.Cortex, reasoner.EpisodicMem,
		reasoner.VCtx, reasoner.Growth, reasoner.Crystals,
	)

	reasoner.AssistantContext = func(input string, recent string) string {
		return buildAssistantContext(assistantStore, input, recent, time.Now())
	}
	reasoner.AssistantAnswer = func(input string, recent string) (string, bool) {
		return answerAssistantQuery(assistantStore, input, recent, time.Now())
	}
	planner := cognitive.NewPlanner(board, nil)
	executor := cognitive.NewExecutor(board, nil, toolReg)
	reflector := cognitive.NewReflector(board, nil)

	// Set up confirmation for dangerous actions
	if *trustMode {
		reasoner.Confirm = cognitive.AutoApprove
	} else {
		reasoner.Confirm = cognitive.TerminalConfirm
	}

	// Load resumed session messages into conversation
	if *sessionID != "" && len(currentSession.Messages) > 0 {
		for _, msg := range currentSession.Messages {
			switch msg.Role {
			case "user":
				reasoner.Conv.User(msg.Content)
			case "assistant":
				reasoner.Conv.Assistant(msg.Content)
			}
		}
	}

	// Enable streaming output — filter tool JSON from display.
	// The model may emit tool-call JSON as text (e.g. {"tool": "read", ...}).
	// We buffer each response chunk and suppress lines matching tool patterns.
	var streamBuf strings.Builder
	inToolCall := false

	toolMarkers := []string{`{"tool"`, "```tool", "```json\n{\"tool\"", "```\n{\"tool\"", "ACT: {\"tool\"", "ACT:{\"tool\""}
	isToolJSON := func(s string) bool {
		t := strings.TrimSpace(s)
		for _, m := range toolMarkers {
			if strings.HasPrefix(t, m) {
				return true
			}
		}
		return false
	}
	couldBeToolJSON := func(s string) bool {
		t := strings.TrimSpace(s)
		if len(t) == 0 {
			return true
		}
		for _, m := range toolMarkers {
			if strings.HasPrefix(m, t) {
				return true
			}
		}
		return false
	}

	// --- Subcommand Mode ---
	// If the first positional arg is a recognized subcommand (understand,
	// generate, reason, remember), execute it and exit immediately.
	// This runs AFTER full cognitive init so all engines are available.
	if SubcommandRouter(flag.Args(), nlu, actions, ltm) {
		os.Exit(0)
	}

	// Spinner for LLM thinking time — stopped on first token
	llmSpinner := cognitive.NewSpinner()
	firstToken := true

	responseStarted := false // tracks whether we've printed any visible response text

	reasoner.OnToken = func(token string, done bool) {
		if firstToken {
			llmSpinner.Stop()
			firstToken = false
		}
		streamBuf.WriteString(token)

		if !done {
			if inToolCall {
				return
			}
			current := streamBuf.String()
			if isToolJSON(current) {
				inToolCall = true
				return
			}
			if couldBeToolJSON(current) {
				return
			}
			// Definitely not a tool call — flush with indentation
			if !responseStarted {
				fmt.Print("  ")
				responseStarted = true
			}
			// Add indentation after newlines for clean formatting
			text := current
			text = strings.ReplaceAll(text, "\n", "\n  ")
			fmt.Print(text)
			streamBuf.Reset()
		} else {
			if !inToolCall {
				remaining := streamBuf.String()
				if !isToolJSON(remaining) && strings.TrimSpace(remaining) != "" {
					if !responseStarted {
						fmt.Print("  ")
						responseStarted = true
					}
					remaining = strings.ReplaceAll(remaining, "\n", "\n  ")
					fmt.Print(remaining)
				}
				fmt.Println()
			}
			inToolCall = false
			streamBuf.Reset()
			firstToken = true
		}
	}

	// Tool status updates with clean formatting
	reasoner.OnStatus = func(status string) {
		// Stop spinner if running so status appears cleanly
		llmSpinner.Stop()
		fmt.Printf("%s%s%s\n", cognitive.ColorGray, status, cognitive.ColorReset)
	}

	// Start cognitive streams
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	streams := []cognitive.Stream{perceiver, reasoner, planner, executor, reflector, learner}
	assistantScheduler := assistant.NewScheduler(assistantStore, board)
	go func() {
		if err := assistantScheduler.Run(ctx); err != nil && err != context.Canceled {
			fmt.Fprintf(os.Stderr, "  %sassistant-scheduler: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
		}
	}()

	for _, s := range streams {
		s := s
		go func() {
			if err := s.Run(ctx); err != nil && err != context.Canceled {
				fmt.Fprintf(os.Stderr, "  %s%s: %v%s\n", cognitive.ColorRed, s.Name(), err, cognitive.ColorReset)
			}
		}()
	}

	// Start Hands system — autonomous background agents
	handsDir := filepath.Join(nousDir, "hands")
	os.MkdirAll(handsDir, 0755)
	handStore := hands.NewStore(handsDir)
	handStateStore, _ := hands.NewHandStateStore(handsDir)
	handRunner := hands.NewRunnerWithState(nil, board, toolReg, handStateStore)
	handManager := hands.NewManager(handStore, handRunner, board)
	// Register builtin hands (only if not already persisted)
	for _, h := range hands.BuiltinHands() {
		if _, err := handManager.Status(h.Name); err != nil {
			handManager.Register(h)
		}
	}
	go func() {
		if err := handManager.Run(ctx); err != nil && err != context.Canceled {
			fmt.Fprintf(os.Stderr, "  %shands: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
		}
	}()

	// Autonomous Agent — goal-driven execution over extended time periods
	agentConfig := agent.DefaultConfig()
	agentConfig.Workspace = filepath.Join(nousDir, "agent")
	autonomousAgent := agent.NewAgent(toolReg, agentConfig)
	agentBrain := agent.NewCognitiveBridge(actions)
	// Wire Ollama for LLM-powered responses if available
	if ollamaClient := tryConnectOllama(); ollamaClient != nil {
		// Agent synthesis — LLM writes reports from web search data
		agentBrain.LLM = ollamaClient
		// Chat pipeline — LLM fills knowledge gaps (topics not in graph)
		actions.LLM = &cognitive.OllamaLLM{Client: ollamaClient}
		fmt.Printf("  %sOllama connected: %s (chat + agent synthesis)%s\n",
			cognitive.ColorGreen, ollamaClient.Model(), cognitive.ColorReset)
		// Pre-warm the model so first user query doesn't wait for model loading
		go func() {
			actions.LLM.Generate("", "warmup", 1)
		}()
	}
	autonomousAgent.SetBrain(agentBrain)
	autonomousAgent.SetReportCallback(func(msg string) {
		fmt.Printf("\n  %s[agent]%s %s\n", cognitive.ColorCyan, cognitive.ColorReset, msg)
	})
	agentScheduler := agent.NewScheduler(autonomousAgent)
	agentScheduler.Start()

	// Proactive Engine — surfaces suggestions during idle time (non-blocking)
	proactive := cognitive.NewProactiveEngine(board)

	// Dream Engine — background cognitive processing during idle time
	// Runs consolidation, pattern mining, and abstraction discovery
	go func() {
		ticker := time.NewTicker(10 * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if actions.CogGraph != nil && actions.CogGraph.NodeCount() > 0 {
					consolidator := cognitive.NewConsolidator(actions.CogGraph)
					insights := consolidator.Consolidate()
					if len(insights) > 0 {
						actions.CogGraph.Save()
					}
					// Run abstraction discovery
					ae := cognitive.NewAbstractionEngine(actions.CogGraph)
					ae.Discover()
				}
			}
		}
	}()

	// Print startup banner
	toolList := toolReg.List()
	vctxSize := ""
	if reasoner.VCtx != nil {
		totalVTokens := reasoner.VCtx.TotalSize()
		switch {
		case totalVTokens >= 1_000_000:
			vctxSize = fmt.Sprintf("%.1fM", float64(totalVTokens)/1_000_000)
		case totalVTokens >= 1_000:
			vctxSize = fmt.Sprintf("%.1fK", float64(totalVTokens)/1_000)
		default:
			vctxSize = fmt.Sprintf("%d", totalVTokens)
		}
	}
	fmt.Print(cognitive.BannerFull(version, "cognitive", "local", len(toolList), 64, vctxSize))

	// Handle graceful shutdown — save session + episodic memory
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Printf("\n\n  %ssaving...%s\n", cognitive.ColorDim, cognitive.ColorReset)
		currentSession.Messages = reasoner.Conv.Messages()
		sessionStore.Save(currentSession)
		assistantStore.Save()
		episodic.Save()
		collector.Save()
		if actions.CogGraph != nil {
			actions.CogGraph.Save()
		}
		if actions.Absorption != nil {
			actions.Absorption.Save()
		}
		if actions.Opinions != nil {
			actions.Opinions.Save()
		}
		if actions.SelfModel != nil {
			actions.SelfModel.Save(filepath.Join(nousDir, "self_model.json"))
		}
		if fileWatcher != nil {
			fileWatcher.Stop()
		}
		cancel()
		os.Exit(0)
	}()

	// --- Telegram / Discord / Matrix channels ---
	// Start channels in the background if configured via ~/.nous/channels.json
	// or NOUS_TELEGRAM_TOKEN environment variable.
	channelConfigPath := channels.DefaultConfigPath()
	if channels.ConfigExists(channelConfigPath) || os.Getenv("NOUS_TELEGRAM_TOKEN") != "" {
		chCfg, err := channels.LoadConfig(channelConfigPath)
		if err != nil && os.Getenv("NOUS_TELEGRAM_TOKEN") != "" {
			// No config file but env var set — create minimal config
			chCfg = &channels.Config{
				Telegram: &channels.TelegramConfig{
					Enabled: true,
					Token:   os.Getenv("NOUS_TELEGRAM_TOKEN"),
				},
			}
		}
		if err == nil || chCfg != nil {
			handler := channelHandler(nlu, actions, reasoner.Conv, board, wm, ltm, episodic, collector)
			chManager := channels.NewManager(handler)

			if chCfg.Telegram != nil && chCfg.Telegram.Enabled && chCfg.Telegram.Token != "" {
				tg := channels.NewTelegram(chCfg.Telegram.Token, chManager, channels.ChannelConfig{
					AllowedUsers: chCfg.Telegram.AllowedUsers,
				})
				chManager.Register(tg)
				fmt.Printf("  %s✓%s Telegram channel enabled\n", cognitive.ColorGreen, cognitive.ColorReset)
			}

			if chCfg.Discord != nil && chCfg.Discord.Enabled && chCfg.Discord.Token != "" {
				dc := channels.NewDiscord(chCfg.Discord.Token, chManager, channels.ChannelConfig{
					AllowedUsers: chCfg.Discord.AllowedUsers,
				})
				chManager.Register(dc)
				fmt.Printf("  %s✓%s Discord channel enabled\n", cognitive.ColorGreen, cognitive.ColorReset)
			}

			if chCfg.Matrix != nil && chCfg.Matrix.Enabled && chCfg.Matrix.Token != "" {
				mx := channels.NewMatrix(chCfg.Matrix.Token, chCfg.Matrix.Homeserver, chManager, channels.ChannelConfig{
					AllowedUsers: chCfg.Matrix.AllowedUsers,
					AllowedRooms: chCfg.Matrix.AllowedRooms,
				})
				chManager.Register(mx)
				fmt.Printf("  %s✓%s Matrix channel enabled\n", cognitive.ColorGreen, cognitive.ColorReset)
			}

			chManager.Start(ctx)
		}
	}

	// --- Server Mode ---
	if *serveMode {
		serveHost := strings.TrimSpace(*listenHost)
		if serveHost == "" {
			serveHost = "127.0.0.1"
		}
		if *publicMode {
			serveHost = "0.0.0.0"
		}
		addr := serveHost + ":" + *servePort
		fmt.Printf("  serving on http://%s\n\n", addr)
		srv := server.New(addr, board, perceiver, assistantStore, *apiKey)
		srv.SetFastPath(&cognitive.FastPathResponder{
			LLM:              nil,
			WorkingMem:        wm,
			LongTermMem:       ltm,
			Knowledge:         reasoner.Knowledge,
			VCtx:              reasoner.VCtx,
			Growth:            reasoner.Growth,
			ResponseCrystals:  responseCrystals,
		}, reasoner.Conv)
		srv.SetNLU(nlu)
		srv.SetActions(actions) // pass the fully-wired action router
		srv.SetPersonalResp(personalResp)
		srv.SetComposer(actions.Composer)
		srv.SetDataSources(wm, ltm, episodic, toolReg, collector, sessionStore)
		srv.SetAgent(autonomousAgent)
		if err := srv.Start(version, *model, len(toolList)); err != nil {
			fmt.Fprintf(os.Stderr, "server error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// --- Federation registry — shared crystal exchange between Nous instances ---
	fedPath := filepath.Join(nousDir, "federation")
	fedRegistry, _ := federation.NewRegistry(fedPath)

	// --- First-run onboarding or welcome-back ---
	if !cognitive.RunOnboarding(os.Stdin, ltm, wm) {
		cognitive.WelcomeBack(ltm)
	}

	// --- REPL Mode ---
	fmt.Print(cognitive.Panel("Quick start", []string{
		cognitive.Styled(cognitive.ColorCyan, "/compass") + " triage view — do now, focus, next anchor, risks",
		cognitive.Styled(cognitive.ColorCyan, "/dashboard") + " overview of the local agent",
		cognitive.Styled(cognitive.ColorCyan, "/briefing") + " morning briefing — what matters today",
		cognitive.Styled(cognitive.ColorCyan, "/now") + " one-line answer for the next best action",
		cognitive.Styled(cognitive.ColorCyan, "/today") + " review reminders, tasks, and upcoming actions",
		cognitive.Styled(cognitive.ColorCyan, "/help") + " browse commands and workflows",
		cognitive.Styled(cognitive.ColorCyan, "/plan <goal>") + " delegate a longer task",
		cognitive.Styled(cognitive.ColorCyan, "/quit") + " save and exit",
	}))

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(cognitive.Prompt())
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		// Strip stray backslashes from terminal escaping (e.g., "thanks\!" → "thanks!")
		input = strings.ReplaceAll(input, "\\!", "!")
		input = strings.ReplaceAll(input, "\\?", "?")
		if input == "" {
			continue
		}

		// Handle commands
		if strings.HasPrefix(input, "/") {
			// /learn — show what Nous has learned from conversations
			if strings.TrimSpace(strings.ToLower(input)) == "/learn" {
				fmt.Printf("  %s\n", strings.ReplaceAll(learningEngine.FormatLearningReport(), "\n", "\n  "))
				continue
			}
			if strings.HasPrefix(strings.ToLower(strings.TrimSpace(input)), "/import-wikidata") {
				cmdParts := strings.Fields(input)
				if len(cmdParts) < 2 {
					fmt.Println("  Usage: /import-wikidata <domain> [limit]")
					fmt.Println("  Domains: philosophy, science, history, technology, geography, art, music, literature, mathematics, biology")
				} else {
					domain := cmdParts[1]
					limit := 500
					if len(cmdParts) > 2 {
						fmt.Sscanf(cmdParts[2], "%d", &limit)
					}
					fmt.Printf("  Importing %s from Wikidata (limit %d)...\n", domain, limit)
					importer := cognitive.NewWikidataImporter()
					pkg, err := importer.ImportDomain(domain, limit)
					if err != nil {
						fmt.Printf("  Error: %v\n", err)
					} else {
						pDir := filepath.Join(workDir, "packages")
						if err := importer.SavePackage(pkg, pDir); err != nil {
							fmt.Printf("  Error saving: %v\n", err)
						} else {
							if pkgLoader != nil {
								pkgLoader.Install(pkg)
							}
							fmt.Printf("  Imported %d facts into %s\n", len(pkg.Facts), domain)
							if actions.CogGraph != nil {
								fmt.Printf("  Knowledge graph: %d nodes, %d edges\n", actions.CogGraph.NodeCount(), actions.CogGraph.EdgeCount())
							}
						}
					}
				}
				continue
			}
			if handleCommand(input, board, model, toolReg, wm, ltm, projMem, undoStack, sessionStore, currentSession, reasoner, learner, projectInfo, episodic, collector, autoTuner, assistantStore, handManager, autonomousAgent, agentScheduler, fedRegistry) {
				continue
			}
		}

		// Extract personal facts from user input (name, role, interests, etc.)
		// Runs on every message — stores facts in long-term + working memory.
		factExtractor := &cognitive.FactExtractor{LTM: ltm, WorkingMem: wm}
		factExtractor.Extract(input)

		// Record activity for proactive engine
		proactive.RecordInput()

		fmt.Println()
		firstToken = true
		responseStarted = false
		start := time.Now()

		// Try cognitive compiler first — compiled handlers from previous responses.
		// This is the fastest path: regex match + template fill, ~0ms.
		var answer string
		if compiled, ok := actions.TryCompiled(input); ok {
			answer = compiled
			if os.Getenv("NOUS_TRACE") != "" {
				fmt.Fprintf(os.Stderr, "  [REPL] TryCompiled HIT len=%d\n", len(answer))
			}
			reasoner.Conv.User(input)
			reasoner.Conv.Assistant(answer)
		} else if os.Getenv("NOUS_TRACE") != "" {
			fmt.Fprintf(os.Stderr, "  [REPL] TryCompiled MISS\n")
		}

		// Try NLU/ActionRouter — handles all 51 tools deterministically (0ms).
		// Supports multi-intent: "do X and Y" splits into separate actions.
		// Only falls through to Composer if the action needs language generation.
		nluResult := nlu.UnderstandMultiWithContext(input, reasoner.Conv)
		// Accept the NLU result if confidence >= 0.5, OR if the intent is one
		// that has specific handler logic (greeting, farewell, affirmation)
		// where even low-confidence classification should be trusted over
		// the generic Composer path.
		socialIntents := nluResult.Intent == "greeting" || nluResult.Intent == "farewell" || nluResult.Intent == "affirmation"
		if answer == "" && (nluResult.Confidence >= 0.5 || socialIntents) {
			if len(nluResult.SubResults) > 0 {
				// Multi-intent: execute each sub-result and combine responses
				var parts []string
				for _, sub := range nluResult.SubResults {
					subAction := actions.Execute(sub, reasoner.Conv)
					if subAction.DirectResponse != "" {
						parts = append(parts, subAction.DirectResponse)
					}
				}
				if len(parts) > 0 {
					answer = strings.Join(parts, "\n\n")
					reasoner.Conv.User(input)
					reasoner.Conv.Assistant(answer)
				}
			} else {
				actionResult := actions.Execute(nluResult, reasoner.Conv)
				// nluResult already used for multi-intent check above
				if actionResult.DirectResponse != "" {
					answer = actionResult.DirectResponse
					reasoner.Conv.User(input)
					reasoner.Conv.Assistant(answer)
				}
			}
		}

		// Composer engine: generates ALL responses without any LLM.
		// Zero external dependencies. Instant. Unique every time.
		if answer == "" && actions.Composer != nil {
			if os.Getenv("NOUS_TRACE") != "" {
				fmt.Fprintf(os.Stderr, "  [REPL] composer path (nlu conf=%.3f intent=%s action=%s)\n",
					nluResult.Confidence, nluResult.Intent, nluResult.Action)
			}
			respType := actions.ClassifyForComposer(input)
			ctx := actions.BuildComposeContext()
			resp := actions.Composer.Compose(input, respType, ctx)
			if resp != nil && resp.Text != "" {
				answer = resp.Text
				reasoner.Conv.User(input)
				reasoner.Conv.Assistant(answer)
			}
		}
		duration := time.Since(start)
		if !responseStarted && strings.TrimSpace(answer) != "" {
			fmt.Printf("  %s\n", strings.ReplaceAll(answer, "\n", "\n  "))
		}

		// Show timing footer
		fmt.Printf("\n%s\n\n", cognitive.TimingFooter(duration))

		// Record turn in Composer for conversation context
		if actions.Composer != nil {
			actions.Composer.RecordTurn(input, answer)
		}

		// Cognitive compiler learning — compile this response into a
		// deterministic handler for future use. Async, no latency impact.
		// Over time, this makes Nous faster: more queries resolve via
		// compiled patterns (~0ms) instead of full NLU+generation.
		go actions.LearnResponse(input, answer, nluResult)

		// Conversational learning — extract facts, patterns, preferences
		// from every user message. Async, no latency impact.
		go learningEngine.LearnFromConversation(input)

		// Record in episodic memory (remembers everything forever)
		go episodic.Record(memory.Episode{
			Timestamp: time.Now(),
			Input:     input,
			Output:    answer,
			Success:   true,
			Duration:  duration.Milliseconds(),
		})

		// Collect training data with auto quality scoring
		msgs := reasoner.Conv.Messages()
		sysPrompt := ""
		if len(msgs) > 0 {
			sysPrompt = msgs[0].Content
		}
		quality := scoreInteractionQuality(answer, duration, board)
		go collector.Collect(sysPrompt, input, answer, nil, quality)

		// Check if auto-tuning should trigger (non-blocking)
		go autoTuner.CheckQuiet()

		// Feed the learner from direct NLU tool usage (async, no latency)
		if nluResult.Action != "" && nluResult.Action != "llm_chat" {
			go learner.LearnFromTools(input, []string{nluResult.Action})
		}

		// Proactive suggestions — only shown during idle, never blocks
		if suggestions := proactive.Check(); len(suggestions) > 0 {
			fmt.Print(cognitive.FormatSuggestions(suggestions))
		}

		// Auto-save session after each exchange
		currentSession.Messages = reasoner.Conv.Messages()
		go sessionStore.Save(currentSession)
	}
}

func handleCommand(input string, board *blackboard.Blackboard, model *string, toolReg *tools.Registry, wm *memory.WorkingMemory, ltm *memory.LongTermMemory, projMem *memory.ProjectMemory, undoStack *memory.UndoStack, sessions *cognitive.SessionStore, current *cognitive.Session, reasoner *cognitive.Reasoner, learner *cognitive.Learner, project *cognitive.ProjectInfo, episodic *memory.EpisodicMemory, collector *training.Collector, autoTuner *training.AutoTuner, assistantStore *assistant.Store, handManager *hands.Manager, autonomousAgent *agent.Agent, agentScheduler *agent.Scheduler, fedRegistry *federation.Registry) bool {
	parts := strings.Fields(input)
	cmd := strings.ToLower(parts[0])

	switch cmd {
	case "/quit", "/exit", "/q":
		fmt.Printf("  %ssaving...%s\n", cognitive.ColorDim, cognitive.ColorReset)
		current.Messages = reasoner.Conv.Messages()
		sessions.Save(current)
		assistantStore.Save()
		projMem.Flush()
		os.Exit(0)

	case "/help", "/h":
		fmt.Print(renderHelp())

	case "/compass":
		fmt.Print(renderCompass(assistantStore, time.Now()))

	case "/dashboard":
		fmt.Print(renderDashboard(board, wm, ltm, projMem, undoStack, current, episodic, collector, autoTuner, assistantStore))

	case "/trace":
		limit := 8
		if len(parts) > 1 {
			parsed, err := strconv.Atoi(parts[1])
			if err != nil || parsed <= 0 {
				fmt.Println("  usage: /trace [count]")
				break
			}
			if parsed > 25 {
				parsed = 25
			}
			limit = parsed
		}
		fmt.Print(renderTrace(board, limit))

	case "/now":
		fmt.Println("  " + whatShouldIDoNow(assistantStore, time.Now()))

	case "/today":
		fmt.Print(renderToday(assistantStore, time.Now()))
		_ = assistantStore.MarkNotificationsRead()

	case "/briefing":
		fmt.Print(renderBriefing(assistantStore, time.Now()))
		_ = assistantStore.MarkNotificationsRead()

	case "/review":
		fmt.Print(renderReview(assistantStore, time.Now()))

	case "/checkin":
		fmt.Println("  " + assistantCheckIn(assistantStore, time.Now(), prefersGerman(assistantStore)))

	case "/focus":
		fmt.Println("  " + assistantFocusReply(assistantStore, time.Now(), prefersGerman(assistantStore)))

	case "/prep":
		fmt.Println("  " + meetingPrepReply(assistantStore, time.Now(), prefersGerman(assistantStore)))

	case "/tasks":
		fmt.Print(renderTasks(assistantStore, time.Now()))

	case "/routines":
		fmt.Print(renderRoutines(assistantStore))

	case "/remind":
		dueAt, recurrence, title, err := parseReminderInput(strings.TrimSpace(strings.TrimPrefix(input, parts[0])), time.Now())
		if err != nil {
			fmt.Printf("  reminder error: %v\n", err)
			break
		}
		task, err := assistantStore.AddTask(title, dueAt, recurrence)
		if err != nil {
			fmt.Printf("  could not create reminder: %v\n", err)
			break
		}
		fmt.Printf("  reminder saved: %s (%s) due %s\n", task.ID, task.Title, task.DueAt.Format("2006-01-02 15:04"))

	case "/routine":
		schedule, clock, title, err := parseRoutineInput(strings.TrimSpace(strings.TrimPrefix(input, parts[0])))
		if err != nil {
			fmt.Printf("  routine error: %v\n", err)
			break
		}
		routine, err := assistantStore.AddRoutine(title, schedule, clock)
		if err != nil {
			fmt.Printf("  could not create routine: %v\n", err)
			break
		}
		fmt.Printf("  routine saved: %s (%s %s)\n", routine.Title, routine.Schedule, routine.TimeOfDay)

	case "/done":
		if len(parts) < 2 {
			fmt.Println("  usage: /done <task-id>")
			break
		}
		task, err := assistantStore.MarkDone(parts[1])
		if err != nil {
			fmt.Printf("  %v\n", err)
			break
		}
		fmt.Printf("  completed: %s (%s)\n", task.Title, task.ID)

	case "/prefs":
		prefs := assistantStore.Preferences()
		if len(prefs) == 0 {
			fmt.Println("  no preferences stored yet")
			break
		}
		fmt.Print(cognitive.Section("Preferences"))
		for _, pref := range prefs {
			fmt.Print(cognitive.KeyValue(pref.Key, pref.Value))
		}

	case "/pref":
		if len(parts) < 3 {
			fmt.Println("  usage: /pref <key> <value...>")
			break
		}
		key := parts[1]
		value := strings.Join(parts[2:], " ")
		if err := assistantStore.SetPreference(key, value); err != nil {
			fmt.Printf("  could not store preference: %v\n", err)
			break
		}
		fmt.Printf("  preference saved: %s = %s\n", key, value)

	case "/capture":
		if len(parts) < 2 {
			fmt.Println("  usage: /capture <personal note>")
			break
		}
		key, value, ok := rememberedProfileNote("remember that " + strings.Join(parts[1:], " "))
		if !ok {
			fmt.Println("  could not capture note")
			break
		}
		if err := assistantStore.SetPreference(key, value); err != nil {
			fmt.Printf("  could not capture note: %v\n", err)
			break
		}
		fmt.Printf("  captured: %s\n", value)

	case "/status":
		fmt.Print(cognitive.Section("System status"))
		fmt.Print(cognitive.KeyValue("Percepts", fmt.Sprintf("%d", len(board.Percepts()))))
		fmt.Print(cognitive.KeyValue("Active goals", fmt.Sprintf("%d", len(board.ActiveGoals()))))
		fmt.Print(cognitive.KeyValue("Working memory", fmt.Sprintf("%d items", wm.Size())))
		fmt.Print(cognitive.KeyValue("Long-term", fmt.Sprintf("%d entries", ltm.Size())))
		fmt.Print(cognitive.KeyValue("Project facts", fmt.Sprintf("%d", projMem.Size())))
		fmt.Print(cognitive.KeyValue("Undo stack", fmt.Sprintf("%d entries", undoStack.Size())))
		fmt.Print(cognitive.KeyValue("Recent actions", fmt.Sprintf("%d", len(board.RecentActions(100)))))
		fmt.Print(cognitive.KeyValue("Conversation", reasoner.Conv.Summary()))
		fmt.Print(cognitive.KeyValue("Session", fmt.Sprintf("%s (%s)", current.Name, current.ID)))

	case "/hand", "/hands":
		if len(parts) < 2 {
			// List all hands
			allHands := handManager.List()
			if len(allHands) == 0 {
				fmt.Println("  no hands registered")
			} else {
				fmt.Print(cognitive.Section("Hands"))
				for _, h := range allHands {
					status := "idle"
					if !h.Enabled {
						status = "disabled"
					} else if h.State == hands.HandRunning {
						status = "running"
					}
					sched := h.Schedule
					if sched == "" {
						sched = "manual"
					}
					fmt.Printf("  %-12s  %s  [%s]  %s\n", h.Name, sched, status, h.Description)
				}
			}
		} else {
			subCmd := strings.ToLower(parts[1])
			switch subCmd {
			case "run":
				if len(parts) < 3 {
					fmt.Println("  usage: /hand run <name>")
				} else {
					name := parts[2]
					if err := handManager.RunNow(context.Background(), name); err != nil {
						fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
					} else {
						fmt.Printf("  %sstarted hand %q%s\n", cognitive.ColorGreen, name, cognitive.ColorReset)
					}
				}
			case "enable":
				if len(parts) < 3 {
					fmt.Println("  usage: /hand enable <name>")
				} else {
					if err := handManager.Activate(parts[2]); err != nil {
						fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
					} else {
						fmt.Printf("  enabled %s\n", parts[2])
					}
				}
			case "disable":
				if len(parts) < 3 {
					fmt.Println("  usage: /hand disable <name>")
				} else {
					if err := handManager.Deactivate(parts[2]); err != nil {
						fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
					} else {
						fmt.Printf("  disabled %s\n", parts[2])
					}
				}
			case "status":
				if len(parts) < 3 {
					fmt.Println("  usage: /hand status <name>")
				} else {
					h, err := handManager.Status(parts[2])
					if err != nil {
						fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
					} else {
						fmt.Printf("  %s: %s (enabled=%v, schedule=%s)\n", h.Name, h.State, h.Enabled, h.Schedule)
						if h.LastError != "" {
							fmt.Printf("  last error: %s\n", h.LastError)
						}
					}
				}
			case "history":
				if len(parts) < 3 {
					fmt.Println("  usage: /hand history <name>")
				} else {
					records := handManager.History(parts[2])
					if len(records) == 0 {
						fmt.Println("  no history")
					}
					for _, r := range records {
						status := "ok"
						if !r.Success {
							status = "FAIL"
						}
						fmt.Printf("  [%s] %s (%dms) %s\n", r.StartedAt.Format("Jan 02 15:04"), status, r.Duration, truncate(r.Output, 80))
					}
				}
			default:
				fmt.Println("  usage: /hand [run|enable|disable|status|history] <name>")
			}
		}

	case "/agent":
		if len(parts) < 2 {
			status := autonomousAgent.Status()
			if status.Running {
				fmt.Print(status.Progress)
			} else if status.Goal != "" {
				fmt.Print(autonomousAgent.Report())
			} else {
				fmt.Println("  no active agent goal")
				fmt.Println("  usage:")
				fmt.Println("    /agent start <goal>     — start autonomous execution")
				fmt.Println("    /agent status            — show progress")
				fmt.Println("    /agent input <text>      — provide human input")
				fmt.Println("    /agent stop              — stop execution")
				fmt.Println("    /agent report            — show full report")
				fmt.Println("    /agent resume            — resume from saved state")
				fmt.Println("    /agent schedule <goal> <schedule> — schedule recurring goal")
				fmt.Println("    /agent jobs              — list scheduled jobs")
			}
		} else {
			subCmd := strings.ToLower(parts[1])
			switch subCmd {
			case "start":
				if len(parts) < 3 {
					fmt.Println("  usage: /agent start <goal description>")
				} else {
					goal := strings.Join(parts[2:], " ")
					// Strip surrounding quotes if present
					goal = strings.Trim(goal, "\"'")
					if err := autonomousAgent.Start(goal); err != nil {
						fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
					} else {
						fmt.Printf("  %sagent started%s — goal: %s\n", cognitive.ColorGreen, cognitive.ColorReset, goal)
					}
				}
			case "status":
				status := autonomousAgent.Status()
				if !status.Running && status.Goal == "" {
					fmt.Println("  agent is idle — no active goal")
				} else {
					fmt.Print(status.Progress)
					if status.PausedFor != "" {
						fmt.Printf("  %swaiting for input:%s %s\n", cognitive.ColorYellow, cognitive.ColorReset, status.PausedFor)
					}
				}
			case "input":
				if len(parts) < 3 {
					fmt.Println("  usage: /agent input <your response>")
				} else {
					input := strings.Join(parts[2:], " ")
					input = strings.Trim(input, "\"'")
					autonomousAgent.Resume(input)
					fmt.Printf("  %sinput sent%s\n", cognitive.ColorGreen, cognitive.ColorReset)
				}
			case "stop":
				autonomousAgent.Stop()
				fmt.Printf("  %sagent stopped%s\n", cognitive.ColorYellow, cognitive.ColorReset)
			case "report":
				fmt.Print(autonomousAgent.Report())
			case "resume":
				if err := autonomousAgent.LoadAndResume(); err != nil {
					fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
				} else {
					fmt.Printf("  %sagent resumed%s\n", cognitive.ColorGreen, cognitive.ColorReset)
				}
			case "schedule":
				if len(parts) < 4 {
					fmt.Println("  usage: /agent schedule <goal> <schedule>")
					fmt.Println("  schedules: \"daily 9:00\", \"hourly\", \"every 30m\", \"weekdays 8:00\"")
				} else {
					// Parse: everything before the last word(s) is the goal, last is schedule
					// Use a simple heuristic: schedule keywords are at the end
					args := strings.Join(parts[2:], " ")
					goal, sched := splitGoalAndSchedule(args)
					id, err := agentScheduler.AddJob(goal, sched)
					if err != nil {
						fmt.Printf("  %serror: %v%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
					} else {
						fmt.Printf("  %sscheduled%s %s [%s] — %s\n", cognitive.ColorGreen, cognitive.ColorReset, id, sched, goal)
					}
				}
			case "jobs":
				jobs := agentScheduler.ListJobs()
				if len(jobs) == 0 {
					fmt.Println("  no scheduled jobs")
				} else {
					for _, j := range jobs {
						status := "enabled"
						if !j.Enabled {
							status = "disabled"
						}
						fmt.Printf("  %-8s  %-15s  [%s]  %s\n", j.ID, j.Schedule, status, j.Goal)
					}
				}
			case "rmjob":
				if len(parts) < 3 {
					fmt.Println("  usage: /agent rmjob <job-id>")
				} else {
					agentScheduler.RemoveJob(parts[2])
					fmt.Printf("  removed %s\n", parts[2])
				}
			default:
				fmt.Printf("  unknown agent command: %s\n", subCmd)
				fmt.Println("  try: start, status, input, stop, report, resume, schedule, jobs, rmjob")
			}
		}

	case "/memory":
		items := wm.MostRelevant(10)
		if len(items) == 0 {
			fmt.Println("  working memory is empty")
		}
		for _, item := range items {
			fmt.Printf("  [%.2f] %s: %v\n", item.Relevance, item.Key, item.Value)
		}

	case "/longterm":
		entries := ltm.All()
		if len(entries) == 0 {
			fmt.Println("  long-term memory is empty")
		}
		for _, e := range entries {
			fmt.Printf("  [%s] %s: %s (accessed %d times)\n", e.Category, e.Key, e.Value, e.AccessCount)
		}

	case "/goals":
		goals := board.ActiveGoals()
		if len(goals) == 0 {
			fmt.Println("  no active goals")
		}
		for _, g := range goals {
			fmt.Printf("  [%s] %s (priority: %d)\n", g.Status, g.Description, g.Priority)
		}

	case "/model":
		fmt.Print(cognitive.Section("Cognitive engine"))
		fmt.Print(cognitive.KeyValue("Active engine", *model))
		fmt.Print(cognitive.KeyValue("Mode", "local cognitive (no LLM)"))
		fmt.Print(cognitive.KeyValue("Pipeline", "NLU → ActionRouter → ReasoningPipeline → Composer"))

	case "/tools":
		fmt.Print(renderToolCatalog(toolReg))

	case "/project":
		fmt.Print(renderProjectView(project))

	case "/sessions":
		list, err := sessions.List()
		if err != nil || len(list) == 0 {
			fmt.Println("  no saved sessions")
		} else {
			fmt.Print(cognitive.Section("Saved sessions"))
			for _, s := range list {
				marker := " "
				if s.ID == current.ID {
					marker = "*"
				}
				fmt.Printf("  %s %s  %-30s  %d msgs  %s\n",
					marker, s.ID, s.Name, len(s.Messages),
					s.UpdatedAt.Format("2006-01-02 15:04"))
			}
			fmt.Println()
			fmt.Print(cognitive.KeyValue("Resume", "nous --resume <ID>"))
		}

	case "/save":
		if len(parts) > 1 {
			current.Name = strings.Join(parts[1:], " ")
		}
		current.Messages = reasoner.Conv.Messages()
		if err := sessions.Save(current); err != nil {
			fmt.Printf("  error saving: %v\n", err)
		} else {
			fmt.Printf("  session saved: %s (%s)\n", current.Name, current.ID)
		}

	case "/remember":
		if len(parts) < 3 {
			fmt.Println("  usage: /remember <key> <value...>")
			break
		}
		key := parts[1]
		value := strings.Join(parts[2:], " ")
		projMem.Remember(key, value, "user", 1.0)
		if err := projMem.Flush(); err != nil {
			fmt.Printf("  remembered but failed to persist: %v\n", err)
		} else {
			fmt.Printf("  remembered: %s = %s\n", key, value)
		}

	case "/recall":
		if len(parts) < 2 {
			fmt.Println("  usage: /recall <query>")
			break
		}
		query := strings.Join(parts[1:], " ")
		// Try exact match first
		if fact, ok := projMem.Recall(query); ok {
			fmt.Printf("  [%.0f%%] %s: %s (source: %s)\n",
				fact.Confidence*100, fact.Key, fact.Value, fact.Source)
		} else {
			// Fall back to keyword search
			results := projMem.Search(query)
			if len(results) == 0 {
				fmt.Println("  no matching facts found")
			} else {
				for _, f := range results {
					fmt.Printf("  [%.0f%%] %s: %s (source: %s)\n",
						f.Confidence*100, f.Key, f.Value, f.Source)
				}
			}
		}

	case "/forget":
		if len(parts) < 2 {
			fmt.Println("  usage: /forget <key>")
			break
		}
		key := parts[1]
		if _, ok := projMem.Recall(key); !ok {
			fmt.Printf("  no fact with key %q\n", key)
		} else {
			projMem.Forget(key)
			projMem.Flush()
			fmt.Printf("  forgot: %s\n", key)
		}

	case "/undo":
		if entry, ok := undoStack.Peek(); ok {
			action := entry.Action
			path := entry.Path
			if _, undoErr := undoStack.Undo(); undoErr != nil {
				fmt.Printf("  undo failed: %v\n", undoErr)
			} else {
				fmt.Printf("  undone: %s %s\n", action, path)
			}
		} else {
			fmt.Println("  nothing to undo")
		}

	case "/history":
		entries := undoStack.List()
		if len(entries) == 0 {
			fmt.Println("  undo stack is empty")
		} else {
			for i, e := range entries {
				age := time.Since(e.Timestamp).Truncate(time.Second)
				tag := ""
				if e.WasNew {
					tag = " (new file)"
				}
				fmt.Printf("  %d. [%s ago] %s %s%s\n", i+1, age, e.Action, e.Path, tag)
			}
		}

	case "/episodes":
		recent := episodic.Recent(10)
		if len(recent) == 0 {
			fmt.Println("  no episodes recorded")
		} else {
			fmt.Printf("  %d total episodes (showing last %d):\n", episodic.Size(), len(recent))
			for _, ep := range recent {
				age := time.Since(ep.Timestamp).Truncate(time.Second)
				status := "OK"
				if !ep.Success {
					status = "FAIL"
				}
				q := ep.Input
				if len(q) > 60 {
					q = q[:60] + "..."
				}
				fmt.Printf("  [%s ago] [%s] %s\n", age, status, q)
			}
			fmt.Printf("  success rate: %.0f%%\n", episodic.SuccessRate()*100)
		}

	case "/search":
		if len(parts) < 2 {
			fmt.Println("  usage: /search <query>")
			break
		}
		query := strings.Join(parts[1:], " ")
		results := episodic.Search(query, 5)
		if len(results) == 0 {
			fmt.Println("  no matching episodes")
		} else {
			for _, ep := range results {
				age := time.Since(ep.Timestamp).Truncate(time.Second)
				q := ep.Input
				if len(q) > 50 {
					q = q[:50] + "..."
				}
				a := ep.Output
				if len(a) > 50 {
					a = a[:50] + "..."
				}
				fmt.Printf("  [%s ago] Q: %s\n", age, q)
				if a != "" {
					fmt.Printf("           A: %s\n", a)
				}
			}
		}

	case "/training":
		fmt.Printf("  training pairs: %d\n", collector.Size())
		dist := collector.QualityDistribution()
		if len(dist) > 0 {
			fmt.Println("  quality distribution:")
			for bucket, count := range dist {
				fmt.Printf("    %s: %d pairs\n", bucket, count)
			}
		}

	case "/autotune":
		stats := autoTuner.Stats()
		fmt.Printf("  auto-tune status:\n")
		fmt.Printf("    training pairs:  %d / %d minimum\n", stats.PairCount, stats.MinPairs)
		fmt.Printf("    avg quality:     %.2f (floor: %.2f)\n", stats.AvgQuality, stats.QualityFloor)
		fmt.Printf("    tuned model:     %s\n", stats.TunedName)
		if !stats.LastTuneAt.IsZero() {
			fmt.Printf("    last tuned:      %s\n", stats.LastTuneAt.Format("2006-01-02 15:04:05"))
			fmt.Printf("    next tune after: %s\n", stats.NextTuneAfter.Format("2006-01-02 15:04:05"))
		} else {
			fmt.Printf("    last tuned:      never\n")
		}
		if stats.Ready {
			fmt.Printf("    status:          READY (conditions met)\n")
		} else {
			fmt.Printf("    status:          waiting\n")
		}
		// Handle /autotune force
		if len(parts) > 1 && strings.ToLower(parts[1]) == "force" {
			fmt.Println()
			if autoTuner.ForceCheck() {
				fmt.Println("  forced tune completed successfully")
			} else {
				fmt.Println("  forced tune did not trigger (need minimum pairs and quality)")
			}
		}

	case "/export":
		if len(parts) < 2 {
			fmt.Println("  usage: /export <format>  (jsonl, alpaca, chatml)")
			break
		}
		format := strings.ToLower(parts[1])
		path := filepath.Join(cognitive.WorkDir, ".nous", "export_"+format)
		var exportErr error
		switch format {
		case "jsonl":
			path += ".jsonl"
			exportErr = collector.ExportJSONL(path)
		case "alpaca":
			path += ".json"
			exportErr = collector.ExportAlpaca(path)
		case "chatml":
			path += ".jsonl"
			exportErr = collector.ExportChatML(path)
		default:
			fmt.Printf("  unknown format: %s (try jsonl, alpaca, chatml)\n", format)
			break
		}
		if exportErr != nil {
			fmt.Printf("  export error: %v\n", exportErr)
		} else {
			fmt.Printf("  exported %d pairs to %s\n", collector.Size(), path)
		}

	case "/finetune":
		cfg := training.DefaultModelfileConfig(*model)
		cfg.System = training.NousSystemPrompt()
		mfPath := filepath.Join(cognitive.WorkDir, ".nous", "Modelfile")
		if err := training.WriteModelfile(mfPath, cfg); err != nil {
			fmt.Printf("  error writing Modelfile: %v\n", err)
			break
		}
		fmt.Printf("  Modelfile written to %s\n", mfPath)
		fmt.Println()
		fmt.Println("  To create your custom model:")
		fmt.Printf("    ollama create %s -f %s\n", cfg.Name, mfPath)
		fmt.Println()
		fmt.Println("  To fine-tune with LoRA (requires Python + unsloth):")
		fmt.Println("    1. /export chatml")
		fmt.Printf("    2. python finetune.py  (generates LoRA adapter)\n")
		fmt.Println("    3. Update Modelfile with ADAPTER path")
		fmt.Printf("    4. ollama create %s -f %s\n", cfg.Name, mfPath)
		fmt.Printf("    5. nous --model %s\n", cfg.Name)

	case "/compile":
		if reasoner.Compiler == nil {
			fmt.Println("  model compiler not initialized")
			break
		}
		modelfile := reasoner.Compiler.GenerateModelfile()
		mfPath := filepath.Join(cognitive.WorkDir, ".nous", "Modelfile.compiled")
		if err := os.WriteFile(mfPath, []byte(modelfile), 0644); err != nil {
			fmt.Printf("  error writing modelfile: %v\n", err)
			break
		}
		modelName := reasoner.Compiler.ModelName()
		fmt.Printf("  compiled model: %s\n", reasoner.Compiler.Versions[len(reasoner.Compiler.Versions)-1].ModelName)
		fmt.Printf("  modelfile: %s\n", mfPath)
		fmt.Printf("  next version: %s\n", modelName)
		fmt.Println()
		fmt.Println("  To create the compiled model:")
		fmt.Printf("    ollama create %s -f %s\n", reasoner.Compiler.Versions[len(reasoner.Compiler.Versions)-1].ModelName, mfPath)
		fmt.Printf("    nous --model %s\n", reasoner.Compiler.Versions[len(reasoner.Compiler.Versions)-1].ModelName)

	case "/ingest":
		if reasoner.Knowledge == nil {
			fmt.Println("  knowledge store not initialized")
			break
		}
		if len(parts) < 2 {
			fmt.Println("  usage: /ingest <file-path>")
			fmt.Println("  ingests a text file into the knowledge vector store")
			break
		}
		filePath := strings.Join(parts[1:], " ")
		fmt.Printf("  ingesting %s...\n", filePath)
		added, err := reasoner.Knowledge.Ingest(filePath)
		if err != nil {
			fmt.Printf("  error: %v\n", err)
			break
		}
		fmt.Printf("  added %d knowledge chunks (total: %d)\n", added, reasoner.Knowledge.Size())

	case "/knowledge":
		if reasoner.Knowledge == nil {
			fmt.Println("  knowledge store not initialized")
			break
		}
		fmt.Printf("  chunks: %d\n", reasoner.Knowledge.Size())
		fmt.Printf("  searches: %d\n", reasoner.Knowledge.SearchCount)
		fmt.Printf("  hits: %d\n", reasoner.Knowledge.HitCount)
		if len(parts) > 1 {
			query := strings.Join(parts[1:], " ")
			results, err := reasoner.Knowledge.Search(query, 3)
			if err != nil {
				fmt.Printf("  search error: %v\n", err)
			} else if len(results) == 0 {
				fmt.Println("  no results")
			} else {
				for _, r := range results {
					fmt.Printf("  [%.2f] %s (%s)\n", r.Score, truncate(r.Text, 80), r.Source)
				}
			}
		}

	case "/packages":
		if pkgLoader == nil {
			fmt.Println("  package loader not initialized")
			break
		}
		loaded := pkgLoader.ListLoaded()
		if len(loaded) == 0 {
			fmt.Println("  no packages loaded")
			fmt.Println("  place .json packages in ./packages/ and restart")
		} else {
			fmt.Printf("  %d packages loaded:\n", len(loaded))
			for name, version := range loaded {
				fmt.Printf("    %s v%s\n", name, version)
			}
		}
		if len(parts) > 1 {
			// /packages load <file>
			if parts[1] == "load" && len(parts) > 2 {
				path := strings.Join(parts[2:], " ")
				result, err := pkgLoader.LoadFile(path)
				if err != nil {
					fmt.Printf("  error: %v\n", err)
				} else {
					fmt.Printf("  %s\n", result)
				}
			}
		}

	case "/vctx":
		if reasoner.VCtx == nil {
			fmt.Println("  virtual context not initialized")
			break
		}
		stats := reasoner.VCtx.Stats()
		fmt.Print("  " + strings.ReplaceAll(stats.FormatStats(), "\n", "\n  "))
		fmt.Println()

	case "/growth":
		if reasoner.Growth == nil {
			fmt.Println("  personal growth not initialized")
			break
		}
		stats := reasoner.Growth.Stats()
		fmt.Printf("  interactions: %d\n", stats.TotalInteractions)
		fmt.Printf("  topics tracked: %d\n", stats.TopicsTracked)
		fmt.Printf("  facts learned: %d\n", stats.FactsLearned)
		fmt.Printf("  days known: %d\n", stats.DaysKnown)
		top := reasoner.Growth.TopInterests(5)
		if len(top) > 0 {
			fmt.Println("  top interests:")
			for _, t := range top {
				fmt.Printf("    %-20s (weight: %.2f, seen: %d times)\n", t.Name, t.Weight, t.Count)
			}
		}

	case "/learn":
		if reasoner.Growth == nil {
			fmt.Println("  personal growth not initialized")
			break
		}
		if len(parts) < 3 {
			fmt.Println("  usage: /learn <category> <fact>")
			fmt.Println("  categories: work, interest, preference, identity")
			break
		}
		category := parts[1]
		fact := strings.Join(parts[2:], " ")
		reasoner.Growth.LearnFact(fact, category)
		fmt.Printf("  learned: %s (%s)\n", fact, category)

	case "/cortex":
		if reasoner.Cortex == nil {
			fmt.Println("  neural cortex not initialized")
			break
		}
		trainCount, paramCount := reasoner.Cortex.Stats()
		fmt.Printf("  parameters: %d\n", paramCount)
		fmt.Printf("  training samples: %d\n", trainCount)
		fmt.Printf("  labels: %s\n", strings.Join(reasoner.Cortex.Labels, ", "))
		if len(parts) > 1 {
			query := strings.Join(parts[1:], " ")
			pred := reasoner.Cortex.Predict(cognitive.CortexInputFromQuery(query, reasoner.Cortex.InputSize))
			fmt.Printf("  prediction for %q: %s (%.1f%%)\n", query, pred.Label, pred.Confidence*100)
		}

	case "/plan":
		if len(parts) < 2 {
			fmt.Println("  usage: /plan <goal description>")
			break
		}
		goalDesc := strings.Join(parts[1:], " ")
		goalID := fmt.Sprintf("goal-%d", time.Now().UnixMilli())
		fmt.Printf("  pushing goal: %s\n", goalDesc)
		fmt.Println("  planning...")
		board.PushGoal(blackboard.Goal{
			ID:          goalID,
			Description: goalDesc,
			Priority:    1,
			Status:      "pending",
			CreatedAt:   time.Now(),
		})
		// Wait for the goal to complete (Planner→Executor→Learner pipeline)
		deadline := time.After(120 * time.Second)
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()
		done := false
		for !done {
			select {
			case <-deadline:
				fmt.Println("  (timeout — goal still in progress)")
				done = true
			case <-ticker.C:
				plan, ok := board.PlanForGoal(goalID)
				if !ok {
					continue
				}
				switch plan.Status {
				case "completed":
					fmt.Println("  goal completed!")
					for _, s := range plan.Steps {
						status := "OK"
						if s.Status == "failed" {
							status = "FAIL"
						}
						desc := s.Description
						if len(desc) > 60 {
							desc = desc[:60] + "..."
						}
						fmt.Printf("    [%s] %s: %s\n", status, s.Tool, desc)
					}
					done = true
				case "failed":
					fmt.Println("  goal failed.")
					for _, s := range plan.Steps {
						if s.Status == "failed" {
							fmt.Printf("    failed at: %s — %s\n", s.Tool, s.Result)
						}
					}
					done = true
				}
			}
		}

	case "/patterns":
		patterns := learner.Patterns()
		if len(patterns) == 0 {
			fmt.Println("  no patterns learned yet")
			fmt.Println("  (patterns are extracted after successful multi-step interactions)")
		} else {
			fmt.Printf("  %d learned patterns:\n", len(patterns))
			for i, p := range patterns {
				chain := strings.Join(p.ToolChain, "→")
				age := time.Since(p.LastUsed).Truncate(time.Second)
				fmt.Printf("  %d. [%.0f%%] %s: %s (used %d times, %s ago)\n",
					i+1, p.Confidence*100, p.Trigger, chain, p.Uses, age)
			}
		}

	case "/clear":
		reasoner.Conv = cognitive.NewConversation(20)
		fmt.Println("  conversation cleared")

	case "/federation":
		if fedRegistry == nil {
			fmt.Println("  federation registry not available")
			break
		}
		sub := ""
		if len(parts) > 1 {
			sub = strings.ToLower(parts[1])
		}
		switch sub {
		case "status":
			stats := fedRegistry.Stats()
			fmt.Println(cognitive.Panel("Federation", []string{
				fmt.Sprintf("Crystals  %d", stats.TotalCrystals),
				fmt.Sprintf("Bundles   %d", stats.TotalBundles),
				fmt.Sprintf("Avg qual  %.1f%%", stats.AvgQuality*100),
			}))
			if len(stats.TopIntents) > 0 {
				fmt.Println("  top intents:")
				for intent, count := range stats.TopIntents {
					fmt.Printf("    %s: %d\n", intent, count)
				}
			}

		case "export":
			ts := time.Now().Format("20060102-150405")
			exportPath := filepath.Join(fedRegistry.Path, "export_"+ts+".json")
			if err := fedRegistry.Export(exportPath, 0.0); err != nil {
				fmt.Printf("  export failed: %v\n", err)
				break
			}
			stats := fedRegistry.Stats()
			fmt.Printf("  exported %d crystals to %s\n", stats.TotalCrystals, exportPath)

		case "import":
			if len(parts) < 3 {
				fmt.Println("  usage: /federation import <path>")
				break
			}
			bundlePath := parts[2]
			bundle, err := federation.ImportBundle(bundlePath)
			if err != nil {
				fmt.Printf("  import failed: %v\n", err)
				break
			}
			if err := bundle.Validate(); err != nil {
				fmt.Printf("  invalid bundle: %v\n", err)
				break
			}
			scorer := federation.NewTrustScorer()
			accepted, rejected := 0, 0
			for _, c := range bundle.Crystals {
				if scorer.ShouldImport(c) {
					fedRegistry.Merge(c)
					accepted++
				} else {
					rejected++
				}
			}
			scorer.ObserveBundle(bundle.Instance, accepted, rejected)
			fmt.Printf("  imported %d crystals (%d rejected by trust scorer)\n", accepted, rejected)

		case "search":
			if len(parts) < 3 {
				fmt.Println("  usage: /federation search <query>")
				break
			}
			query := strings.Join(parts[2:], " ")
			results := fedRegistry.Search(query, 10)
			if len(results) == 0 {
				fmt.Println("  no crystals matched")
				break
			}
			fmt.Printf("  %d results for %q:\n", len(results), query)
			for i, c := range results {
				fmt.Printf("  %d. [%.0f%% q, %d votes] %s — %s\n",
					i+1, c.Quality*100, c.Votes, c.Intent, c.Pattern)
			}

		case "top":
			n := 10
			if len(parts) > 2 {
				parsed, err := strconv.Atoi(parts[2])
				if err == nil && parsed > 0 {
					n = parsed
				}
			}
			top := fedRegistry.TopCrystals(n)
			if len(top) == 0 {
				fmt.Println("  no crystals in registry")
				break
			}
			fmt.Printf("  top %d crystals:\n", len(top))
			for i, c := range top {
				fmt.Printf("  %d. [%.0f%% q, %d votes] %s — %s\n",
					i+1, c.Quality*100, c.Votes, c.Intent, c.Pattern)
			}

		default:
			fmt.Println("  usage: /federation <status|export|import|search|top>")
			fmt.Println("  subcommands:")
			fmt.Println("    status         show federation stats")
			fmt.Println("    export         export crystals to a bundle file")
			fmt.Println("    import <path>  import a crystal bundle")
			fmt.Println("    search <query> search registry for crystals")
			fmt.Println("    top [n]        show top N crystals by quality")
		}

	default:
		fmt.Printf("  unknown command: %s (try /help)\n", cmd)
	}

	return true
}

func renderHelp() string {
	var b strings.Builder
	b.WriteString(cognitive.Panel("Core workflows", []string{
		cognitive.Styled(cognitive.ColorCyan, "/compass") + " triage panel for the next action, focus cue, and risk signals",
		cognitive.Styled(cognitive.ColorCyan, "/dashboard") + " snapshot of memory, sessions, training, and uptime signals",
		cognitive.Styled(cognitive.ColorCyan, "/trace [n]") + " inspect the latest tool actions and outcomes",
		cognitive.Styled(cognitive.ColorCyan, "/now") + " deterministic answer for what to do next",
		cognitive.Styled(cognitive.ColorCyan, "/focus") + " calm, personalized focus guidance when the day feels noisy",
		cognitive.Styled(cognitive.ColorCyan, "/today") + " open your assistant inbox with due reminders and upcoming tasks",
		cognitive.Styled(cognitive.ColorCyan, "/status") + " low-level runtime counters and current session state",
		cognitive.Styled(cognitive.ColorCyan, "/plan <goal>") + " hand a longer task to the planner/executor pipeline",
		cognitive.Styled(cognitive.ColorCyan, "/tools") + " browse built-in tools by category",
	}))
	b.WriteString(cognitive.Panel("Assistant operations", []string{
		cognitive.Styled(cognitive.ColorCyan, "/briefing") + " morning briefing — overdue, today's schedule, routines",
		cognitive.Styled(cognitive.ColorCyan, "/checkin") + " quick pulse: done today, overdue, and what comes next",
		cognitive.Styled(cognitive.ColorCyan, "/prep") + " prepare for your next meeting or time-boxed task",
		cognitive.Styled(cognitive.ColorCyan, "/review") + " evening review — completed, still pending, overdue",
		"/remind <when> <task>, /tasks, /done <task-id>",
		"/routine <daily|weekdays> <HH:MM> <task>, /routines",
		"/pref <key> <value>, /prefs, /capture <note>",
		"Examples: /remind in 2h stretch · /remind tomorrow 09:00 dentist",
	}))
	b.WriteString(cognitive.Panel("Memory and recall", []string{
		"/memory, /longterm, /episodes, /search <query>",
		"/remember <key> <value>, /recall <query>, /forget <key>",
	}))
	b.WriteString(cognitive.Panel("Training and tuning", []string{
		"/training, /autotune [force], /export <jsonl|alpaca|chatml>",
		"/finetune to generate a Modelfile and local tuning guide",
		"/compile to generate an experience-compiled Modelfile",
		"/cortex [query] — neural cortex stats and prediction test",
	}))
	b.WriteString(cognitive.Panel("Knowledge and growth", []string{
		"/ingest <file> — add text file to knowledge vector store",
		"/knowledge [query] — stats and search the knowledge store",
		"/vctx — virtual context stats (your 200K+ token window)",
		"/growth — your interests, topics, and interaction patterns",
		"/learn <category> <fact> — teach Nous about yourself",
	}))
	b.WriteString(cognitive.Panel("Federation", []string{
		cognitive.Styled(cognitive.ColorCyan, "/federation status") + " crystals, bundles, avg quality",
		cognitive.Styled(cognitive.ColorCyan, "/federation export") + " export local crystals to a bundle file",
		cognitive.Styled(cognitive.ColorCyan, "/federation import <path>") + " import a crystal bundle",
		cognitive.Styled(cognitive.ColorCyan, "/federation search <query>") + " search registry for crystals",
		cognitive.Styled(cognitive.ColorCyan, "/federation top [n]") + " show top N crystals by quality",
	}))
	b.WriteString(cognitive.Panel("Sessions and safety", []string{
		"/sessions, /save [name], /clear, /undo, /history, /quit",
		"Tip: use /dashboard first when reconnecting to a long-lived session.",
	}))
	return b.String()
}

func renderDashboard(board *blackboard.Blackboard, wm *memory.WorkingMemory, ltm *memory.LongTermMemory, projMem *memory.ProjectMemory, undoStack *memory.UndoStack, current *cognitive.Session, episodic *memory.EpisodicMemory, collector *training.Collector, autoTuner *training.AutoTuner, assistantStore *assistant.Store) string {
	stats := autoTuner.Stats()
	left := cognitive.Panel("Runtime", []string{
		fmt.Sprintf("Percepts       %d", len(board.Percepts())),
		fmt.Sprintf("Active goals   %d", len(board.ActiveGoals())),
		fmt.Sprintf("Recent actions %d", len(board.RecentActions(25))),
		fmt.Sprintf("Undo entries   %d", undoStack.Size()),
	})
	right := cognitive.Panel("Memory", []string{
		fmt.Sprintf("Working   %d items", wm.Size()),
		fmt.Sprintf("Long-term %d entries", ltm.Size()),
		fmt.Sprintf("Project   %d facts", projMem.Size()),
		fmt.Sprintf("Episodes  %d total (%.0f%% success)", episodic.Size(), episodic.SuccessRate()*100),
	})
	train := cognitive.Panel("Learning loop", []string{
		fmt.Sprintf("Training pairs   %d", collector.Size()),
		fmt.Sprintf("Avg quality      %.2f", stats.AvgQuality),
		fmt.Sprintf("Tuned model      %s", stats.TunedName),
		fmt.Sprintf("Ready to tune    %t", stats.Ready),
	})
	session := cognitive.Panel("Session", []string{
		fmt.Sprintf("Name      %s", current.Name),
		fmt.Sprintf("ID        %s", current.ID),
		fmt.Sprintf("Messages  %d", len(current.Messages)),
	})
	assistantPanel := cognitive.Panel("Assistant", []string{
		fmt.Sprintf("Pending tasks  %d", len(assistantStore.PendingTasks())),
		fmt.Sprintf("Unread inbox   %d", len(assistantStore.UnreadNotifications())),
		fmt.Sprintf("Routines       %d", len(assistantStore.Routines())),
		fmt.Sprintf("Preferences    %d", len(assistantStore.Preferences())),
	})

	return left + right + assistantPanel + train + session
}

func renderTrace(board *blackboard.Blackboard, limit int) string {
	if limit <= 0 {
		limit = 8
	}
	actions := board.RecentActions(limit)
	if len(actions) == 0 {
		return cognitive.Panel("Operator trace", []string{"No actions recorded yet."})
	}

	lines := make([]string, 0, len(actions))
	for i := len(actions) - 1; i >= 0; i-- {
		action := actions[i]
		status := "OK"
		if !action.Success {
			status = "FAIL"
		}
		input := strings.TrimSpace(action.Input)
		if input == "" {
			input = "(no input)"
		}
		if len(input) > 36 {
			input = input[:36] + "..."
		}
		output := strings.TrimSpace(action.Output)
		if idx := strings.IndexByte(output, '\n'); idx >= 0 {
			output = output[:idx]
		}
		if output == "" {
			output = "(no output)"
		}
		if len(output) > 48 {
			output = output[:48] + "..."
		}
		lines = append(lines, fmt.Sprintf("%s  %s  %s  %s  -> %s", action.Timestamp.Format("15:04:05"), status, action.Tool, input, output))
	}

	return cognitive.Panel("Operator trace", lines)
}

func renderCompass(store *assistant.Store, now time.Time) string {
	de := prefersGerman(store)
	title := "Compass"
	if de {
		title = "Kompass"
	}

	lines := []string{
		whatShouldIDoNow(store, now),
		assistantFocusReply(store, now, de),
	}

	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			lines = append(lines, fmt.Sprintf("Nächster Anker: %s um %s", task.Title, task.DueAt.Format("15:04")))
		} else {
			lines = append(lines, fmt.Sprintf("Next anchor: %s at %s", task.Title, task.DueAt.Format("15:04")))
		}
	}

	overdue := len(store.Overdue(now))
	unread := len(store.UnreadNotifications())
	if de {
		lines = append(lines, fmt.Sprintf("Risiken: %d überfällig · %d ungelesen", overdue, unread))
	} else {
		lines = append(lines, fmt.Sprintf("Risks: %d overdue · %d unread", overdue, unread))
	}

	return cognitive.Panel(title, lines)
}

func renderToday(store *assistant.Store, now time.Time) string {
	unread := store.UnreadNotifications()
	today := store.Today(now)
	upcoming := store.Upcoming(5, now)

	noteLines := []string{"No unread reminders."}
	if len(unread) > 0 {
		noteLines = noteLines[:0]
		for _, note := range unread {
			noteLines = append(noteLines, fmt.Sprintf("%s  %s", note.CreatedAt.Format("15:04"), note.Message))
		}
	}

	todayLines := []string{"Nothing scheduled for today."}
	if len(today) > 0 {
		todayLines = todayLines[:0]
		for _, task := range today {
			todayLines = append(todayLines, fmt.Sprintf("%s  %s (%s)", task.DueAt.Format("15:04"), task.Title, task.ID))
		}
	}

	upcomingLines := []string{"No upcoming tasks."}
	if len(upcoming) > 0 {
		upcomingLines = upcomingLines[:0]
		for _, task := range upcoming {
			upcomingLines = append(upcomingLines, fmt.Sprintf("%s  %s (%s)", task.DueAt.Format("2006-01-02 15:04"), task.Title, task.ID))
		}
	}

	return cognitive.Panel("Inbox", noteLines) +
		cognitive.Panel("Today", todayLines) +
		cognitive.Panel("Upcoming", upcomingLines)
}

func renderTasks(store *assistant.Store, now time.Time) string {
	tasks := store.PendingTasks()
	lines := []string{"No pending tasks."}
	if len(tasks) > 0 {
		lines = lines[:0]
		for _, task := range tasks {
			status := "upcoming"
			if !task.DueAt.After(now) {
				status = "due"
			}
			recur := ""
			if task.Recurrence != "" {
				recur = " · " + task.Recurrence
			}
			lines = append(lines, fmt.Sprintf("%s  %s (%s%s)", task.DueAt.Format("2006-01-02 15:04"), task.Title, status, recur))
		}
	}
	return cognitive.Panel("Tasks", lines)
}

func renderRoutines(store *assistant.Store) string {
	routines := store.Routines()
	lines := []string{"No routines configured."}
	if len(routines) > 0 {
		lines = lines[:0]
		for _, routine := range routines {
			state := "enabled"
			if !routine.Enabled {
				state = "disabled"
			}
			lines = append(lines, fmt.Sprintf("%s  %s (%s, %s)", routine.TimeOfDay, routine.Title, routine.Schedule, state))
		}
	}
	return cognitive.Panel("Routines", lines)
}

// splitGoalAndSchedule splits "/agent schedule <goal> <schedule>" args.
// Schedule keywords appear at the end: "daily 9:00", "hourly", "every 30m", etc.
func splitGoalAndSchedule(s string) (string, string) {
	lower := strings.ToLower(s)
	// Try to find schedule pattern at the end
	schedPatterns := []string{
		"daily ", "hourly", "every ", "weekdays ", "weekly ",
	}
	for _, pat := range schedPatterns {
		idx := strings.LastIndex(lower, pat)
		if idx > 0 {
			return strings.TrimSpace(s[:idx]), strings.TrimSpace(s[idx:])
		}
	}
	// Fallback: last word is schedule
	parts := strings.Fields(s)
	if len(parts) >= 2 {
		return strings.Join(parts[:len(parts)-1], " "), parts[len(parts)-1]
	}
	return s, "daily 9:00"
}

// tryConnectOllama attempts to connect to a local Ollama instance.
// Returns nil if Ollama is not running or not available.
func tryConnectOllama() *ollama.Client {
	client := ollama.New()
	if err := client.Ping(); err != nil {
		return nil
	}
	return client
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func prefersGerman(store *assistant.Store) bool {
	if store == nil {
		return false
	}
	return strings.HasPrefix(strings.ToLower(preferenceValue(store, "language")), "de")
}

func renderBriefing(store *assistant.Store, now time.Time) string {
	de := prefersGerman(store)
	greeting := "Good morning"
	hour := now.Hour()
	switch {
	case hour >= 17:
		greeting = "Good evening"
	case hour >= 12:
		greeting = "Good afternoon"
	}
	if de {
		switch {
		case hour >= 17:
			greeting = "Guten Abend"
		case hour >= 12:
			greeting = "Guten Tag"
		default:
			greeting = "Guten Morgen"
		}
	}

	var sections []string

	// Overdue
	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		lines := make([]string, 0, len(overdue))
		for _, task := range overdue {
			ago := friendlyDuration(now.Sub(task.DueAt), de)
			if de {
				lines = append(lines, fmt.Sprintf("  %s  %s (%s überfällig)", task.DueAt.Format("15:04"), task.Title, ago))
			} else {
				lines = append(lines, fmt.Sprintf("  %s  %s (overdue %s)", task.DueAt.Format("15:04"), task.Title, ago))
			}
		}
		title := fmt.Sprintf("Overdue (%d)", len(overdue))
		if de {
			title = fmt.Sprintf("Überfällig (%d)", len(overdue))
		}
		sections = append(sections, cognitive.Styled(cognitive.ColorRed, title)+"\n"+strings.Join(lines, "\n"))
	}

	// Today's schedule
	today := store.Today(now)
	if len(today) > 0 {
		lines := make([]string, 0, len(today))
		for _, task := range today {
			lines = append(lines, fmt.Sprintf("  %s  %s", task.DueAt.Format("15:04"), task.Title))
		}
		title := fmt.Sprintf("Today (%d)", len(today))
		if de {
			title = fmt.Sprintf("Heute (%d)", len(today))
		}
		sections = append(sections, title+"\n"+strings.Join(lines, "\n"))
	}

	// Active routines for today
	routines := store.ActiveRoutinesForDay(now)
	if len(routines) > 0 {
		lines := make([]string, 0, len(routines))
		for _, routine := range routines {
			lines = append(lines, fmt.Sprintf("  %s  %s", routine.TimeOfDay, routine.Title))
		}
		title := fmt.Sprintf("Routines (%d)", len(routines))
		if de {
			title = fmt.Sprintf("Routinen (%d)", len(routines))
		}
		sections = append(sections, title+"\n"+strings.Join(lines, "\n"))
	}

	// Unread notifications
	unread := store.UnreadNotifications()
	if len(unread) > 0 {
		lines := make([]string, 0, len(unread))
		for _, note := range unread {
			lines = append(lines, fmt.Sprintf("  %s", note.Message))
		}
		title := fmt.Sprintf("Unread (%d)", len(unread))
		if de {
			title = fmt.Sprintf("Ungelesen (%d)", len(unread))
		}
		sections = append(sections, title+"\n"+strings.Join(lines, "\n"))
	}

	// Upcoming (next 3, not today)
	upcoming := store.Upcoming(3, now)
	var futureOnly []assistant.Task
	y, m, d := now.Date()
	for _, task := range upcoming {
		ty, tm, td := task.DueAt.Date()
		if !(y == ty && m == tm && d == td) {
			futureOnly = append(futureOnly, task)
		}
	}
	if len(futureOnly) > 0 {
		lines := make([]string, 0, len(futureOnly))
		for _, task := range futureOnly {
			lines = append(lines, fmt.Sprintf("  %s  %s", task.DueAt.Format("Mon 15:04"), task.Title))
		}
		title := "Coming up"
		if de {
			title = "Später"
		}
		sections = append(sections, title+"\n"+strings.Join(lines, "\n"))
	}

	if len(sections) == 0 {
		if de {
			return cognitive.Panel(greeting, []string{"Dein Zeitplan ist frei. Keine Aufgaben, nichts überfällig."})
		}
		return cognitive.Panel(greeting, []string{"Your schedule is clear. No tasks, no overdue items."})
	}

	return cognitive.Panel(greeting, sections)
}

func renderReview(store *assistant.Store, now time.Time) string {
	de := prefersGerman(store)
	var sections []string

	// Completed today
	done := store.CompletedToday(now)
	if len(done) > 0 {
		lines := make([]string, 0, len(done))
		for _, task := range done {
			lines = append(lines, fmt.Sprintf("  %s  %s", task.CompletedAt.Format("15:04"), task.Title))
		}
		title := fmt.Sprintf("Completed today (%d)", len(done))
		if de {
			title = fmt.Sprintf("Heute erledigt (%d)", len(done))
		}
		sections = append(sections, cognitive.Styled(cognitive.ColorGreen, title)+"\n"+strings.Join(lines, "\n"))
	} else {
		if de {
			sections = append(sections, "Heute wurde noch nichts erledigt.")
		} else {
			sections = append(sections, "No tasks completed today.")
		}
	}

	// Still pending today
	today := store.Today(now)
	if len(today) > 0 {
		lines := make([]string, 0, len(today))
		for _, task := range today {
			lines = append(lines, fmt.Sprintf("  %s  %s", task.DueAt.Format("15:04"), task.Title))
		}
		title := fmt.Sprintf("Still pending (%d)", len(today))
		if de {
			title = fmt.Sprintf("Noch offen (%d)", len(today))
		}
		sections = append(sections, title+"\n"+strings.Join(lines, "\n"))
	}

	// Overdue
	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		lines := make([]string, 0, len(overdue))
		for _, task := range overdue {
			lines = append(lines, fmt.Sprintf("  %s  %s", task.DueAt.Format("15:04"), task.Title))
		}
		title := fmt.Sprintf("Overdue (%d)", len(overdue))
		if de {
			title = fmt.Sprintf("Überfällig (%d)", len(overdue))
		}
		sections = append(sections, cognitive.Styled(cognitive.ColorRed, title)+"\n"+strings.Join(lines, "\n"))
	}

	// Tomorrow preview
	tomorrow := now.Add(24 * time.Hour)
	tomorrowTasks := store.Today(tomorrow)
	tomorrowRoutines := store.ActiveRoutinesForDay(tomorrow)
	if len(tomorrowTasks) > 0 || len(tomorrowRoutines) > 0 {
		lines := make([]string, 0)
		for _, task := range tomorrowTasks {
			lines = append(lines, fmt.Sprintf("  %s  %s", task.DueAt.Format("15:04"), task.Title))
		}
		for _, routine := range tomorrowRoutines {
			if de {
				lines = append(lines, fmt.Sprintf("  %s  %s (Routine)", routine.TimeOfDay, routine.Title))
			} else {
				lines = append(lines, fmt.Sprintf("  %s  %s (routine)", routine.TimeOfDay, routine.Title))
			}
		}
		title := "Tomorrow"
		if de {
			title = "Morgen"
		}
		sections = append(sections, title+"\n"+strings.Join(lines, "\n"))
	}

	panelTitle := "Evening review"
	if de {
		panelTitle = "Abendlicher Rückblick"
	}
	return cognitive.Panel(panelTitle, sections)
}

func whatShouldIDoNow(store *assistant.Store, now time.Time) string {
	de := prefersGerman(store)
	// 1. Overdue tasks — most urgent
	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		task := overdue[0]
		ago := friendlyDuration(now.Sub(task.DueAt), de)
		if de {
			return fmt.Sprintf("Du hast eine überfällige Aufgabe: \"%s\" (seit %s überfällig, %s). Kümmere dich zuerst darum.", task.Title, ago, task.DueAt.Format("15:04"))
		}
		return fmt.Sprintf("You have an overdue task: \"%s\" (due %s ago, %s). Take care of that first.", task.Title, ago, task.DueAt.Format("15:04"))
	}

	// 2. Tasks due within the next hour
	today := store.Today(now)
	for _, task := range today {
		if task.DueAt.After(now) && task.DueAt.Before(now.Add(time.Hour)) {
			if looksLikeMeetingTask(task.Title) && meetingAnxietyNote(store) {
				if de {
					return fmt.Sprintf("Dein nächster wichtiger Punkt ist \"%s\" in %s. Nimm dir jetzt 10 ruhige Minuten für Notizen, den wichtigsten Punkt und eine erste Frage.", task.Title, friendlyDuration(task.DueAt.Sub(now), true))
				}
				return fmt.Sprintf("Your next important thing is %q in %s. Take 10 calm minutes now for notes, the main point you want to land, and one opening question.", task.Title, friendlyDuration(task.DueAt.Sub(now), false))
			}
			until := friendlyDuration(task.DueAt.Sub(now), de)
			if de {
				return fmt.Sprintf("In %s steht \"%s\" um %s an.", until, task.Title, task.DueAt.Format("15:04"))
			}
			return fmt.Sprintf("Coming up in %s: \"%s\" at %s.", until, task.Title, task.DueAt.Format("15:04"))
		}
	}

	// 3. Next task on today's schedule
	for _, task := range today {
		if task.DueAt.After(now) {
			if de {
				return fmt.Sprintf("Als Nächstes auf deinem Plan: \"%s\" um %s.", task.Title, task.DueAt.Format("15:04"))
			}
			return fmt.Sprintf("Next on your schedule: \"%s\" at %s.", task.Title, task.DueAt.Format("15:04"))
		}
	}

	// 4. Any pending tasks at all
	pending := store.PendingTasks()
	if len(pending) > 0 {
		task := pending[0]
		if de {
			return fmt.Sprintf("Gerade ist nichts fällig. Nächste Aufgabe: \"%s\" am %s.", task.Title, task.DueAt.Format("Mon 15:04"))
		}
		return fmt.Sprintf("No tasks due right now. Next task: \"%s\" on %s.", task.Title, task.DueAt.Format("Mon 15:04"))
	}

	if de {
		return "Alles frei — keine offenen Aufgaben oder Erinnerungen."
	}
	return "You're all clear — no pending tasks or reminders."
}

func nextScheduledTask(store *assistant.Store, now time.Time) (assistant.Task, bool) {
	today := store.Today(now)
	for _, task := range today {
		if task.DueAt.After(now) {
			return task, true
		}
	}
	pending := store.PendingTasks()
	if len(pending) == 0 {
		return assistant.Task{}, false
	}
	return pending[0], true
}

func looksLikeMeetingTask(title string) bool {
	lower := strings.ToLower(strings.TrimSpace(title))
	for _, nonMeeting := range []string{"dentist", "doctor", "arzt", "therapy", "therapist", "appointment", "termin beim", "zahnarzt"} {
		if strings.Contains(lower, nonMeeting) {
			return false
		}
	}
	keywords := []string{"meeting", "1:1", "1on1", "standup", "review", "sync", "call", "interview", "retro", "planung", "besprechung", "termin"}
	for _, keyword := range keywords {
		if strings.Contains(lower, keyword) {
			return true
		}
	}
	return false
}

func nextMeetingTask(store *assistant.Store, now time.Time) (assistant.Task, bool) {
	pending := store.PendingTasks()
	for _, task := range pending {
		if task.DueAt.Before(now) {
			continue
		}
		if looksLikeMeetingTask(task.Title) {
			return task, true
		}
	}
	return assistant.Task{}, false
}

func meetingAnxietyNote(store *assistant.Store) bool {
	for _, note := range profileNotes(store) {
		lower := strings.ToLower(note)
		if (strings.Contains(lower, "meeting") || strings.Contains(lower, "meetings") || strings.Contains(lower, "besprech") || strings.Contains(lower, "termin")) &&
			(strings.Contains(lower, "anxious") || strings.Contains(lower, "nervous") || strings.Contains(lower, "stress") || strings.Contains(lower, "nervös") || strings.Contains(lower, "nervoes") || strings.Contains(lower, "angst")) {
			return true
		}
	}
	return false
}

func meetingPrepReply(store *assistant.Store, now time.Time, de bool) string {
	task, ok := nextMeetingTask(store, now)
	if !ok {
		if de {
			return "Lass uns die Vorbereitung ruhig halten: nimm dir 10 Minuten für Notizen, den wichtigsten Punkt und eine erste Frage."
		}
		return "Let's keep the preparation calm: take 10 minutes for notes, the main point you want to land, and one first question."
	}
	until := friendlyDuration(task.DueAt.Sub(now), de)
	if de {
		if meetingAnxietyNote(store) {
			return fmt.Sprintf("Für \"%s\" in %s würde ich es bewusst klein halten: notiere jetzt den wichtigsten Punkt, zwei Stichworte und eine erste Frage. So gehst du ruhiger hinein.", task.Title, until)
		}
		return fmt.Sprintf("Für \"%s\" in %s: notiere jetzt den wichtigsten Punkt, zwei Stichworte und eine erste Frage. Dann bist du mit einem klaren Einstieg vorbereitet.", task.Title, until)
	}
	if meetingAnxietyNote(store) {
		return fmt.Sprintf("For %q in %s, I would keep it deliberately small: write the main point you want to land, two bullet notes, and one opening question. That should help you go in a bit calmer.", task.Title, until)
	}
	return fmt.Sprintf("For %q in %s: write the main point you want to land, two bullet notes, and one opening question. That gives you a clear starting point.", task.Title, until)
}

func isPlainGreeting(input string) bool {
	switch strings.TrimSpace(input) {
	case "hi", "hello", "hey", "hi nous", "hello nous", "hey nous", "how are you", "how are you?", "you there", "you there?":
		return true
	default:
		return false
	}
}

func friendlyDuration(d time.Duration, de bool) string {
	if d < 0 {
		d = -d
	}
	d = d.Round(time.Minute)
	days := d / (24 * time.Hour)
	d -= days * 24 * time.Hour
	hours := d / time.Hour
	d -= hours * time.Hour
	minutes := d / time.Minute

	plural := func(n int64, one, many string) string {
		if n == 1 {
			return one
		}
		return many
	}

	if days > 0 {
		if de {
			return fmt.Sprintf("%d %s", days, plural(int64(days), "Tag", "Tage"))
		}
		return fmt.Sprintf("%d %s", days, plural(int64(days), "day", "days"))
	}
	if hours > 0 {
		if minutes > 0 {
			if de {
				return fmt.Sprintf("%d Std. %d Min.", hours, minutes)
			}
			return fmt.Sprintf("%dh %dm", hours, minutes)
		}
		if de {
			return fmt.Sprintf("%d Std.", hours)
		}
		return fmt.Sprintf("%dh", hours)
	}
	if de {
		return fmt.Sprintf("%d Min.", minutes)
	}
	return fmt.Sprintf("%dm", minutes)
}

func lastAssistantLine(recent string) string {
	lines := strings.Split(strings.TrimSpace(recent), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "Assistant: ") {
			return strings.TrimPrefix(line, "Assistant: ")
		}
	}
	return strings.TrimSpace(recent)
}

func lastUserLine(recent string) string {
	lines := strings.Split(strings.TrimSpace(recent), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "User: ") {
			return strings.TrimPrefix(line, "User: ")
		}
	}
	return ""
}

func topicFromText(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	if first := strings.Index(trimmed, "\""); first >= 0 {
		if second := strings.Index(trimmed[first+1:], "\""); second >= 0 {
			quoted := strings.TrimSpace(trimmed[first+1 : first+1+second])
			if len(strings.Fields(quoted)) > 0 && len(strings.Fields(quoted)) <= 4 {
				return quoted
			}
		}
	}

	lower := strings.ToLower(trimmed)
	markers := []string{" on my ", " on the ", " my ", " the ", " bei meinem ", " bei meinem ", " bei meiner ", " bei meinen ", " mit meinem ", " mit meiner ", " mit meinen ", " an meinem ", " an meiner ", " an meinen "}
	stopWords := map[string]bool{
		"a": true, "an": true, "and": true, "because": true, "before": true,
		"but": true, "day": true, "days": true, "feels": true, "for": true,
		"from": true, "has": true, "have": true, "if": true, "is": true,
		"it": true, "of": true, "right": true, "since": true, "that": true,
		"the": true, "this": true, "today": true, "tomorrow": true, "week": true,
		"weeks": true, "with": true, "dem": true, "den": true, "der": true,
		"die": true, "das": true, "einem": true, "einer": true, "einen": true,
		"meinem": true, "meiner": true, "meinen": true,
	}
	for _, marker := range markers {
		idx := strings.Index(lower, marker)
		if idx < 0 {
			continue
		}
		rest := trimmed[idx+len(marker):]
		words := strings.Fields(rest)
		topicWords := make([]string, 0, 3)
		for _, word := range words {
			clean := strings.TrimFunc(word, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '-'
			})
			if clean == "" {
				continue
			}
			if stopWords[strings.ToLower(clean)] {
				break
			}
			topicWords = append(topicWords, clean)
			if len(topicWords) >= 3 {
				break
			}
		}
		if len(topicWords) > 0 {
			return strings.Join(topicWords, " ")
		}
	}

	for _, topic := range []string{"report", "meeting", "invoice", "design review"} {
		if strings.Contains(lower, topic) {
			return topic
		}
	}
	return ""
}

func conversationFocusTopic(recent string) string {
	if topic := topicFromText(lastUserLine(recent)); topic != "" {
		return topic
	}
	if topic := topicFromText(lastAssistantLine(recent)); topic != "" {
		return topic
	}
	lower := strings.ToLower(recent)
	for _, topic := range []string{"report", "meeting", "invoice", "design review"} {
		if strings.Contains(lower, topic) {
			return topic
		}
	}
	return ""
}

func assistantGreeting(store *assistant.Store, now time.Time, de bool) string {
	prefix := "Good morning"
	if now.Hour() >= 12 && now.Hour() < 18 {
		prefix = "Good afternoon"
	} else if now.Hour() >= 18 {
		prefix = "Good evening"
	}
	if de {
		prefix = "Hallo"
	}

	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		task := overdue[0]
		ago := friendlyDuration(now.Sub(task.DueAt), de)
		if de {
			return fmt.Sprintf("%s — ich bin da. Das Wichtigste gerade ist \"%s\". Die Aufgabe ist seit %s überfällig.", prefix, task.Title, ago)
		}
		return fmt.Sprintf("%s — I'm here. The main thing right now is %q. It's overdue by %s.", prefix, task.Title, ago)
	}

	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			return fmt.Sprintf("%s — ich bin da. Als Nächstes kommt \"%s\" um %s.", prefix, task.Title, task.DueAt.Format("15:04"))
		}
		return fmt.Sprintf("%s — I'm here. Next up is %q at %s.", prefix, task.Title, task.DueAt.Format("15:04"))
	}

	routines := store.ActiveRoutinesForDay(now)
	if len(routines) > 0 {
		routine := routines[0]
		if de {
			return fmt.Sprintf("%s — ich bin da. Dein erster Anker heute ist \"%s\" um %s.", prefix, routine.Title, routine.TimeOfDay)
		}
		return fmt.Sprintf("%s — I'm here. Your first anchor today is %q at %s.", prefix, routine.Title, routine.TimeOfDay)
	}

	if de {
		return prefix + " — ich bin da. Heute ist im Moment nichts Dringendes offen."
	}
	return prefix + " — I'm here. Nothing urgent is on your plate right now."
}

func assistantPreferenceSummary(store *assistant.Store, now time.Time, de bool) string {
	prefs := visiblePreferences(store)
	notes := profileNotes(store)
	if len(prefs) == 0 && len(notes) == 0 {
		if de {
			return "Ich habe noch keine gespeicherten Präferenzen über dich. Wenn du möchtest, kannst du mir welche geben."
		}
		return "I do not have any saved preferences about you yet. If you want, you can teach me some."
	}

	parts := make([]string, 0, min(len(prefs), 3))
	for i, pref := range prefs {
		if i >= 3 {
			break
		}
		parts = append(parts, fmt.Sprintf("%s=%s", pref.Key, pref.Value))
	}
	noteText := ""
	if len(notes) > 0 {
		kept := notes
		if len(kept) > 2 {
			kept = kept[:2]
		}
		if de {
			noteText = " Persönliche Hinweise: " + strings.Join(kept, "; ") + "."
		} else {
			noteText = " Personal notes: " + strings.Join(kept, "; ") + "."
		}
	}

	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			base := fmt.Sprintf("Ich kenne bisher diese Präferenzen: %s.", strings.Join(parts, ", "))
			if len(parts) == 0 {
				base = "Ich kenne bisher schon ein paar Dinge über dich."
			}
			return base + noteText + fmt.Sprintf(" Im Moment ist besonders relevant, dass als Nächstes \"%s\" um %s ansteht.", task.Title, task.DueAt.Format("15:04"))
		}
		base := fmt.Sprintf("So far I know these preferences: %s.", strings.Join(parts, ", "))
		if len(parts) == 0 {
			base = "So far I already know a few things about you."
		}
		return base + noteText + fmt.Sprintf(" The most relevant one right now is that %q is coming up at %s.", task.Title, task.DueAt.Format("15:04"))
	}

	if de {
		if len(parts) == 0 {
			return "Ich kenne bisher schon ein paar Dinge über dich." + noteText
		}
		return "Ich kenne bisher diese Präferenzen: " + strings.Join(parts, ", ") + "." + noteText
	}
	if len(parts) == 0 {
		return "So far I already know a few things about you." + noteText
	}
	return "So far I know these preferences: " + strings.Join(parts, ", ") + "." + noteText
}

func visiblePreferences(store *assistant.Store) []assistant.Preference {
	if store == nil {
		return nil
	}
	prefs := store.Preferences()
	out := make([]assistant.Preference, 0, len(prefs))
	for _, pref := range prefs {
		if strings.HasPrefix(pref.Key, "profile.") {
			continue
		}
		out = append(out, pref)
	}
	return out
}

func profileNotes(store *assistant.Store) []string {
	if store == nil {
		return nil
	}
	prefs := store.Preferences()
	out := make([]string, 0, len(prefs))
	for _, pref := range prefs {
		if strings.HasPrefix(pref.Key, "profile.") && strings.TrimSpace(pref.Value) != "" {
			out = append(out, pref.Value)
		}
	}
	return out
}

func rememberedProfileNote(input string) (string, string, bool) {
	trimmed := strings.TrimSpace(input)
	lower := strings.ToLower(trimmed)
	prefixes := []string{"remember that ", "remember this: ", "remember this ", "please remember that "}
	statement := ""
	for _, prefix := range prefixes {
		if strings.HasPrefix(lower, prefix) {
			statement = strings.TrimSpace(trimmed[len(prefix):])
			break
		}
	}
	if statement == "" {
		return "", "", false
	}
	statement = strings.TrimSpace(strings.TrimRight(statement, ".!?"))
	if statement == "" {
		return "", "", false
	}

	words := strings.FieldsFunc(strings.ToLower(statement), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	if len(words) == 0 {
		return "", "", false
	}
	if len(words) > 5 {
		words = words[:5]
	}
	return "profile." + strings.Join(words, "_"), statement, true
}

func isSignalWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_'
}

func containsIntentSignal(text string, signal string) bool {
	if signal == "" {
		return false
	}
	if strings.IndexFunc(signal, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}) >= 0 {
		return strings.Contains(text, signal)
	}
	for start := 0; start < len(text); {
		idx := strings.Index(text[start:], signal)
		if idx < 0 {
			return false
		}
		idx += start
		beforeOK := idx == 0 || !isSignalWordRune(rune(text[idx-1]))
		afterIdx := idx + len(signal)
		afterOK := afterIdx >= len(text) || !isSignalWordRune(rune(text[afterIdx]))
		if beforeOK && afterOK {
			return true
		}
		start = idx + len(signal)
	}
	return false
}

func assistantCheckIn(store *assistant.Store, now time.Time, de bool) string {
	done := store.CompletedToday(now)
	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		task := overdue[0]
		if de {
			return fmt.Sprintf("Kurzer Check-in: %d Aufgaben heute erledigt. Das Wichtigste jetzt ist \"%s\" — diese Aufgabe ist überfällig.", len(done), task.Title)
		}
		return fmt.Sprintf("Quick check-in: you've finished %d task(s) today. The main thing now is %q — it's overdue.", len(done), task.Title)
	}
	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			return fmt.Sprintf("Kurzer Check-in: %d Aufgaben heute erledigt. Als Nächstes kommt \"%s\" um %s.", len(done), task.Title, task.DueAt.Format("15:04"))
		}
		return fmt.Sprintf("Quick check-in: you've finished %d task(s) today. Next up is %q at %s.", len(done), task.Title, task.DueAt.Format("15:04"))
	}
	if de {
		return fmt.Sprintf("Kurzer Check-in: %d Aufgaben heute erledigt und gerade nichts Dringendes offen.", len(done))
	}
	return fmt.Sprintf("Quick check-in: you've finished %d task(s) today and nothing urgent is open right now.", len(done))
}

func assistantPlanReply(store *assistant.Store, now time.Time, de bool) string {
	overdue := store.Overdue(now)
	routines := store.ActiveRoutinesForDay(now)
	if de {
		var parts []string
		if len(overdue) > 0 {
			parts = append(parts, fmt.Sprintf("zuerst \"%s\" erledigen", overdue[0].Title))
		} else if task, ok := nextScheduledTask(store, now); ok {
			parts = append(parts, fmt.Sprintf("mit \"%s\" um %s anfangen", task.Title, task.DueAt.Format("15:04")))
		}
		if focus := preferenceValue(store, "focus"); focus != "" {
			parts = append(parts, fmt.Sprintf("deine Fokus-Präferenz beachten: %s", focus))
		}
		if len(routines) > 0 {
			parts = append(parts, fmt.Sprintf("\"%s\" um %s als Anker nutzen", routines[0].Title, routines[0].TimeOfDay))
		}
		if len(parts) == 0 {
			return "Lass uns es einfach halten: Heute ist nichts Dringendes offen. Wähle eine wichtige Sache und arbeite 25 Minuten fokussiert daran."
		}
		return "Lass uns deinen Tag einfach halten: " + strings.Join(parts, "; ") + "."
	}

	var parts []string
	if len(overdue) > 0 {
		parts = append(parts, fmt.Sprintf("first handle \"%s\"", overdue[0].Title))
	} else if task, ok := nextScheduledTask(store, now); ok {
		parts = append(parts, fmt.Sprintf("start with \"%s\" at %s", task.Title, task.DueAt.Format("15:04")))
	}
	if focus := preferenceValue(store, "focus"); focus != "" {
		parts = append(parts, fmt.Sprintf("keep your focus preference in mind: %s", focus))
	}
	if len(routines) > 0 {
		parts = append(parts, fmt.Sprintf("use \"%s\" at %s as your anchor", routines[0].Title, routines[0].TimeOfDay))
	}
	if len(parts) == 0 {
		return "Let's keep it simple: nothing urgent is open today. Pick one meaningful thing and give it 25 focused minutes."
	}
	return "Let's keep your day simple: " + strings.Join(parts, "; ") + "."
}

func assistantFocusReply(store *assistant.Store, now time.Time, de bool) string {
	focus := preferenceValue(store, "focus")
	if task, ok := nextMeetingTask(store, now); ok && task.DueAt.Before(now.Add(4*time.Hour)) && meetingAnxietyNote(store) {
		if de {
			return fmt.Sprintf("Lass es uns für \"%s\" leicht machen: schreibe jetzt nur den wichtigsten Punkt, zwei Stichworte und eine erste Frage auf. Mehr musst du gerade nicht vorbereiten.", task.Title)
		}
		return fmt.Sprintf("Let's make %q feel lighter: write down the main point you want to land, two short notes, and one opening question. You do not need to prepare more than that right now.", task.Title)
	}
	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		if de {
			if focus != "" {
				return fmt.Sprintf("Lass uns es verkleinern: zuerst nur \"%s\". Danach kannst du dich an deine Präferenz halten: %s.", overdue[0].Title, focus)
			}
			return fmt.Sprintf("Lass uns es verkleinern: zuerst nur \"%s\". Mehr musst du gerade nicht lösen.", overdue[0].Title)
		}
		if focus != "" {
			return fmt.Sprintf("Let's make it smaller: do %q first. After that, return to your preference: %s.", overdue[0].Title, focus)
		}
		return fmt.Sprintf("Let's make it smaller: do %q first. You do not need to solve everything at once.", overdue[0].Title)
	}
	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			if focus != "" {
				return fmt.Sprintf("Ein ruhiger Fokusplan: bis %s an \"%s\" arbeiten. Denk dabei an deine Präferenz: %s.", task.DueAt.Format("15:04"), task.Title, focus)
			}
			return fmt.Sprintf("Ein ruhiger Fokusplan: bis %s an \"%s\" arbeiten. Nur dieser eine nächste Schritt zählt gerade.", task.DueAt.Format("15:04"), task.Title)
		}
		if focus != "" {
			return fmt.Sprintf("A calmer focus plan: work on %q until %s. Keep your preference in mind: %s.", task.Title, task.DueAt.Format("15:04"), focus)
		}
		return fmt.Sprintf("A calmer focus plan: work on %q until %s. Only the next step matters right now.", task.Title, task.DueAt.Format("15:04"))
	}
	if de {
		return "Gerade ist nichts Dringendes offen. Nimm dir eine Aufgabe, stelle einen kurzen Fokusblock ein und beginne klein."
	}
	return "Nothing urgent is open right now. Pick one task, set a short focus block, and start small."
}

func assistantProcrastinationReply(input string, recent string, de bool) string {
	topic := topicFromText(input)
	if topic == "" {
		topic = conversationFocusTopic(recent)
	}
	if topic == "" {
		if de {
			return "Das klingt nicht nach Faulheit. Meist bedeutet Prokrastination, dass die Aufgabe zu groß, zu unklar oder emotional aufgeladen wirkt. Lass uns den Einstieg kleiner und sicherer machen."
		}
		return "That does not sound like laziness. Procrastination usually means the task feels too big, too vague, or too loaded. Let's make the entry point smaller and safer."
	}
	if de {
		return fmt.Sprintf("Es klingt so, als hätte %q inzwischen zu viel Druck bekommen. Das ist meistens kein Faulheitsproblem, sondern ein Zeichen dafür, dass der Einstieg zu groß oder zu unklar wirkt. Lass uns %q auf einen winzigen Anfang reduzieren: öffne es und schreibe nur drei grobe Stichpunkte.", topic, topic)
	}
	return fmt.Sprintf("It sounds like %q has picked up too much pressure. That usually is not laziness; it means the starting point feels too big or too unclear. Let's shrink %q to a tiny entry point: open it and write just three rough bullet points.", topic, topic)
}

func assistantReflectionReply(recent string, de bool) string {
	topic := conversationFocusTopic(recent)
	if topic == "" {
		if de {
			return "Meine Vermutung: Dein Kopf schützt dich gerade vor etwas, das schwer, unklar oder zu bewertbar wirkt. Das heißt nicht, dass mit dir etwas nicht stimmt — die Aufgabe braucht nur einen kleineren Einstiegspunkt."
		}
		return "My guess is that your brain is protecting you from something that feels heavy, unclear, or easy to judge yourself over. That does not mean something is wrong with you — it usually means the task needs a smaller entry point."
	}
	if de {
		return fmt.Sprintf("Meine Vermutung: %q ist inzwischen zu einer Druck-Aufgabe geworden. Wenn etwas wichtig wirkt, aber noch keinen klaren Einstieg hat, weichen wir oft aus, um Anspannung oder Selbstkritik kurz zu vermeiden. Das heißt nicht, dass du faul bist — %q braucht nur einen kleineren, sichereren Anfang.", topic, topic)
	}
	return fmt.Sprintf("My guess is that %q has turned into a pressure task. When something feels important but still vague, people often avoid it for a while to get relief from tension or self-judgment. That does not mean you're lazy — %q just needs a smaller, safer starting point.", topic, topic)
}

func topicSmallStep(topic string, de bool) string {
	lower := strings.ToLower(strings.TrimSpace(topic))
	if lower == "" {
		if de {
			return "wähle eine Sache und gib ihr jetzt 10 ruhige Minuten"
		}
		return "pick one thing and give it 10 calm minutes right now"
	}
	if strings.Contains(lower, "meeting") {
		if de {
			return fmt.Sprintf("bereite %q jetzt 10 Minuten lang ruhig vor", topic)
		}
		return fmt.Sprintf("prepare for %q for just 10 calm minutes now", topic)
	}
	if strings.Contains(lower, "invoice") || strings.Contains(lower, "rechnung") {
		if de {
			return fmt.Sprintf("öffne %q jetzt und bringe sie 10 Minuten lang voran", topic)
		}
		return fmt.Sprintf("open %q now and move it forward for just 10 minutes", topic)
	}
	if de {
		return fmt.Sprintf("öffne %q jetzt und arbeite nur 10 Minuten daran", topic)
	}
	return fmt.Sprintf("open %q now and work on it for just 10 minutes", topic)
}

func isGermanAffirmation(lower string) bool {
	switch lower {
	case "das klingt richtig", "das klingt gut", "genau", "macht sinn", "ja das passt", "okay", "ok", "ja":
		return true
	default:
		return false
	}
}

func assistantSmallStepReply(store *assistant.Store, recent string, now time.Time, de bool) string {
	if topic := conversationFocusTopic(recent); topic != "" {
		if de {
			return "Gut. Der nächste kleine Schritt ist ganz konkret: " + topicSmallStep(topic, true) + "."
		}
		return "Good. The next small step is concrete: " + topicSmallStep(topic, false) + "."
	}
	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		if de {
			return fmt.Sprintf("Gut. Dein nächster kleiner Schritt ist ganz konkret: öffne \"%s\" jetzt und arbeite nur 10 Minuten daran.", overdue[0].Title)
		}
		return fmt.Sprintf("Good. Your next small step is concrete: open %q now and work on it for just 10 minutes.", overdue[0].Title)
	}
	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			return fmt.Sprintf("Gut. Der nächste kleine Schritt ist: bereite \"%s\" jetzt vor, damit du bis %s einen klaren Einstieg hast.", task.Title, task.DueAt.Format("15:04"))
		}
		return fmt.Sprintf("Good. The next small step is: set up %q now so you have a clean starting point before %s.", task.Title, task.DueAt.Format("15:04"))
	}
	if de {
		return "Gut. Der nächste kleine Schritt ist: wähle eine Sache und gib ihr jetzt 10 ruhige Minuten."
	}
	return "Good. The next small step is: pick one thing and give it 10 calm minutes right now."
}

func buildAssistantContext(store *assistant.Store, input string, recent string, now time.Time) string {
	if store == nil || !shouldInjectAssistantContext(input) {
		return ""
	}

	var lines []string
	lines = append(lines, "Current time: "+now.Format("2006-01-02 15:04 Mon"))
	if recent != "" {
		lines = append(lines, "Recent conversation: "+recent)
	}
	prefs := visiblePreferences(store)
	if len(prefs) > 0 {
		prefLines := make([]string, 0, min(len(prefs), 5))
		for i, pref := range prefs {
			if i >= 5 {
				break
			}
			prefLines = append(prefLines, fmt.Sprintf("%s=%s", pref.Key, pref.Value))
		}
		lines = append(lines, "Preferences: "+strings.Join(prefLines, ", "))
	}

	notes := profileNotes(store)
	if len(notes) > 0 {
		noteLines := make([]string, 0, min(len(notes), 3))
		for i, note := range notes {
			if i >= 3 {
				break
			}
			noteLines = append(noteLines, note)
		}
		lines = append(lines, "Personal notes: "+strings.Join(noteLines, " | "))
	}

	overdue := store.Overdue(now)
	if len(overdue) > 0 {
		overdueLines := make([]string, 0, min(len(overdue), 3))
		for i, task := range overdue {
			if i >= 3 {
				break
			}
			overdueLines = append(overdueLines, fmt.Sprintf("%s %s", task.DueAt.Format("2006-01-02 15:04"), task.Title))
		}
		lines = append(lines, "Overdue: "+strings.Join(overdueLines, " | "))
	}

	unread := store.UnreadNotifications()
	if len(unread) > 0 {
		noteLines := make([]string, 0, min(len(unread), 3))
		for i, note := range unread {
			if i >= 3 {
				break
			}
			noteLines = append(noteLines, note.Message)
		}
		lines = append(lines, "Unread reminders: "+strings.Join(noteLines, " | "))
	}

	upcoming := store.Upcoming(3, now)
	if len(upcoming) > 0 {
		reminderLines := make([]string, 0, len(upcoming))
		for _, task := range upcoming {
			reminderLines = append(reminderLines, fmt.Sprintf("%s %s", task.DueAt.Format("2006-01-02 15:04"), task.Title))
		}
		lines = append(lines, "Active reminders/tasks: "+strings.Join(reminderLines, " | "))
	}

	today := store.Today(now)
	if len(today) > 0 {
		todayLines := make([]string, 0, min(len(today), 3))
		for i, task := range today {
			if i >= 3 {
				break
			}
			todayLines = append(todayLines, fmt.Sprintf("%s %s", task.DueAt.Format("15:04"), task.Title))
		}
		lines = append(lines, "Today: "+strings.Join(todayLines, " | "))
	}

	doneToday := store.CompletedToday(now)
	if len(doneToday) > 0 {
		doneLines := make([]string, 0, min(len(doneToday), 3))
		for i, task := range doneToday {
			if i >= 3 {
				break
			}
			doneLines = append(doneLines, task.Title)
		}
		lines = append(lines, "Completed today: "+strings.Join(doneLines, " | "))
	}

	routines := store.ActiveRoutinesForDay(now)
	if len(routines) > 0 {
		routineLines := make([]string, 0, min(len(routines), 3))
		for i, routine := range routines {
			if i >= 3 {
				break
			}
			routineLines = append(routineLines, fmt.Sprintf("%s %s (%s)", routine.TimeOfDay, routine.Title, routine.Schedule))
		}
		lines = append(lines, "Active routines today: "+strings.Join(routineLines, " | "))
	}

	if len(lines) == 0 {
		return ""
	}

	return "[Assistant Memory]\nUse this for personal-assistant questions. Sound warm, natural, and practical. Use memory selectively instead of dumping it. Respect explicit preferences such as language. If reminders or tasks are listed below, do not say there are none. If the user sounds overwhelmed, reduce things to the next small step.\n" + strings.Join(lines, "\n")
}

func answerAssistantQuery(store *assistant.Store, input string, recent string, now time.Time) (string, bool) {
	if store == nil {
		return "", false
	}

	lower := strings.ToLower(strings.TrimSpace(input))

	// Code-intent signals — if any are present, NEVER hijack into assistant answers.
	// This prevents "what does renderBriefing include" from triggering the briefing flow.
	codeSignals := []string{
		"function", "func ", "method", "struct", "type ", "variable", "constant",
		"implement", "code", "source", "definition", "defined", "return", "parameter",
		"argument", "signature", "package", "import", "module",
		"file ", "file?", ".go", ".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml",
		"read file", "show file", "open file", "read the", "show me the", "show the",
		"grep", "search for", "find in", "look for", "where is",
		"class", "interface", "enum", "const ", "var ",
		"compile", "build", "test", "debug", "fix", "bug", "error",
		"repo", "repository", "codebase", "project structure",
		"directory", "folder", "path",
		"explain", "how does", "what does", "what is the",
		"semantic", "ranking", "working memory", "decay",
	}
	for _, sig := range codeSignals {
		if containsIntentSignal(lower, sig) {
			return "", false
		}
	}

	lang := preferenceValue(store, "language")
	de := strings.HasPrefix(strings.ToLower(lang), "de")

	if key, value, ok := rememberedProfileNote(input); ok {
		if err := store.SetPreference(key, value); err == nil {
			if de {
				return fmt.Sprintf("Verstanden. Ich merke mir: %s.", value), true
			}
			return fmt.Sprintf("Got it. I'll remember that %s.", value), true
		}
	}

	if isPlainGreeting(lower) {
		return assistantGreeting(store, now, de), true
	}

	if strings.Contains(lower, "thank you") || strings.HasPrefix(lower, "thanks") || strings.Contains(lower, "danke") {
		if task, ok := nextScheduledTask(store, now); ok {
			if de {
				return fmt.Sprintf("Gern. Als Nächstes steht \"%s\" um %s an.", task.Title, task.DueAt.Format("15:04")), true
			}
			return fmt.Sprintf("Anytime. Next up is %q at %s.", task.Title, task.DueAt.Format("15:04")), true
		}
		if de {
			return "Gern. Ich bin hier, wenn du den nächsten Schritt sortieren willst.", true
		}
		return "Anytime. I'm here if you want to sort out the next step.", true
	}

	if strings.Contains(lower, "what do you know about my preferences") || strings.Contains(lower, "what do you know about me") {
		return assistantPreferenceSummary(store, now, de), true
	}
	if strings.Contains(lower, "was weißt du über mich") || strings.Contains(lower, "was weisst du über mich") || strings.Contains(lower, "was weißt du über meine präferenzen") || strings.Contains(lower, "was weisst du ueber mich") {
		return assistantPreferenceSummary(store, now, true), true
	}

	if recent != "" {
		if lower == "tell me more" || lower == "go on" || lower == "and then" || lower == "what do you mean" || lower == "why" {
			last := lastAssistantLine(recent)
			if de {
				return "Klar. Ich knüpfe an unseren letzten Punkt an: " + last + "\n\nWenn du willst, vertiefe ich den wichtigsten Punkt oder formuliere direkt den nächsten Schritt.", true
			}
			return "Sure. Picking up from our last point: " + last + "\n\nIf you want, I can either deepen the main point or turn it into the next concrete step.", true
		}
		if lower == "yes" || lower == "yeah" || lower == "ok" || lower == "okay" {
			if de {
				return assistantSmallStepReply(store, recent, now, de), true
			}
			return assistantSmallStepReply(store, recent, now, de), true
		}
		if lower == "that sounds right" || lower == "exactly" || lower == "makes sense" || lower == "yes that fits" {
			return assistantSmallStepReply(store, recent, now, de), true
		}
		if isGermanAffirmation(lower) {
			return assistantSmallStepReply(store, recent, now, true), true
		}
	}

	if strings.Contains(lower, "help me plan my day") || strings.Contains(lower, "plan my day") || strings.Contains(lower, "organize my day") || strings.Contains(lower, "prioritize my day") {
		return assistantPlanReply(store, now, de), true
	}
	if strings.Contains(lower, "prepare for my meeting") || strings.Contains(lower, "prepare for the meeting") || strings.Contains(lower, "help me prepare for my meeting") || strings.Contains(lower, "wie bereite ich mich auf mein meeting vor") || strings.Contains(lower, "hilf mir bei meinem meeting") || strings.Contains(lower, "bereite mich auf mein meeting vor") {
		return meetingPrepReply(store, now, de), true
	}

	if strings.Contains(lower, "procrastinating") {
		return assistantProcrastinationReply(input, recent, de), true
	}
	if strings.Contains(lower, "ich prokrastiniere") || strings.Contains(lower, "ich schiebe") || strings.Contains(lower, "ich vermeide") {
		return assistantProcrastinationReply(input, recent, true), true
	}

	if strings.Contains(lower, "why do you think that keeps happening") || strings.Contains(lower, "why does that keep happening") || strings.Contains(lower, "why is that keeping happening") {
		return assistantReflectionReply(recent, de), true
	}
	if strings.Contains(lower, "warum passiert das immer wieder") || strings.Contains(lower, "warum passiert das dauernd") || strings.Contains(lower, "warum passiert mir das immer wieder") {
		return assistantReflectionReply(recent, true), true
	}

	if strings.Contains(lower, "help me focus") || strings.Contains(lower, "i feel overwhelmed") || strings.Contains(lower, "i'm overwhelmed") || strings.Contains(lower, "i am overwhelmed") || strings.Contains(lower, "i feel stressed") || strings.Contains(lower, "i'm stressed") {
		return assistantFocusReply(store, now, de), true
	}

	if strings.Contains(lower, "how am i doing") || strings.Contains(lower, "check in") || strings.Contains(lower, "how is my day going") {
		return assistantCheckIn(store, now, de), true
	}

	if strings.Contains(lower, "answer in my preferred language") || strings.Contains(lower, "mention my current reminders") {
		upcoming := store.Upcoming(3, now)
		if de {
			if len(upcoming) == 0 {
				return "Deine bevorzugte Sprache ist Deutsch. Aktuell hast du keine aktiven Erinnerungen.", true
			}
			parts := make([]string, 0, len(upcoming))
			for _, task := range upcoming {
				parts = append(parts, fmt.Sprintf("%s %s", task.DueAt.Format("15:04"), task.Title))
			}
			return "Deine bevorzugte Sprache ist Deutsch. Aktuelle Erinnerungen: " + strings.Join(parts, "; ") + ".", true
		}
		if len(upcoming) == 0 {
			return "Your preferred language is " + fallbackPref(lang, "English") + ". You currently have no active reminders.", true
		}
		parts := make([]string, 0, len(upcoming))
		for _, task := range upcoming {
			parts = append(parts, fmt.Sprintf("%s %s", task.DueAt.Format("15:04"), task.Title))
		}
		return "Your preferred language is " + fallbackPref(lang, "English") + ". Current reminders: " + strings.Join(parts, "; ") + ".", true
	}

	if strings.Contains(lower, "what language do i prefer") || strings.Contains(lower, "preferred language") {
		if lang == "" {
			if de {
				return "Ich habe noch keine bevorzugte Sprache gespeichert.", true
			}
			return "I do not have a saved language preference yet.", true
		}
		if de {
			return fmt.Sprintf("Deine bevorzugte Sprache ist %s.", lang), true
		}
		return fmt.Sprintf("Your preferred language is %s.", lang), true
	}

	if strings.Contains(lower, "what reminder do i currently have") || strings.Contains(lower, "current reminder") || strings.Contains(lower, "current reminders") || strings.Contains(lower, "what reminders do i have") {
		upcoming := store.Upcoming(3, now)
		if len(upcoming) == 0 {
			if de {
				return "Du hast aktuell keine aktiven Erinnerungen.", true
			}
			return "You currently have no active reminders.", true
		}
		parts := make([]string, 0, len(upcoming))
		for _, task := range upcoming {
			parts = append(parts, fmt.Sprintf("%s %s", task.DueAt.Format("15:04"), task.Title))
		}
		if de {
			return "Deine aktuellen Erinnerungen sind: " + strings.Join(parts, "; ") + ".", true
		}
		return "Your current reminders are: " + strings.Join(parts, "; ") + ".", true
	}

	if strings.Contains(lower, "what matters today") || strings.Contains(lower, "what do i have today") || strings.Contains(lower, "what's on my plate") {
		today := store.Today(now)
		if len(today) == 0 {
			if de {
				return "Heute ist noch nichts geplant.", true
			}
			return "You do not have anything scheduled for today yet.", true
		}
		parts := make([]string, 0, len(today))
		for _, task := range today {
			parts = append(parts, fmt.Sprintf("%s %s", task.DueAt.Format("15:04"), task.Title))
		}
		if de {
			return "Heute wichtig: " + strings.Join(parts, "; ") + ".", true
		}
		return "What matters today: " + strings.Join(parts, "; ") + ".", true
	}

	// "what should I do" / "what now" / "what's next" — deterministic priority answer
	if strings.Contains(lower, "what should i do") || strings.Contains(lower, "what now") ||
		strings.Contains(lower, "what's next") || strings.Contains(lower, "what do i do next") ||
		strings.Contains(lower, "what is next") {
		answer := whatShouldIDoNow(store, now)
		if de {
			// Translate key phrases for German
			answer = strings.ReplaceAll(answer, "You have an overdue task:", "Du hast eine überfällige Aufgabe:")
			answer = strings.ReplaceAll(answer, "Take care of that first.", "Kümmere dich zuerst darum.")
			answer = strings.ReplaceAll(answer, "Coming up in", "In Kürze in")
			answer = strings.ReplaceAll(answer, "Next on your schedule:", "Als Nächstes auf deinem Plan:")
			answer = strings.ReplaceAll(answer, "No tasks due right now. Next task:", "Gerade nichts fällig. Nächste Aufgabe:")
			answer = strings.ReplaceAll(answer, "You're all clear — no pending tasks or reminders.", "Alles erledigt — keine offenen Aufgaben.")
		}
		return answer, true
	}

	// "morning briefing" / "good morning" — deterministic briefing
	if strings.Contains(lower, "morning briefing") || strings.Contains(lower, "good morning") ||
		strings.Contains(lower, "briefing") || strings.Contains(lower, "brief me") {
		overdue := store.Overdue(now)
		today := store.Today(now)
		routines := store.ActiveRoutinesForDay(now)

		var parts []string
		if len(overdue) > 0 {
			names := make([]string, 0, len(overdue))
			for _, t := range overdue {
				names = append(names, t.Title)
			}
			parts = append(parts, fmt.Sprintf("%d overdue: %s", len(overdue), strings.Join(names, ", ")))
		}
		if len(today) > 0 {
			names := make([]string, 0, len(today))
			for _, t := range today {
				names = append(names, fmt.Sprintf("%s %s", t.DueAt.Format("15:04"), t.Title))
			}
			parts = append(parts, fmt.Sprintf("Today: %s", strings.Join(names, "; ")))
		}
		if len(routines) > 0 {
			names := make([]string, 0, len(routines))
			for _, r := range routines {
				names = append(names, fmt.Sprintf("%s %s", r.TimeOfDay, r.Title))
			}
			parts = append(parts, fmt.Sprintf("Routines: %s", strings.Join(names, "; ")))
		}
		if len(parts) == 0 {
			if de {
				return "Guten Morgen. Dein Zeitplan ist frei — keine Aufgaben, keine Erinnerungen.", true
			}
			return "Good morning. Your schedule is clear — no tasks, no reminders.", true
		}
		prefix := "Good morning. "
		if de {
			prefix = "Guten Morgen. "
		}
		return prefix + strings.Join(parts, ". ") + ".", true
	}

	// "evening review" / "how did today go" — deterministic review
	if strings.Contains(lower, "evening review") || strings.Contains(lower, "how did today go") ||
		strings.Contains(lower, "daily review") || strings.Contains(lower, "end of day") {
		done := store.CompletedToday(now)
		overdue := store.Overdue(now)
		pending := store.Today(now)

		var parts []string
		if len(done) > 0 {
			names := make([]string, 0, len(done))
			for _, t := range done {
				names = append(names, t.Title)
			}
			parts = append(parts, fmt.Sprintf("Completed %d: %s", len(done), strings.Join(names, ", ")))
		} else {
			parts = append(parts, "No tasks completed today")
		}
		if len(overdue) > 0 {
			parts = append(parts, fmt.Sprintf("%d overdue", len(overdue)))
		}
		if len(pending) > 0 {
			parts = append(parts, fmt.Sprintf("%d still pending", len(pending)))
		}
		return strings.Join(parts, ". ") + ".", true
	}

	// "what's overdue" / "overdue tasks"
	if strings.Contains(lower, "overdue") {
		overdue := store.Overdue(now)
		if len(overdue) == 0 {
			if de {
				return "Keine überfälligen Aufgaben.", true
			}
			return "No overdue tasks.", true
		}
		parts := make([]string, 0, len(overdue))
		for _, task := range overdue {
			ago := friendlyDuration(now.Sub(task.DueAt), de)
			if de {
				parts = append(parts, fmt.Sprintf("%s (%s überfällig)", task.Title, ago))
			} else {
				parts = append(parts, fmt.Sprintf("%s (%s overdue)", task.Title, ago))
			}
		}
		if de {
			return fmt.Sprintf("%d überfällige Aufgaben: %s.", len(overdue), strings.Join(parts, "; ")), true
		}
		return fmt.Sprintf("%d overdue: %s.", len(overdue), strings.Join(parts, "; ")), true
	}

	return "", false
}

func shouldInjectAssistantContext(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	if lower == "" {
		return false
	}

	// Code signals — if ANY are present, do not inject assistant context.
	// Must check these FIRST to prevent false positives.
	codeSignals := []string{
		"function", "func ", "method", "struct", "type ", "variable", "constant",
		"implement", "code", "source", "definition", "defined", "return", "parameter",
		"file ", "file?", ".go", ".py", ".js", ".ts", ".md", ".json",
		"read file", "show file", "open file", "read the", "show me the", "show the",
		"grep", "search for", "find in", "look for", "where is",
		"class", "interface", "enum", "const ", "var ",
		"compile", "build", "test", "debug", "fix", "bug", "error",
		"repo", "repository", "codebase", "project structure",
		"directory", "folder", "path",
		"explain", "how does", "what does", "what is the",
		"semantic", "ranking", "working memory", "decay",
		"edit", "write ", "grep",
	}
	for _, sig := range codeSignals {
		if containsIntentSignal(lower, sig) {
			return false
		}
	}

	assistantHints := []string{
		"today", "reminder", "remind", "task", "tasks", "routine", "routines", "schedule", "calendar",
		"appointment", "prefer", "preference", "language", "what matters", "my day", "plan my", "inbox",
		"what do i", "do i have", "personal", "assistant", "briefing", "brief me", "good morning",
		"what should i do", "what now", "what's next", "overdue", "evening review", "daily review",
		"end of day", "how did today go", "what's on my plate",
	}
	for _, hint := range assistantHints {
		if strings.Contains(lower, hint) {
			return true
		}
	}

	return true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func preferenceValue(store *assistant.Store, key string) string {
	for _, pref := range store.Preferences() {
		if strings.EqualFold(pref.Key, key) {
			return pref.Value
		}
	}
	return ""
}

func fallbackPref(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func parseReminderInput(input string, now time.Time) (time.Time, string, string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return time.Time{}, "", "", fmt.Errorf("usage: /remind <when> <task>")
	}

	parts := strings.Fields(input)
	if len(parts) < 2 {
		return time.Time{}, "", "", fmt.Errorf("reminder needs a time and task description")
	}

	recurrence := ""
	if parts[0] == "daily" {
		if len(parts) < 3 {
			return time.Time{}, "", "", fmt.Errorf("daily reminders use: /remind daily HH:MM <task>")
		}
		tm, err := time.Parse("15:04", parts[1])
		if err != nil {
			return time.Time{}, "", "", fmt.Errorf("invalid daily time: %w", err)
		}
		due := time.Date(now.Year(), now.Month(), now.Day(), tm.Hour(), tm.Minute(), 0, 0, now.Location())
		if !due.After(now) {
			due = due.Add(24 * time.Hour)
		}
		return due, "daily", strings.Join(parts[2:], " "), nil
	}

	if parts[0] == "in" {
		if len(parts) < 3 {
			return time.Time{}, "", "", fmt.Errorf("relative reminders use: /remind in 2h <task>")
		}
		dur, err := time.ParseDuration(parts[1])
		if err != nil {
			return time.Time{}, "", "", fmt.Errorf("invalid duration: %w", err)
		}
		return now.Add(dur), recurrence, strings.Join(parts[2:], " "), nil
	}

	if parts[0] == "today" || parts[0] == "tomorrow" {
		if len(parts) < 3 {
			return time.Time{}, "", "", fmt.Errorf("use: /remind %s HH:MM <task>", parts[0])
		}
		tm, err := time.Parse("15:04", parts[1])
		if err != nil {
			return time.Time{}, "", "", fmt.Errorf("invalid time: %w", err)
		}
		due := time.Date(now.Year(), now.Month(), now.Day(), tm.Hour(), tm.Minute(), 0, 0, now.Location())
		if parts[0] == "tomorrow" {
			due = due.Add(24 * time.Hour)
		}
		if parts[0] == "today" && !due.After(now) {
			return time.Time{}, "", "", fmt.Errorf("today reminder time is already in the past")
		}
		return due, recurrence, strings.Join(parts[2:], " "), nil
	}

	if len(parts) >= 3 {
		due, err := time.ParseInLocation("2006-01-02 15:04", parts[0]+" "+parts[1], now.Location())
		if err == nil {
			return due, recurrence, strings.Join(parts[2:], " "), nil
		}
	}

	return time.Time{}, "", "", fmt.Errorf("unsupported reminder format")
}

func parseRoutineInput(input string) (string, string, string, error) {
	parts := strings.Fields(strings.TrimSpace(input))
	if len(parts) < 3 {
		return "", "", "", fmt.Errorf("usage: /routine <daily|weekdays> <HH:MM> <task>")
	}
	schedule := strings.ToLower(parts[0])
	if schedule != "daily" && schedule != "weekdays" {
		return "", "", "", fmt.Errorf("routine schedule must be daily or weekdays")
	}
	if _, err := time.Parse("15:04", parts[1]); err != nil {
		return "", "", "", fmt.Errorf("invalid routine time: %w", err)
	}
	return schedule, parts[1], strings.Join(parts[2:], " "), nil
}

func renderToolCatalog(toolReg *tools.Registry) string {
	toolsByName := make(map[string]tools.Tool)
	for _, tool := range toolReg.List() {
		toolsByName[tool.Name] = tool
	}

	var b strings.Builder
	for _, category := range cognitive.AllCategories {
		names := cognitive.CategoryNames(category, toolReg.List())
		sort.Strings(names)
		if len(names) == 0 {
			continue
		}
		lines := make([]string, 0, len(names))
		for _, name := range names {
			tool := toolsByName[name]
			lines = append(lines, fmt.Sprintf("%s%-14s%s %s", cognitive.ColorCyan, name, cognitive.ColorReset, tool.Description))
		}
		b.WriteString(cognitive.Panel(strings.Title(string(category)), lines))
	}
	return b.String()
}

func renderProjectView(project *cognitive.ProjectInfo) string {
	lines := []string{
		fmt.Sprintf("Name      %s", project.Name),
		fmt.Sprintf("Language  %s", project.Language),
		fmt.Sprintf("Files     %d", project.FileCount),
	}
	if len(project.KeyFiles) > 0 {
		keys := append([]string(nil), project.KeyFiles...)
		sort.Strings(keys)
		if len(keys) > 4 {
			keys = keys[:4]
		}
		lines = append(lines, fmt.Sprintf("Key files %s", strings.Join(keys, ", ")))
	}

	var b strings.Builder
	b.WriteString(cognitive.Panel("Project", lines))
	if strings.TrimSpace(project.Tree) != "" {
		b.WriteString(cognitive.Section("Structure"))
		for _, line := range strings.Split(strings.TrimSuffix(project.Tree, "\n"), "\n") {
			b.WriteString("  ")
			b.WriteString(line)
			b.WriteString("\n")
		}
	}
	return b.String()
}

// scoreInteractionQuality rates an interaction for training data collection.
// Higher quality = more useful for fine-tuning.
func scoreInteractionQuality(answer string, duration time.Duration, board *blackboard.Blackboard) float64 {
	quality := 0.5 // base score

	// Bonus: fast response (under 10 seconds) = likely straightforward success
	if duration < 10*time.Second {
		quality += 0.15
	}

	// Bonus: substantive answer (not empty, not an error)
	if len(answer) > 50 {
		quality += 0.1
	}

	// Penalty: answer looks like an error or failure
	lower := strings.ToLower(answer)
	if strings.Contains(lower, "error") || strings.Contains(lower, "failed") ||
		strings.Contains(lower, "i can't") || strings.Contains(lower, "unable to") {
		quality -= 0.2
	}

	// Penalty: reached max iterations
	if strings.Contains(lower, "maximum tool iterations") {
		quality -= 0.3
	}

	// Bonus: no reflector warnings during this interaction
	if _, hasReflection := board.Get("reflection"); !hasReflection {
		quality += 0.1
	}

	// Bonus: used tools successfully (actions recorded)
	actions := board.RecentActions(10)
	successCount := 0
	for _, a := range actions {
		if a.Success {
			successCount++
		}
	}
	if successCount > 0 && successCount <= 4 {
		quality += 0.15 // efficient multi-step
	}

	// Clamp to 0.0 - 1.0
	if quality < 0.0 {
		quality = 0.0
	}
	if quality > 1.0 {
		quality = 1.0
	}

	return quality
}

func waitForAnswerStr(board *blackboard.Blackboard) string {
	deadline := time.After(300 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			fmt.Println("  (timeout waiting for response)")
			return ""
		case <-ticker.C:
			if answer, ok := board.Get("last_answer"); ok {
				board.Delete("last_answer")
				if s, ok := answer.(string); ok {
					return s
				}
				return fmt.Sprintf("%v", answer)
			}
		}
	}
}

func defaultMemoryPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".nous"
	}
	return filepath.Join(home, ".nous")
}

