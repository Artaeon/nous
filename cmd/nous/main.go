package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/cognitive"
	"github.com/artaeon/nous/internal/compress"
	"github.com/artaeon/nous/internal/index"
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/sentinel"
	"github.com/artaeon/nous/internal/server"
	"github.com/artaeon/nous/internal/tools"
	"github.com/artaeon/nous/internal/training"
)

const banner = `
                ╔═══════════════════════════════════╗
                ║             N O U S               ║
                ║   Native Orchestration of         ║
                ║       Unified Streams             ║
                ╚═══════════════════════════════════╝
`

const version = "0.6.0"

func main() {
	// Flags
	model := flag.String("model", ollama.DefaultModel, "Ollama model to use")
	host := flag.String("host", ollama.DefaultHost, "Ollama server address")
	memoryPath := flag.String("memory", defaultMemoryPath(), "Path for persistent memory storage")
	allowShell := flag.Bool("allow-shell", false, "Enable shell command execution")
	trustMode := flag.Bool("trust", false, "Skip confirmation prompts for file operations")
	sessionID := flag.String("resume", "", "Resume a previous session by ID")
	serveMode := flag.Bool("serve", false, "Run as HTTP server instead of REPL")
	servePort := flag.String("port", "3333", "HTTP server port (with --serve)")
	showVersion := flag.Bool("version", false, "Show version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("nous %s\n", version)
		os.Exit(0)
	}

	fmt.Print(banner)
	fmt.Printf("  version %s | %s | %d cores | %s RAM\n\n",
		version, runtime.GOARCH, runtime.NumCPU(), formatMemory())

	// Initialize Ollama client
	llm := ollama.New(
		ollama.WithHost(*host),
		ollama.WithModel(*model),
	)

	// Check Ollama connectivity
	fmt.Print("  connecting to ollama... ")
	if err := llm.Ping(); err != nil {
		fmt.Printf("FAILED\n  %v\n", err)
		fmt.Println("\n  Make sure Ollama is running: ollama serve")
		os.Exit(1)
	}
	fmt.Printf("OK (%s)\n", *model)

	// Verify model is available
	models, err := llm.ListModels()
	if err == nil {
		found := false
		for _, m := range models {
			if strings.HasPrefix(m.Name, *model) {
				found = true
				break
			}
		}
		if !found {
			fmt.Printf("  warning: model '%s' not found locally\n", *model)
			fmt.Printf("  pull it with: ollama pull %s\n", *model)
		}
	}

	// Multi-model routing
	router := cognitive.NewModelRouter(*host, *model)
	if err := router.Discover(context.Background()); err != nil {
		fmt.Printf("  model router: %v (using default)\n", err)
	} else {
		fmt.Printf("  %s\n", router.Status())
	}

	// Get working directory
	workDir, _ := os.Getwd()
	cognitive.WorkDir = workDir

	// Project auto-scan
	fmt.Print("  scanning project... ")
	projectInfo := cognitive.ScanProject(workDir)
	cognitive.CurrentProject = projectInfo
	fmt.Printf("%s (%s, %d files)\n", projectInfo.Name, projectInfo.Language, projectInfo.FileCount)

	// Build codebase index (Go AST parsing for structural context)
	nousDir := filepath.Join(workDir, ".nous")
	codeIndex := index.NewCodebaseIndex(nousDir)
	if projectInfo.Language == "Go" {
		if err := codeIndex.Build(workDir); err != nil {
			fmt.Printf("  index: %v\n", err)
		} else {
			fmt.Printf("  codebase index: %d symbols\n", codeIndex.Size())
		}
	}

	// Filesystem Sentinel — ambient file watching with inotify
	var fileWatcher *sentinel.Watcher
	fileWatcher, err = sentinel.NewWatcher(workDir, 500*time.Millisecond, func(events []sentinel.FileEvent) {
		// Auto-update codebase index on Go file changes
		if projectInfo.Language == "Go" {
			changed := sentinel.ChangedGoFiles(events)
			if len(changed) > 0 {
				codeIndex.IncrementalUpdate(workDir, changed)
			}
		}
	})
	if err != nil {
		fmt.Printf("  sentinel: %v (disabled)\n", err)
	} else {
		go fileWatcher.Run()
		fmt.Printf("  sentinel: watching %d dirs\n", fileWatcher.WatchCount())
	}

	// Tool Choreography — learned multi-step recipes
	recipeBook := cognitive.NewRecipeBook(nousDir)
	if recipeBook.Size() > 0 {
		fmt.Printf("  recipes: %d learned\n", recipeBook.Size())
	}

	// Initialize undo stack and tool registry
	undoStack := memory.NewUndoStack(100)
	toolReg := tools.NewRegistry()
	tools.RegisterBuiltins(toolReg, workDir, *allowShell, undoStack)

	// Predictive Pre-computation
	predictor := cognitive.NewPredictor(toolReg)

	// Initialize core systems
	board := blackboard.New()

	// Memory systems
	wm := memory.NewWorkingMemory(64)
	ltm := memory.NewLongTermMemory(*memoryPath)

	// Project-level memory (stored in the project's .nous/ directory)
	projMem := memory.NewProjectMemory(workDir)
	fmt.Printf("  project memory: %d facts\n", projMem.Size())

	// Episodic memory — remembers every interaction forever (semantic search via embeddings)
	episodic := memory.NewEpisodicMemory(nousDir, llm.Embed)
	if episodic.Size() > 0 {
		fmt.Printf("  episodic memory: %d episodes (%.0f%% success rate)\n",
			episodic.Size(), episodic.SuccessRate()*100)
	}

	// Training data collector — gathers successful interactions for fine-tuning
	collector := training.NewCollector(nousDir)
	if collector.Size() > 0 {
		fmt.Printf("  training data: %d pairs\n", collector.Size())
	}

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
			fmt.Printf("  warning: could not resume session %s: %v\n", *sessionID, err)
		} else {
			currentSession = loaded
			fmt.Printf("  resumed session: %s (%d messages)\n", loaded.Name, len(loaded.Messages))
		}
	}

	// Create cognitive streams
	perceiver := cognitive.NewPerceiver(board, llm)
	perceiver.Router = router
	reasoner := cognitive.NewReasoner(board, llm, toolReg)
	reasoner.WorkingMem = wm
	reasoner.LongTermMem = ltm
	reasoner.ProjectMem = projMem
	reasoner.Compressor = compress.NewCompressor(llm)
	reasoner.CodeIndex = codeIndex
	reasoner.Recipes = recipeBook
	reasoner.Predictor = predictor
	planner := cognitive.NewPlanner(board, llm)
	executor := cognitive.NewExecutor(board, llm, toolReg)
	reflector := cognitive.NewReflector(board, llm)
	learner := cognitive.NewLearner(board, llm, *memoryPath)

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

	reasoner.OnToken = func(token string, done bool) {
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
				// Still ambiguous — keep buffering
				return
			}
			// Definitely not a tool call — flush everything buffered
			fmt.Print(current)
			streamBuf.Reset()
		} else {
			if !inToolCall {
				remaining := streamBuf.String()
				if !isToolJSON(remaining) && strings.TrimSpace(remaining) != "" {
					fmt.Print(remaining)
				}
				fmt.Println()
			}
			inToolCall = false
			streamBuf.Reset()
		}
	}

	// Tool status updates
	reasoner.OnStatus = func(status string) {
		fmt.Printf("\033[90m%s\033[0m\n", status)
	}

	// Start cognitive streams
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	streams := []cognitive.Stream{perceiver, reasoner, planner, executor, reflector, learner}

	for _, s := range streams {
		s := s
		go func() {
			if err := s.Run(ctx); err != nil && err != context.Canceled {
				fmt.Fprintf(os.Stderr, "\n  [%s] error: %v\n", s.Name(), err)
			}
		}()
	}

	// Print status
	toolList := toolReg.List()
	toolNames := make([]string, len(toolList))
	for i, t := range toolList {
		toolNames[i] = t.Name
	}

	fmt.Printf("  %d cognitive streams active\n", len(streams))
	fmt.Printf("  %d tools: %s\n", len(toolList), strings.Join(toolNames, ", "))
	if *allowShell {
		fmt.Print("  shell: ENABLED")
		if *trustMode {
			fmt.Print(" | trust mode: ON")
		}
		fmt.Println()
	}
	fmt.Printf("  session: %s\n", currentSession.ID)
	fmt.Println()
	fmt.Println("  I am Nous. I think, therefore I am — locally.")

	// Handle graceful shutdown — save session + episodic memory
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\n\n  saving session...")
		currentSession.Messages = reasoner.Conv.Messages()
		sessionStore.Save(currentSession)
		episodic.Save()
		collector.Save()
		if fileWatcher != nil {
			fileWatcher.Stop()
		}
		fmt.Println("  goodbye.")
		cancel()
		os.Exit(0)
	}()

	// --- Server Mode ---
	if *serveMode {
		addr := ":" + *servePort
		fmt.Printf("  serving on http://0.0.0.0%s\n\n", addr)
		srv := server.New(addr, board, perceiver)
		if err := srv.Start(version, *model, len(toolList)); err != nil {
			fmt.Fprintf(os.Stderr, "server error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// --- REPL Mode ---
	fmt.Println("  type /help for commands, /quit to exit")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("  nous> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Handle commands
		if strings.HasPrefix(input, "/") {
			if handleCommand(input, board, llm, toolReg, wm, ltm, projMem, undoStack, sessionStore, currentSession, reasoner, projectInfo, episodic, collector) {
				continue
			}
		}

		// Submit to the perceiver and wait for reasoning to complete
		fmt.Println()
		start := time.Now()
		perceiver.Submit(input)

		// Wait for the answer to appear on the blackboard
		answer := waitForAnswerStr(board)
		duration := time.Since(start)
		fmt.Println()

		// Record in episodic memory (remembers everything forever)
		go episodic.Record(memory.Episode{
			Timestamp: time.Now(),
			Input:     input,
			Output:    answer,
			Success:   true,
			Duration:  duration.Milliseconds(),
		})

		// Collect training data from successful interactions
		msgs := reasoner.Conv.Messages()
		sysPrompt := ""
		if len(msgs) > 0 {
			sysPrompt = msgs[0].Content
		}
		go collector.Collect(sysPrompt, input, answer, nil, 0.7)

		// Auto-save session after each exchange
		currentSession.Messages = reasoner.Conv.Messages()
		go sessionStore.Save(currentSession)
	}
}

func handleCommand(input string, board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry, wm *memory.WorkingMemory, ltm *memory.LongTermMemory, projMem *memory.ProjectMemory, undoStack *memory.UndoStack, sessions *cognitive.SessionStore, current *cognitive.Session, reasoner *cognitive.Reasoner, project *cognitive.ProjectInfo, episodic *memory.EpisodicMemory, collector *training.Collector) bool {
	parts := strings.Fields(input)
	cmd := strings.ToLower(parts[0])

	switch cmd {
	case "/quit", "/exit", "/q":
		fmt.Println("  saving session...")
		current.Messages = reasoner.Conv.Messages()
		sessions.Save(current)
		projMem.Flush()
		fmt.Println("  goodbye.")
		os.Exit(0)

	case "/help", "/h":
		fmt.Print(`
  Commands:
    /help              Show this help
    /status            Show cognitive system status
    /memory            Show working memory contents
    /longterm          Show long-term memory entries
    /episodes          Show recent episodic memories
    /search <query>    Semantic search through all memories
    /remember <k> <v>  Remember a project fact
    /recall <query>    Search project memory
    /forget <key>      Forget a project fact
    /training          Show training data stats
    /export <fmt>      Export training data (jsonl/alpaca/chatml)
    /finetune          Generate Modelfile + fine-tuning guide
    /undo              Revert the last file change
    /history           Show undo stack
    /goals             Show active goals
    /model             Show current model info
    /tools             List available tools
    /project           Show project info
    /sessions          List saved sessions
    /save [name]       Save current session with a name
    /clear             Clear conversation context
    /quit              Exit Nous (auto-saves session)
`)

	case "/status":
		fmt.Printf("  Percepts: %d\n", len(board.Percepts()))
		fmt.Printf("  Active goals: %d\n", len(board.ActiveGoals()))
		fmt.Printf("  Working memory: %d items\n", wm.Size())
		fmt.Printf("  Long-term memory: %d entries\n", ltm.Size())
		fmt.Printf("  Project memory: %d facts\n", projMem.Size())
		fmt.Printf("  Undo stack: %d entries\n", undoStack.Size())
		fmt.Printf("  Recent actions: %d\n", len(board.RecentActions(100)))
		fmt.Printf("  Conversation: %s\n", reasoner.Conv.Summary())
		fmt.Printf("  Session: %s (%s)\n", current.Name, current.ID)

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
		fmt.Printf("  model: %s\n", llm.Model())
		models, err := llm.ListModels()
		if err == nil {
			fmt.Println("  available models:")
			for _, m := range models {
				fmt.Printf("    - %s (%.1f MB)\n", m.Name, float64(m.Size)/(1024*1024))
			}
		}

	case "/tools":
		for _, t := range toolReg.List() {
			fmt.Printf("  %-10s %s\n", t.Name, t.Description)
		}

	case "/project":
		fmt.Print(project.ContextString())

	case "/sessions":
		list, err := sessions.List()
		if err != nil || len(list) == 0 {
			fmt.Println("  no saved sessions")
		} else {
			for _, s := range list {
				marker := " "
				if s.ID == current.ID {
					marker = "*"
				}
				fmt.Printf("  %s %s  %-30s  %d msgs  %s\n",
					marker, s.ID, s.Name, len(s.Messages),
					s.UpdatedAt.Format("2006-01-02 15:04"))
			}
			fmt.Println("\n  resume with: nous --resume <ID>")
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
		cfg := training.DefaultModelfileConfig(llm.Model())
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

	case "/clear":
		reasoner.Conv = cognitive.NewConversation(20)
		fmt.Println("  conversation cleared")

	default:
		fmt.Printf("  unknown command: %s (try /help)\n", cmd)
	}

	return true
}

func waitForAnswer(board *blackboard.Blackboard) {
	_ = waitForAnswerStr(board)
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

func formatMemory() string {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	totalMB := m.Sys / 1024 / 1024
	if totalMB < 1024 {
		return fmt.Sprintf("%d MB", totalMB)
	}
	return fmt.Sprintf("%.1f GB", float64(totalMB)/1024)
}
