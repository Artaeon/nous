package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/artaeon/nous/internal/assistant"
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

	// Initialize Ollama client
	llm := ollama.New(
		ollama.WithHost(*host),
		ollama.WithModel(*model),
	)

	// Check Ollama connectivity with spinner
	spinner := cognitive.NewSpinner()
	spinner.Start("connecting...")
	if err := llm.Ping(); err != nil {
		spinner.Stop()
		fmt.Printf("  %s%s%s\n", cognitive.ColorRed, err, cognitive.ColorReset)
		fmt.Printf("  %sollama serve%s to start\n", cognitive.ColorDim, cognitive.ColorReset)
		os.Exit(1)
	}
	spinner.Stop()

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
			fmt.Printf("  %smodel '%s' not found — ollama pull %s%s\n", cognitive.ColorYellow, *model, *model, cognitive.ColorReset)
		}
	}

	// Multi-model routing
	router := cognitive.NewModelRouter(*host, *model)
	if err := router.Discover(context.Background()); err != nil {
		// silently use default
		_ = err
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

	// Predictive Pre-computation
	predictor := cognitive.NewPredictor(toolReg)

	// Initialize core systems
	board := blackboard.New()

	// Memory systems
	wm := memory.NewWorkingMemory(64)
	wm.SetEmbedFunc(llm.Embed) // enable semantic retrieval
	ltm := memory.NewLongTermMemory(*memoryPath)

	// Project-level memory (stored in the project's .nous/ directory)
	projMem := memory.NewProjectMemory(workDir)

	// Episodic memory — remembers every interaction forever (semantic search via embeddings)
	episodic := memory.NewEpisodicMemory(nousDir, llm.Embed)
	assistantStore := assistant.NewStore(*memoryPath)

	// Training data collector — gathers successful interactions for fine-tuning
	collector := training.NewCollector(nousDir)

	// Auto-tuner — monitors training data and triggers Modelfile-based tuning
	autoTuner := training.NewAutoTuner(collector, *model).
		WithCreator(llm).
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
	perceiver := cognitive.NewPerceiver(board, llm)
	perceiver.Router = router
	learner := cognitive.NewLearner(board, llm, *memoryPath)
	reasoner := cognitive.NewReasoner(board, llm, toolReg)
	reasoner.WorkingMem = wm
	reasoner.LongTermMem = ltm
	reasoner.ProjectMem = projMem
	reasoner.Compressor = compress.NewCompressor(llm)
	reasoner.CodeIndex = codeIndex
	reasoner.Recipes = recipeBook
	reasoner.Predictor = predictor
	reasoner.Learner = learner
	reasoner.EpisodicMem = episodic
	reasoner.Router = router
	reasoner.AssistantContext = func(input string, recent string) string {
		return buildAssistantContext(assistantStore, input, recent, time.Now())
	}
	reasoner.AssistantAnswer = func(input string, recent string) (string, bool) {
		return answerAssistantQuery(assistantStore, input, recent, time.Now())
	}
	planner := cognitive.NewPlanner(board, llm)
	executor := cognitive.NewExecutor(board, llm, toolReg)
	reflector := cognitive.NewReflector(board, llm)

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

	// Print startup banner
	toolList := toolReg.List()
	fmt.Print(cognitive.Banner(version, *model, *host, len(toolList), 64))

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
		if fileWatcher != nil {
			fileWatcher.Stop()
		}
		cancel()
		os.Exit(0)
	}()

	// --- Server Mode ---
	if *serveMode {
		addr := ":" + *servePort
		fmt.Printf("  serving on http://0.0.0.0%s\n\n", addr)
		srv := server.New(addr, board, perceiver, assistantStore)
		if err := srv.Start(version, *model, len(toolList)); err != nil {
			fmt.Fprintf(os.Stderr, "server error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// --- REPL Mode ---
	fmt.Print(cognitive.Panel("Quick start", []string{
		cognitive.Styled(cognitive.ColorCyan, "/dashboard") + " overview of the local agent",
		cognitive.Styled(cognitive.ColorCyan, "/briefing") + " morning briefing — what matters today",
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
		if input == "" {
			continue
		}

		// Handle commands
		if strings.HasPrefix(input, "/") {
			if handleCommand(input, board, llm, toolReg, wm, ltm, projMem, undoStack, sessionStore, currentSession, reasoner, learner, projectInfo, episodic, collector, autoTuner, assistantStore) {
				continue
			}
		}

		// Submit to the perceiver and wait for reasoning to complete
		fmt.Println()
		firstToken = true
		responseStarted = false
		llmSpinner.Start("thinking...")
		start := time.Now()
		perceiver.Submit(input)

		// Wait for the answer to appear on the blackboard
		answer := waitForAnswerStr(board)
		duration := time.Since(start)
		if !responseStarted && strings.TrimSpace(answer) != "" {
			fmt.Printf("  %s\n", strings.ReplaceAll(answer, "\n", "\n  "))
		}

		// Show timing footer
		fmt.Printf("\n%s\n\n", cognitive.TimingFooter(duration))

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
		go autoTuner.Check()

		// Auto-save session after each exchange
		currentSession.Messages = reasoner.Conv.Messages()
		go sessionStore.Save(currentSession)
	}
}

func handleCommand(input string, board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry, wm *memory.WorkingMemory, ltm *memory.LongTermMemory, projMem *memory.ProjectMemory, undoStack *memory.UndoStack, sessions *cognitive.SessionStore, current *cognitive.Session, reasoner *cognitive.Reasoner, learner *cognitive.Learner, project *cognitive.ProjectInfo, episodic *memory.EpisodicMemory, collector *training.Collector, autoTuner *training.AutoTuner, assistantStore *assistant.Store) bool {
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

	case "/dashboard":
		fmt.Print(renderDashboard(board, wm, ltm, projMem, undoStack, current, episodic, collector, autoTuner, assistantStore))

	case "/today":
		fmt.Print(renderToday(assistantStore, time.Now()))
		_ = assistantStore.MarkNotificationsRead()

	case "/briefing":
		fmt.Print(renderBriefing(assistantStore, time.Now()))
		_ = assistantStore.MarkNotificationsRead()

	case "/review":
		fmt.Print(renderReview(assistantStore, time.Now()))

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
		fmt.Print(cognitive.Section("Model routing"))
		fmt.Print(cognitive.KeyValue("Active model", llm.Model()))
		models, err := llm.ListModels()
		if err == nil {
			fmt.Print(cognitive.Section("Available local models"))
			for _, m := range models {
				fmt.Printf("  • %s%s%s  %s%.1f MB%s\n", cognitive.ColorCyan, m.Name, cognitive.ColorReset, cognitive.ColorDim, float64(m.Size)/(1024*1024), cognitive.ColorReset)
			}
		}

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

	default:
		fmt.Printf("  unknown command: %s (try /help)\n", cmd)
	}

	return true
}

func renderHelp() string {
	var b strings.Builder
	b.WriteString(cognitive.Panel("Core workflows", []string{
		cognitive.Styled(cognitive.ColorCyan, "/dashboard") + " snapshot of memory, sessions, training, and uptime signals",
		cognitive.Styled(cognitive.ColorCyan, "/today") + " open your assistant inbox with due reminders and upcoming tasks",
		cognitive.Styled(cognitive.ColorCyan, "/status") + " low-level runtime counters and current session state",
		cognitive.Styled(cognitive.ColorCyan, "/plan <goal>") + " hand a longer task to the planner/executor pipeline",
		cognitive.Styled(cognitive.ColorCyan, "/tools") + " browse built-in tools by category",
	}))
	b.WriteString(cognitive.Panel("Assistant operations", []string{
		cognitive.Styled(cognitive.ColorCyan, "/briefing") + " morning briefing — overdue, today's schedule, routines",
		cognitive.Styled(cognitive.ColorCyan, "/review") + " evening review — completed, still pending, overdue",
		"/remind <when> <task>, /tasks, /done <task-id>",
		"/routine <daily|weekdays> <HH:MM> <task>, /routines",
		"/pref <key> <value>, /prefs",
		"Examples: /remind in 2h stretch · /remind tomorrow 09:00 dentist",
	}))
	b.WriteString(cognitive.Panel("Memory and recall", []string{
		"/memory, /longterm, /episodes, /search <query>",
		"/remember <key> <value>, /recall <query>, /forget <key>",
	}))
	b.WriteString(cognitive.Panel("Training and tuning", []string{
		"/training, /autotune [force], /export <jsonl|alpaca|chatml>",
		"/finetune to generate a Modelfile and local tuning guide",
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
	prefs := store.Preferences()
	if len(prefs) == 0 {
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

	if task, ok := nextScheduledTask(store, now); ok {
		if de {
			return fmt.Sprintf("Ich kenne bisher diese Präferenzen: %s. Im Moment ist besonders relevant, dass als Nächstes \"%s\" um %s ansteht.", strings.Join(parts, ", "), task.Title, task.DueAt.Format("15:04"))
		}
		return fmt.Sprintf("So far I know these preferences: %s. The most relevant one right now is that %q is coming up at %s.", strings.Join(parts, ", "), task.Title, task.DueAt.Format("15:04"))
	}

	if de {
		return "Ich kenne bisher diese Präferenzen: " + strings.Join(parts, ", ") + "."
	}
	return "So far I know these preferences: " + strings.Join(parts, ", ") + "."
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

func buildAssistantContext(store *assistant.Store, input string, recent string, now time.Time) string {
	if store == nil || !shouldInjectAssistantContext(input) {
		return ""
	}

	var lines []string
	lines = append(lines, "Current time: "+now.Format("2006-01-02 15:04 Mon"))
	if recent != "" {
		lines = append(lines, "Recent conversation: "+recent)
	}
	prefs := store.Preferences()
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
		if strings.Contains(lower, sig) {
			return "", false
		}
	}

	lang := preferenceValue(store, "language")
	de := strings.HasPrefix(strings.ToLower(lang), "de")

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
				return "Gut — wir bleiben bei diesem Faden. Ich kann jetzt den nächsten kleinen Schritt daraus machen.", true
			}
			return "Alright — we'll stay with this thread. I can turn it into the next small step now.", true
		}
	}

	if strings.Contains(lower, "help me plan my day") || strings.Contains(lower, "plan my day") || strings.Contains(lower, "organize my day") || strings.Contains(lower, "prioritize my day") {
		return assistantPlanReply(store, now, de), true
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
		if strings.Contains(lower, sig) {
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
