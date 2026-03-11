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
		cognitive.Styled(cognitive.ColorCyan, "/today") + " review reminders, tasks, and upcoming actions",
		cognitive.Styled(cognitive.ColorCyan, "/routines") + " inspect recurring assistant routines",
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
