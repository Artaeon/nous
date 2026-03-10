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
	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/ollama"
	"github.com/artaeon/nous/internal/tools"
)

const banner = `
                ╔═══════════════════════════════════╗
                ║             N O U S               ║
                ║   Native Orchestration of         ║
                ║       Unified Streams             ║
                ╚═══════════════════════════════════╝
`

const version = "0.2.0"

func main() {
	// Flags
	model := flag.String("model", ollama.DefaultModel, "Ollama model to use")
	host := flag.String("host", ollama.DefaultHost, "Ollama server address")
	memoryPath := flag.String("memory", defaultMemoryPath(), "Path for persistent memory storage")
	allowShell := flag.Bool("allow-shell", false, "Enable shell command execution")
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

	// Get working directory
	workDir, _ := os.Getwd()

	// Initialize tool registry
	toolReg := tools.NewRegistry()
	tools.RegisterBuiltins(toolReg, workDir, *allowShell)

	// Initialize core systems
	board := blackboard.New()

	// Memory systems
	wm := memory.NewWorkingMemory(64)
	ltm := memory.NewLongTermMemory(*memoryPath)

	// Create cognitive streams
	perceiver := cognitive.NewPerceiver(board, llm)
	reasoner := cognitive.NewReasoner(board, llm, toolReg)
	planner := cognitive.NewPlanner(board, llm)
	executor := cognitive.NewExecutor(board, llm)
	reflector := cognitive.NewReflector(board, llm)
	learner := cognitive.NewLearner(board, llm, *memoryPath)

	executor.AllowShell = *allowShell

	// Enable streaming output
	reasoner.OnToken = func(token string, done bool) {
		fmt.Print(token)
		if done {
			fmt.Println()
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

	fmt.Printf("  6 cognitive streams active\n")
	fmt.Printf("  %d tools: %s\n", len(toolList), strings.Join(toolNames, ", "))
	if *allowShell {
		fmt.Println("  shell execution: ENABLED")
	}
	fmt.Printf("  working directory: %s\n", workDir)
	fmt.Println()
	fmt.Println("  I am Nous. I think, therefore I am — locally.")
	fmt.Println("  type /help for commands, /quit to exit")
	fmt.Println()

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\n\n  shutting down...")
		cancel()
		os.Exit(0)
	}()

	// REPL
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
			if handleCommand(input, board, llm, toolReg, wm, ltm) {
				continue
			}
		}

		// Submit to the perceiver and wait for reasoning to complete
		fmt.Println()
		perceiver.Submit(input)

		// Wait for the answer to appear on the blackboard
		waitForAnswer(board)
		fmt.Println()
	}
}

func handleCommand(input string, board *blackboard.Blackboard, llm *ollama.Client, toolReg *tools.Registry, wm *memory.WorkingMemory, ltm *memory.LongTermMemory) bool {
	parts := strings.Fields(input)
	cmd := strings.ToLower(parts[0])

	switch cmd {
	case "/quit", "/exit", "/q":
		fmt.Println("  goodbye.")
		os.Exit(0)

	case "/help", "/h":
		fmt.Println(`
  Commands:
    /help          Show this help
    /status        Show cognitive system status
    /memory        Show working memory contents
    /longterm      Show long-term memory entries
    /goals         Show active goals
    /model         Show current model info
    /tools         List available tools
    /clear         Clear working memory
    /quit          Exit Nous
`)

	case "/status":
		fmt.Printf("  Percepts: %d\n", len(board.Percepts()))
		fmt.Printf("  Active goals: %d\n", len(board.ActiveGoals()))
		fmt.Printf("  Working memory: %d items\n", wm.Size())
		fmt.Printf("  Long-term memory: %d entries\n", ltm.Size())
		fmt.Printf("  Recent actions: %d\n", len(board.RecentActions(100)))

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

	case "/clear":
		fmt.Println("  working memory cleared")

	default:
		fmt.Printf("  unknown command: %s (try /help)\n", cmd)
	}

	return true
}

func waitForAnswer(board *blackboard.Blackboard) {
	deadline := time.After(300 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			fmt.Println("  (timeout waiting for response)")
			return
		case <-ticker.C:
			if answer, ok := board.Get("last_answer"); ok {
				board.Delete("last_answer")
				_ = answer
				return
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
