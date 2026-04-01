package main

import (
	"bufio"
	"os"
	"strings"

	"github.com/artaeon/nous/internal/cognitive"
	"github.com/artaeon/nous/internal/memory"
)

// SubcommandRouter checks if the first positional argument is a recognized
// subcommand and executes it. Returns true if a subcommand was handled
// (caller should exit).
func SubcommandRouter(args []string, nlu *cognitive.NLU, actions *cognitive.ActionRouter, ltm *memory.LongTermMemory) bool {
	if len(args) == 0 {
		return false
	}

	switch args[0] {
	case "understand":
		return runUnderstand(args[1:], nlu)
	case "generate":
		return runGenerate(args[1:], actions)
	case "reason":
		return runReason(args[1:], actions)
	case "remember":
		return runRemember(args[1:], ltm)
	default:
		return false
	}
}

// Subcommand stubs — each is implemented in subsequent commits.
// They exist here as forward declarations so the router compiles.

func runUnderstand(_ []string, _ *cognitive.NLU) bool          { return true }
func runGenerate(_ []string, _ *cognitive.ActionRouter) bool   { return true }
func runReason(_ []string, _ *cognitive.ActionRouter) bool     { return true }
func runRemember(_ []string, _ *memory.LongTermMemory) bool    { return true }

// readStdin reads all of stdin when data is piped in.
// Returns an empty string if stdin is a terminal (not piped).
func readStdin() string {
	info, err := os.Stdin.Stat()
	if err != nil {
		return ""
	}
	if info.Mode()&os.ModeCharDevice != 0 {
		return "" // interactive terminal, not piped
	}
	scanner := bufio.NewScanner(os.Stdin)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}
