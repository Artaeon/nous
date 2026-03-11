package cognitive

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// ConfirmFunc asks the user to confirm a dangerous action.
// Returns true if the user approves, false if denied.
type ConfirmFunc func(action, detail string) bool

// TerminalConfirm prompts the user in the terminal for confirmation.
func TerminalConfirm(action, detail string) bool {
	fmt.Println()
	if detail != "" {
		// Detail may contain ANSI-colored diff preview
		for _, line := range strings.Split(detail, "\n") {
			fmt.Printf("  %s\n", line)
		}
	}
	fmt.Printf("\n  \033[33m⚡ %s\033[0m\n", action)
	fmt.Print("  \033[90mallow? [y/N]\033[0m ")

	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(strings.ToLower(input))

	return input == "y" || input == "yes"
}

// AutoApprove always returns true (for non-interactive / trusted mode).
func AutoApprove(action, detail string) bool {
	return true
}

// DangerousTools lists tools that require user confirmation.
var DangerousTools = map[string]string{
	"write":        "Will create or overwrite a file",
	"edit":         "Will modify a file",
	"patch":        "Will apply a multi-line edit to a file",
	"find_replace": "Will regex find and replace in a file",
	"shell":        "Will execute a shell command",
	"mkdir":        "Will create a directory",
}

// IsDangerous checks if a tool call requires confirmation.
func IsDangerous(toolName string) (string, bool) {
	reason, ok := DangerousTools[toolName]
	return reason, ok
}
