package cli

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// Input provides enhanced line reading with history integration and
// multi-line continuation. It works with any terminal (no raw mode),
// relying on bufio.Scanner for portable input.
type Input struct {
	history     *History
	completions []string // slash commands, tool names
	prompt      string
	scanner     *bufio.Scanner
}

// NewInput creates an Input tied to the given history and reader.
// completions are used for the /help listing and future tab-completion.
func NewInput(history *History, completions []string, reader io.Reader) *Input {
	return &Input{
		history:     history,
		completions: completions,
		scanner:     bufio.NewScanner(reader),
	}
}

// SetCompletions replaces the completion candidates (slash commands, tool names).
func (inp *Input) SetCompletions(completions []string) {
	inp.completions = completions
}

// ReadLine reads a line of input, handling multi-line continuation with
// trailing backslash. Returns the assembled line and any error.
// Returns io.EOF when the input stream is exhausted (Ctrl+D).
func (inp *Input) ReadLine(prompt string) (string, error) {
	var parts []string
	currentPrompt := prompt

	for {
		fmt.Print(currentPrompt)
		if !inp.scanner.Scan() {
			if err := inp.scanner.Err(); err != nil {
				return "", err
			}
			// Ctrl+D / EOF
			if len(parts) > 0 {
				return strings.Join(parts, "\n"), nil
			}
			return "", io.EOF
		}

		line := inp.scanner.Text()

		// Multi-line continuation: trailing backslash
		if strings.HasSuffix(strings.TrimRight(line, " \t"), "\\") {
			// Strip the trailing backslash and accumulate
			line = strings.TrimRight(line, " \t")
			line = line[:len(line)-1]
			parts = append(parts, line)
			currentPrompt = "  ... "
			continue
		}

		parts = append(parts, line)
		return strings.Join(parts, "\n"), nil
	}
}

// Completions returns the current completion candidates.
func (inp *Input) Completions() []string {
	return inp.completions
}
