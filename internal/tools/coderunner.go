package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

const (
	defaultCodeTimeout = 10 * time.Second
	maxCodeOutput      = 5000
)

// RunCode executes code in the specified language with a timeout.
// Supported languages: python/python3, bash/shell/sh, node/javascript/js.
func RunCode(language, code string, timeout time.Duration) (string, error) {
	if code == "" {
		return "", fmt.Errorf("coderunner: no code provided")
	}
	if timeout <= 0 {
		timeout = defaultCodeTimeout
	}

	var cmdName string
	var cmdArgs []string

	switch strings.ToLower(language) {
	case "python", "python3":
		cmdName = "python3"
		cmdArgs = []string{"-c", code}
	case "bash", "shell", "sh", "":
		cmdName = "bash"
		cmdArgs = []string{"-c", code}
	case "node", "javascript", "js":
		cmdName = "node"
		cmdArgs = []string{"-e", code}
	default:
		return "", fmt.Errorf("coderunner: unsupported language %q (supported: python, bash, node)", language)
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, cmdName, cmdArgs...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	stdoutStr := stdout.String()
	stderrStr := stderr.String()

	// Truncate output
	if len(stdoutStr) > maxCodeOutput {
		stdoutStr = stdoutStr[:maxCodeOutput] + "\n... (truncated)"
	}
	if len(stderrStr) > maxCodeOutput {
		stderrStr = stderrStr[:maxCodeOutput] + "\n... (truncated)"
	}

	// Check for timeout
	if ctx.Err() == context.DeadlineExceeded {
		return formatCodeOutput(stdoutStr, stderrStr), fmt.Errorf("coderunner: execution timed out after %s", timeout)
	}

	if err != nil {
		return formatCodeOutput(stdoutStr, stderrStr), fmt.Errorf("coderunner: %v", err)
	}

	return formatCodeOutput(stdoutStr, stderrStr), nil
}

// formatCodeOutput formats stdout and stderr into a single output string.
func formatCodeOutput(stdout, stderr string) string {
	var sb strings.Builder

	if stdout != "" {
		sb.WriteString("Output:\n")
		sb.WriteString(stdout)
	}

	if stderr != "" {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString("Errors:\n")
		sb.WriteString(stderr)
	}

	if sb.Len() == 0 {
		return "(no output)"
	}

	return sb.String()
}

// DetectLanguage uses heuristics to guess the programming language of a code snippet.
func DetectLanguage(code string) string {
	trimmed := strings.TrimSpace(code)

	// Python indicators
	if strings.HasPrefix(trimmed, "import ") ||
		strings.HasPrefix(trimmed, "from ") ||
		strings.HasPrefix(trimmed, "def ") ||
		strings.HasPrefix(trimmed, "class ") ||
		strings.Contains(trimmed, "print(") {
		return "python"
	}

	// Shebang
	if strings.HasPrefix(trimmed, "#!") {
		return "bash"
	}

	// JavaScript/Node indicators
	if strings.Contains(trimmed, "console.log") ||
		strings.HasPrefix(trimmed, "const ") ||
		strings.HasPrefix(trimmed, "let ") ||
		strings.HasPrefix(trimmed, "var ") ||
		strings.Contains(trimmed, "require(") ||
		strings.Contains(trimmed, "=> {") {
		return "node"
	}

	// Default to bash
	return "bash"
}

// RegisterCodeRunnerTools adds the coderunner tool to the registry.
func RegisterCodeRunnerTools(r *Registry) {
	r.Register(Tool{
		Name:        "coderunner",
		Description: "Execute code in a sandboxed environment. Args: code (required), language (optional: python, bash, node — auto-detected if omitted).",
		Execute: func(args map[string]string) (string, error) {
			code := args["code"]
			if code == "" {
				return "", fmt.Errorf("coderunner requires 'code' argument")
			}

			language := args["language"]
			if language == "" {
				language = DetectLanguage(code)
			}

			return RunCode(language, code, defaultCodeTimeout)
		},
	})
}
