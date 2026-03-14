package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/artaeon/nous/internal/memory"
	"github.com/artaeon/nous/internal/sandbox"
)

// BuiltinOpts holds optional configuration for RegisterBuiltinsWithSandbox.
type BuiltinOpts struct {
	UndoStack *memory.UndoStack
	Sandbox   *sandbox.NativeSandbox
}

// RegisterBuiltinsWithSandbox is like RegisterBuiltins but wraps shell and run
// tools through the sandbox when one is provided.
func RegisterBuiltinsWithSandbox(r *Registry, workDir string, allowShell bool, opts BuiltinOpts) {
	// Register all standard tools first
	RegisterBuiltins(r, workDir, allowShell, opts.UndoStack)

	// If no sandbox, we're done
	if opts.Sandbox == nil {
		return
	}

	sb := opts.Sandbox

	// Override shell tool with sandboxed version
	r.Register(Tool{
		Name:        "shell",
		Description: "Execute a shell command (sandboxed). Args: command (required). Only available with --allow-shell flag.",
		Execute: func(args map[string]string) (string, error) {
			if !allowShell {
				return "", fmt.Errorf("shell execution disabled — start Nous with --allow-shell to enable")
			}
			return toolShellSandboxed(workDir, args, sb)
		},
	})

	// Override run tool with sandboxed version
	r.Register(Tool{
		Name:        "run",
		Description: "Execute a command and capture output (sandboxed). Args: command (required), stdin (optional). Requires --allow-shell.",
		Execute: func(args map[string]string) (string, error) {
			if !allowShell {
				return "", fmt.Errorf("command execution disabled — start Nous with --allow-shell to enable")
			}
			return toolRunSandboxed(workDir, args, sb)
		},
	})
}

// toolShellSandboxed executes a shell command through the sandbox.
func toolShellSandboxed(workDir string, args map[string]string, sb *sandbox.NativeSandbox) (string, error) {
	command := args["command"]
	if command == "" {
		return "", fmt.Errorf("shell requires 'command' argument")
	}

	result, err := sb.Execute(context.Background(), command, nil, sandbox.ExecOpts{
		WorkDir: workDir,
		Timeout: 60 * time.Second,
	})

	output := result.Stdout
	if result.Stderr != "" {
		if output != "" {
			output += "\n"
		}
		output += "STDERR:\n" + result.Stderr
	}

	if err != nil {
		return output, fmt.Errorf("exit %v", err)
	}

	if len(output) > 8192 {
		output = output[:8192] + "\n... (truncated)"
	}

	return output, nil
}

// toolRunSandboxed executes a command through the sandbox with stdin support.
func toolRunSandboxed(workDir string, args map[string]string, sb *sandbox.NativeSandbox) (string, error) {
	command := args["command"]
	if command == "" {
		return "", fmt.Errorf("run requires 'command' argument")
	}

	opts := sandbox.ExecOpts{
		WorkDir: workDir,
		Timeout: 60 * time.Second,
	}

	if stdin, ok := args["stdin"]; ok && stdin != "" {
		opts.Stdin = stdin
	}

	result, err := sb.Execute(context.Background(), command, nil, opts)

	output := result.Stdout
	if result.Stderr != "" {
		if output != "" {
			output += "\n"
		}
		output += result.Stderr
	}

	if len(output) > 8192 {
		output = output[:8192] + "\n... (truncated)"
	}

	if err != nil {
		return output, fmt.Errorf("run: %v", err)
	}

	return output, nil
}
