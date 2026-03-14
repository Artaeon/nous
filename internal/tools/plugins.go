package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const pluginTimeout = 30 * time.Second

// PluginDescriptor holds the metadata returned by a plugin script's --describe flag.
type PluginDescriptor struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Args        []string `json:"args"`
}

// PluginLoader scans a directory for executable scripts and registers them as tools.
type PluginLoader struct {
	Dir string
}

// NewPluginLoader creates a loader that scans the given directory.
func NewPluginLoader(dir string) *PluginLoader {
	return &PluginLoader{Dir: dir}
}

// LoadPlugins scans dir for executable scripts and registers each as a tool.
// Returns the number of plugins loaded and any error encountered during scanning.
func LoadPlugins(dir string, registry *Registry) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("read plugin dir %s: %w", dir, err)
	}

	loaded := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		path := filepath.Join(dir, entry.Name())

		info, err := entry.Info()
		if err != nil {
			continue
		}
		// Skip non-executable files
		if info.Mode()&0111 == 0 {
			continue
		}

		desc, err := describePlugin(path)
		if err != nil {
			continue
		}

		// Use script name (without extension) as tool name if descriptor doesn't provide one
		toolName := desc.Name
		if toolName == "" {
			toolName = strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		}

		// Capture path for closure
		scriptPath := path
		pluginArgs := desc.Args

		registry.Register(Tool{
			Name:        toolName,
			Description: desc.Description,
			Execute: func(args map[string]string) (string, error) {
				return executePlugin(scriptPath, pluginArgs, args)
			},
		})

		loaded++
	}

	return loaded, nil
}

// describePlugin runs a script with --describe and parses the JSON output.
func describePlugin(path string) (PluginDescriptor, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, path, "--describe")
	var stdout bytes.Buffer
	cmd.Stdout = &stdout

	if err := cmd.Run(); err != nil {
		return PluginDescriptor{}, fmt.Errorf("describe %s: %w", path, err)
	}

	var desc PluginDescriptor
	if err := json.Unmarshal(stdout.Bytes(), &desc); err != nil {
		return PluginDescriptor{}, fmt.Errorf("parse descriptor from %s: %w", path, err)
	}

	return desc, nil
}

// executePlugin runs a plugin script with args passed as NOUS_ARG_ environment variables.
func executePlugin(path string, declaredArgs []string, args map[string]string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), pluginTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, path)

	// Pass arguments as environment variables
	cmd.Env = os.Environ()
	for key, value := range args {
		envKey := "NOUS_ARG_" + key
		cmd.Env = append(cmd.Env, envKey+"="+value)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	if ctx.Err() == context.DeadlineExceeded {
		return "", fmt.Errorf("plugin %s timed out after %v", filepath.Base(path), pluginTimeout)
	}

	if err != nil {
		errMsg := stderr.String()
		if errMsg == "" {
			errMsg = err.Error()
		}
		return "", fmt.Errorf("plugin %s failed: %s", filepath.Base(path), strings.TrimSpace(errMsg))
	}

	return stdout.String(), nil
}
