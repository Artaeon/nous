package main

import (
	"bufio"
	"encoding/json"
	goflag "flag"
	"fmt"
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

// --- understand subcommand ---

// understandResult is the JSON output for "nous understand".
type understandResult struct {
	Intent     string            `json:"intent"`
	Action     string            `json:"action"`
	Entities   map[string]string `json:"entities"`
	Confidence float64           `json:"confidence"`
	Raw        string            `json:"raw"`
}

func runUnderstand(args []string, nlu *cognitive.NLU) bool {
	// Collect input: positional args or stdin
	text := strings.Join(args, " ")
	if text == "" {
		text = readStdin()
	}
	if text == "" {
		fmt.Fprintln(os.Stderr, "usage: nous understand <text>")
		os.Exit(1)
	}

	result := nlu.Understand(text)

	out := understandResult{
		Intent:     result.Intent,
		Action:     result.Action,
		Entities:   result.Entities,
		Confidence: result.Confidence,
		Raw:        result.Raw,
	}
	if out.Entities == nil {
		out.Entities = make(map[string]string)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(out)
	return true
}

// --- generate subcommand ---

func runGenerate(args []string, actions *cognitive.ActionRouter) bool {
	fs := goflag.NewFlagSet("generate", goflag.ExitOnError)
	facts := fs.String("facts", "", "Facts to compose text from")
	style := fs.String("style", "paragraph", "Output style: paragraph, bullet, brief, detailed")
	jsonOut := fs.Bool("json", false, "Emit JSON instead of plain text")
	jsonShort := fs.Bool("j", false, "Emit JSON instead of plain text (short)")
	quiet := fs.Bool("quiet", false, "Suppress headers, raw output only")
	quietShort := fs.Bool("q", false, "Suppress headers, raw output only (short)")
	fs.Parse(args)

	wantJSON := *jsonOut || *jsonShort
	wantQuiet := *quiet || *quietShort
	_ = wantQuiet // reserved for future header suppression

	// Collect facts: --facts flag, positional args, or stdin
	input := *facts
	if input == "" {
		input = strings.Join(fs.Args(), " ")
	}
	if input == "" {
		input = readStdin()
	}
	if input == "" {
		fmt.Fprintln(os.Stderr, "usage: nous generate --facts \"...\" [--style paragraph|bullet|brief|detailed]")
		os.Exit(1)
	}

	// Feed facts into the Composer as a factual query
	var text string
	if actions.Composer != nil {
		ctx := actions.BuildComposeContext()
		resp := actions.Composer.Compose(input, cognitive.RespFactual, ctx)
		if resp != nil && resp.Text != "" {
			text = resp.Text
		}
	}

	// Fallback: if the Composer produced nothing, echo the input as-is
	if text == "" {
		text = input
	}

	// Apply style transformation
	text = applyStyle(text, *style)

	if wantJSON {
		out := map[string]string{
			"style": *style,
			"text":  text,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
	} else {
		fmt.Println(text)
	}
	return true
}

// applyStyle formats composed text according to the requested style.
func applyStyle(text, style string) string {
	switch style {
	case "bullet":
		sentences := splitSentences(text)
		var bullets []string
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if s != "" {
				bullets = append(bullets, "- "+s)
			}
		}
		if len(bullets) > 0 {
			return strings.Join(bullets, "\n")
		}
		return "- " + text
	case "brief":
		sentences := splitSentences(text)
		if len(sentences) > 2 {
			return strings.Join(sentences[:2], " ")
		}
		return text
	case "detailed":
		// For detailed, return the full text as-is (already maximally expanded)
		return text
	default: // "paragraph"
		return text
	}
}

// splitSentences naively splits text on sentence-ending punctuation.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder
	for i, r := range text {
		current.WriteRune(r)
		if (r == '.' || r == '!' || r == '?') && i+1 < len(text) && text[i+1] == ' ' {
			s := strings.TrimSpace(current.String())
			if s != "" {
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}
	if s := strings.TrimSpace(current.String()); s != "" {
		sentences = append(sentences, s)
	}
	return sentences
}

// --- reason subcommand ---

// reasonResult is the JSON output for "nous reason".
type reasonResult struct {
	Question   string   `json:"question"`
	Analysis   []string `json:"analysis"`
	Conclusion string   `json:"conclusion"`
	Confidence float64  `json:"confidence"`
}

func runReason(args []string, actions *cognitive.ActionRouter) bool {
	// Collect input: positional args or stdin
	text := strings.Join(args, " ")
	if text == "" {
		text = readStdin()
	}
	if text == "" {
		fmt.Fprintln(os.Stderr, "usage: nous reason <question>")
		os.Exit(1)
	}

	out := reasonResult{
		Question: text,
	}

	// Run the full reasoning pipeline if available
	if actions.Pipeline != nil {
		result := actions.Pipeline.Process(text)
		if result != nil {
			// Gather analysis points from all engines that contributed
			for _, fact := range result.DirectFacts {
				if fact != "" {
					out.Analysis = append(out.Analysis, fact)
				}
			}
			for _, fact := range result.InferredFacts {
				if fact != "" {
					out.Analysis = append(out.Analysis, fact)
				}
			}
			if result.ReasoningTrace != "" {
				out.Analysis = append(out.Analysis, result.ReasoningTrace)
			}
			if result.CausalTrace != "" {
				out.Analysis = append(out.Analysis, result.CausalTrace)
			}
			if result.AnalogyTrace != "" {
				out.Analysis = append(out.Analysis, result.AnalogyTrace)
			}
			out.Confidence = result.Confidence

			// Use the composed response as the conclusion
			conclusion := actions.Pipeline.ComposeResponse(text, result)
			if conclusion != "" {
				out.Conclusion = conclusion
			}
		}
	}

	// Fallback: use the Composer for a factual response
	if out.Conclusion == "" && actions.Composer != nil {
		ctx := actions.BuildComposeContext()
		resp := actions.Composer.Compose(text, cognitive.RespFactual, ctx)
		if resp != nil && resp.Text != "" {
			out.Conclusion = resp.Text
		}
	}

	if out.Analysis == nil {
		out.Analysis = []string{}
	}
	if out.Conclusion == "" {
		out.Conclusion = "Insufficient knowledge to draw a conclusion."
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(out)
	return true
}

// --- remember subcommand ---

func runRemember(args []string, ltm *memory.LongTermMemory) bool {
	fs := goflag.NewFlagSet("remember", goflag.ExitOnError)
	list := fs.Bool("list", false, "List all stored key-value pairs")
	fs.Parse(args)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")

	// --list mode: dump all entries
	if *list {
		entries := ltm.All()
		type listEntry struct {
			Key      string `json:"key"`
			Value    string `json:"value"`
			Category string `json:"category,omitempty"`
		}
		out := make([]listEntry, 0, len(entries))
		for _, e := range entries {
			out = append(out, listEntry{
				Key:      e.Key,
				Value:    e.Value,
				Category: e.Category,
			})
		}
		enc.Encode(out)
		return true
	}

	remaining := fs.Args()

	// Store mode: key + value
	if len(remaining) >= 2 {
		key := remaining[0]
		value := strings.Join(remaining[1:], " ")
		ltm.Store(key, value, "cli")

		out := map[string]interface{}{
			"key":    key,
			"value":  value,
			"stored": true,
		}
		enc.Encode(out)
		return true
	}

	// Recall mode: key only
	if len(remaining) == 1 {
		key := remaining[0]
		value, ok := ltm.Retrieve(key)
		if !ok {
			out := map[string]interface{}{
				"key":   key,
				"found": false,
			}
			enc.Encode(out)
			os.Exit(1)
		}
		out := map[string]interface{}{
			"key":   key,
			"value": value,
		}
		enc.Encode(out)
		return true
	}

	fmt.Fprintln(os.Stderr, "usage: nous remember <key> [value]")
	fmt.Fprintln(os.Stderr, "       nous remember --list")
	os.Exit(1)
	return true
}

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
