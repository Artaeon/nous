package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	goflag "flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/agent"
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
	case "summarize":
		return runSummarize(args[1:])
	case "transform":
		return runTransform(args[1:])
	case "draft":
		return runDraft(args[1:])
	case "code":
		return runCode(args[1:])
	case "simulate":
		return runSimulate(args[1:], actions)
	case "expand":
		return runExpand(args[1:], actions)
	case "infer":
		return runInfer(args[1:], actions)
	case "dream":
		return runDream(args[1:], actions)
	case "research":
		return runResearch(args[1:], actions)
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

// --- summarize subcommand ---

func runSummarize(args []string) bool {
	fs := goflag.NewFlagSet("summarize", goflag.ContinueOnError)
	sentences := fs.Int("sentences", 5, "Number of sentences to extract")
	fs.IntVar(sentences, "s", 5, "Number of sentences (short)")
	bullet := fs.Bool("bullet", false, "Output as bullet points")
	fs.BoolVar(bullet, "b", false, "Bullet points (short)")
	oneliner := fs.Bool("oneliner", false, "Single sentence summary")
	fs.BoolVar(oneliner, "1", false, "One-liner (short)")
	jsonOut := fs.Bool("json", false, "JSON output")
	fs.BoolVar(jsonOut, "j", false, "JSON output (short)")
	fs.Parse(args)

	text := strings.Join(fs.Args(), " ")
	if text == "" {
		text = readStdin()
	}
	if text == "" {
		fmt.Fprintln(os.Stderr, "usage: nous summarize [--sentences N] [--bullet] [--oneliner] <text>")
		os.Exit(1)
	}

	var result string
	switch {
	case *oneliner:
		result = cognitive.ExtractOneLiner(text)
	case *bullet:
		result = cognitive.ExtractBullets(text, *sentences)
	default:
		result = cognitive.ExtractSummary(text, *sentences)
	}

	if *jsonOut {
		out := map[string]interface{}{
			"summary":   result,
			"mode":      "extractive",
			"sentences": *sentences,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
	} else {
		fmt.Println(result)
	}
	return true
}

// --- transform subcommand ---

func runTransform(args []string) bool {
	fs := goflag.NewFlagSet("transform", goflag.ContinueOnError)
	mode := fs.String("mode", "formal", "Transform mode: formal, casual, simple, short, bullet")
	fs.StringVar(mode, "m", "formal", "Transform mode (short)")
	fs.Parse(args)

	text := strings.Join(fs.Args(), " ")
	if text == "" {
		text = readStdin()
	}
	if text == "" {
		fmt.Fprintln(os.Stderr, "usage: nous transform --mode formal|casual|simple|short|bullet <text>")
		os.Exit(1)
	}

	engine := cognitive.NewTextTransformEngine()
	result := engine.Transform(text, *mode)
	fmt.Println(result)
	return true
}

// --- draft subcommand ---

func runDraft(args []string) bool {
	// Extract the document type (positional arg) before flag parsing.
	// Go's flag package stops at the first non-flag argument, so
	// "nous draft email --about foo" would leave --about unparsed
	// if we didn't consume "email" first.
	docType := ""
	flagArgs := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		docType = args[0]
		flagArgs = args[1:]
	}

	fs := goflag.NewFlagSet("draft", goflag.ContinueOnError)
	to := fs.String("to", "", "Recipient")
	about := fs.String("about", "", "Subject/topic")
	tone := fs.String("tone", "formal", "Tone: formal, casual, friendly, urgent")
	facts := fs.String("facts", "", "Key points (comma-separated)")
	attendees := fs.String("attendees", "", "Meeting attendees (comma-separated)")
	decisions := fs.String("decisions", "", "Decisions made (comma-separated)")
	actions := fs.String("actions", "", "Action items (comma-separated)")
	fs.Parse(flagArgs)

	// If docType wasn't a leading positional arg, check remaining args.
	if docType == "" && len(fs.Args()) > 0 {
		docType = fs.Args()[0]
	}
	if docType == "" {
		fmt.Fprintln(os.Stderr, "usage: nous draft <type> [flags]")
		fmt.Fprintln(os.Stderr, "types: email, report, meeting-notes, proposal, status")
		os.Exit(1)
	}

	split := func(s string) []string {
		if s == "" {
			return nil
		}
		parts := strings.Split(s, ",")
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		return parts
	}

	dc := cognitive.NewDocComposer()
	params := cognitive.DocParams{
		Type:      docType,
		To:        *to,
		Subject:   *about,
		Tone:      *tone,
		Facts:     split(*facts),
		Attendees: split(*attendees),
		Decisions: split(*decisions),
		Actions:   split(*actions),
	}

	result := dc.Draft(params)
	fmt.Println(result)
	return true
}

// --- code subcommand ---

func runCode(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, `Usage: nous code <command> [args]

Commands:
  explain  <file>[:<func|line>]   Explain Go source code
  review   <file|dir>             Find code issues (static analysis)
  generate <type> [flags]         Generate code boilerplate
  run      <file.go>              Compile and execute a Go file
  fix      <file.go> [--write]    Auto-fix common Go issues
  test     <package> [-run regex] Run tests with summary
  doc      <file>[:<func>]        Generate doc comments for undocumented code
  deps     <file>:<func>          Show dependency graph (callers/callees)
  diff     [--staged|--commit H]  Explain code changes in natural language`)
		os.Exit(1)
	}

	switch args[0] {
	case "explain":
		return runCodeExplain(args[1:])
	case "review":
		return runCodeReview(args[1:])
	case "generate":
		return runCodeGenerate(args[1:])
	case "run":
		return runCodeRun(args[1:])
	case "fix":
		return runCodeFix(args[1:])
	case "test":
		return runCodeTest(args[1:])
	case "doc":
		return runCodeDoc(args[1:])
	case "deps":
		return runCodeDeps(args[1:])
	case "diff":
		return runCodeDiffCmd(args[1:])
	case "build":
		return runCodeBuild(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "unknown code command: %s\n", args[0])
		os.Exit(1)
	}
	return true
}

func runCodeExplain(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code explain <file>[:<func|line>]")
		os.Exit(1)
	}

	jsonOut := false
	target := ""
	for _, a := range args {
		if a == "--json" || a == "-j" {
			jsonOut = true
		} else if target == "" {
			target = a
		}
	}

	explainer := cognitive.NewCodeExplainer()
	file, selector := target, ""
	if idx := strings.LastIndex(target, ":"); idx > 0 {
		file = target[:idx]
		selector = target[idx+1:]
	}

	src, readErr := os.ReadFile(file)
	if readErr != nil {
		fmt.Fprintf(os.Stderr, "error reading %s: %v\n", file, readErr)
		os.Exit(1)
	}

	var result string

	switch {
	case selector == "":
		// Explain whole file — show all functions
		explained, err := explainer.ExplainSource(string(src))
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("File: %s\n\n", file))
		for _, ef := range explained {
			sb.WriteString(fmt.Sprintf("## %s\n%s\n", ef.Signature, ef.Summary))
			if ef.DataFlow != "" {
				sb.WriteString(fmt.Sprintf("Flow: %s\n", ef.DataFlow))
			}
			sb.WriteString("\n")
		}
		result = sb.String()
	case isDigit(selector):
		line, _ := strconv.Atoi(selector)
		lines := strings.Split(string(src), "\n")
		if line >= 1 && line <= len(lines) {
			result = fmt.Sprintf("Line %d: %s", line, strings.TrimSpace(lines[line-1]))
		} else {
			result = fmt.Sprintf("Line %d out of range (file has %d lines)", line, len(lines))
		}
	default:
		ef, err := explainer.ExplainFunc(string(src))
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		result = fmt.Sprintf("%s\n\n%s", ef.Signature, ef.Summary)
		if ef.DataFlow != "" {
			result += fmt.Sprintf("\n\nFlow: %s", ef.DataFlow)
		}
	}

	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(map[string]string{"explanation": result})
	} else {
		fmt.Println(result)
	}
	return true
}

func isDigit(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

func runCodeReview(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code review <file|dir>")
		os.Exit(1)
	}

	jsonOut := false
	target := ""
	for _, a := range args {
		if a == "--json" || a == "-j" {
			jsonOut = true
		} else if target == "" {
			target = a
		}
	}

	reviewer := &cognitive.CodeReviewer{}
	var findings []cognitive.ReviewResult
	var err error

	info, statErr := os.Stat(target)
	if statErr != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", statErr)
		os.Exit(1)
	}

	if info.IsDir() {
		findings, err = reviewer.ReviewDir(target)
	} else {
		findings, err = reviewer.ReviewFile(target)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	if jsonOut {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(findings)
	} else if len(findings) == 0 {
		fmt.Println("no issues found")
	} else {
		for _, f := range findings {
			sev := "[" + f.Severity + "]"
			fmt.Printf("  %-40s %-10s %-22s %s\n",
				fmt.Sprintf("%s:%d", f.File, f.Line), sev, f.Rule+":", f.Message)
		}
		fmt.Printf("\n  %d issue(s) found\n", len(findings))
	}
	return true
}

func runCodeGenerate(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code generate handler|test|struct [flags]")
		os.Exit(1)
	}

	switch args[0] {
	case "handler":
		return genHandler(args[1:])
	case "test":
		return genTest(args[1:])
	case "struct":
		return genStruct(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "unknown generate type: %s\n", args[0])
		os.Exit(1)
	}
	return true
}

func genHandler(args []string) bool {
	fs := goflag.NewFlagSet("handler", goflag.ContinueOnError)
	name := fs.String("name", "Handler", "Handler function name")
	method := fs.String("method", "GET", "HTTP method")
	path := fs.String("path", "/api/endpoint", "URL path")
	fs.Parse(args)

	fmt.Printf(`// %sHandler handles %s %s requests.
func %sHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "%s" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// TODO: implement %s logic

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
`, *name, *method, *path, *name, *method, *name)
	return true
}

func genTest(args []string) bool {
	fs := goflag.NewFlagSet("test", goflag.ContinueOnError)
	file := fs.String("file", "", "Source file to generate tests for")
	fs.Parse(args)

	if *file == "" {
		fmt.Fprintln(os.Stderr, "usage: nous code generate test --file <file.go>")
		os.Exit(1)
	}

	// Use SmartTestGenerator for signature-based test generation
	stg := &cognitive.SmartTestGenerator{}
	output, err := stg.GenerateTests(*file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Print(output)
	return true
}

func extractFuncName(line string) string {
	// Try to find a function name in the line
	if idx := strings.Index(line, "func "); idx >= 0 {
		rest := line[idx+5:]
		// Skip receiver
		if strings.HasPrefix(rest, "(") {
			end := strings.Index(rest, ")")
			if end >= 0 {
				rest = strings.TrimSpace(rest[end+1:])
			}
		}
		end := strings.IndexAny(rest, "( ")
		if end > 0 {
			return rest[:end]
		}
	}
	// Try "- FuncName" pattern
	if idx := strings.Index(line, "- "); idx >= 0 {
		rest := strings.TrimSpace(line[idx+2:])
		end := strings.IndexAny(rest, " (")
		if end > 0 {
			name := rest[:end]
			if len(name) > 0 && name[0] >= 'A' && name[0] <= 'Z' {
				return name
			}
		}
	}
	return ""
}

func genStruct(args []string) bool {
	fs := goflag.NewFlagSet("struct", goflag.ContinueOnError)
	name := fs.String("name", "MyStruct", "Struct name")
	fields := fs.String("fields", "", "Fields as name:type,name:type")
	fs.Parse(args)

	if *fields == "" {
		fmt.Fprintln(os.Stderr, "usage: nous code generate struct --name Name --fields \"host:string,port:int\"")
		os.Exit(1)
	}

	fmt.Printf("// %s holds configuration.\ntype %s struct {\n", *name, *name)

	var fieldNames []string
	var fieldTypes []string
	for _, f := range strings.Split(*fields, ",") {
		parts := strings.SplitN(strings.TrimSpace(f), ":", 2)
		if len(parts) != 2 {
			continue
		}
		fname := strings.TrimSpace(parts[0])
		ftype := strings.TrimSpace(parts[1])
		// Capitalize field name
		capName := strings.ToUpper(fname[:1]) + fname[1:]
		fmt.Printf("\t%s %s\n", capName, ftype)
		fieldNames = append(fieldNames, capName)
		fieldTypes = append(fieldTypes, ftype)
	}
	fmt.Println("}")

	// Constructor
	fmt.Printf("\n// New%s creates a new %s.\nfunc New%s(", *name, *name, *name)
	for i, fn := range fieldNames {
		if i > 0 {
			fmt.Print(", ")
		}
		// lowercase for param
		paramName := strings.ToLower(fn[:1]) + fn[1:]
		fmt.Printf("%s %s", paramName, fieldTypes[i])
	}
	fmt.Printf(") *%s {\n\treturn &%s{\n", *name, *name)
	for _, fn := range fieldNames {
		paramName := strings.ToLower(fn[:1]) + fn[1:]
		fmt.Printf("\t\t%s: %s,\n", fn, paramName)
	}
	fmt.Println("\t}\n}")

	return true
}

// --- code run subcommand ---

func runCodeRun(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code run <file.go>")
		os.Exit(1)
	}
	file := args[0]
	if _, err := os.Stat(file); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Compiling... ")
	start := time.Now()

	cmd := exec.Command("go", "run", file)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	dur := time.Since(start)

	if err != nil {
		fmt.Printf("failed (%.1fs)\n\n", dur.Seconds())
		errText := stderr.String()
		if errText != "" {
			for _, line := range strings.Split(strings.TrimRight(errText, "\n"), "\n") {
				fmt.Printf("  %s\n", line)
			}
		}
		// Highlight source lines from error
		errLineRe := regexp.MustCompile(`([^:\s]+\.go):(\d+)(?::\d+)?:`)
		src, _ := os.ReadFile(file)
		lines := strings.Split(string(src), "\n")
		for _, m := range errLineRe.FindAllStringSubmatch(errText, 3) {
			if ln, e := strconv.Atoi(m[2]); e == nil && ln >= 1 && ln <= len(lines) {
				fmt.Printf("\n  Line %d: %s\n", ln, strings.TrimSpace(lines[ln-1]))
			}
		}
		exitCode := 1
		if cmd.ProcessState != nil {
			exitCode = cmd.ProcessState.ExitCode()
		}
		fmt.Printf("\nExit: %d (%.1fs)\n", exitCode, dur.Seconds())
	} else {
		fmt.Printf("ok (%.1fs)\n", dur.Seconds())
		if out := stdout.String(); out != "" {
			fmt.Println()
			fmt.Print(out)
		}
		fmt.Printf("\nExit: 0 (%.1fs)\n", dur.Seconds())
	}
	return true
}

// --- code fix subcommand ---

func runCodeFix(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code fix <file.go> [--write]")
		os.Exit(1)
	}

	file := ""
	write := false
	for _, a := range args {
		if a == "--write" || a == "-w" {
			write = true
		} else if file == "" {
			file = a
		}
	}

	src, err := os.ReadFile(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// Run gofmt first
	formatted, fmtErr := goFormat(src)
	var changes []string

	if fmtErr == nil && string(formatted) != string(src) {
		changes = append(changes, "formatted code (gofmt)")
	}

	// Run go vet for additional issues
	cmd := exec.Command("go", "vet", file)
	var vetOut bytes.Buffer
	cmd.Stderr = &vetOut
	cmd.Run()
	if vetText := vetOut.String(); vetText != "" {
		for _, line := range strings.Split(strings.TrimSpace(vetText), "\n") {
			if strings.Contains(line, file) {
				changes = append(changes, "vet: "+line)
			}
		}
	}

	if len(changes) == 0 {
		fmt.Println("no fixes needed")
		return true
	}

	fmt.Printf("Found %d issue(s):\n", len(changes))
	for _, c := range changes {
		fmt.Printf("  - %s\n", c)
	}

	if write && fmtErr == nil && string(formatted) != string(src) {
		os.WriteFile(file, formatted, 0644)
		fmt.Printf("\nWrote %s\n", file)
	} else if !write && fmtErr == nil && string(formatted) != string(src) {
		fmt.Println("\nRun with --write to apply formatting fix")
	}
	return true
}

func goFormat(src []byte) ([]byte, error) {
	// Use go/format from stdlib
	cmd := exec.Command("gofmt")
	cmd.Stdin = bytes.NewReader(src)
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// --- code test subcommand ---

func runCodeTest(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code test <package> [-run regex]")
		os.Exit(1)
	}

	testArgs := []string{"test", "-v", "-count=1"}
	testArgs = append(testArgs, args...)

	cmd := exec.Command("go", testArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	start := time.Now()
	err := cmd.Run()
	dur := time.Since(start)

	fmt.Println()
	if err != nil {
		fmt.Printf("FAILED (%.1fs)\n", dur.Seconds())
	} else {
		fmt.Printf("All tests passed (%.1fs)\n", dur.Seconds())
	}
	return true
}

// --- code doc subcommand ---

func runCodeDoc(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code doc <file>[:<func>] [--write]")
		os.Exit(1)
	}

	write := false
	target := ""
	for _, a := range args {
		if a == "--write" || a == "-w" {
			write = true
		} else if target == "" {
			target = a
		}
	}

	file, funcName := target, ""
	if idx := strings.LastIndex(target, ":"); idx > 0 {
		file = target[:idx]
		funcName = target[idx+1:]
	}

	gen := cognitive.NewAutoDocGenerator()

	if funcName != "" {
		// Document a single function
		doc := gen.GenerateDoc(file, funcName)
		if doc == "" {
			fmt.Println("no documentation needed (already documented or not found)")
		} else {
			fmt.Println(doc)
		}
		return true
	}

	if write {
		// Write docs for entire directory
		info, err := os.Stat(file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		dir := file
		if !info.IsDir() {
			dir = "."
		}
		count, err := gen.DocumentPackage(dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Documented %d function(s)\n", count)
	} else {
		// Show suggested docs as diff
		diff := gen.GenerateDocForFile(file)
		if diff == "" {
			fmt.Println("all exported functions are already documented")
		} else {
			fmt.Print(diff)
			fmt.Println("\nRun with --write to apply")
		}
	}
	return true
}

// --- code deps subcommand ---

func runCodeDeps(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: nous code deps <file>:<funcName> [--impact]")
		os.Exit(1)
	}

	impact := false
	target := ""
	for _, a := range args {
		if a == "--impact" {
			impact = true
		} else if target == "" {
			target = a
		}
	}

	file, funcName := target, ""
	if idx := strings.LastIndex(target, ":"); idx > 0 {
		file = target[:idx]
		funcName = target[idx+1:]
	}
	if funcName == "" {
		fmt.Fprintln(os.Stderr, "usage: nous code deps <file>:<funcName>")
		os.Exit(1)
	}

	// Build dep graph from the file's directory
	info, _ := os.Stat(file)
	dir := "."
	if info != nil && !info.IsDir() {
		dir = file[:strings.LastIndex(file, "/")]
		if dir == "" {
			dir = "."
		}
	}

	graph := cognitive.NewDepGraph()
	if err := graph.Build(dir); err != nil {
		fmt.Fprintf(os.Stderr, "error building dependency graph: %v\n", err)
		os.Exit(1)
	}

	if impact {
		affected := graph.Impact(funcName)
		if len(affected) == 0 {
			fmt.Printf("No transitive callers found for %s\n", funcName)
		} else {
			fmt.Printf("If %s changes, these are affected:\n", funcName)
			for _, a := range affected {
				fmt.Printf("  - %s\n", a)
			}
		}
	} else {
		fmt.Print(graph.Render(funcName, 2))
	}
	return true
}

// --- code diff subcommand ---

func runCodeDiffCmd(args []string) bool {
	staged := false
	commitHash := ""
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--staged":
			staged = true
		case "--commit":
			if i+1 < len(args) {
				i++
				commitHash = args[i]
			}
		}
	}

	var diffText string
	var err error

	if commitHash != "" {
		cmd := exec.Command("git", "show", commitHash)
		out, e := cmd.Output()
		diffText, err = string(out), e
	} else if staged {
		cmd := exec.Command("git", "diff", "--staged")
		out, e := cmd.Output()
		diffText, err = string(out), e
	} else {
		// Check stdin first
		piped := readStdin()
		if piped != "" {
			diffText = piped
		} else {
			cmd := exec.Command("git", "diff")
			out, e := cmd.Output()
			diffText, err = string(out), e
		}
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error getting diff: %v\n", err)
		os.Exit(1)
	}

	if strings.TrimSpace(diffText) == "" {
		fmt.Println("no changes to explain")
		return true
	}

	explainer := &cognitive.DiffExplainer{}
	result := explainer.ExplainDiff(diffText)

	fmt.Printf("Intent:   %s\n", result.Intent)
	fmt.Printf("Risk:     %s\n", result.Risk)
	if result.Breaking {
		fmt.Printf("Breaking: YES\n")
	}
	fmt.Println()
	fmt.Println(result.Summary)

	if len(result.Files) > 0 {
		fmt.Println()
		fmt.Println("Files:")
		for _, f := range result.Files {
			fmt.Printf("  %-50s +%d -%d  %s\n", f.Path, f.Added, f.Removed, f.Description)
		}
	}

	return true
}

// --- code build subcommand ---

func runCodeBuild(args []string) bool {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, `Usage: nous code build "<description>"

Examples:
  nous code build "REST API for managing todos"
  nous code build "CLI tool for file management"
  nous code build "worker pool for image processing"`)
		os.Exit(1)
	}

	description := strings.Join(args, " ")
	description = strings.Trim(description, "\"'")

	// Parse to get project name for output dir
	tmpAgent := agent.NewCodeAgent("")
	plan := tmpAgent.ParseRequest(description)
	outputDir := "./" + plan.ProjectName

	ca := agent.NewCodeAgent(outputDir)
	fmt.Printf("Building: %s\nOutput:   %s/\n\n", description, outputDir)

	result, err := ca.Build(description)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nSummary: %d files, compiled=%v, tests=%d/%d passed (%s)\n",
		len(result.Files), result.Compiled,
		result.TestsPassed, result.TestsPassed+result.TestsFailed,
		result.Duration.Round(time.Millisecond))
	return true
}

// --- simulate subcommand ---

func runSimulate(args []string, actions *cognitive.ActionRouter) bool {
	fs := goflag.NewFlagSet("simulate", goflag.ContinueOnError)
	steps := fs.Int("steps", 5, "Number of simulation steps (1-10)")
	removal := fs.Bool("remove", false, "Simulate removal of an entity")
	fs.Parse(args)

	scenario := strings.Join(fs.Args(), " ")
	if scenario == "" {
		scenario = readStdin()
	}
	if scenario == "" {
		fmt.Fprintln(os.Stderr, "usage: nous simulate [--steps N] [--remove] <scenario>")
		fmt.Fprintln(os.Stderr, "examples:")
		fmt.Fprintln(os.Stderr, "  nous simulate \"What if renewable energy becomes cheaper than coal?\"")
		fmt.Fprintln(os.Stderr, "  nous simulate --remove \"gravity\"")
		fmt.Fprintln(os.Stderr, "  nous simulate --steps 3 \"What if Einstein never published relativity?\"")
		os.Exit(1)
	}

	// Build simulation engine from action router components.
	sim := cognitive.NewSimulationEngine(
		actions.CogGraph,
		actions.CausalReasoner,
		actions.Council,
		actions.MultiHop,
	)
	if actions.SelfTeacher != nil {
		sim.KnowledgeDir = actions.SelfTeacher.KnowledgeDir()
	}

	var result *cognitive.SimulationResult
	if *removal {
		result = sim.SimulateRemoval(scenario)
	} else {
		result = sim.Simulate(scenario, *steps)
	}

	if result == nil {
		fmt.Fprintln(os.Stderr, "Could not simulate: topic not found in knowledge graph.")
		os.Exit(1)
	}

	fmt.Println(result.Report)
	return true
}

// --- expand subcommand ---

func runExpand(args []string, actions *cognitive.ActionRouter) bool {
	fs := goflag.NewFlagSet("expand", goflag.ContinueOnError)
	generations := fs.Int("generations", 3, "Max expansion generations (1-5)")
	dryRun := fs.Bool("dry-run", false, "Only discover frontier, don't fill gaps")
	fs.Parse(args)

	if actions.Expander == nil {
		fmt.Fprintln(os.Stderr, "Knowledge expander not available. Requires knowledge graph and SelfTeacher.")
		os.Exit(1)
	}

	if *dryRun {
		fmt.Println("Discovering frontier topics...")
		frontier := actions.Expander.DiscoverFrontier()
		fmt.Printf("\nFound %d frontier topics (mentioned but not well-covered):\n\n", len(frontier))
		limit := 50
		if len(frontier) < limit {
			limit = len(frontier)
		}
		for i, ft := range frontier[:limit] {
			status := "new"
			if ft.HasNode {
				status = fmt.Sprintf("%d edges", ft.EdgeCount)
			}
			fmt.Printf("  %3d. %-40s (mentioned %dx, %s, priority %.1f)\n",
				i+1, ft.Name, ft.Mentions, status, ft.Priority)
		}
		if len(frontier) > limit {
			fmt.Printf("\n  ... and %d more\n", len(frontier)-limit)
		}
		return true
	}

	fmt.Printf("Running knowledge expansion (up to %d generations)...\n\n", *generations)
	reports := actions.Expander.Expand(*generations)
	fmt.Print(cognitive.FormatExpansionReport(reports))

	// Run causal inference after expansion to discover new causal edges.
	if actions.CausalInfer != nil {
		fmt.Println("\nRunning causal inference on expanded graph...")
		inferReport := actions.CausalInfer.InferAll()
		fmt.Printf("Inferred %d new causal edges (%d temporal, %d dependency, %d inhibition, %d production)\n",
			inferReport.AddedCount, inferReport.TemporalCount, inferReport.DependencyCount,
			inferReport.InhibitionCount, inferReport.ProductionCount)
	}

	// Save the updated graph.
	if actions.CogGraph != nil {
		actions.CogGraph.Save()
		fmt.Println("\nKnowledge graph saved.")
	}

	return true
}

// --- infer subcommand ---

func runInfer(args []string, actions *cognitive.ActionRouter) bool {
	if actions.CausalInfer == nil {
		fmt.Fprintln(os.Stderr, "Causal inference engine not available. Requires knowledge graph.")
		os.Exit(1)
	}

	fmt.Println("Running structural causal inference...")
	report := actions.CausalInfer.InferAll()

	fmt.Printf("\n# Causal Inference Report\n\n")
	fmt.Printf("- Temporal ordering edges: %d\n", report.TemporalCount)
	fmt.Printf("- Dependency chain edges:  %d\n", report.DependencyCount)
	fmt.Printf("- Inhibition edges:        %d\n", report.InhibitionCount)
	fmt.Printf("- Production chain edges:  %d\n", report.ProductionCount)
	fmt.Printf("- **Total new edges added: %d**\n\n", report.AddedCount)

	if len(report.Edges) > 0 {
		fmt.Println("Sample inferred edges:")
		limit := 20
		if len(report.Edges) < limit {
			limit = len(report.Edges)
		}
		for _, e := range report.Edges[:limit] {
			fmt.Printf("  %s —[%s]→ %s (conf=%.2f) %s\n",
				e.From, e.Relation, e.To, e.Confidence, e.Reason)
		}
	}

	if actions.CogGraph != nil {
		actions.CogGraph.Save()
		fmt.Println("\nKnowledge graph saved.")
	}

	return true
}

// --- dream subcommand ---

func runDream(args []string, actions *cognitive.ActionRouter) bool {
	cycles := 50
	if len(args) > 0 {
		if n, err := fmt.Sscanf(args[0], "%d", &cycles); n == 0 || err != nil {
			cycles = 50
		}
	}

	dream := cognitive.NewDreamEngine(
		actions.CogGraph,
		actions.EpisodicMem,
		actions.CausalInfer,
		actions.WikiLoader,
		actions.Expander,
	)

	fmt.Printf("Dreaming (%d cycles)...\n\n", cycles)
	report := dream.Dream(cycles)
	if report == nil {
		fmt.Println("Dream engine not available.")
		return true
	}

	fmt.Print(cognitive.FormatDreamReport(report))

	if actions.CogGraph != nil {
		actions.CogGraph.Save()
	}
	return true
}

// --- research subcommand ---

func runResearch(args []string, actions *cognitive.ActionRouter) bool {
	depth := "standard"
	remaining := args

	if len(args) >= 2 && args[0] == "--depth" {
		depth = args[1]
		remaining = args[2:]
	}

	topic := strings.Join(remaining, " ")
	if topic == "" {
		fmt.Fprintln(os.Stderr, "usage: nous research [--depth quick|standard|deep] <topic>")
		os.Exit(1)
	}

	agent := cognitive.NewDeepResearchAgent(
		actions.CogGraph,
		actions.WikiLoader,
		actions.CausalInfer,
		actions.MultiHop,
	)

	fmt.Printf("Researching %q (depth: %s)...\n\n", topic, depth)
	result := agent.Research(topic, depth)
	if result == nil {
		fmt.Println("Research failed.")
		return true
	}

	fmt.Print(result.Report)

	if actions.CogGraph != nil {
		actions.CogGraph.Save()
	}
	return true
}
