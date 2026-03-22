package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/artaeon/nous/internal/cognitive"
)

func main() {
	inputDir := flag.String("input", "./packages/wiki/", "Directory containing wiki-batch-*.json files")
	outputPath := flag.String("output", "./packages/templates/wiki-templates.json", "Output path for mined templates")
	minFreq := flag.Int("min-freq", 3, "Minimum frequency threshold for templates")
	verbose := flag.Bool("v", false, "Verbose output")
	flag.Parse()

	// Find all wiki batch files
	pattern := filepath.Join(*inputDir, "wiki-batch-*.json")
	files, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error globbing %s: %v\n", pattern, err)
		os.Exit(1)
	}
	if len(files) == 0 {
		fmt.Fprintf(os.Stderr, "No wiki batch files found at %s\n", pattern)
		os.Exit(1)
	}
	sort.Strings(files)

	fmt.Printf("Found %d wiki batch files\n", len(files))

	miner := cognitive.NewTemplateMiner()

	totalSentences := 0
	totalMined := 0

	// Process each batch file
	for i, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", f, err)
			continue
		}

		var batch struct {
			Facts []struct {
				S string `json:"s"`
				R string `json:"r"`
				O string `json:"o"`
			} `json:"facts"`
		}
		if err := json.Unmarshal(data, &batch); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing %s: %v\n", f, err)
			continue
		}

		batchMined := 0
		for _, fact := range batch.Facts {
			if fact.R != "described_as" {
				continue
			}
			totalSentences++

			rel, tmpl, ok := miner.ProcessSentence(fact.O, fact.S)
			if ok {
				miner.AddTemplate(rel, tmpl)
				totalMined++
				batchMined++
				if *verbose {
					fmt.Printf("  [%s] %s\n    → %s\n", rel, truncate(fact.O, 80), tmpl)
				}
			}
		}

		if (i+1)%100 == 0 || i == len(files)-1 {
			fmt.Printf("  Processed %d/%d files (%d sentences, %d templates mined)\n",
				i+1, len(files), totalSentences, totalMined)
		}
		_ = batchMined
	}

	// Export with frequency threshold
	result := miner.Export(*minFreq)

	// Print summary
	fmt.Printf("\nResults (min frequency = %d):\n", *minFreq)
	totalTemplates := 0
	for relStr, tmpls := range result.Templates {
		fmt.Printf("  %-15s: %d templates\n", relStr, len(tmpls))
		totalTemplates += len(tmpls)
		if *verbose {
			// Show top 5 by frequency
			sort.Slice(tmpls, func(i, j int) bool { return tmpls[i].Freq > tmpls[j].Freq })
			for j, t := range tmpls {
				if j >= 5 {
					break
				}
				fmt.Printf("    [%4d] %s\n", t.Freq, t.Pattern)
			}
		}
	}
	fmt.Printf("  Total: %d unique templates from %d sentences\n", totalTemplates, totalSentences)

	// Write output
	outData, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling output: %v\n", err)
		os.Exit(1)
	}

	if err := os.MkdirAll(filepath.Dir(*outputPath), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*outputPath, outData, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing %s: %v\n", *outputPath, err)
		os.Exit(1)
	}

	fmt.Printf("\nWritten to %s\n", *outputPath)
}

func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
