package main

import (
	"compress/bzip2"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/artaeon/nous/internal/cognitive"
)

const (
	defaultBatchSize = 500
	dumpURL          = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
)

func main() {
	dumpPath := flag.String("dump", "", "Path to simplewiki-*-pages-articles.xml.bz2")
	outputDir := flag.String("output", "./packages/wiki/", "Output directory for package JSON files")
	batchSize := flag.Int("batch", defaultBatchSize, "Articles per package file")
	limit := flag.Int("limit", 0, "Max articles to process (0 = all)")
	download := flag.Bool("download", false, "Download Simple English Wikipedia dump first")
	flag.Parse()

	// Download if requested
	if *download {
		dest := "simplewiki-latest-pages-articles.xml.bz2"
		if *dumpPath != "" {
			dest = *dumpPath
		}
		fmt.Printf("Downloading Simple English Wikipedia dump...\n")
		fmt.Printf("  URL: %s\n", dumpURL)
		if err := downloadDump(dumpURL, dest); err != nil {
			fmt.Fprintf(os.Stderr, "download failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("  Saved to: %s\n", dest)
		*dumpPath = dest
	}

	if *dumpPath == "" {
		fmt.Fprintf(os.Stderr, "Error: --dump path required (or use --download)\n")
		flag.Usage()
		os.Exit(1)
	}

	// Create output directory
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "create output dir: %v\n", err)
		os.Exit(1)
	}

	// Open dump file
	f, err := os.Open(*dumpPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open dump: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// Decompress bzip2
	bzReader := bzip2.NewReader(f)

	stats := &cognitive.WikiImportStats{
		StartTime: time.Now(),
	}

	var currentFacts []cognitive.PackageFact
	batchNum := 1
	articlesInBatch := 0

	// Sentence corpus for Layer 2 retrieval-based generation.
	corpus := cognitive.NewSentenceCorpus()
	// Discourse corpus for Layer 2b — sentences indexed by discourse function.
	discCorpus := cognitive.NewDiscourseCorpus()

	fmt.Printf("Processing Wikipedia dump: %s\n", *dumpPath)
	fmt.Printf("Output: %s | Batch size: %d", *outputDir, *batchSize)
	if *limit > 0 {
		fmt.Printf(" | Limit: %d", *limit)
	}
	fmt.Println()
	fmt.Println()

	err = cognitive.ParseWikiDump(bzReader, func(article cognitive.WikiArticle) error {
		// Check limit
		if *limit > 0 && stats.ArticlesProcessed >= *limit {
			return fmt.Errorf("limit reached")
		}

		// Extract facts
		facts := cognitive.ArticleToFacts(article.Title, article.Text)
		if len(facts) == 0 {
			stats.ArticlesSkipped++
			return nil
		}

		// Extract sentence exemplars for Layer 2 corpus.
		exemplars := cognitive.ArticleToExemplars(article.Title, article.Text)
		for _, ex := range exemplars {
			corpus.Add(ex)
		}
		// Extract discourse-typed sentences for Layer 2b.
		discSents := cognitive.ExtractDiscourseSentences(article.Title, article.Text)
		for _, ds := range discSents {
			discCorpus.Add(ds)
		}

		stats.ArticlesProcessed++
		stats.FactsExtracted += len(facts)
		currentFacts = append(currentFacts, facts...)
		articlesInBatch++

		// Write batch when full
		if articlesInBatch >= *batchSize {
			if err := writeBatch(*outputDir, batchNum, currentFacts); err != nil {
				return fmt.Errorf("write batch %d: %w", batchNum, err)
			}
			stats.PackagesWritten++

			// Progress
			fmt.Printf("  [batch %04d] %d facts | total: %s\n",
				batchNum, len(currentFacts), stats)

			batchNum++
			currentFacts = nil
			articlesInBatch = 0
		}

		// Periodic progress for large dumps
		if stats.ArticlesProcessed%1000 == 0 {
			fmt.Printf("  ... %s\n", stats)
		}

		return nil
	})

	// Handle "limit reached" as non-error
	if err != nil && err.Error() != "limit reached" {
		fmt.Fprintf(os.Stderr, "parse error: %v\n", err)
		os.Exit(1)
	}

	// Write remaining facts
	if len(currentFacts) > 0 {
		if err := writeBatch(*outputDir, batchNum, currentFacts); err != nil {
			fmt.Fprintf(os.Stderr, "write final batch: %v\n", err)
			os.Exit(1)
		}
		stats.PackagesWritten++
		fmt.Printf("  [batch %04d] %d facts (final)\n", batchNum, len(currentFacts))
	}

	// Save sentence corpus for Layer 2 retrieval.
	corpusPath := filepath.Join(*outputDir, "sentence_corpus.json")
	if corpus.Size() > 0 {
		if err := corpus.Save(corpusPath); err != nil {
			fmt.Fprintf(os.Stderr, "save sentence corpus: %v\n", err)
		} else {
			fmt.Printf("\n  Sentence corpus: %d exemplars → %s\n", corpus.Size(), corpusPath)
			for rel, count := range corpus.RelationCounts() {
				if count > 100 {
					fmt.Printf("    %-14s %d sentences\n", rel, count)
				}
			}
		}
	}

	// Save discourse corpus for Layer 2b.
	discPath := filepath.Join(*outputDir, "discourse_corpus.json")
	if discCorpus.Size() > 0 {
		if err := discCorpus.Save(discPath); err != nil {
			fmt.Fprintf(os.Stderr, "save discourse corpus: %v\n", err)
		} else {
			fmt.Printf("\n  Discourse corpus: %d sentences → %s\n", discCorpus.Size(), discPath)
			for fn, count := range discCorpus.FunctionCounts() {
				if count > 100 {
					fmt.Printf("    %-14s %d sentences\n", fn, count)
				}
			}
		}
	}

	// Summary
	fmt.Println()
	fmt.Println("=== Import Complete ===")
	fmt.Printf("  Articles processed: %d\n", stats.ArticlesProcessed)
	fmt.Printf("  Articles skipped:   %d\n", stats.ArticlesSkipped)
	fmt.Printf("  Facts extracted:    %d\n", stats.FactsExtracted)
	fmt.Printf("  Packages written:   %d\n", stats.PackagesWritten)
	fmt.Printf("  Sentence exemplars: %d\n", corpus.Size())
	fmt.Printf("  Output directory:   %s\n", *outputDir)
	fmt.Printf("  Elapsed:            %s\n", time.Since(stats.StartTime).Round(time.Millisecond))
}

func writeBatch(outputDir string, batchNum int, facts []cognitive.PackageFact) error {
	pkg := cognitive.BatchToPackage(batchNum, "wikipedia", facts)

	data, err := json.MarshalIndent(pkg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	filename := filepath.Join(outputDir, fmt.Sprintf("wiki-batch-%04d.json", batchNum))
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", filename, err)
	}

	return nil
}

func downloadDump(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("http get: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("http status %d", resp.StatusCode)
	}

	out, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer out.Close()

	size, err := io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}

	fmt.Printf("  Downloaded %.1f MB\n", float64(size)/1024/1024)
	return nil
}
