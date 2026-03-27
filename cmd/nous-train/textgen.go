package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/cognitive"
)

// trainTextGen trains the GRU text generation model from knowledge graph data.
// It loads all knowledge packages into a cognitive graph, generates training
// examples from the edges, then trains the model via BPTT.
//
// Usage: nous-train textgen [-o model.bin] [-epochs 100] [-lr 0.01] [-large]
func trainTextGen(args []string) {
	fs := flag.NewFlagSet("textgen", flag.ExitOnError)
	output := fs.String("o", "nous-textgen.bin", "Output path for the trained model")
	packDir := fs.String("packages", "packages", "Directory containing knowledge packages")
	epochs := fs.Int("epochs", 100, "Number of training epochs")
	lr := fs.Float64("lr", 0.003, "Initial learning rate")
	large := fs.Bool("large", false, "Use large config (hiddenDim=512, embedDim=128)")
	augment := fs.Int("augment", 3, "Data augmentation factor (1=no augmentation)")
	maxCorpus := fs.Int("max-corpus", 50000, "Max examples from sentence/discourse corpus")
	fs.Parse(args)

	fmt.Println("═══ Nous Text Generation Model — GRU Training ═══")
	start := time.Now()

	// 1. Load sentence and discourse corpora (real Wikipedia sentences).
	// These are the PRIMARY training data — human-written text, not templates.
	var sc *cognitive.SentenceCorpus
	var dc *cognitive.DiscourseCorpus

	scPath := filepath.Join(*packDir, "wiki", "sentence_corpus.json")
	sc = cognitive.NewSentenceCorpus()
	if err := sc.Load(scPath); err == nil && sc.Size() > 0 {
		fmt.Printf("Loaded sentence corpus: %d exemplars\n", sc.Size())
	} else {
		sc = nil
		fmt.Println("No sentence corpus found — using graph-only training")
	}

	dcPath := filepath.Join(*packDir, "wiki", "discourse_corpus.json")
	dc = cognitive.NewDiscourseCorpus()
	if err := dc.Load(dcPath); err == nil && dc.Size() > 0 {
		fmt.Printf("Loaded discourse corpus: %d sentences\n", dc.Size())
	} else {
		dc = nil
	}

	// 2. Generate training examples from corpus (real Wikipedia sentences)
	var examples []cognitive.TextGenExample
	if sc != nil || dc != nil {
		fmt.Printf("Generating corpus examples (max %d)... ", *maxCorpus)
		examples = cognitive.GenerateTextGenFromCorpus(sc, dc, *maxCorpus)
		fmt.Printf("%d examples from real Wikipedia text\n", len(examples))
	}

	// 3. Supplement with graph-generated examples if corpus is small
	if len(examples) < 1000 {
		fmt.Print("Loading knowledge packages for supplemental data... ")
		graph := cognitive.NewCognitiveGraph("")
		engine := cognitive.NewGenerativeEngine()
		composer := cognitive.NewComposer(graph, nil, nil, nil)
		loader := cognitive.NewPackageLoader(graph, engine, composer, *packDir)
		loader.MaxStartupFacts = 0
		loader.LoadAll()

		graphExamples := cognitive.GenerateTextGenTrainingData(graph)
		fmt.Printf("%d graph examples\n", len(graphExamples))
		examples = append(examples, graphExamples...)
	}

	fmt.Printf("Total training examples: %d\n", len(examples))

	if len(examples) == 0 {
		fmt.Fprintln(os.Stderr, "No training examples generated — check your packages directory")
		os.Exit(1)
	}

	// 3. Data augmentation — create variations of each example
	if *augment > 1 {
		fmt.Printf("Augmenting data (%dx)... ", *augment)
		augmented := augmentExamples(examples, *augment)
		fmt.Printf("%d → %d examples\n", len(examples), len(augmented))
		examples = augmented
	}

	// 4. Choose model config
	var cfg cognitive.TextGenConfig
	if *large {
		cfg = cognitive.DefaultTextGenConfig()
		fmt.Printf("Model: large (embed=%d, hidden=%d, cond=%d)\n", cfg.EmbedDim, cfg.HiddenDim, cfg.CondDim)
	} else {
		cfg = cognitive.SmallTextGenConfig()
		fmt.Printf("Model: small (embed=%d, hidden=%d, cond=%d)\n", cfg.EmbedDim, cfg.HiddenDim, cfg.CondDim)
	}

	// Calculate parameter count
	inputDim := cfg.EmbedDim + cfg.CondDim + cfg.HiddenDim
	params := cfg.VocabSize*cfg.EmbedDim + // Embed
		cfg.NumRels*cfg.CondDim + // RelEmbed
		3*inputDim*cfg.HiddenDim + // Wr, Wz, Wn
		3*cfg.HiddenDim + // Br, Bz, Bn
		cfg.HiddenDim*cfg.VocabSize + // Wout
		cfg.VocabSize // Bout
	fmt.Printf("Parameters: %d (%.1f KB weights)\n", params, float64(params*4)/1024)

	// 5. Build vocabulary and train
	model := cognitive.NewTextGenModel(cfg)
	model.BuildVocab(examples)
	fmt.Printf("Vocabulary: %d words\n", model.Config.VocabSize)

	// Recalculate params with actual vocab size
	cfg = model.Config
	inputDim = cfg.EmbedDim + cfg.CondDim + cfg.HiddenDim
	params = cfg.VocabSize*cfg.EmbedDim +
		cfg.NumRels*cfg.CondDim +
		3*inputDim*cfg.HiddenDim +
		3*cfg.HiddenDim +
		cfg.HiddenDim*cfg.VocabSize +
		cfg.VocabSize
	fmt.Printf("Parameters: %d (%.1f KB weights)\n", params, float64(params*4)/1024)

	fmt.Printf("Training for %d epochs with lr=%.4f...\n", *epochs, *lr)
	result := model.Train(examples, *epochs, float32(*lr))

	fmt.Printf("\nTraining complete:\n")
	fmt.Printf("  Final loss: %.4f\n", result.FinalLoss)
	fmt.Printf("  Tokens:     %d\n", result.TotalTokens)
	fmt.Printf("  Duration:   %s\n", time.Since(start).Round(time.Millisecond))

	// 6. Save model
	if err := model.Save(*output); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to save model: %v\n", err)
		os.Exit(1)
	}

	info, _ := os.Stat(*output)
	fmt.Printf("  Model file: %s (%d bytes, %.1f KB)\n", *output, info.Size(), float64(info.Size())/1024)

	// 7. Sample a few generations to show quality
	fmt.Println("\n═══ Sample Generations ═══")
	samples := []struct {
		subj string
		rel  cognitive.RelType
		obj  string
	}{
		{"Albert Einstein", cognitive.RelIsA, "physicist"},
		{"Python", cognitive.RelUsedFor, "web development"},
		{"Tokyo", cognitive.RelLocatedIn, "Japan"},
		{"Linux", cognitive.RelCreatedBy, "Linus Torvalds"},
		{"Mathematics", cognitive.RelHas, "algebra"},
	}

	for _, s := range samples {
		greedy := model.Generate(s.subj, s.rel, s.obj, 0)
		sampled := model.Generate(s.subj, s.rel, s.obj, 0.7)
		fmt.Printf("  [%s, %s, %s]\n", s.subj, s.rel, s.obj)
		fmt.Printf("    greedy:  %s\n", greedy)
		fmt.Printf("    temp=0.7: %s\n\n", sampled)
	}
}

// augmentExamples creates variations of training examples.
func augmentExamples(examples []cognitive.TextGenExample, factor int) []cognitive.TextGenExample {
	result := make([]cognitive.TextGenExample, 0, len(examples)*factor)
	result = append(result, examples...)

	for aug := 1; aug < factor; aug++ {
		for _, ex := range examples {
			var variant cognitive.TextGenExample
			variant.Subject = ex.Subject
			variant.Relation = ex.Relation
			variant.Object = ex.Object

			switch aug % 3 {
			case 0:
				variant.Target = rephraseWithPronoun(ex)
			case 1:
				variant.Target = rephraseReorder(ex)
			case 2:
				variant.Target = rephraseWithAdverb(ex)
			}

			if variant.Target != "" && variant.Target != ex.Target {
				result = append(result, variant)
			}
		}
	}

	return result
}

func rephraseWithPronoun(ex cognitive.TextGenExample) string {
	switch ex.Relation {
	case cognitive.RelIsA:
		return "It is a " + ex.Object + "."
	case cognitive.RelLocatedIn:
		return "It is located in " + ex.Object + "."
	case cognitive.RelUsedFor:
		return "It is used for " + ex.Object + "."
	case cognitive.RelHas:
		return ex.Subject + " has " + ex.Object + "."
	case cognitive.RelCreatedBy:
		return "It was created by " + ex.Object + "."
	default:
		return ""
	}
}

func rephraseReorder(ex cognitive.TextGenExample) string {
	switch ex.Relation {
	case cognitive.RelCreatedBy:
		return ex.Object + " created " + ex.Subject + "."
	case cognitive.RelFoundedBy:
		return ex.Object + " founded " + ex.Subject + "."
	case cognitive.RelLocatedIn:
		return "In " + ex.Object + ", you can find " + ex.Subject + "."
	default:
		return ""
	}
}

func rephraseWithAdverb(ex cognitive.TextGenExample) string {
	if len(ex.Target) == 0 {
		return ""
	}
	adverbs := []string{"Notably, ", "Interestingly, ", "In particular, ", "Specifically, "}
	adv := adverbs[len(ex.Subject)%len(adverbs)]
	return adv + strings.ToLower(ex.Target[:1]) + ex.Target[1:]
}
