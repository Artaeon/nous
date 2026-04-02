// nous-train trains Nous neural models and saves them to disk.
// Used by CI/CD to produce model artifacts for releases.
//
// Usage:
//
//	nous-train              — train the neural intent classifier (default)
//	nous-train nlu          — same as above
//	nous-train textgen      — train the GRU text generation model
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/artaeon/nous/internal/cognitive"
	"github.com/artaeon/nous/internal/micromodel"
)

func main() {
	// Check for subcommand
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "textgen":
			trainTextGen(os.Args[2:])
			return
		case "micromodel":
			trainMicroModel(os.Args[2:])
			return
		case "mamba":
			trainMamba(os.Args[2:])
			return
		case "nlu":
			os.Args = append(os.Args[:1], os.Args[2:]...)
			// fall through to default NLU training
		case "help", "-h", "--help":
			fmt.Println("Usage: nous-train [command] [flags]")
			fmt.Println()
			fmt.Println("Commands:")
			fmt.Println("  nlu         Train the neural intent classifier (default)")
			fmt.Println("  textgen     Train the GRU text generation model")
			fmt.Println("  micromodel  Train the micro language model (transformer)")
			fmt.Println("  mamba       Train the Mamba language model (recommended)")
			fmt.Println()
			fmt.Println("NLU flags:")
			fmt.Println("  -o path   Output path (default: nous-neural.bin)")
			fmt.Println()
			fmt.Println("Mamba flags:")
			fmt.Println("  -knowledge dir  Knowledge text directory (default: knowledge)")
			fmt.Println("  -o path         Output path (default: mamba.bin)")
			fmt.Println("  -epochs n       Training epochs (default: 50)")
			fmt.Println("  -lr rate        Learning rate (default: 0.001)")
			fmt.Println("  -small          Use small config (faster, for testing)")
			return
		}
	}

	// Default: train NLU
	trainNLU()
}

func trainMamba(args []string) {
	fs := flag.NewFlagSet("mamba", flag.ExitOnError)
	knowledgeDir := fs.String("knowledge", "knowledge", "Knowledge text directory")
	output := fs.String("o", "mamba.bin", "Output model path")
	epochs := fs.Int("epochs", 50, "Training epochs")
	lr := fs.Float64("lr", 0.001, "Learning rate")
	small := fs.Bool("small", false, "Use small config (faster, for testing)")
	fs.Parse(args)

	// Generate training pairs — triples + freeform paragraphs
	fmt.Println("Generating training data...")
	pairs := micromodel.GenerateTrainingPairs(*knowledgeDir)
	triplePairs := len(pairs)
	fmt.Printf("  Triple pairs: %d\n", triplePairs)

	// Add freeform paragraph pairs for paragraph-level generation
	paraPairs := micromodel.ExtractParagraphPairs(*knowledgeDir, 600)
	pairs = append(pairs, paraPairs...)
	fmt.Printf("  Paragraph pairs: %d\n", len(paraPairs))
	fmt.Printf("  Total: %d training pairs\n", len(pairs))

	if len(pairs) == 0 {
		fmt.Fprintln(os.Stderr, "No training data found")
		os.Exit(1)
	}

	// Build vocabulary
	var texts []string
	for _, p := range pairs {
		texts = append(texts, p.Input, p.Target)
	}

	cfg := micromodel.DefaultMambaConfig()
	if *small {
		cfg = micromodel.SmallMambaConfig()
	}

	tok := micromodel.NewTokenizer()
	tok.BuildVocab(texts, cfg.VocabSize)
	cfg.VocabSize = tok.VocabSize()
	fmt.Printf("  Vocabulary: %d tokens\n", cfg.VocabSize)

	// Create Mamba model
	model := micromodel.NewMambaModel(cfg)
	model.Tok = tok

	fmt.Printf("  Architecture: Mamba (selective state spaces)\n")
	fmt.Printf("  Parameters:   %d (%.1f MB)\n", model.ParamCount(), float64(model.ParamCount())*4/1024/1024)
	fmt.Printf("  Layers: %d, dim: %d, state: %d, conv: %d, expand: %d\n",
		cfg.NumLayers, cfg.ModelDim, cfg.StateDim, cfg.ConvDim, cfg.Expand)
	fmt.Printf("\nTraining for %d epochs (lr=%.4f)...\n", *epochs, *lr)

	result := model.Train(pairs, *epochs, float32(*lr))

	fmt.Printf("\nTraining complete:\n")
	fmt.Printf("  Epochs:     %d\n", result.Epochs)
	fmt.Printf("  Final loss: %.4f\n", result.FinalLoss)
	fmt.Printf("  Duration:   %s\n", result.Duration.Round(time.Second))

	if err := model.Save(*output); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to save: %v\n", err)
		os.Exit(1)
	}

	info, _ := os.Stat(*output)
	fmt.Printf("  Model size: %.1f MB\n", float64(info.Size())/1024/1024)
	fmt.Printf("  Output:     %s\n", *output)

	// Show example generations
	fmt.Println("\nExample generations:")
	examples := [][3]string{
		{"quantum mechanics", "is_a", "branch of physics"},
		{"Bitcoin", "created_by", "Satoshi Nakamoto"},
		{"Python", "used_for", "web development"},
		{"DNA", "is_a", "molecule"},
		{"Google", "founded_in", "1998"},
	}
	for _, e := range examples {
		sent := model.Generate(e[0], e[1], e[2], 30, 0.7)
		fmt.Printf("  (%s, %s, %s) → %s\n", e[0], e[1], e[2], sent)
	}
}

func trainMicroModel(args []string) {
	fs := flag.NewFlagSet("micromodel", flag.ExitOnError)
	knowledgeDir := fs.String("knowledge", "knowledge", "Knowledge text directory")
	output := fs.String("o", "micromodel.bin", "Output model path")
	epochs := fs.Int("epochs", 50, "Training epochs")
	lr := fs.Float64("lr", 0.001, "Learning rate")
	small := fs.Bool("small", false, "Use small config (faster, for testing)")
	fs.Parse(args)

	// Generate training pairs
	fmt.Println("Generating training data...")
	pairs := micromodel.GenerateTrainingPairs(*knowledgeDir)
	fmt.Printf("  Extracted %d pairs from knowledge corpus\n", len(pairs))

	fmt.Printf("  Total: %d training pairs\n", len(pairs))

	if len(pairs) == 0 {
		fmt.Fprintln(os.Stderr, "No training data found")
		os.Exit(1)
	}

	// Build vocabulary from all texts
	var texts []string
	for _, p := range pairs {
		texts = append(texts, p.Input, p.Target)
	}

	cfg := micromodel.DefaultConfig()
	if *small {
		cfg = micromodel.SmallConfig()
	}

	tok := micromodel.NewTokenizer()
	tok.BuildVocab(texts, cfg.VocabSize)
	cfg.VocabSize = tok.VocabSize()
	fmt.Printf("  Vocabulary: %d tokens\n", cfg.VocabSize)

	// Create and initialize model
	model := micromodel.NewMicroModel(cfg)
	model.Tok = tok

	fmt.Printf("  Parameters: %d (%.1f MB)\n", model.ParamCount(), float64(model.ParamCount())*4/1024/1024)
	fmt.Printf("\nTraining for %d epochs (lr=%.4f)...\n", *epochs, *lr)

	result := model.Train(pairs, *epochs, float32(*lr))

	fmt.Printf("\nTraining complete:\n")
	fmt.Printf("  Epochs:     %d\n", result.Epochs)
	fmt.Printf("  Final loss: %.4f\n", result.FinalLoss)
	fmt.Printf("  Duration:   %s\n", result.Duration.Round(time.Second))

	// Save
	if err := model.Save(*output); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to save: %v\n", err)
		os.Exit(1)
	}

	info, _ := os.Stat(*output)
	fmt.Printf("  Model size: %.1f MB\n", float64(info.Size())/1024/1024)
	fmt.Printf("  Output:     %s\n", *output)

	// Show a few example generations
	fmt.Println("\nExample generations:")
	examples := [][3]string{
		{"quantum mechanics", "is_a", "branch of physics"},
		{"Bitcoin", "created_by", "Satoshi Nakamoto"},
		{"Python", "used_for", "web development"},
		{"DNA", "is_a", "molecule"},
		{"Google", "founded_in", "1998"},
	}
	for _, e := range examples {
		sent := model.Generate(e[0], e[1], e[2], 30, 0.7)
		fmt.Printf("  (%s, %s, %s) → %s\n", e[0], e[1], e[2], sent)
	}
}

func trainNLU() {
	output := flag.String("o", "nous-neural.bin", "Output path for the trained model")
	flag.Parse()

	start := time.Now()

	nlu := cognitive.NewNLU()
	nn := cognitive.NewNeuralNLU(*output)
	if err := nn.LoadOrTrain(nlu); err != nil {
		fmt.Fprintf(os.Stderr, "training failed: %v\n", err)
		os.Exit(1)
	}

	info, err := os.Stat(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model file not found: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Neural model trained successfully\n")
	fmt.Printf("  Intents:  %d\n", nn.Classifier.NumIntents())
	fmt.Printf("  Size:     %d bytes (%.0f KB)\n", info.Size(), float64(info.Size())/1024)
	fmt.Printf("  Duration: %s\n", time.Since(start).Round(time.Millisecond))
	fmt.Printf("  Output:   %s\n", *output)
}
