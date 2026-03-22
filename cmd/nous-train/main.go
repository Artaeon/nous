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
)

func main() {
	// Check for subcommand
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "textgen":
			trainTextGen(os.Args[2:])
			return
		case "nlu":
			os.Args = append(os.Args[:1], os.Args[2:]...)
			// fall through to default NLU training
		case "help", "-h", "--help":
			fmt.Println("Usage: nous-train [command] [flags]")
			fmt.Println()
			fmt.Println("Commands:")
			fmt.Println("  nlu       Train the neural intent classifier (default)")
			fmt.Println("  textgen   Train the GRU text generation model")
			fmt.Println()
			fmt.Println("NLU flags:")
			fmt.Println("  -o path   Output path (default: nous-neural.bin)")
			fmt.Println()
			fmt.Println("TextGen flags:")
			fmt.Println("  -o path       Output path (default: nous-textgen.bin)")
			fmt.Println("  -packages dir Package directory (default: packages)")
			fmt.Println("  -epochs n     Training epochs (default: 100)")
			fmt.Println("  -lr rate      Learning rate (default: 0.003)")
			fmt.Println("  -large        Use large model config")
			fmt.Println("  -augment n    Data augmentation factor (default: 3)")
			return
		}
	}

	// Default: train NLU
	trainNLU()
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
