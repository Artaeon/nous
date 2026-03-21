// nous-train trains the neural intent classifier and saves the model to disk.
// Used by CI/CD to produce the model artifact for releases.
//
// Usage: nous-train [-o output.bin]
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/artaeon/nous/internal/cognitive"
)

func main() {
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
