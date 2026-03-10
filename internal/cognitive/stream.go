package cognitive

import (
	"context"
	"log"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Stream is the interface all cognitive modules implement.
type Stream interface {
	Name() string
	Run(ctx context.Context) error
}

// Base provides shared dependencies for all cognitive streams.
type Base struct {
	Board  *blackboard.Blackboard
	LLM    *ollama.Client
	Logger *log.Logger
}
