package cognitive

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Perceiver processes raw input and posts structured percepts to the blackboard.
// It extracts intent and entities using the LLM, transforming unstructured
// natural language into actionable cognitive representations.
type Perceiver struct {
	Base
	Router *ModelRouter
	input  chan string
}

func NewPerceiver(board *blackboard.Blackboard, llm *ollama.Client) *Perceiver {
	return &Perceiver{
		Base:  Base{Board: board, LLM: llm},
		input: make(chan string, 16),
	}
}

func (p *Perceiver) Name() string { return "perceiver" }

// Submit sends raw input to the perceiver for processing.
func (p *Perceiver) Submit(text string) {
	p.input <- text
}

func (p *Perceiver) Run(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case raw := <-p.input:
			percept, err := p.perceive(raw)
			if err != nil {
				p.Board.Set("perceiver_error", err.Error())
				// Fall back to a basic percept without LLM analysis
				percept = blackboard.Percept{
					Raw:       raw,
					Intent:    "unknown",
					Entities:  map[string]string{},
					Timestamp: time.Now(),
				}
			}
			p.Board.PostPercept(percept)
		}
	}
}

func (p *Perceiver) perceive(raw string) (blackboard.Percept, error) {
	prompt := fmt.Sprintf(`Analyze this user input and extract the intent and key entities.
Respond in exactly this format (no other text):
INTENT: <one word: question, command, request, statement, greeting>
ENTITIES: <key1=value1, key2=value2>

User input: %s`, raw)

	// Use the router's perception-optimized client if available
	client := p.LLM
	if p.Router != nil {
		client = p.Router.ClientFor(TaskPerception)
	}

	resp, err := client.Chat([]ollama.Message{
		{Role: "system", Content: PerceivePrompt},
		{Role: "user", Content: prompt},
	}, &ollama.ModelOptions{
		Temperature: 0.1,
		NumPredict:  40,
	})
	if err != nil {
		return blackboard.Percept{}, fmt.Errorf("perceiver llm: %w", err)
	}

	intent, entities := parsePerception(resp.Message.Content)

	return blackboard.Percept{
		Raw:       raw,
		Intent:    intent,
		Entities:  entities,
		Timestamp: time.Now(),
	}, nil
}

func parsePerception(response string) (string, map[string]string) {
	intent := "unknown"
	entities := make(map[string]string)

	for _, line := range strings.Split(response, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(line), "INTENT:") {
			intent = strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
			intent = strings.ToLower(strings.TrimSpace(intent))
		}
		if strings.HasPrefix(strings.ToUpper(line), "ENTITIES:") {
			entStr := strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
			for _, pair := range strings.Split(entStr, ",") {
				parts := strings.SplitN(strings.TrimSpace(pair), "=", 2)
				if len(parts) == 2 {
					entities[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
				}
			}
		}
	}

	return intent, entities
}
