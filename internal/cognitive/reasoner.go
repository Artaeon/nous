package cognitive

import (
	"context"
	"fmt"
	"strings"

	"github.com/artaeon/nous/internal/blackboard"
	"github.com/artaeon/nous/internal/ollama"
)

// Reasoner performs chain-of-thought inference using the local LLM.
// It listens for new percepts on the blackboard, constructs a reasoning
// prompt with relevant context, and posts the result back.
type Reasoner struct {
	Base
	// OnToken is called for each streamed token during reasoning.
	// Set this to enable real-time output.
	OnToken func(token string, done bool)
}

func NewReasoner(board *blackboard.Blackboard, llm *ollama.Client) *Reasoner {
	return &Reasoner{
		Base: Base{Board: board, LLM: llm},
	}
}

func (r *Reasoner) Name() string { return "reasoner" }

func (r *Reasoner) Run(ctx context.Context) error {
	events := r.Board.Subscribe("percept")

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev := <-events:
			percept, ok := ev.Payload.(blackboard.Percept)
			if !ok {
				continue
			}
			if err := r.reason(ctx, percept); err != nil {
				r.Board.Set("reasoner_error", err.Error())
			}
		}
	}
}

func (r *Reasoner) reason(ctx context.Context, percept blackboard.Percept) error {
	// Build context from working memory
	contextStr := r.buildContext()

	systemPrompt := Persona + `

## Response Format

When you need to perform an action, respond with:
THINK: <your reasoning>
ACTION: <tool_name> <args>

When you have a final answer, respond with:
THINK: <your reasoning>
ANSWER: <your response to the user>

You may omit the THINK line for simple, direct responses.`

	messages := []ollama.Message{
		{Role: "system", Content: systemPrompt},
	}

	if contextStr != "" {
		messages = append(messages, ollama.Message{
			Role: "system", Content: "Context:\n" + contextStr,
		})
	}

	messages = append(messages, ollama.Message{
		Role: "user", Content: percept.Raw,
	})

	if r.OnToken != nil {
		// Streaming mode
		resp, err := r.LLM.ChatStream(messages, &ollama.ModelOptions{
			Temperature: 0.7,
			NumPredict:  1024,
		}, r.OnToken)
		if err != nil {
			return fmt.Errorf("reasoner stream: %w", err)
		}
		r.processResponse(resp.Message.Content)
	} else {
		// Batch mode
		resp, err := r.LLM.Chat(messages, &ollama.ModelOptions{
			Temperature: 0.7,
			NumPredict:  1024,
		})
		if err != nil {
			return fmt.Errorf("reasoner chat: %w", err)
		}
		r.processResponse(resp.Message.Content)
	}

	return nil
}

func (r *Reasoner) buildContext() string {
	var parts []string

	// Include active goals
	goals := r.Board.ActiveGoals()
	if len(goals) > 0 {
		parts = append(parts, "Active goals:")
		for _, g := range goals {
			parts = append(parts, fmt.Sprintf("  - [%s] %s", g.Status, g.Description))
		}
	}

	// Include recent actions
	actions := r.Board.RecentActions(5)
	if len(actions) > 0 {
		parts = append(parts, "Recent actions:")
		for _, a := range actions {
			status := "success"
			if !a.Success {
				status = "failed"
			}
			parts = append(parts, fmt.Sprintf("  - %s %s (%s)", a.Tool, a.Input, status))
		}
	}

	return strings.Join(parts, "\n")
}

func (r *Reasoner) processResponse(content string) {
	// Extract structured parts from the response
	var think, answer, action string

	for _, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		upper := strings.ToUpper(trimmed)

		if strings.HasPrefix(upper, "THINK:") {
			think = strings.TrimSpace(trimmed[6:])
		} else if strings.HasPrefix(upper, "ANSWER:") {
			answer = strings.TrimSpace(trimmed[7:])
		} else if strings.HasPrefix(upper, "ACTION:") {
			action = strings.TrimSpace(trimmed[7:])
		}
	}

	if think != "" {
		r.Board.Set("last_thought", think)
	}

	if action != "" {
		// Post action request for the executor
		r.Board.Set("pending_action", action)
	}

	if answer != "" {
		r.Board.Set("last_answer", answer)
	} else if action == "" {
		// If no structured output was found, treat the whole response as the answer
		r.Board.Set("last_answer", content)
	}
}
