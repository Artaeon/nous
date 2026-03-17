package cognitive

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

// ConstrainedDecoder provides helper functions for getting structured output
// from small LLMs. Instead of free-form text generation (where 1.5B models
// waste capacity on grammar/syntax), these functions constrain the output
// space so the model focuses 100% on decisions.
type ConstrainedDecoder struct {
	LLM     *ollama.Client
	Timeout time.Duration
}

// NewConstrainedDecoder creates a constrained decoder with sensible defaults.
func NewConstrainedDecoder(llm *ollama.Client) *ConstrainedDecoder {
	return &ConstrainedDecoder{
		LLM:     llm,
		Timeout: 5 * time.Second,
	}
}

// AskChoice presents numbered options and returns the selected index (0-based).
// Returns -1 if the model fails to choose. Much more reliable than free-form
// tool selection since the model only needs to output a single number.
func (cd *ConstrainedDecoder) AskChoice(prompt string, options []string) int {
	if cd.LLM == nil || len(options) == 0 {
		return -1
	}

	var sb strings.Builder
	sb.WriteString(prompt)
	sb.WriteString("\n\nOptions:\n")
	for i, opt := range options {
		fmt.Fprintf(&sb, "%d. %s\n", i+1, opt)
	}
	sb.WriteString("\nRespond with ONLY the number of your choice (e.g. 1, 2, 3):")

	result := cd.quickChat(sb.String(), 10)
	result = strings.TrimSpace(result)

	// Parse the number
	num, err := strconv.Atoi(strings.TrimRight(result, ".):"))
	if err != nil {
		// Try to find a number in the response
		for _, word := range strings.Fields(result) {
			if n, e := strconv.Atoi(strings.TrimRight(word, ".):,")); e == nil {
				num = n
				err = nil
				break
			}
		}
	}
	if err != nil || num < 1 || num > len(options) {
		return -1
	}
	return num - 1
}

// AskYesNo asks a yes/no question and returns the boolean result.
// Returns the defaultVal if the model's response is ambiguous.
func (cd *ConstrainedDecoder) AskYesNo(prompt string, defaultVal bool) bool {
	if cd.LLM == nil {
		return defaultVal
	}

	fullPrompt := prompt + "\n\nRespond with ONLY 'yes' or 'no':"
	result := cd.quickChat(fullPrompt, 5)
	result = strings.ToLower(strings.TrimSpace(result))

	switch {
	case strings.HasPrefix(result, "yes"), result == "y", result == "true":
		return true
	case strings.HasPrefix(result, "no"), result == "n", result == "false":
		return false
	default:
		return defaultVal
	}
}

// AskJSON asks the model to respond in JSON format and unmarshals into target.
// Uses Ollama's format:"json" option for constrained decoding.
func (cd *ConstrainedDecoder) AskJSON(prompt string, target interface{}) error {
	if cd.LLM == nil {
		return fmt.Errorf("no LLM configured")
	}

	fullPrompt := prompt + "\n\nRespond with ONLY valid JSON, no other text:"

	type resp struct {
		text string
		err  error
	}
	ch := make(chan resp, 1)
	go func() {
		r, err := cd.LLM.ChatJSON([]ollama.Message{
			{Role: "user", Content: fullPrompt},
		}, &ollama.ModelOptions{
			Temperature: 0.1,
			NumPredict:  200,
		})
		if err != nil {
			ch <- resp{err: err}
			return
		}
		ch <- resp{text: r.Message.Content}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return r.err
		}
		return json.Unmarshal([]byte(strings.TrimSpace(r.text)), target)
	case <-time.After(cd.Timeout):
		return fmt.Errorf("timeout waiting for JSON response")
	}
}

// BinaryCascade decomposes a complex decision into a cascade of yes/no questions.
// Returns the index of the matching path. Each question narrows the space by half.
// Much more reliable than asking a 1.5B model to make a complex multi-way choice.
func (cd *ConstrainedDecoder) BinaryCascade(context string, questions []string) []bool {
	results := make([]bool, len(questions))
	for i, q := range questions {
		prompt := context + "\n\n" + q
		results[i] = cd.AskYesNo(prompt, false)
	}
	return results
}

// quickChat sends a fast, low-token chat to the LLM with timeout.
func (cd *ConstrainedDecoder) quickChat(prompt string, maxTokens int) string {
	type resp struct {
		text string
		err  error
	}
	ch := make(chan resp, 1)
	go func() {
		r, err := cd.LLM.Chat([]ollama.Message{
			{Role: "user", Content: prompt},
		}, &ollama.ModelOptions{
			Temperature: 0.1,
			NumPredict:  maxTokens,
		})
		if err != nil {
			ch <- resp{err: err}
			return
		}
		ch <- resp{text: r.Message.Content}
	}()

	select {
	case r := <-ch:
		if r.err != nil {
			return ""
		}
		return r.text
	case <-time.After(cd.Timeout):
		return ""
	}
}
