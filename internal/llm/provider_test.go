package llm

import (
	"context"
	"testing"
)

// mockProvider implements Provider for testing.
type mockProvider struct {
	model     string
	response  *Response
	embedding []float64
	err       error
}

func (m *mockProvider) Chat(messages []Message, opts *Options) (*Response, error) {
	return m.response, m.err
}

func (m *mockProvider) ChatCtx(ctx context.Context, messages []Message, opts *Options) (*Response, error) {
	return m.response, m.err
}

func (m *mockProvider) Embed(text string) ([]float64, error) {
	return m.embedding, m.err
}

func (m *mockProvider) Model() string {
	return m.model
}

func TestProviderInterface(t *testing.T) {
	// Verify mockProvider satisfies Provider interface
	var _ Provider = &mockProvider{}

	mock := &mockProvider{
		model: "test-model",
		response: &Response{
			Content:    "Hello, world!",
			TokensUsed: 10,
		},
		embedding: []float64{0.1, 0.2, 0.3},
	}

	t.Run("Chat", func(t *testing.T) {
		resp, err := mock.Chat([]Message{
			{Role: "user", Content: "hi"},
		}, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "Hello, world!" {
			t.Errorf("got content %q, want %q", resp.Content, "Hello, world!")
		}
		if resp.TokensUsed != 10 {
			t.Errorf("got tokens %d, want %d", resp.TokensUsed, 10)
		}
	})

	t.Run("ChatCtx", func(t *testing.T) {
		resp, err := mock.ChatCtx(context.Background(), []Message{
			{Role: "user", Content: "hi"},
		}, &Options{Temperature: 0.7})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content != "Hello, world!" {
			t.Errorf("got content %q, want %q", resp.Content, "Hello, world!")
		}
	})

	t.Run("Embed", func(t *testing.T) {
		vec, err := mock.Embed("test text")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(vec) != 3 {
			t.Errorf("got %d dimensions, want 3", len(vec))
		}
	})

	t.Run("Model", func(t *testing.T) {
		if mock.Model() != "test-model" {
			t.Errorf("got model %q, want %q", mock.Model(), "test-model")
		}
	})
}

func TestOptionsDefaults(t *testing.T) {
	opts := &Options{}
	if opts.Temperature != 0 {
		t.Errorf("default temperature should be 0, got %f", opts.Temperature)
	}
	if opts.MaxTokens != 0 {
		t.Errorf("default max tokens should be 0, got %d", opts.MaxTokens)
	}
}
