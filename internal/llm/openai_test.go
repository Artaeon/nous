package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestOpenAIProvider_Chat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", 404)
			return
		}

		if r.Method != "POST" {
			t.Errorf("unexpected method: %s", r.Method)
		}

		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("unexpected auth header: %s", r.Header.Get("Authorization"))
		}

		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected content type: %s", r.Header.Get("Content-Type"))
		}

		var req openaiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		if req.Model != "gpt-4o-mini" {
			t.Errorf("got model %q, want %q", req.Model, "gpt-4o-mini")
		}

		if len(req.Messages) != 1 || req.Messages[0].Content != "hello" {
			t.Errorf("unexpected messages: %+v", req.Messages)
		}

		resp := openaiResponse{
			ID: "test-id",
			Choices: []openaiChoice{
				{
					Index:        0,
					Message:      openaiMessage{Role: "assistant", Content: "Hello! How can I help?"},
					FinishReason: "stop",
				},
			},
			Usage: openaiUsage{
				PromptTokens:     5,
				CompletionTokens: 8,
				TotalTokens:      13,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "test-key", "gpt-4o-mini")

	resp, err := provider.Chat([]Message{
		{Role: "user", Content: "hello"},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Content != "Hello! How can I help?" {
		t.Errorf("got content %q, want %q", resp.Content, "Hello! How can I help?")
	}
	if resp.TokensUsed != 13 {
		t.Errorf("got tokens %d, want %d", resp.TokensUsed, 13)
	}
}

func TestOpenAIProvider_ChatWithOptions(t *testing.T) {
	var capturedReq openaiRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedReq)

		resp := openaiResponse{
			Choices: []openaiChoice{
				{Message: openaiMessage{Role: "assistant", Content: "ok"}},
			},
			Usage: openaiUsage{TotalTokens: 5},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "key", "model")

	_, err := provider.Chat([]Message{
		{Role: "user", Content: "test"},
	}, &Options{
		Temperature: 0.5,
		TopP:        0.9,
		MaxTokens:   100,
		Stop:        []string{"\n"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedReq.Temperature == nil || *capturedReq.Temperature != 0.5 {
		t.Error("temperature not set correctly")
	}
	if capturedReq.TopP == nil || *capturedReq.TopP != 0.9 {
		t.Error("top_p not set correctly")
	}
	if capturedReq.MaxTokens == nil || *capturedReq.MaxTokens != 100 {
		t.Error("max_tokens not set correctly")
	}
	if len(capturedReq.Stop) != 1 || capturedReq.Stop[0] != "\n" {
		t.Error("stop not set correctly")
	}
}

func TestOpenAIProvider_ChatCtx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openaiResponse{
			Choices: []openaiChoice{
				{Message: openaiMessage{Role: "assistant", Content: "response"}},
			},
			Usage: openaiUsage{TotalTokens: 3},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "", "model")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := provider.ChatCtx(ctx, []Message{
		{Role: "user", Content: "hi"},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "response" {
		t.Errorf("got %q, want %q", resp.Content, "response")
	}
}

func TestOpenAIProvider_ChatError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]string{
				"message": "invalid api key",
				"type":    "invalid_request_error",
			},
		})
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "bad-key", "model")

	_, err := provider.Chat([]Message{
		{Role: "user", Content: "hi"},
	}, nil)
	if err == nil {
		t.Fatal("expected error for unauthorized request")
	}
}

func TestOpenAIProvider_ChatNoChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openaiResponse{
			Choices: []openaiChoice{},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "", "model")

	_, err := provider.Chat([]Message{
		{Role: "user", Content: "hi"},
	}, nil)
	if err == nil {
		t.Fatal("expected error for empty choices")
	}
}

func TestOpenAIProvider_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			http.Error(w, "not found", 404)
			return
		}

		var req openaiEmbedRequest
		json.NewDecoder(r.Body).Decode(&req)

		if req.Input != "test text" {
			t.Errorf("got input %q, want %q", req.Input, "test text")
		}

		resp := openaiEmbedResponse{
			Data: []openaiEmbedData{
				{Embedding: []float64{0.1, 0.2, 0.3, 0.4}, Index: 0},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "key", "model")

	vec, err := provider.Embed("test text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vec) != 4 {
		t.Errorf("got %d dimensions, want 4", len(vec))
	}
	if vec[0] != 0.1 {
		t.Errorf("got vec[0]=%f, want 0.1", vec[0])
	}
}

func TestOpenAIProvider_EmbedError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "", "model")

	_, err := provider.Embed("test")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestOpenAIProvider_Model(t *testing.T) {
	provider := NewOpenAI("http://example.com", "key", "gpt-4")
	if provider.Model() != "gpt-4" {
		t.Errorf("got %q, want %q", provider.Model(), "gpt-4")
	}
}

func TestOpenAIProvider_DefaultURL(t *testing.T) {
	provider := NewOpenAI("", "key", "model")
	if provider.baseURL != DefaultOpenAIURL {
		t.Errorf("got %q, want %q", provider.baseURL, DefaultOpenAIURL)
	}
}

func TestOpenAIProvider_NoAuthHeader(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		resp := openaiResponse{
			Choices: []openaiChoice{
				{Message: openaiMessage{Role: "assistant", Content: "ok"}},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := NewOpenAI(server.URL, "", "model")
	provider.Chat([]Message{{Role: "user", Content: "hi"}}, nil)

	if gotAuth != "" {
		t.Errorf("expected no auth header, got %q", gotAuth)
	}
}
