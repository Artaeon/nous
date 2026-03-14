// Package client provides an HTTP client for connecting to a remote nous server.
// This enables CLI usage from anywhere without running a local Ollama instance.
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client connects to a remote nous server over HTTP.
type Client struct {
	BaseURL string
	APIKey  string
	client  *http.Client
}

// New creates a client targeting the given nous server URL.
// If apiKey is non-empty it is sent as a Bearer token on every request.
func New(baseURL, apiKey string) *Client {
	return &Client{
		BaseURL: baseURL,
		APIKey:  apiKey,
		client: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
}

// --- Response types ---

// ChatResponse is returned by POST /api/chat.
type ChatResponse struct {
	Answer   string `json:"answer"`
	Duration int64  `json:"duration_ms"`
}

// StatusResponse is returned by GET /api/status.
type StatusResponse struct {
	Version     string `json:"version"`
	Model       string `json:"model"`
	Uptime      string `json:"uptime"`
	Percepts    int    `json:"percepts"`
	Goals       int    `json:"goals"`
	ToolCount   int    `json:"tool_count"`
	QueuedJobs  int    `json:"queued_jobs"`
	RunningJobs int    `json:"running_jobs"`
}

// TodayResponse is returned by GET /api/assistant/today.
type TodayResponse struct {
	Notifications json.RawMessage `json:"notifications"`
	Today         json.RawMessage `json:"today"`
	Upcoming      json.RawMessage `json:"upcoming"`
}

// HandInfo describes a hand returned by GET /api/hands.
type HandInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Schedule    string `json:"schedule"`
	Enabled     bool   `json:"enabled"`
	State       string `json:"state"`
}

// HandsResponse wraps the list returned by the server.
type HandsResponse struct {
	Hands []HandInfo `json:"hands"`
}

// --- API methods ---

// Chat sends a message to the remote server and returns the response.
func (c *Client) Chat(message string) (string, error) {
	body, err := json.Marshal(map[string]string{"message": message})
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	resp, err := c.do("POST", "/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var cr ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	return cr.Answer, nil
}

// Status returns the remote server's system status.
func (c *Client) Status() (StatusResponse, error) {
	resp, err := c.do("GET", "/api/status", nil)
	if err != nil {
		return StatusResponse{}, err
	}
	defer resp.Body.Close()

	var sr StatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return StatusResponse{}, fmt.Errorf("decode status: %w", err)
	}
	return sr, nil
}

// Health checks whether the remote server is alive.
func (c *Client) Health() error {
	resp, err := c.do("GET", "/api/health", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// ListHands returns all hands registered on the remote server.
func (c *Client) ListHands() ([]HandInfo, error) {
	resp, err := c.do("GET", "/api/hands", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var hr HandsResponse
	if err := json.NewDecoder(resp.Body).Decode(&hr); err != nil {
		return nil, fmt.Errorf("decode hands: %w", err)
	}
	return hr.Hands, nil
}

// RunHand triggers an immediate run of the named hand.
func (c *Client) RunHand(name string) error {
	resp, err := c.do("POST", "/api/hands/"+name+"/run", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// ActivateHand enables the named hand on the remote server.
func (c *Client) ActivateHand(name string) error {
	resp, err := c.do("POST", "/api/hands/"+name+"/activate", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// Today returns the assistant inbox for the current day.
func (c *Client) Today() (TodayResponse, error) {
	resp, err := c.do("GET", "/api/assistant/today", nil)
	if err != nil {
		return TodayResponse{}, err
	}
	defer resp.Body.Close()

	var tr TodayResponse
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return TodayResponse{}, fmt.Errorf("decode today: %w", err)
	}
	return tr, nil
}

// CreateTask adds a new task on the remote server.
func (c *Client) CreateTask(title, due string) error {
	body, err := json.Marshal(map[string]string{"title": title, "due_at": due})
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	resp, err := c.do("POST", "/api/assistant/tasks", bytes.NewReader(body))
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// --- internal helpers ---

// do executes an HTTP request, adding auth headers and checking for errors.
func (c *Client) do(method, path string, body io.Reader) (*http.Response, error) {
	url := c.BaseURL + path
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request %s %s: %w", method, path, err)
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("%s %s returned %d: %s", method, path, resp.StatusCode, string(b))
	}

	return resp, nil
}
