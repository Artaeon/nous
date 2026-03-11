package training

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// TrainingPair is a single (input, output) pair suitable for fine-tuning.
// This is the bridge from prompt engineering to actual weight modification.
type TrainingPair struct {
	// ChatML format for fine-tuning frameworks (unsloth, axolotl, etc.)
	System    string   `json:"system"`
	Input     string   `json:"input"`
	Output    string   `json:"output"`
	ToolCalls []string `json:"tool_calls,omitempty"` // tool names used
	Quality   float64  `json:"quality"`              // 0.0-1.0 quality score
	Timestamp string   `json:"timestamp"`
}

// AlpacaPair is the Alpaca format used by most fine-tuning frameworks.
type AlpacaPair struct {
	Instruction string `json:"instruction"`
	Input       string `json:"input"`
	Output      string `json:"output"`
}

// Collector gathers successful interactions as training data.
// When enough high-quality pairs accumulate, they can be exported
// for LoRA fine-tuning — changing the model's weights, not just its prompts.
type Collector struct {
	mu        sync.RWMutex
	pairs     []TrainingPair
	storePath string
	minQuality float64
}

// NewCollector creates a training data collector.
func NewCollector(storePath string) *Collector {
	c := &Collector{
		storePath:  storePath,
		minQuality: 0.6, // only collect above this quality threshold
	}
	c.load()
	return c
}

// Collect adds a training pair if it meets the quality threshold.
func (c *Collector) Collect(system, input, output string, toolCalls []string, quality float64) {
	if quality < c.minQuality {
		return
	}

	pair := TrainingPair{
		System:    system,
		Input:     input,
		Output:    output,
		ToolCalls: toolCalls,
		Quality:   quality,
		Timestamp: time.Now().Format(time.RFC3339),
	}

	c.mu.Lock()
	c.pairs = append(c.pairs, pair)
	c.mu.Unlock()

	// Auto-save periodically
	c.mu.RLock()
	count := len(c.pairs)
	c.mu.RUnlock()
	if count%5 == 0 {
		go c.Save()
	}
}

// Size returns the number of collected pairs.
func (c *Collector) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.pairs)
}

// ExportJSONL exports training data in JSONL format (one JSON object per line).
// This is the standard format for fine-tuning tools like unsloth and axolotl.
func (c *Collector) ExportJSONL(outputPath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	for _, pair := range c.pairs {
		if err := encoder.Encode(pair); err != nil {
			return err
		}
	}

	return nil
}

// ExportAlpaca exports in Alpaca format for compatibility with most fine-tuning frameworks.
func (c *Collector) ExportAlpaca(outputPath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var alpaca []AlpacaPair
	for _, pair := range c.pairs {
		alpaca = append(alpaca, AlpacaPair{
			Instruction: pair.System,
			Input:       pair.Input,
			Output:      pair.Output,
		})
	}

	data, err := json.MarshalIndent(alpaca, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(outputPath, data, 0644)
}

// ExportChatML exports in ChatML format for models that use chat templates.
func (c *Collector) ExportChatML(outputPath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, pair := range c.pairs {
		entry := map[string]interface{}{
			"messages": []map[string]string{
				{"role": "system", "content": pair.System},
				{"role": "user", "content": pair.Input},
				{"role": "assistant", "content": pair.Output},
			},
		}
		data, err := json.Marshal(entry)
		if err != nil {
			continue
		}
		fmt.Fprintln(f, string(data))
	}

	return nil
}

// AverageQuality returns the mean quality score of all collected pairs.
func (c *Collector) AverageQuality() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.pairs) == 0 {
		return 0
	}

	var sum float64
	for _, p := range c.pairs {
		sum += p.Quality
	}
	return sum / float64(len(c.pairs))
}

// HighQualityPairs returns pairs with quality >= threshold.
func (c *Collector) HighQualityPairs(threshold float64) []TrainingPair {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var result []TrainingPair
	for _, p := range c.pairs {
		if p.Quality >= threshold {
			result = append(result, p)
		}
	}
	return result
}

// Pairs returns all collected training pairs.
func (c *Collector) Pairs() []TrainingPair {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]TrainingPair, len(c.pairs))
	copy(result, c.pairs)
	return result
}

// QualityDistribution returns the count of pairs at each quality level (bucketed by 0.1).
func (c *Collector) QualityDistribution() map[string]int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	dist := make(map[string]int)
	for _, pair := range c.pairs {
		bucket := fmt.Sprintf("%.1f", float64(int(pair.Quality*10))/10)
		dist[bucket]++
	}
	return dist
}

// PurgeBelow removes all pairs below the given quality threshold.
func (c *Collector) PurgeBelow(threshold float64) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	filtered := c.pairs[:0]
	removed := 0
	for _, pair := range c.pairs {
		if pair.Quality >= threshold {
			filtered = append(filtered, pair)
		} else {
			removed++
		}
	}
	c.pairs = filtered
	return removed
}

// Save persists collected pairs to disk.
func (c *Collector) Save() error {
	c.mu.RLock()
	data, err := json.MarshalIndent(c.pairs, "", "  ")
	c.mu.RUnlock()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(c.storePath, 0755); err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(c.storePath, "training_data.json"), data, 0644)
}

func (c *Collector) load() {
	if c.storePath == "" {
		return
	}
	data, err := os.ReadFile(filepath.Join(c.storePath, "training_data.json"))
	if err != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	_ = json.Unmarshal(data, &c.pairs)
}
