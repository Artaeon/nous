package eval

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// KPITracker tracks online key performance indicators.
type KPITracker struct {
	mu        sync.RWMutex
	metrics   map[string]*KPIMetric
	window    []KPIEvent
	maxWindow int // rolling window size
}

// KPIMetric is one tracked metric.
type KPIMetric struct {
	Name      string  `json:"name"`
	Value     float64 `json:"value"`
	Count     int     `json:"count"`
	Sum       float64 `json:"sum"`
	Min       float64 `json:"min"`
	Max       float64 `json:"max"`
	Threshold float64 `json:"threshold"`
	Direction string  `json:"direction"` // "higher_is_better" or "lower_is_better"
}

// KPIEvent represents one recorded event.
type KPIEvent struct {
	Metric    string            `json:"metric"`
	Value     float64           `json:"value"`
	Timestamp time.Time         `json:"timestamp"`
	Tags      map[string]string `json:"tags,omitempty"`
}

// KPISnapshot captures all current KPI values at a point in time.
type KPISnapshot struct {
	Metrics    map[string]*KPIMetric `json:"metrics"`
	Alerts     []KPIAlert            `json:"alerts,omitempty"`
	Timestamp  time.Time             `json:"timestamp"`
	WindowSize int                   `json:"window_size"`
}

// KPIAlert signals that a metric has breached its threshold.
type KPIAlert struct {
	Metric    string  `json:"metric"`
	Current   float64 `json:"current"`
	Threshold float64 `json:"threshold"`
	Direction string  `json:"direction"`
	Message   string  `json:"message"`
}

// NewKPITracker creates a new KPI tracker with the given rolling window size.
func NewKPITracker(maxWindow int) *KPITracker {
	if maxWindow <= 0 {
		maxWindow = 10000
	}
	kt := &KPITracker{
		metrics:   make(map[string]*KPIMetric),
		window:    make([]KPIEvent, 0, maxWindow),
		maxWindow: maxWindow,
	}
	// Initialize with default KPIs
	for name, m := range DefaultKPIs() {
		kt.metrics[name] = m
	}
	return kt
}

// DefaultKPIs returns the standard KPI set with thresholds.
func DefaultKPIs() map[string]*KPIMetric {
	return map[string]*KPIMetric{
		"task_success_rate": {
			Name:      "task_success_rate",
			Threshold: 0.90,
			Direction: "higher_is_better",
		},
		"retry_rate": {
			Name:      "retry_rate",
			Threshold: 0.05,
			Direction: "lower_is_better",
		},
		"user_correction_rate": {
			Name:      "user_correction_rate",
			Threshold: 0.08,
			Direction: "lower_is_better",
		},
		"filler_response_rate": {
			Name:      "filler_response_rate",
			Threshold: 0.02,
			Direction: "lower_is_better",
		},
		"p50_latency_ms": {
			Name:      "p50_latency_ms",
			Threshold: 50,
			Direction: "lower_is_better",
		},
		"p95_latency_ms": {
			Name:      "p95_latency_ms",
			Threshold: 200,
			Direction: "lower_is_better",
		},
		"intent_accuracy": {
			Name:      "intent_accuracy",
			Threshold: 0.92,
			Direction: "higher_is_better",
		},
		"hallucination_rate": {
			Name:      "hallucination_rate",
			Threshold: 0.03,
			Direction: "lower_is_better",
		},
	}
}

// Record adds a new metric observation.
func (kt *KPITracker) Record(metric string, value float64, tags map[string]string) {
	kt.mu.Lock()
	defer kt.mu.Unlock()

	// Ensure the metric exists
	m, ok := kt.metrics[metric]
	if !ok {
		m = &KPIMetric{
			Name:      metric,
			Direction: "higher_is_better",
		}
		kt.metrics[metric] = m
	}

	// Update running statistics
	m.Count++
	m.Sum += value
	m.Value = m.Sum / float64(m.Count)

	if m.Count == 1 {
		m.Min = value
		m.Max = value
	} else {
		if value < m.Min {
			m.Min = value
		}
		if value > m.Max {
			m.Max = value
		}
	}

	// Append to rolling window
	event := KPIEvent{
		Metric:    metric,
		Value:     value,
		Timestamp: time.Now(),
		Tags:      tags,
	}

	if len(kt.window) >= kt.maxWindow {
		// Evict oldest event
		kt.window = kt.window[1:]
	}
	kt.window = append(kt.window, event)
}

// Get returns the current state of a metric. Returns nil if not found.
func (kt *KPITracker) Get(metric string) *KPIMetric {
	kt.mu.RLock()
	defer kt.mu.RUnlock()

	m, ok := kt.metrics[metric]
	if !ok {
		return nil
	}
	// Return a copy
	cp := *m
	return &cp
}

// Snapshot returns all current KPI values.
func (kt *KPITracker) Snapshot() *KPISnapshot {
	kt.mu.RLock()
	defer kt.mu.RUnlock()

	metrics := make(map[string]*KPIMetric, len(kt.metrics))
	for name, m := range kt.metrics {
		cp := *m
		metrics[name] = &cp
	}

	snap := &KPISnapshot{
		Metrics:    metrics,
		Timestamp:  time.Now(),
		WindowSize: len(kt.window),
	}

	// Check alerts while holding the lock
	snap.Alerts = kt.checkAlertsLocked()

	return snap
}

// CheckAlerts evaluates all metrics against thresholds.
func (kt *KPITracker) CheckAlerts() []KPIAlert {
	kt.mu.RLock()
	defer kt.mu.RUnlock()
	return kt.checkAlertsLocked()
}

// checkAlertsLocked evaluates alerts without acquiring the lock.
// Caller must hold at least a read lock.
func (kt *KPITracker) checkAlertsLocked() []KPIAlert {
	var alerts []KPIAlert

	for _, m := range kt.metrics {
		if m.Count == 0 || m.Threshold == 0 {
			continue
		}

		var breached bool
		switch m.Direction {
		case "higher_is_better":
			breached = m.Value < m.Threshold
		case "lower_is_better":
			breached = m.Value > m.Threshold
		}

		if breached {
			alerts = append(alerts, KPIAlert{
				Metric:    m.Name,
				Current:   m.Value,
				Threshold: m.Threshold,
				Direction: m.Direction,
				Message:   formatAlertMessage(m),
			})
		}
	}

	return alerts
}

// formatAlertMessage creates a human-readable alert message.
func formatAlertMessage(m *KPIMetric) string {
	switch m.Direction {
	case "higher_is_better":
		return fmt.Sprintf("%s is %.4f, below threshold %.4f", m.Name, m.Value, m.Threshold)
	case "lower_is_better":
		return fmt.Sprintf("%s is %.4f, above threshold %.4f", m.Name, m.Value, m.Threshold)
	default:
		return fmt.Sprintf("%s is %.4f (threshold %.4f)", m.Name, m.Value, m.Threshold)
	}
}

// RecordTaskSuccess records a task success or failure.
func (kt *KPITracker) RecordTaskSuccess(success bool) {
	val := 0.0
	if success {
		val = 1.0
	}
	kt.Record("task_success_rate", val, nil)
}

// RecordRetry records that a retry occurred for a query.
func (kt *KPITracker) RecordRetry(query string) {
	kt.Record("retry_rate", 1.0, map[string]string{"query": truncateTag(query, 200)})
}

// RecordUserCorrection records a user correction event.
func (kt *KPITracker) RecordUserCorrection(query, correction string) {
	kt.Record("user_correction_rate", 1.0, map[string]string{
		"query":      truncateTag(query, 200),
		"correction": truncateTag(correction, 200),
	})
}

// RecordFillerResponse records a filler response event.
func (kt *KPITracker) RecordFillerResponse(query, response string) {
	kt.Record("filler_response_rate", 1.0, map[string]string{
		"query":    truncateTag(query, 200),
		"response": truncateTag(response, 200),
	})
}

// RecordLatency records a latency observation in milliseconds.
func (kt *KPITracker) RecordLatency(latencyMs int64) {
	kt.Record("p50_latency_ms", float64(latencyMs), nil)
	kt.Record("p95_latency_ms", float64(latencyMs), nil)
}

// RecordIntentAccuracy records whether an intent classification was correct.
func (kt *KPITracker) RecordIntentAccuracy(correct bool) {
	val := 0.0
	if correct {
		val = 1.0
	}
	kt.Record("intent_accuracy", val, nil)
}

// RecordHallucination records whether a hallucination was detected.
func (kt *KPITracker) RecordHallucination(detected bool) {
	val := 0.0
	if detected {
		val = 1.0
	}
	kt.Record("hallucination_rate", val, nil)
}

// truncateTag truncates a string for use in tags.
func truncateTag(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}

// Report generates a human-readable KPI report.
func (kt *KPITracker) Report() string {
	kt.mu.RLock()
	defer kt.mu.RUnlock()

	var b strings.Builder
	b.WriteString("=== KPI Report ===\n")
	b.WriteString(fmt.Sprintf("Timestamp: %s\n", time.Now().Format(time.RFC3339)))
	b.WriteString(fmt.Sprintf("Window: %d events\n\n", len(kt.window)))

	// Gather metric names for deterministic ordering
	names := make([]string, 0, len(kt.metrics))
	for name := range kt.metrics {
		names = append(names, name)
	}
	sortStrings(names)

	for _, name := range names {
		m := kt.metrics[name]
		status := "OK"

		if m.Count > 0 && m.Threshold > 0 {
			switch m.Direction {
			case "higher_is_better":
				if m.Value < m.Threshold {
					status = "ALERT"
				}
			case "lower_is_better":
				if m.Value > m.Threshold {
					status = "ALERT"
				}
			}
		}

		b.WriteString(fmt.Sprintf("  %-25s  value=%.4f  count=%d  min=%.4f  max=%.4f  threshold=%.4f  [%s]  %s\n",
			name, m.Value, m.Count, m.Min, m.Max, m.Threshold, m.Direction, status))
	}

	alerts := kt.checkAlertsLocked()
	if len(alerts) > 0 {
		b.WriteString(fmt.Sprintf("\nAlerts (%d):\n", len(alerts)))
		for _, a := range alerts {
			b.WriteString(fmt.Sprintf("  - %s\n", a.Message))
		}
	} else {
		b.WriteString("\nNo alerts.\n")
	}

	return b.String()
}

// ExportJSON exports KPIs as JSON for dashboards.
func (kt *KPITracker) ExportJSON() ([]byte, error) {
	snap := kt.Snapshot()
	return json.MarshalIndent(snap, "", "  ")
}

// sortStrings sorts a string slice in place using insertion sort
// to avoid importing sort package (keeps this file self-contained).
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j] > key {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}
