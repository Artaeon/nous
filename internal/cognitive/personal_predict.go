package cognitive

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/memory"
)

// -----------------------------------------------------------------------
// Predictive Personal Intelligence — anticipate, don't just respond.
//
// Analyzes episodic memory and conversation patterns to predict user
// behavior and proactively surface insights. Turns Nous from reactive
// ("answer when asked") to anticipatory ("predict what you'll need").
// -----------------------------------------------------------------------

// PersonalPredictor analyzes patterns for proactive intelligence.
type PersonalPredictor struct {
	Episodic *memory.EpisodicMemory
}

// PersonalPrediction is a proactive insight.
type PersonalPrediction struct {
	Type       string  // "topic", "behavioral", "temporal", "anomaly", "recurring"
	Summary    string
	Confidence float64
	ActionItem string // suggested action (optional)
	BasedOn    int    // number of data points
}

// PersonalForecast is the full predictive analysis.
type PersonalForecast struct {
	TopicPredictions   []PersonalPrediction
	TemporalPatterns   []PersonalPrediction
	RecurringInterests []PersonalPrediction
	Anomalies          []PersonalPrediction
}

// NewPersonalPredictor creates a predictive intelligence engine.
func NewPersonalPredictor(episodic *memory.EpisodicMemory) *PersonalPredictor {
	return &PersonalPredictor{Episodic: episodic}
}

// Forecast generates predictions based on historical patterns.
func (pp *PersonalPredictor) Forecast() *PersonalForecast {
	if pp.Episodic == nil {
		return nil
	}

	episodes := pp.Episodic.Recent(200)
	if len(episodes) < 5 {
		return &PersonalForecast{}
	}

	return &PersonalForecast{
		TopicPredictions:   pp.predictTopics(episodes),
		TemporalPatterns:   pp.detectTemporalPatterns(episodes),
		RecurringInterests: pp.findRecurringInterests(episodes),
		Anomalies:          pp.detectAnomalies(episodes),
	}
}

// PredictNextTopic predicts the most likely next topic.
func (pp *PersonalPredictor) PredictNextTopic() *PersonalPrediction {
	if pp.Episodic == nil {
		return nil
	}
	predictions := pp.predictTopics(pp.Episodic.Recent(50))
	if len(predictions) == 0 {
		return nil
	}
	return &predictions[0]
}

// FormatPersonalForecast returns a human-readable forecast.
func FormatPersonalForecast(f *PersonalForecast) string {
	if f == nil {
		return "Not enough data for predictions yet."
	}

	var b strings.Builder
	b.WriteString("# Personal Intelligence Forecast\n\n")

	if len(f.TopicPredictions) > 0 {
		b.WriteString("## Predicted Interests\n")
		for _, p := range f.TopicPredictions {
			fmt.Fprintf(&b, "- %s (%.0f%% confidence)\n", p.Summary, p.Confidence*100)
		}
		b.WriteString("\n")
	}

	if len(f.TemporalPatterns) > 0 {
		b.WriteString("## Your Patterns\n")
		for _, p := range f.TemporalPatterns {
			fmt.Fprintf(&b, "- %s\n", p.Summary)
			if p.ActionItem != "" {
				fmt.Fprintf(&b, "  *Suggestion: %s*\n", p.ActionItem)
			}
		}
		b.WriteString("\n")
	}

	if len(f.RecurringInterests) > 0 {
		b.WriteString("## Deep Interests\n")
		for _, p := range f.RecurringInterests {
			fmt.Fprintf(&b, "- %s\n", p.Summary)
		}
		b.WriteString("\n")
	}

	if len(f.Anomalies) > 0 {
		b.WriteString("## Notable Changes\n")
		for _, p := range f.Anomalies {
			fmt.Fprintf(&b, "- %s\n", p.Summary)
		}
	}

	return b.String()
}

// -----------------------------------------------------------------------
// Analysis methods
// -----------------------------------------------------------------------

func (pp *PersonalPredictor) predictTopics(episodes []memory.Episode) []PersonalPrediction {
	type scored struct {
		topic string
		score float64
		count int
	}
	scores := make(map[string]*scored)
	now := time.Now()

	for _, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic == "" {
			continue
		}
		age := now.Sub(ep.Timestamp).Hours()
		recency := math.Exp(-age / 168.0) // 1 week decay

		if s, ok := scores[topic]; ok {
			s.score += recency
			s.count++
		} else {
			scores[topic] = &scored{topic, recency, 1}
		}
	}

	var sorted []*scored
	for _, s := range scores {
		if s.count >= 2 {
			sorted = append(sorted, s)
		}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].score > sorted[j].score
	})

	var predictions []PersonalPrediction
	for _, s := range sorted {
		if len(predictions) >= 5 {
			break
		}
		predictions = append(predictions, PersonalPrediction{
			Type:       "topic",
			Summary:    fmt.Sprintf("Likely to explore %s next (asked %d times recently)", s.topic, s.count),
			Confidence: math.Min(0.9, 0.3+s.score*0.3),
			BasedOn:    s.count,
		})
	}
	return predictions
}

func (pp *PersonalPredictor) detectTemporalPatterns(episodes []memory.Episode) []PersonalPrediction {
	hourCounts := make(map[int]int)
	for _, ep := range episodes {
		hourCounts[ep.Timestamp.Hour()]++
	}

	var predictions []PersonalPrediction

	peakHour, peakCount := 0, 0
	for h, c := range hourCounts {
		if c > peakCount {
			peakCount = c
			peakHour = h
		}
	}
	if peakCount >= 3 {
		period := "morning"
		switch {
		case peakHour >= 12 && peakHour < 17:
			period = "afternoon"
		case peakHour >= 17 && peakHour < 21:
			period = "evening"
		case peakHour >= 21:
			period = "late evening"
		}
		predictions = append(predictions, PersonalPrediction{
			Type:       "temporal",
			Summary:    fmt.Sprintf("Peak curiosity: %s (%d:00-%d:00, %d interactions)", period, peakHour, peakHour+1, peakCount),
			Confidence: 0.7,
			ActionItem: fmt.Sprintf("I'll prepare deeper content around %d:00", peakHour),
			BasedOn:    peakCount,
		})
	}

	return predictions
}

func (pp *PersonalPredictor) findRecurringInterests(episodes []memory.Episode) []PersonalPrediction {
	type weekTopic struct {
		topic string
		weeks map[int]bool
	}
	topics := make(map[string]*weekTopic)

	for _, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic == "" {
			continue
		}
		_, week := ep.Timestamp.ISOWeek()
		if wt, ok := topics[topic]; ok {
			wt.weeks[week] = true
		} else {
			topics[topic] = &weekTopic{topic, map[int]bool{week: true}}
		}
	}

	var predictions []PersonalPrediction
	for _, wt := range topics {
		if len(wt.weeks) >= 2 {
			predictions = append(predictions, PersonalPrediction{
				Type:       "recurring",
				Summary:    fmt.Sprintf("%s — persistent interest across %d weeks", wt.topic, len(wt.weeks)),
				Confidence: math.Min(0.9, 0.4+float64(len(wt.weeks))*0.15),
				BasedOn:    len(wt.weeks),
			})
		}
	}

	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].BasedOn > predictions[j].BasedOn
	})
	if len(predictions) > 5 {
		predictions = predictions[:5]
	}
	return predictions
}

func (pp *PersonalPredictor) detectAnomalies(episodes []memory.Episode) []PersonalPrediction {
	if len(episodes) < 10 {
		return nil
	}

	var predictions []PersonalPrediction
	recent := make(map[string]int)
	older := make(map[string]int)
	half := len(episodes) / 2

	for i, ep := range episodes {
		topic := extractDreamTopic(ep.Input)
		if topic == "" {
			continue
		}
		if i < half {
			recent[topic]++
		} else {
			older[topic]++
		}
	}

	for topic, count := range recent {
		if count >= 2 && older[topic] == 0 {
			predictions = append(predictions, PersonalPrediction{
				Type:       "anomaly",
				Summary:    fmt.Sprintf("New interest: %s (appeared %d times recently)", topic, count),
				Confidence: 0.6,
				BasedOn:    count,
			})
		}
	}

	for topic, count := range older {
		if count >= 3 && recent[topic] == 0 {
			predictions = append(predictions, PersonalPrediction{
				Type:       "anomaly",
				Summary:    fmt.Sprintf("Fading interest: %s (asked %d times before, silent recently)", topic, count),
				Confidence: 0.5,
				BasedOn:    count,
			})
		}
	}

	return predictions
}
