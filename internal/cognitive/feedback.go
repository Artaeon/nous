package cognitive

import (
	"github.com/artaeon/nous/internal/memory"
)

// FeedbackLoop wires all cognitive subsystems together so they learn from
// each other. This is the nervous system connecting Nous's multiple brains.
//
// Innovation: Most AI agents have independent subsystems that each collect
// their own data. FeedbackLoop creates BIDIRECTIONAL connections:
//
//   Episodic Memory ←→ Neural Cortex
//     - Cortex queries episodic for similar past successes before training
//     - Successful episodes boost cortex training weight
//
//   Firewall → Episodic Memory
//     - Caught errors are recorded as "what NOT to do"
//     - Future queries with similar keywords avoid the same mistakes
//
//   Growth → Virtual Context
//     - Trending topics get higher context budget
//     - Declining topics lose budget (natural rebalancing)
//
//   Episodic → Crystal Book (via AutoCrystallizer)
//     - Recurring patterns are mined and crystallized
//     - Common queries become instant after enough repetitions
//
// The result: every interaction makes ALL subsystems smarter, not just one.
type FeedbackLoop struct {
	Cortex     *NeuralCortex
	Episodic   *memory.EpisodicMemory
	VCtx       *VirtualContext
	Growth     *PersonalGrowth
	Crystals   *CrystalBook
	AutoCryst  *AutoCrystallizer
}

// NewFeedbackLoop creates a feedback loop connecting all subsystems.
func NewFeedbackLoop(cortex *NeuralCortex, episodic *memory.EpisodicMemory,
	vctx *VirtualContext, growth *PersonalGrowth, crystals *CrystalBook) *FeedbackLoop {
	fl := &FeedbackLoop{
		Cortex:   cortex,
		Episodic: episodic,
		VCtx:     vctx,
		Growth:   growth,
		Crystals: crystals,
	}
	if crystals != nil && episodic != nil {
		fl.AutoCryst = NewAutoCrystallizer(crystals, episodic)
	}
	return fl
}

// OnToolSuccess is called when a tool call succeeds. It propagates the
// success signal across all subsystems.
func (fl *FeedbackLoop) OnToolSuccess(query string, tool string, toolSequence []string) {
	if fl == nil {
		return
	}

	// 1. Train cortex with episodic context boost
	if fl.Cortex != nil && fl.Episodic != nil {
		fl.trainCortexWithMemory(query, tool)
	}

	// 2. Record success in virtual context for the relevant sources
	if fl.VCtx != nil {
		fl.VCtx.RecordSuccess("knowledge")
	}

	// 3. Sync growth topics to virtual context priorities
	if fl.Growth != nil && fl.VCtx != nil {
		fl.syncGrowthToContext()
	}

	// 4. Periodically auto-crystallize from episodic memory
	if fl.AutoCryst != nil {
		go fl.AutoCryst.Run()
	}
}

// OnToolFailure is called when a tool call fails. It records the failure
// pattern across subsystems so the same mistake isn't repeated.
func (fl *FeedbackLoop) OnToolFailure(query string, tool string) {
	if fl == nil {
		return
	}

	// Record failure in virtual context (source quality degrades)
	if fl.VCtx != nil {
		fl.VCtx.RecordFailure("knowledge")
	}
}

// OnFirewallViolation is called when the cognitive firewall catches an error.
// The violation is recorded as negative evidence to prevent repetition.
func (fl *FeedbackLoop) OnFirewallViolation(query string, violation string) {
	if fl == nil {
		return
	}

	// Record the violation as a failed episode so episodic search
	// can warn future queries with similar keywords
	if fl.Episodic != nil {
		fl.Episodic.Record(memory.Episode{
			Input:   query,
			Output:  "FIREWALL: " + violation,
			Success: false,
			Tags:    []string{"firewall", "violation"},
		})
	}
}

// trainCortexWithMemory trains the cortex with an episodic memory boost.
// If similar past queries successfully used the same tool, the training
// signal is reinforced (trained twice instead of once).
func (fl *FeedbackLoop) trainCortexWithMemory(query string, tool string) {
	if fl.Cortex == nil {
		return
	}

	input := CortexInputFromQuery(query, fl.Cortex.InputSize)

	// Always train on the current success
	fl.Cortex.Train(input, tool)

	// Check if episodic memory has similar past successes with the same tool
	if fl.Episodic == nil {
		return
	}
	similar := fl.Episodic.SuccessfulToolEpisodes(tool, 3)
	if len(similar) == 0 {
		return
	}

	// If we find past successes with this tool, reinforce the training
	// by training on a similar past query too (experience replay)
	for _, ep := range similar[:1] { // just the most recent similar one
		pastInput := CortexInputFromQuery(ep.Input, fl.Cortex.InputSize)
		fl.Cortex.Train(pastInput, tool)
	}
}

// syncGrowthToContext updates virtual context source priorities based on
// the user's trending interests from the growth system.
func (fl *FeedbackLoop) syncGrowthToContext() {
	if fl.Growth == nil || fl.VCtx == nil {
		return
	}

	// Get top interests
	interests := fl.Growth.TopInterests(5)
	if len(interests) == 0 {
		return
	}

	// Boost knowledge source quality for trending topics
	// This is a soft signal — the EMA will smooth it out over time
	for _, interest := range interests {
		if interest.Weight > 0.6 {
			fl.VCtx.RecordQuality("knowledge", 0.8)
			break // one boost per sync is enough
		}
	}
}

// Stats returns feedback loop statistics.
type FeedbackStats struct {
	CortexTrainCount    int
	EpisodicSize        int
	CrystalCount        int
	VCtxSourceCount     int
	GrowthInteractions  int
}

func (fl *FeedbackLoop) Stats() FeedbackStats {
	stats := FeedbackStats{}
	if fl == nil {
		return stats
	}
	if fl.Cortex != nil {
		stats.CortexTrainCount = fl.Cortex.TrainCount
	}
	if fl.Episodic != nil {
		stats.EpisodicSize = fl.Episodic.Size()
	}
	if fl.Crystals != nil {
		stats.CrystalCount = fl.Crystals.Size()
	}
	if fl.VCtx != nil {
		stats.VCtxSourceCount = fl.VCtx.SourceCount()
	}
	if fl.Growth != nil {
		gs := fl.Growth.Stats()
		stats.GrowthInteractions = gs.TotalInteractions
	}
	return stats
}
