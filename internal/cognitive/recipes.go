package cognitive

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/artaeon/nous/internal/safefile"
)

// RecipeStep is a single tool invocation within a recipe.
type RecipeStep struct {
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
	// Parameterized placeholders: keys like "$FILE" get replaced at replay time
}

// Recipe is a learned multi-step tool sequence that can be replayed.
// Recipes are generalized from observed successful tool chains by replacing
// concrete values with parameterized placeholders.
type Recipe struct {
	ID        string       `json:"id"`
	Name      string       `json:"name"`
	Trigger   string       `json:"trigger"`   // intent pattern that activates this recipe
	Keywords  []string     `json:"keywords"`  // keywords from the original query
	Steps     []RecipeStep `json:"steps"`
	Params    []string     `json:"params"`    // list of parameter names (e.g., "$FILE", "$PATTERN")
	Uses      int          `json:"uses"`
	Successes int          `json:"successes"` // how many times it ran without error
	LastUsed  time.Time    `json:"last_used"`
	CreatedAt time.Time    `json:"created_at"`
}

// Confidence returns a score 0.0-1.0 based on usage success rate.
func (r *Recipe) Confidence() float64 {
	if r.Uses == 0 {
		return 0.5 // new recipe, neutral confidence
	}
	return float64(r.Successes) / float64(r.Uses)
}

// RecipeBook stores and retrieves tool choreography recipes.
type RecipeBook struct {
	mu        sync.RWMutex
	recipes   []Recipe
	storePath string
}

// NewRecipeBook creates or loads a recipe book from disk.
func NewRecipeBook(storePath string) *RecipeBook {
	rb := &RecipeBook{storePath: storePath}
	rb.load()
	return rb
}

// Record observes a completed pipeline and extracts a recipe if the sequence
// was successful (2+ steps, no errors).
func (rb *RecipeBook) Record(pipe *Pipeline, intent string, rawInput string) {
	if pipe.StepCount() < 2 {
		return // single-step sequences aren't worth learning
	}

	// Check for errors in the pipeline
	for _, step := range pipe.steps {
		if strings.Contains(step.Summary, "Error:") {
			return // don't learn from failures
		}
	}

	// Check if a similar recipe already exists
	if existing := rb.findSimilar(pipe); existing != nil {
		rb.mu.Lock()
		existing.Uses++
		existing.Successes++
		existing.LastUsed = time.Now()
		rb.mu.Unlock()
		rb.save()
		return
	}

	// Extract keywords from input
	keywords := extractKeywords(rawInput)

	// Build parameterized steps
	steps, params := parameterize(pipe.steps)

	recipe := Recipe{
		ID:        fmt.Sprintf("recipe_%d", time.Now().UnixNano()),
		Name:      generateRecipeName(pipe.steps),
		Trigger:   intent,
		Keywords:  keywords,
		Steps:     steps,
		Params:    params,
		Uses:      1,
		Successes: 1,
		LastUsed:  time.Now(),
		CreatedAt: time.Now(),
	}

	rb.mu.Lock()
	rb.recipes = append(rb.recipes, recipe)
	// Cap at 50 recipes, prune lowest confidence
	if len(rb.recipes) > 50 {
		rb.prune()
	}
	rb.mu.Unlock()

	rb.save()
}

// Match finds recipes that could handle the given query.
// Returns recipes sorted by relevance (keyword overlap + confidence).
func (rb *RecipeBook) Match(intent string, rawInput string) []Recipe {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	inputWords := extractKeywords(rawInput)
	if len(inputWords) == 0 {
		return nil
	}

	type scored struct {
		recipe Recipe
		score  float64
	}

	var matches []scored

	for _, r := range rb.recipes {
		score := 0.0

		// Intent match
		if r.Trigger == intent {
			score += 1.0
		}

		// Keyword overlap
		overlap := keywordOverlap(r.Keywords, inputWords)
		score += overlap * 2.0

		// Confidence bonus
		score += r.Confidence() * 0.5

		// Recency bonus (recipes used in last hour get a boost)
		if time.Since(r.LastUsed) < time.Hour {
			score += 0.3
		}

		if score >= 1.5 { // minimum threshold
			matches = append(matches, scored{recipe: r, score: score})
		}
	}

	// Sort by score descending
	for i := 0; i < len(matches); i++ {
		for j := i + 1; j < len(matches); j++ {
			if matches[j].score > matches[i].score {
				matches[i], matches[j] = matches[j], matches[i]
			}
		}
	}

	result := make([]Recipe, 0, len(matches))
	for _, m := range matches {
		result = append(result, m.recipe)
		if len(result) >= 3 {
			break
		}
	}

	return result
}

// Replay returns the steps of a recipe with parameters substituted.
// paramValues maps parameter names (e.g., "$FILE") to concrete values.
func (rb *RecipeBook) Replay(recipeID string, paramValues map[string]string) ([]RecipeStep, error) {
	rb.mu.RLock()
	var recipe *Recipe
	for i := range rb.recipes {
		if rb.recipes[i].ID == recipeID {
			recipe = &rb.recipes[i]
			break
		}
	}
	rb.mu.RUnlock()

	if recipe == nil {
		return nil, fmt.Errorf("recipe not found: %s", recipeID)
	}

	// Substitute parameters
	var steps []RecipeStep
	for _, step := range recipe.Steps {
		newArgs := make(map[string]string)
		for k, v := range step.Args {
			resolved := v
			for param, val := range paramValues {
				resolved = strings.ReplaceAll(resolved, param, val)
			}
			newArgs[k] = resolved
		}
		steps = append(steps, RecipeStep{
			Tool: step.Tool,
			Args: newArgs,
		})
	}

	// Track usage
	rb.mu.Lock()
	recipe.Uses++
	recipe.LastUsed = time.Now()
	rb.mu.Unlock()

	return steps, nil
}

// ReportSuccess marks the last replay of a recipe as successful.
func (rb *RecipeBook) ReportSuccess(recipeID string) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	for i := range rb.recipes {
		if rb.recipes[i].ID == recipeID {
			rb.recipes[i].Successes++
			break
		}
	}
	rb.save()
}

// Size returns the number of stored recipes.
func (rb *RecipeBook) Size() int {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return len(rb.recipes)
}

// List returns all recipes.
func (rb *RecipeBook) List() []Recipe {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	out := make([]Recipe, len(rb.recipes))
	copy(out, rb.recipes)
	return out
}

// --- internal helpers ---

// findSimilar checks if an existing recipe matches the same tool sequence.
func (rb *RecipeBook) findSimilar(pipe *Pipeline) *Recipe {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	for i := range rb.recipes {
		r := &rb.recipes[i]
		if len(r.Steps) != pipe.StepCount() {
			continue
		}
		match := true
		for j, step := range r.Steps {
			if step.Tool != pipe.steps[j].ToolName {
				match = false
				break
			}
		}
		if match {
			return r
		}
	}
	return nil
}

// prune removes the lowest-confidence recipes to stay under the cap.
func (rb *RecipeBook) prune() {
	// Sort by confidence ascending
	for i := 0; i < len(rb.recipes); i++ {
		for j := i + 1; j < len(rb.recipes); j++ {
			if rb.recipes[j].Confidence() < rb.recipes[i].Confidence() {
				rb.recipes[i], rb.recipes[j] = rb.recipes[j], rb.recipes[i]
			}
		}
	}
	// Keep the best 40
	if len(rb.recipes) > 40 {
		rb.recipes = rb.recipes[len(rb.recipes)-40:]
	}
}

// parameterize extracts concrete values from pipeline steps and replaces
// them with named parameters like $FILE, $PATTERN, $DIR.
func parameterize(steps []StepResult) ([]RecipeStep, []string) {
	paramSet := make(map[string]bool)
	var recipeSteps []RecipeStep

	for _, step := range steps {
		rs := RecipeStep{
			Tool: step.ToolName,
			Args: make(map[string]string),
		}

		// Extract args from summary (heuristic: look for path-like values)
		summary := step.Summary

		// Detect file paths and replace with $FILE
		words := strings.Fields(summary)
		for _, w := range words {
			w = strings.TrimRight(w, ":,")
			if looksLikePath(w) {
				param := "$FILE"
				if strings.HasSuffix(w, "/") {
					param = "$DIR"
				}
				paramSet[param] = true
				rs.Args["path"] = param
				break
			}
		}

		recipeSteps = append(recipeSteps, rs)
	}

	var params []string
	for p := range paramSet {
		params = append(params, p)
	}

	return recipeSteps, params
}

func looksLikePath(s string) bool {
	return strings.Contains(s, "/") || strings.Contains(s, ".go") ||
		strings.Contains(s, ".py") || strings.Contains(s, ".js") ||
		strings.Contains(s, ".ts") || strings.Contains(s, ".md")
}

// generateRecipeName creates a human-readable name from the step sequence.
func generateRecipeName(steps []StepResult) string {
	if len(steps) == 0 {
		return "empty"
	}

	tools := make([]string, 0, len(steps))
	seen := make(map[string]bool)
	for _, s := range steps {
		if !seen[s.ToolName] {
			seen[s.ToolName] = true
			tools = append(tools, s.ToolName)
		}
	}

	if len(tools) <= 3 {
		return strings.Join(tools, "→")
	}
	return fmt.Sprintf("%s→...→%s (%d steps)", tools[0], tools[len(tools)-1], len(steps))
}

// extractKeywords pulls meaningful words from input (3+ chars, lowercased).
func extractKeywords(input string) []string {
	words := strings.Fields(strings.ToLower(input))
	var keywords []string
	stopWords := map[string]bool{
		"the": true, "and": true, "for": true, "that": true, "this": true,
		"with": true, "from": true, "are": true, "was": true, "have": true,
		"has": true, "can": true, "will": true, "how": true, "what": true,
		"where": true, "when": true, "who": true, "which": true, "does": true,
		"please": true, "could": true, "would": true, "should": true,
	}

	for _, w := range words {
		w = strings.Trim(w, ".,!?;:'\"()[]")
		if len(w) >= 3 && !stopWords[w] {
			keywords = append(keywords, w)
		}
	}

	return keywords
}

// keywordOverlap returns 0.0-1.0 representing how many keywords match.
func keywordOverlap(a, b []string) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	setB := make(map[string]bool)
	for _, w := range b {
		setB[w] = true
	}

	matches := 0
	for _, w := range a {
		if setB[w] {
			matches++
		}
	}

	return float64(matches) / float64(len(a))
}

func (rb *RecipeBook) load() {
	if rb.storePath == "" {
		return
	}
	path := filepath.Join(rb.storePath, "recipes.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	rb.mu.Lock()
	defer rb.mu.Unlock()
	_ = json.Unmarshal(data, &rb.recipes)
}

func (rb *RecipeBook) save() {
	if rb.storePath == "" {
		return
	}
	rb.mu.RLock()
	data, err := json.MarshalIndent(rb.recipes, "", "  ")
	rb.mu.RUnlock()
	if err != nil {
		return
	}
	_ = safefile.WriteAtomic(filepath.Join(rb.storePath, "recipes.json"), data, 0644)
}
