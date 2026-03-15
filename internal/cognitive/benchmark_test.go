package cognitive

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// --- Pipeline Benchmarks ---

func BenchmarkCompressStep_Read(b *testing.B) {
	var lines []string
	for i := 0; i < 100; i++ {
		lines = append(lines, fmt.Sprintf("line %d: func Handle(w http.ResponseWriter, r *http.Request) {", i))
	}
	result := strings.Join(lines, "\n")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CompressStep("read", result)
	}
}

func BenchmarkCompressStep_Grep(b *testing.B) {
	var lines []string
	for i := 0; i < 50; i++ {
		lines = append(lines, fmt.Sprintf("internal/pkg%d/handler.go:42:func Handle() error {", i))
	}
	result := strings.Join(lines, "\n")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CompressStep("grep", result)
	}
}

func BenchmarkCompressStep_Glob(b *testing.B) {
	var lines []string
	for i := 0; i < 30; i++ {
		lines = append(lines, fmt.Sprintf("internal/pkg%d/handler.go", i))
	}
	result := strings.Join(lines, "\n")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CompressStep("glob", result)
	}
}

func BenchmarkPipelineAddStep(b *testing.B) {
	result := strings.Repeat("package main\nfunc main() {}\n", 50)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := NewPipeline("what does this code do?")
		for j := 0; j < 5; j++ {
			p.AddStep("read", result)
		}
	}
}

func BenchmarkPipelineBuildContext(b *testing.B) {
	p := NewPipeline("explain the architecture")
	for i := 0; i < 10; i++ {
		p.AddStep("read", fmt.Sprintf("content of file %d with many lines of code\n", i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.BuildContext()
	}
}

// --- Grounding Benchmarks ---

func BenchmarkSmartTruncate_ShortRead(b *testing.B) {
	content := "package main\nfunc main() {}\n"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SmartTruncate("read", content)
	}
}

func BenchmarkSmartTruncate_LongRead(b *testing.B) {
	var lines []string
	for i := 0; i < 200; i++ {
		lines = append(lines, fmt.Sprintf("line %d: %s", i, strings.Repeat("code", 20)))
	}
	content := strings.Join(lines, "\n")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SmartTruncate("read", content)
	}
}

func BenchmarkContextBudgetEstimate(b *testing.B) {
	budget := DefaultBudget()
	text := strings.Repeat("hello world this is a test ", 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		budget.EstimateTokens(text)
	}
}

func BenchmarkReflectionGateCheck(b *testing.B) {
	g := &ReflectionGate{}
	result := "some file content that was read from disk successfully"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%6 == 0 {
			g.Reset()
		}
		g.Check("read", result, nil)
	}
}

// --- Query Classification Benchmarks ---

func BenchmarkClassifyQuery_Fast(b *testing.B) {
	c := &FastPathClassifier{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ClassifyQuery("hello!")
	}
}

func BenchmarkClassifyQuery_Medium(b *testing.B) {
	c := &FastPathClassifier{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ClassifyQuery("explain how garbage collection works in Go")
	}
}

func BenchmarkClassifyQuery_Full(b *testing.B) {
	c := &FastPathClassifier{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ClassifyQuery("read file main.go and show me the main function")
	}
}

func BenchmarkClassifyQuery_AllPatterns(b *testing.B) {
	c := &FastPathClassifier{}
	// Worst case: query matches no patterns and falls through all checks
	query := "this is a moderately long ambiguous query that doesn't match any specific pattern in the classifier system"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ClassifyQuery(query)
	}
}

// --- Diff Algorithm Benchmarks ---

func BenchmarkLCSEditScript_Small(b *testing.B) {
	old := []string{"a", "b", "c", "d", "e"}
	new := []string{"a", "X", "c", "Y", "e"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lcsEditScript(old, new)
	}
}

func BenchmarkLCSEditScript_Medium(b *testing.B) {
	var old, new []string
	for i := 0; i < 50; i++ {
		old = append(old, fmt.Sprintf("line %d original", i))
		if i%5 == 0 {
			new = append(new, fmt.Sprintf("line %d modified", i))
		} else {
			new = append(new, fmt.Sprintf("line %d original", i))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lcsEditScript(old, new)
	}
}

func BenchmarkLCSEditScript_Large(b *testing.B) {
	var old, new []string
	for i := 0; i < 200; i++ {
		old = append(old, fmt.Sprintf("line %d content here", i))
		if i%10 == 0 {
			new = append(new, fmt.Sprintf("line %d changed", i))
		} else {
			new = append(new, fmt.Sprintf("line %d content here", i))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lcsEditScript(old, new)
	}
}

func BenchmarkDiffPreview_SmallChange(b *testing.B) {
	old := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	new := "line 1\nline 2 changed\nline 3\nline 4\nline 5\n"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DiffPreview(old, new, "test.go")
	}
}

// --- Recipe Benchmarks ---

func BenchmarkExtractKeywords(b *testing.B) {
	query := "How can I find the function definition for NewReasoner in the cognitive package?"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		extractKeywords(query)
	}
}

func BenchmarkKeywordOverlap(b *testing.B) {
	a := []string{"find", "function", "definition", "reasoner", "cognitive", "package"}
	bSlice := []string{"find", "method", "definition", "perceiver", "cognitive", "module"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		keywordOverlap(a, bSlice)
	}
}

func BenchmarkRecipeBookMatch(b *testing.B) {
	rb := NewRecipeBook("")

	// Populate with 40 recipes
	for i := 0; i < 40; i++ {
		rb.mu.Lock()
		rb.recipes = append(rb.recipes, Recipe{
			ID:        fmt.Sprintf("recipe_%d", i),
			Trigger:   "question",
			Keywords:  []string{"find", "function", fmt.Sprintf("word%d", i)},
			Steps:     []RecipeStep{{Tool: "grep"}, {Tool: "read"}},
			Uses:      i + 1,
			Successes: i,
			LastUsed:  time.Now(),
		})
		rb.mu.Unlock()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rb.Match("question", "find the function definition for NewReasoner")
	}
}

// --- Predictor Benchmarks ---

func BenchmarkCacheKey(b *testing.B) {
	args := map[string]string{"path": "internal/cognitive/reasoner.go", "pattern": "func.*New"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheKey("grep", args)
	}
}

func BenchmarkPredictorLookup(b *testing.B) {
	p := NewPredictor(mockRegistry())

	// Pre-populate cache
	p.mu.Lock()
	for i := 0; i < 20; i++ {
		key := cacheKey("read", map[string]string{"path": fmt.Sprintf("file%d.go", i)})
		p.cache[key] = Prediction{
			ToolName:  "read",
			Result:    "content",
			CreatedAt: time.Now(),
		}
	}
	p.mu.Unlock()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Lookup("read", map[string]string{"path": fmt.Sprintf("file%d.go", i%20)})
	}
}

// --- Helper Benchmarks ---

func BenchmarkIsReadOnly(b *testing.B) {
	tools := []string{"read", "ls", "write", "grep", "shell", "glob"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		isReadOnly(tools[i%len(tools)])
	}
}

func BenchmarkLooksLikeFile(b *testing.B) {
	inputs := []string{"main.go", "internal/pkg/handler.go", "-rw-r--r--", "README.md", "just-text"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		looksLikeFile(inputs[i%len(inputs)])
	}
}

func BenchmarkShortHash(b *testing.B) {
	data := strings.Repeat("some tool output content that gets hashed for repetition detection", 10)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		shortHash(data)
	}
}
