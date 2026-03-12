package cognitive

import "testing"

func TestParseStepArgsParsesNamedArgs(t *testing.T) {
	args := parseStepArgs(`path=bitcoin.md, content="# Bitcoin\nSummary", url=https://example.com`)

	if args["path"] != "bitcoin.md" {
		t.Fatalf("expected path to be parsed, got %v", args)
	}
	if args["content"] != "# Bitcoin\nSummary" {
		t.Fatalf("expected content to preserve newline, got %q", args["content"])
	}
	if args["url"] != "https://example.com" {
		t.Fatalf("expected url to be parsed, got %v", args)
	}
}

func TestParsePlanExtractsNamedArgs(t *testing.T) {
	p := &Planner{}
	plan := p.parsePlan("goal-1", `STEP: Fetch source | TOOL: fetch | ARGS: url=https://example.com
STEP: Write file | TOOL: write | ARGS: path=bitcoin.md, content="# Bitcoin\nSummary"`)

	if len(plan.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(plan.Steps))
	}
	if plan.Steps[0].Args["url"] != "https://example.com" {
		t.Fatalf("expected fetch url arg, got %v", plan.Steps[0].Args)
	}
	if plan.Steps[1].Args["path"] != "bitcoin.md" {
		t.Fatalf("expected write path arg, got %v", plan.Steps[1].Args)
	}
	if plan.Steps[1].Args["content"] != "# Bitcoin\nSummary" {
		t.Fatalf("expected write content arg, got %v", plan.Steps[1].Args)
	}
}
