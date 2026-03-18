package tools

import (
	"strings"
	"testing"
)

func TestCalculatorBasicArithmetic(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"2 + 3", "5"},
		{"10 - 4", "6"},
		{"3 * 7", "21"},
		{"10 / 3", "3.333333"},
		{"10 % 3", "1"},
		{"100 + 200", "300"},
		{"-5 + 3", "-2"},
		{"2.5 * 4", "10"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorParentheses(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"(2 + 3) * 4", "20"},
		{"2 * (3 + 4)", "14"},
		{"((1 + 2) * (3 + 4))", "21"},
		{"(10 - 2) / (4 - 2)", "4"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorExponents(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"2^10", "1024"},
		{"2**8", "256"},
		{"3^2", "9"},
		{"2^0", "1"},
		{"2^-1", "0.5"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorFunctions(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"sqrt(144)", "12"},
		{"abs(-5)", "5"},
		{"abs(5)", "5"},
		{"log2(8)", "3"},
		{"log10(1000)", "3"},
		{"ceil(3.2)", "4"},
		{"floor(3.9)", "3"},
		{"round(3.5)", "4"},
		{"round(3.4)", "3"},
		{"min(3, 7)", "3"},
		{"max(3, 7)", "7"},
		{"pow(2, 10)", "1024"},
		{"min(1, 2, 3)", "1"},
		{"max(5, 2, 8, 1)", "8"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorConstants(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"pi * 2", "6.283185"},
		{"e^2", "7.389056"},
		{"pi", "3.141593"},
		{"e", "2.718282"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorPercentages(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"15% of 847", "127.05"},
		{"20% off 100", "80"},
		{"50% of 200", "100"},
		{"10% off 50", "45"},
		{"100% of 42", "42"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorComplex(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"sqrt(144) + 3^4", "93"},
		{"2 * (3 + 4) - 1", "13"},
		{"(1 + 2) * (3 + 4) / 3", "7"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorNested(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"sqrt(abs(-16))", "4"},
		{"abs(min(-3, -7))", "7"},
		{"sqrt(pow(3, 2) + pow(4, 2))", "5"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorImplicitMultiplication(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"2(3+4)", "14"},
		{"3pi", "9.424778"},
		{"2e", "5.436564"},
	}

	for _, tt := range tests {
		result, err := EvalExpression(tt.expr)
		if err != nil {
			t.Errorf("EvalExpression(%q) error: %v", tt.expr, err)
			continue
		}
		got := FormatResult(result)
		if got != tt.expected {
			t.Errorf("EvalExpression(%q) = %q, want %q", tt.expr, got, tt.expected)
		}
	}
}

func TestCalculatorDivisionByZero(t *testing.T) {
	_, err := EvalExpression("1 / 0")
	if err == nil {
		t.Error("expected error for division by zero, got nil")
	}
	if !strings.Contains(err.Error(), "division by zero") {
		t.Errorf("expected 'division by zero' error, got: %v", err)
	}

	_, err = EvalExpression("10 % 0")
	if err == nil {
		t.Error("expected error for modulo by zero, got nil")
	}
}

func TestCalculatorInvalidExpressions(t *testing.T) {
	invalids := []string{
		"",
		"2 +",
		"* 3",
		"(2 + 3",
		"hello",
		"2 @ 3",
		"sqrt()",
		"min(1)",
	}

	for _, expr := range invalids {
		_, err := EvalExpression(expr)
		if err == nil {
			t.Errorf("expected error for %q, got nil", expr)
		}
	}
}

func TestCalculatorToolRegistration(t *testing.T) {
	r := NewRegistry()
	RegisterCalculatorTools(r)

	tool, err := r.Get("calculator")
	if err != nil {
		t.Fatalf("calculator tool not registered: %v", err)
	}

	if tool.Name != "calculator" {
		t.Errorf("tool name = %q, want %q", tool.Name, "calculator")
	}

	// Test via tool execution
	result, err := tool.Execute(map[string]string{"expression": "2 + 3"})
	if err != nil {
		t.Fatalf("tool execution error: %v", err)
	}
	if result != "5" {
		t.Errorf("tool result = %q, want %q", result, "5")
	}

	// Test missing expression
	_, err = tool.Execute(map[string]string{})
	if err == nil {
		t.Error("expected error for missing expression, got nil")
	}
}
