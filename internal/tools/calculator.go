package tools

import (
	"fmt"
	"math"
	"strings"
	"unicode"
)

// tokenType represents the type of a lexer token.
type tokenType int

const (
	tokNumber tokenType = iota
	tokPlus
	tokMinus
	tokMul
	tokDiv
	tokMod
	tokPow
	tokLParen
	tokRParen
	tokIdent
	tokComma
	tokEOF
)

type token struct {
	typ tokenType
	val string
	num float64
}

// tokenizer breaks an expression string into tokens.
type tokenizer struct {
	input []rune
	pos   int
}

func newTokenizer(input string) *tokenizer {
	return &tokenizer{input: []rune(input), pos: 0}
}

func (t *tokenizer) peek() rune {
	if t.pos >= len(t.input) {
		return 0
	}
	return t.input[t.pos]
}

func (t *tokenizer) advance() rune {
	r := t.input[t.pos]
	t.pos++
	return r
}

func (t *tokenizer) skipWhitespace() {
	for t.pos < len(t.input) && unicode.IsSpace(t.input[t.pos]) {
		t.pos++
	}
}

func (t *tokenizer) tokenize() ([]token, error) {
	var tokens []token
	for {
		t.skipWhitespace()
		if t.pos >= len(t.input) {
			tokens = append(tokens, token{typ: tokEOF})
			return tokens, nil
		}

		r := t.peek()

		// Number (including decimals like .5)
		if unicode.IsDigit(r) || (r == '.' && t.pos+1 < len(t.input) && unicode.IsDigit(t.input[t.pos+1])) {
			tok, err := t.readNumber()
			if err != nil {
				return nil, err
			}
			tokens = append(tokens, tok)
			continue
		}

		// Identifier (function name or constant)
		if unicode.IsLetter(r) || r == '_' {
			tok := t.readIdent()
			tokens = append(tokens, tok)
			continue
		}

		switch r {
		case '+':
			t.advance()
			tokens = append(tokens, token{typ: tokPlus, val: "+"})
		case '-':
			t.advance()
			tokens = append(tokens, token{typ: tokMinus, val: "-"})
		case '*':
			t.advance()
			if t.pos < len(t.input) && t.input[t.pos] == '*' {
				t.advance()
				tokens = append(tokens, token{typ: tokPow, val: "**"})
			} else {
				tokens = append(tokens, token{typ: tokMul, val: "*"})
			}
		case '/':
			t.advance()
			tokens = append(tokens, token{typ: tokDiv, val: "/"})
		case '%':
			t.advance()
			tokens = append(tokens, token{typ: tokMod, val: "%"})
		case '^':
			t.advance()
			tokens = append(tokens, token{typ: tokPow, val: "^"})
		case '(':
			t.advance()
			tokens = append(tokens, token{typ: tokLParen, val: "("})
		case ')':
			t.advance()
			tokens = append(tokens, token{typ: tokRParen, val: ")"})
		case ',':
			t.advance()
			tokens = append(tokens, token{typ: tokComma, val: ","})
		default:
			return nil, fmt.Errorf("unexpected character: %c", r)
		}
	}
}

func (t *tokenizer) readNumber() (token, error) {
	start := t.pos
	hasDot := false
	for t.pos < len(t.input) {
		r := t.input[t.pos]
		if unicode.IsDigit(r) {
			t.pos++
		} else if r == '.' && !hasDot {
			hasDot = true
			t.pos++
		} else {
			break
		}
	}
	s := string(t.input[start:t.pos])
	var num float64
	_, err := fmt.Sscanf(s, "%f", &num)
	if err != nil {
		return token{}, fmt.Errorf("invalid number: %s", s)
	}
	return token{typ: tokNumber, val: s, num: num}, nil
}

func (t *tokenizer) readIdent() token {
	start := t.pos
	for t.pos < len(t.input) && (unicode.IsLetter(t.input[t.pos]) || unicode.IsDigit(t.input[t.pos]) || t.input[t.pos] == '_') {
		t.pos++
	}
	s := string(t.input[start:t.pos])
	return token{typ: tokIdent, val: s}
}

// parser implements a recursive descent parser for math expressions.
type parser struct {
	tokens []token
	pos    int
}

func newParser(tokens []token) *parser {
	return &parser{tokens: tokens, pos: 0}
}

func (p *parser) peek() token {
	if p.pos >= len(p.tokens) {
		return token{typ: tokEOF}
	}
	return p.tokens[p.pos]
}

func (p *parser) advance() token {
	t := p.tokens[p.pos]
	p.pos++
	return t
}

func (p *parser) expect(typ tokenType) (token, error) {
	t := p.peek()
	if t.typ != typ {
		return t, fmt.Errorf("expected token type %d, got %d (%q)", typ, t.typ, t.val)
	}
	return p.advance(), nil
}

// parse is the entry point: handles the full expression.
func (p *parser) parse() (float64, error) {
	result, err := p.parseExpr()
	if err != nil {
		return 0, err
	}
	if p.peek().typ != tokEOF {
		return 0, fmt.Errorf("unexpected token: %q", p.peek().val)
	}
	return result, nil
}

// parseExpr handles addition and subtraction.
func (p *parser) parseExpr() (float64, error) {
	left, err := p.parseTerm()
	if err != nil {
		return 0, err
	}

	for {
		t := p.peek()
		if t.typ == tokPlus {
			p.advance()
			right, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left += right
		} else if t.typ == tokMinus {
			p.advance()
			right, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left -= right
		} else {
			break
		}
	}
	return left, nil
}

// parseTerm handles multiplication, division, modulo, and implicit multiplication.
func (p *parser) parseTerm() (float64, error) {
	left, err := p.parsePower()
	if err != nil {
		return 0, err
	}

	for {
		t := p.peek()
		if t.typ == tokMul {
			p.advance()
			right, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			left *= right
		} else if t.typ == tokDiv {
			p.advance()
			right, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
		} else if t.typ == tokMod {
			p.advance()
			right, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left = math.Mod(left, right)
		} else if t.typ == tokLParen || t.typ == tokIdent {
			// Implicit multiplication: 2(3+4) or 2pi
			right, err := p.parsePower()
			if err != nil {
				return 0, err
			}
			left *= right
		} else {
			break
		}
	}
	return left, nil
}

// parsePower handles exponentiation (right-associative).
func (p *parser) parsePower() (float64, error) {
	base, err := p.parseUnary()
	if err != nil {
		return 0, err
	}

	if p.peek().typ == tokPow {
		p.advance()
		exp, err := p.parsePower() // right-associative
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

// parseUnary handles unary plus/minus.
func (p *parser) parseUnary() (float64, error) {
	if p.peek().typ == tokMinus {
		p.advance()
		val, err := p.parseUnary()
		if err != nil {
			return 0, err
		}
		return -val, nil
	}
	if p.peek().typ == tokPlus {
		p.advance()
		return p.parseUnary()
	}
	return p.parsePrimary()
}

// parsePrimary handles numbers, constants, functions, and parenthesized expressions.
func (p *parser) parsePrimary() (float64, error) {
	t := p.peek()

	switch t.typ {
	case tokNumber:
		p.advance()
		return t.num, nil

	case tokLParen:
		p.advance()
		val, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		_, err = p.expect(tokRParen)
		if err != nil {
			return 0, fmt.Errorf("missing closing parenthesis")
		}
		return val, nil

	case tokIdent:
		name := strings.ToLower(t.val)

		// Check if it's a function call (next token is '(')
		if p.pos+1 < len(p.tokens) && p.tokens[p.pos+1].typ == tokLParen {
			return p.parseFunction(name)
		}

		// Otherwise it's a constant
		p.advance()
		switch name {
		case "pi":
			return math.Pi, nil
		case "e":
			return math.E, nil
		default:
			return 0, fmt.Errorf("unknown identifier: %s", t.val)
		}

	default:
		return 0, fmt.Errorf("unexpected token: %q", t.val)
	}
}

// parseFunction parses a function call like sqrt(x) or max(a, b).
func (p *parser) parseFunction(name string) (float64, error) {
	p.advance() // consume function name
	p.advance() // consume '('

	// Collect arguments
	var args []float64
	if p.peek().typ != tokRParen {
		val, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		args = append(args, val)

		for p.peek().typ == tokComma {
			p.advance() // consume ','
			val, err = p.parseExpr()
			if err != nil {
				return 0, err
			}
			args = append(args, val)
		}
	}

	_, err := p.expect(tokRParen)
	if err != nil {
		return 0, fmt.Errorf("missing closing parenthesis in function %s", name)
	}

	return evalFunction(name, args)
}

func evalFunction(name string, args []float64) (float64, error) {
	switch name {
	case "sqrt":
		if len(args) != 1 {
			return 0, fmt.Errorf("sqrt requires 1 argument")
		}
		if args[0] < 0 {
			return 0, fmt.Errorf("sqrt of negative number")
		}
		return math.Sqrt(args[0]), nil
	case "abs":
		if len(args) != 1 {
			return 0, fmt.Errorf("abs requires 1 argument")
		}
		return math.Abs(args[0]), nil
	case "sin":
		if len(args) != 1 {
			return 0, fmt.Errorf("sin requires 1 argument")
		}
		return math.Sin(args[0]), nil
	case "cos":
		if len(args) != 1 {
			return 0, fmt.Errorf("cos requires 1 argument")
		}
		return math.Cos(args[0]), nil
	case "tan":
		if len(args) != 1 {
			return 0, fmt.Errorf("tan requires 1 argument")
		}
		return math.Tan(args[0]), nil
	case "log":
		if len(args) != 1 {
			return 0, fmt.Errorf("log requires 1 argument")
		}
		if args[0] <= 0 {
			return 0, fmt.Errorf("log of non-positive number")
		}
		return math.Log(args[0]), nil
	case "log2":
		if len(args) != 1 {
			return 0, fmt.Errorf("log2 requires 1 argument")
		}
		if args[0] <= 0 {
			return 0, fmt.Errorf("log2 of non-positive number")
		}
		return math.Log2(args[0]), nil
	case "log10":
		if len(args) != 1 {
			return 0, fmt.Errorf("log10 requires 1 argument")
		}
		if args[0] <= 0 {
			return 0, fmt.Errorf("log10 of non-positive number")
		}
		return math.Log10(args[0]), nil
	case "ceil":
		if len(args) != 1 {
			return 0, fmt.Errorf("ceil requires 1 argument")
		}
		return math.Ceil(args[0]), nil
	case "floor":
		if len(args) != 1 {
			return 0, fmt.Errorf("floor requires 1 argument")
		}
		return math.Floor(args[0]), nil
	case "round":
		if len(args) != 1 {
			return 0, fmt.Errorf("round requires 1 argument")
		}
		return math.Round(args[0]), nil
	case "min":
		if len(args) < 2 {
			return 0, fmt.Errorf("min requires at least 2 arguments")
		}
		result := args[0]
		for _, a := range args[1:] {
			if a < result {
				result = a
			}
		}
		return result, nil
	case "max":
		if len(args) < 2 {
			return 0, fmt.Errorf("max requires at least 2 arguments")
		}
		result := args[0]
		for _, a := range args[1:] {
			if a > result {
				result = a
			}
		}
		return result, nil
	case "pow":
		if len(args) != 2 {
			return 0, fmt.Errorf("pow requires 2 arguments")
		}
		return math.Pow(args[0], args[1]), nil
	default:
		return 0, fmt.Errorf("unknown function: %s", name)
	}
}

// preProcessPercentage handles "X% of Y" and "X% off Y" patterns before tokenization.
func preProcessPercentage(input string) string {
	lower := strings.ToLower(input)

	// Match "X% of Y" → (X/100)*Y
	// Match "X% off Y" → Y - (X/100)*Y = Y*(1 - X/100)
	// We need to find the pattern and replace it.

	// Find "% of " or "% off "
	for {
		idx := strings.Index(lower, "% of ")
		if idx == -1 {
			break
		}
		// Find the number before %
		numStart := idx - 1
		for numStart >= 0 && (lower[numStart] == '.' || (lower[numStart] >= '0' && lower[numStart] <= '9')) {
			numStart--
		}
		numStart++
		if numStart >= idx {
			break
		}
		numStr := input[numStart:idx]
		rest := input[idx+len("% of "):]
		input = fmt.Sprintf("(%s/100)*(%s)", numStr, rest)
		lower = strings.ToLower(input)
	}

	for {
		idx := strings.Index(lower, "% off ")
		if idx == -1 {
			break
		}
		numStart := idx - 1
		for numStart >= 0 && (lower[numStart] == '.' || (lower[numStart] >= '0' && lower[numStart] <= '9')) {
			numStart--
		}
		numStart++
		if numStart >= idx {
			break
		}
		numStr := input[numStart:idx]
		rest := input[idx+len("% off "):]
		input = fmt.Sprintf("(%s)*(1-%s/100)", rest, numStr)
		lower = strings.ToLower(input)
	}

	return input
}

// EvalExpression evaluates a mathematical expression string and returns the result.
func EvalExpression(input string) (float64, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return 0, fmt.Errorf("empty expression")
	}

	// Pre-process percentage patterns
	input = preProcessPercentage(input)

	// Handle remaining % as modulo (already handled by tokenizer)

	t := newTokenizer(input)
	tokens, err := t.tokenize()
	if err != nil {
		return 0, err
	}

	p := newParser(tokens)
	result, err := p.parse()
	if err != nil {
		return 0, err
	}

	// Check for NaN/Inf
	if math.IsNaN(result) {
		return 0, fmt.Errorf("result is not a number")
	}
	if math.IsInf(result, 0) {
		return 0, fmt.Errorf("result is infinite")
	}

	return result, nil
}

// FormatResult formats a calculation result: integers without decimals, floats to reasonable precision.
func FormatResult(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return fmt.Sprintf("%.0f", v)
	}
	// Format with 6 decimal places, trim trailing zeros
	s := fmt.Sprintf("%.6f", v)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

// RegisterCalculatorTools adds the calculator tool to the registry.
func RegisterCalculatorTools(r *Registry) {
	r.Register(Tool{
		Name:        "calculator",
		Description: "Evaluate math expressions. Args: expression (required, e.g. '2+3', 'sqrt(144)', '15% of 847').",
		Execute: func(args map[string]string) (string, error) {
			expr := args["expression"]
			if expr == "" {
				return "", fmt.Errorf("calculator requires 'expression' argument")
			}

			result, err := EvalExpression(expr)
			if err != nil {
				return "", err
			}

			return FormatResult(result), nil
		},
	})
}
