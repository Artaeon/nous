package tools

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Unit conversion tables: each unit maps to its base unit factor.
// To convert A -> B: value * factorA / factorB (both relative to base).

// Length base unit: meter
var lengthUnits = map[string]float64{
	"mm": 0.001,
	"cm": 0.01,
	"m":  1,
	"km": 1000,
	"in": 0.0254,
	"ft": 0.3048,
	"yd": 0.9144,
	"mi": 1609.344,
	"nm": 1852, // nautical mile
}

// Weight base unit: gram
var weightUnits = map[string]float64{
	"mg":    0.001,
	"g":     1,
	"kg":    1000,
	"lb":    453.59237,
	"oz":    28.349523125,
	"ton":   907184.74, // US short ton
	"stone": 6350.29318,
}

// Volume base unit: milliliter
var volumeUnits = map[string]float64{
	"ml":    1,
	"l":     1000,
	"gal":   3785.41178,
	"qt":    946.352946,
	"pt":    473.176473,
	"cup":   236.588236,
	"fl_oz": 29.5735296,
	"tbsp":  14.7867648,
	"tsp":   4.92892159,
}

// Speed base unit: m/s
var speedUnits = map[string]float64{
	"m/s":   1,
	"km/h":  1.0 / 3.6,
	"mph":   0.44704,
	"knots": 0.514444,
}

// Data base unit: byte (1024-based)
var dataUnits = map[string]float64{
	"b":  1,
	"kb": 1024,
	"mb": 1024 * 1024,
	"gb": 1024 * 1024 * 1024,
	"tb": 1024 * 1024 * 1024 * 1024,
	"pb": 1024 * 1024 * 1024 * 1024 * 1024,
}

// Time base unit: second
var timeUnits = map[string]float64{
	"s":     1,
	"min":   60,
	"h":     3600,
	"day":   86400,
	"week":  604800,
	"month": 2592000,  // 30 days
	"year":  31536000, // 365 days
}

// Area base unit: square meter
var areaUnits = map[string]float64{
	"m2":      1,
	"km2":     1e6,
	"ft2":     0.09290304,
	"acre":    4046.8564224,
	"hectare": 10000,
}

// Temperature units need special handling (not simple ratios).
var temperatureUnits = map[string]bool{
	"c": true, "f": true, "k": true,
}

// allCategories maps each unit to its category table for lookup.
var allCategories = []struct {
	name  string
	units map[string]float64
}{
	{"length", lengthUnits},
	{"weight", weightUnits},
	{"volume", volumeUnits},
	{"speed", speedUnits},
	{"data", dataUnits},
	{"time", timeUnits},
	{"area", areaUnits},
}

func normalizeUnit(u string) string {
	return strings.ToLower(strings.TrimSpace(u))
}

// findCategory returns the category name and conversion table containing the given unit.
func findCategory(unit string) (string, map[string]float64) {
	for _, cat := range allCategories {
		if _, ok := cat.units[unit]; ok {
			return cat.name, cat.units
		}
	}
	return "", nil
}

// ConvertUnits converts a value from one unit to another.
func ConvertUnits(value float64, from, to string) (float64, error) {
	from = normalizeUnit(from)
	to = normalizeUnit(to)

	if from == to {
		return value, nil
	}

	// Temperature special case
	if temperatureUnits[from] && temperatureUnits[to] {
		return convertTemperature(value, from, to)
	}

	// Check if one is temperature but the other isn't
	if temperatureUnits[from] || temperatureUnits[to] {
		return 0, fmt.Errorf("cannot convert between temperature and non-temperature units")
	}

	nameFrom, catFrom := findCategory(from)
	nameTo, catTo := findCategory(to)

	if catFrom == nil {
		return 0, fmt.Errorf("unknown unit: %s", from)
	}
	if catTo == nil {
		return 0, fmt.Errorf("unknown unit: %s", to)
	}

	// Both must be in the same category
	if nameFrom != nameTo {
		return 0, fmt.Errorf("incompatible units: %s and %s are in different categories", from, to)
	}

	// Convert: from -> base -> to
	baseValue := value * catFrom[from]
	result := baseValue / catTo[to]

	return result, nil
}

func convertTemperature(value float64, from, to string) (float64, error) {
	// First convert to Celsius as base
	var celsius float64
	switch from {
	case "c":
		celsius = value
	case "f":
		celsius = (value - 32) * 5.0 / 9.0
	case "k":
		celsius = value - 273.15
	default:
		return 0, fmt.Errorf("unknown temperature unit: %s", from)
	}

	// Then convert from Celsius to target
	switch to {
	case "c":
		return celsius, nil
	case "f":
		return celsius*9.0/5.0 + 32, nil
	case "k":
		return celsius + 273.15, nil
	default:
		return 0, fmt.Errorf("unknown temperature unit: %s", to)
	}
}

// Unit aliases for natural language parsing.
var unitAliases = map[string]string{
	// Length
	"millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
	"centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
	"meter": "m", "meters": "m", "metre": "m", "metres": "m",
	"kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
	"inch": "in", "inches": "in",
	"foot": "ft", "feet": "ft",
	"yard": "yd", "yards": "yd",
	"mile": "mi", "miles": "mi",
	"nautical mile": "nm", "nautical miles": "nm",
	// Weight
	"milligram": "mg", "milligrams": "mg",
	"gram": "g", "grams": "g",
	"kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
	"pound": "lb", "pounds": "lb", "lbs": "lb",
	"ounce": "oz", "ounces": "oz",
	"tons": "ton",
	"stones": "stone",
	// Temperature
	"celsius": "c", "fahrenheit": "f", "kelvin": "k",
	// Volume
	"milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
	"liter": "l", "liters": "l", "litre": "l", "litres": "l",
	"gallon": "gal", "gallons": "gal",
	"quart": "qt", "quarts": "qt",
	"pint": "pt", "pints": "pt",
	"cups": "cup",
	"fluid ounce": "fl_oz", "fluid ounces": "fl_oz", "fluid_ounce": "fl_oz", "fluid_ounces": "fl_oz",
	"tablespoon": "tbsp", "tablespoons": "tbsp",
	"teaspoon": "tsp", "teaspoons": "tsp",
	// Speed
	"kph": "km/h", "kmh": "km/h", "kmph": "km/h",
	"knot": "knots",
	// Data
	"byte": "b", "bytes": "b",
	"kilobyte": "kb", "kilobytes": "kb",
	"megabyte": "mb", "megabytes": "mb",
	"gigabyte": "gb", "gigabytes": "gb",
	"terabyte": "tb", "terabytes": "tb",
	"petabyte": "pb", "petabytes": "pb",
	// Time
	"second": "s", "seconds": "s", "sec": "s", "secs": "s",
	"minute": "min", "minutes": "min", "mins": "min",
	"hour": "h", "hours": "h", "hr": "h", "hrs": "h",
	"days": "day",
	"weeks": "week",
	"months": "month",
	"years": "year",
	// Area
	"square meter": "m2", "square meters": "m2", "square metre": "m2", "square metres": "m2", "sq m": "m2",
	"square kilometer": "km2", "square kilometers": "km2", "sq km": "km2",
	"square foot": "ft2", "square feet": "ft2", "sq ft": "ft2",
	"acres": "acre",
	"hectares": "hectare",
}

func resolveUnit(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if alias, ok := unitAliases[s]; ok {
		return alias
	}
	return s
}

// reConversion matches patterns like "100 km to miles", "5.5 pounds in kg"
var reConversion = regexp.MustCompile(`(?i)^\s*(-?[\d.]+)\s+(.+?)\s+(?:to|in|into|as)\s+(.+?)\s*$`)

// ParseConversion parses natural language conversion requests.
func ParseConversion(input string) (float64, string, string, error) {
	m := reConversion.FindStringSubmatch(input)
	if m == nil {
		return 0, "", "", fmt.Errorf("could not parse conversion from %q — expected format like '100 km to miles'", input)
	}

	value, err := strconv.ParseFloat(m[1], 64)
	if err != nil {
		return 0, "", "", fmt.Errorf("invalid number: %s", m[1])
	}

	from := resolveUnit(m[2])
	to := resolveUnit(m[3])

	return value, from, to, nil
}

// FormatConversion formats a conversion result nicely.
func FormatConversion(value float64, from string, result float64, to string) string {
	// Use appropriate precision
	fromStr := formatNumber(value)
	toStr := formatNumber(result)
	return fmt.Sprintf("%s %s = %s %s", fromStr, from, toStr, to)
}

func formatNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatFloat(v, 'f', 0, 64)
	}
	// Up to 6 significant decimal places, trim trailing zeros
	s := strconv.FormatFloat(v, 'f', 6, 64)
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	return s
}

// RegisterConvertTool adds the convert tool to the registry.
func RegisterConvertTool(r *Registry) {
	r.Register(Tool{
		Name:        "convert",
		Description: "Convert between units. Args: input (required, e.g. '100 km to miles', '72 fahrenheit to celsius').",
		Execute: func(args map[string]string) (string, error) {
			input := args["input"]
			if input == "" {
				// Try positional: value, from, to
				v := args["value"]
				from := args["from"]
				to := args["to"]
				if v == "" || from == "" || to == "" {
					return "", fmt.Errorf("convert requires 'input' (e.g. '100 km to miles') or 'value', 'from', 'to' arguments")
				}
				val, err := strconv.ParseFloat(v, 64)
				if err != nil {
					return "", fmt.Errorf("invalid value: %s", v)
				}
				from = resolveUnit(from)
				to = resolveUnit(to)
				result, err := ConvertUnits(val, from, to)
				if err != nil {
					return "", err
				}
				return FormatConversion(val, from, result, to), nil
			}

			value, from, to, err := ParseConversion(input)
			if err != nil {
				return "", err
			}

			result, err := ConvertUnits(value, from, to)
			if err != nil {
				return "", err
			}

			return FormatConversion(value, from, result, to), nil
		},
	})
}
