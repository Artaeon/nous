package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// weatherCodeDescription maps Open-Meteo WMO weather codes to human-readable descriptions.
var weatherCodeDescription = map[int]string{
	0:  "clear sky",
	1:  "mainly clear",
	2:  "partly cloudy",
	3:  "overcast",
	45: "fog",
	48: "depositing rime fog",
	51: "light drizzle",
	53: "moderate drizzle",
	55: "dense drizzle",
	61: "slight rain",
	63: "moderate rain",
	65: "heavy rain",
	71: "slight snow",
	73: "moderate snow",
	75: "heavy snow",
	80: "slight rain showers",
	81: "moderate rain showers",
	82: "violent rain showers",
	95: "thunderstorm",
}

// WeatherCodeToDescription converts a WMO weather code to a human-readable string.
func WeatherCodeToDescription(code int) string {
	if desc, ok := weatherCodeDescription[code]; ok {
		return desc
	}
	return fmt.Sprintf("unknown (code %d)", code)
}

// geocodeResult holds the response from the Open-Meteo geocoding API.
type geocodeResult struct {
	Results []struct {
		Name      string  `json:"name"`
		Latitude  float64 `json:"latitude"`
		Longitude float64 `json:"longitude"`
		Country   string  `json:"country"`
	} `json:"results"`
}

// currentWeatherResponse holds the response from the Open-Meteo forecast API.
type currentWeatherResponse struct {
	Current struct {
		Temperature  float64 `json:"temperature_2m"`
		Humidity     float64 `json:"relative_humidity_2m"`
		WindSpeed    float64 `json:"wind_speed_10m"`
		WeatherCode int     `json:"weather_code"`
	} `json:"current"`
}

// forecastResponse holds the multi-day forecast from the Open-Meteo API.
type forecastResponse struct {
	Daily struct {
		Time         []string  `json:"time"`
		TempMax      []float64 `json:"temperature_2m_max"`
		TempMin      []float64 `json:"temperature_2m_min"`
		WeatherCode  []int     `json:"weather_code"`
		Precipitation []float64 `json:"precipitation_sum"`
	} `json:"daily"`
}

// weatherHTTPClient allows overriding the HTTP client for testing.
var weatherHTTPClient *http.Client

func getWeatherHTTPClient() *http.Client {
	if weatherHTTPClient != nil {
		return weatherHTTPClient
	}
	return &http.Client{Timeout: 10 * time.Second}
}

// geocodeLocation looks up coordinates for a location name using the Open-Meteo geocoding API.
func geocodeLocation(location string) (name string, lat, lon float64, err error) {
	apiURL := "https://geocoding-api.open-meteo.com/v1/search?name=" + url.QueryEscape(location) + "&count=1"

	client := getWeatherHTTPClient()
	resp, err := client.Get(apiURL)
	if err != nil {
		return "", 0, 0, fmt.Errorf("weather: geocoding request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", 0, 0, fmt.Errorf("weather: geocoding HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", 0, 0, fmt.Errorf("weather: reading geocoding response: %w", err)
	}

	var result geocodeResult
	if err := json.Unmarshal(body, &result); err != nil {
		return "", 0, 0, fmt.Errorf("weather: invalid geocoding JSON: %w", err)
	}

	if len(result.Results) == 0 {
		return "", 0, 0, fmt.Errorf("weather: location %q not found", location)
	}

	r := result.Results[0]
	return r.Name, r.Latitude, r.Longitude, nil
}

// GetWeather fetches current weather for a location using the Open-Meteo API.
func GetWeather(location string) (string, error) {
	if location == "" {
		return "", fmt.Errorf("weather: location is required")
	}

	name, lat, lon, err := geocodeLocation(location)
	if err != nil {
		return "", err
	}

	apiURL := fmt.Sprintf(
		"https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&timezone=auto",
		lat, lon,
	)

	client := getWeatherHTTPClient()
	resp, err := client.Get(apiURL)
	if err != nil {
		return "", fmt.Errorf("weather: forecast request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("weather: forecast HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("weather: reading forecast response: %w", err)
	}

	var weather currentWeatherResponse
	if err := json.Unmarshal(body, &weather); err != nil {
		return "", fmt.Errorf("weather: invalid forecast JSON: %w", err)
	}

	c := weather.Current
	desc := WeatherCodeToDescription(c.WeatherCode)

	return FormatCurrentWeather(name, desc, c.Temperature, c.Humidity, c.WindSpeed), nil
}

// FormatCurrentWeather formats weather data into a human-readable string.
func FormatCurrentWeather(city, description string, temp, humidity, wind float64) string {
	return fmt.Sprintf("Weather in %s: %s, %.1f°C, humidity %.0f%%, wind %.1f km/h",
		city, description, temp, humidity, wind)
}

// GetForecast fetches a multi-day weather forecast for a location.
func GetForecast(location string, days int) (string, error) {
	if location == "" {
		return "", fmt.Errorf("weather: location is required")
	}
	if days <= 0 || days > 16 {
		days = 7
	}

	name, lat, lon, err := geocodeLocation(location)
	if err != nil {
		return "", err
	}

	apiURL := fmt.Sprintf(
		"https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum&timezone=auto&forecast_days=%d",
		lat, lon, days,
	)

	client := getWeatherHTTPClient()
	resp, err := client.Get(apiURL)
	if err != nil {
		return "", fmt.Errorf("weather: forecast request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("weather: forecast HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("weather: reading forecast response: %w", err)
	}

	var forecast forecastResponse
	if err := json.Unmarshal(body, &forecast); err != nil {
		return "", fmt.Errorf("weather: invalid forecast JSON: %w", err)
	}

	return FormatForecast(name, forecast.Daily.Time, forecast.Daily.TempMax, forecast.Daily.TempMin, forecast.Daily.WeatherCode, forecast.Daily.Precipitation), nil
}

// FormatForecast formats multi-day forecast data into a human-readable string.
func FormatForecast(city string, dates []string, tempMax, tempMin []float64, codes []int, precip []float64) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Forecast for %s:\n", city)

	n := len(dates)
	for i := 0; i < n; i++ {
		desc := WeatherCodeToDescription(codes[i])
		fmt.Fprintf(&sb, "  %s: %s, %.0f-%.0f°C", dates[i], desc, tempMin[i], tempMax[i])
		if precip != nil && i < len(precip) && precip[i] > 0 {
			fmt.Fprintf(&sb, ", %.1f mm precip", precip[i])
		}
		sb.WriteString("\n")
	}

	return strings.TrimRight(sb.String(), "\n")
}

// RegisterWeatherTools adds weather tools to the registry.
func RegisterWeatherTools(r *Registry) {
	r.Register(Tool{
		Name:        "weather",
		Description: "Get current weather or forecast for a location. Args: location (required), days (optional, for forecast).",
		Execute: func(args map[string]string) (string, error) {
			location := args["location"]
			if location == "" {
				return "", fmt.Errorf("weather requires 'location' argument")
			}

			if daysStr, ok := args["days"]; ok && daysStr != "" {
				days, err := strconv.Atoi(daysStr)
				if err != nil {
					return "", fmt.Errorf("weather: invalid days value %q", daysStr)
				}
				return GetForecast(location, days)
			}

			return GetWeather(location)
		},
	})
}
