package main

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type runFile struct {
	Filename string `json:"filename"`
	Modified string `json:"modified"`
}

type server struct {
	telemetryDir string
	staticDir    string
}

func main() {
	telemetryDir := getenv("TELEMETRY_DIR", "/telemetry/runs")
	staticDir := getenv("STATIC_DIR", "/app/static")
	listenAddress := getenv("LISTEN_ADDRESS", "0.0.0.0:8080")

	s := &server{telemetryDir: telemetryDir, staticDir: staticDir}

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/api/runs", s.handleRuns)
	mux.HandleFunc("/api/runs/", s.handleRunFitness)
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))

	log.Printf("serving GA telemetry from %s on %s", telemetryDir, listenAddress)
	if err := http.ListenAndServe(listenAddress, mux); err != nil {
		log.Fatal(err)
	}
}

func getenv(name string, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
}

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, filepath.Join(s.staticDir, "index.html"))
}

func (s *server) handleRuns(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	entries, err := os.ReadDir(s.telemetryDir)
	if err != nil {
		http.Error(w, "could not read telemetry directory", http.StatusInternalServerError)
		return
	}

	runs := make([]runFile, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".sqlite" {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}
		runs = append(runs, runFile{
			Filename: entry.Name(),
			Modified: info.ModTime().UTC().Format(time.RFC3339),
		})
	}

	sort.Slice(runs, func(i, j int) bool {
		return runs[i].Modified > runs[j].Modified
	})

	writeJSON(w, runs)
}

func (s *server) handleRunFitness(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	const prefix = "/api/runs/"
	const suffix = "/fitness"
	if !strings.HasPrefix(r.URL.Path, prefix) || !strings.HasSuffix(r.URL.Path, suffix) {
		http.NotFound(w, r)
		return
	}

	filename := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, prefix), suffix)
	filename = strings.Trim(filename, "/")
	dbPath, err := s.databasePath(filename)
	if err != nil {
		http.Error(w, "invalid telemetry database", http.StatusBadRequest)
		return
	}

	query, err := fitnessQuery(dbPath)
	if err != nil {
		http.Error(w, "could not inspect telemetry database", http.StatusInternalServerError)
		return
	}
	command := exec.Command("sqlite3", "-readonly", "-json", dbPath, query)
	output, err := command.Output()
	if err != nil {
		http.Error(w, "could not query telemetry database", http.StatusInternalServerError)
		return
	}
	if len(strings.TrimSpace(string(output))) == 0 {
		output = []byte("[]\n")
	}

	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(output)
}

func fitnessQuery(dbPath string) (string, error) {
	columns, err := generationFitnessColumns(dbPath)
	if err != nil {
		return "", err
	}

	expr := func(column string, fallback string) string {
		if columns[column] {
			return column
		}
		return fallback
	}

	return `SELECT generation AS generation,
` + expr("population_size", "0") + ` AS population_size,
` + expr("training_word_count", "0") + ` AS training_word_count,
fitness_min AS min,
fitness_mean AS mean,
fitness_median AS median,
` + expr("fitness_p90", "fitness_max") + ` AS p90,
` + expr("fitness_p99", "fitness_max") + ` AS p99,
fitness_max AS max,
` + expr("fitness_stddev", "0") + ` AS stddev,
` + expr("distinct_fitness_count", "0") + ` AS distinct_values
FROM generation_fitness
ORDER BY generation;`, nil
}

func generationFitnessColumns(dbPath string) (map[string]bool, error) {
	command := exec.Command("sqlite3", "-readonly", "-json", dbPath, "PRAGMA table_info(generation_fitness);")
	output, err := command.Output()
	if err != nil {
		return nil, err
	}

	var rows []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(output, &rows); err != nil {
		return nil, err
	}

	columns := make(map[string]bool, len(rows))
	for _, row := range rows {
		columns[row.Name] = true
	}
	return columns, nil
}

func (s *server) databasePath(filename string) (string, error) {
	if filename == "" || filepath.Base(filename) != filename || strings.Contains(filename, `\`) ||
		filepath.Ext(filename) != ".sqlite" {
		return "", errors.New("invalid filename")
	}

	dir, err := filepath.EvalSymlinks(s.telemetryDir)
	if err != nil {
		return "", err
	}
	dbPath := filepath.Join(dir, filename)
	realDBPath, err := filepath.EvalSymlinks(dbPath)
	if err != nil {
		return "", err
	}

	dirWithSeparator := dir + string(os.PathSeparator)
	if !strings.HasPrefix(realDBPath, dirWithSeparator) {
		return "", errors.New("database path escapes telemetry directory")
	}
	return realDBPath, nil
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	_ = encoder.Encode(value)
}
