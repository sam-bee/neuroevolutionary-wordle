package main

import (
	"bytes"
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
	mux.HandleFunc("/api/runs/", s.handleRunMetric)
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
		log.Printf("could not read telemetry directory dir=%q err=%v", s.telemetryDir, err)
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
			log.Printf("could not inspect telemetry run entry dir=%q name=%q err=%v", s.telemetryDir, entry.Name(), err)
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

func (s *server) handleRunMetric(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	const prefix = "/api/runs/"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		http.NotFound(w, r)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, prefix)
	if strings.HasSuffix(path, "/fitness") {
		filename := strings.TrimSuffix(path, "/fitness")
		s.handleRunFitness(w, filename)
		return
	}
	if strings.HasSuffix(path, "/genetic-convergence") {
		filename := strings.TrimSuffix(path, "/genetic-convergence")
		s.handleRunGeneticConvergence(w, filename)
		return
	}

	http.NotFound(w, r)
}

func (s *server) handleRunFitness(w http.ResponseWriter, filename string) {
	filename = strings.Trim(filename, "/")
	dbPath, err := s.databasePath(filename)
	if err != nil {
		log.Printf("invalid fitness telemetry database filename=%q err=%v", filename, err)
		http.Error(w, "invalid telemetry database", http.StatusBadRequest)
		return
	}

	hasTable, err := tableExists(dbPath, "generation_fitness")
	if err != nil {
		log.Printf("could not inspect generation fitness telemetry table filename=%q db=%q err=%v", filename, dbPath, err)
		http.Error(w, "could not inspect telemetry database", http.StatusInternalServerError)
		return
	}
	if !hasTable {
		log.Printf("legacy telemetry database lacks generation_fitness table filename=%q db=%q", filename, dbPath)
		writeEmptyJSON(w)
		return
	}

	query, err := fitnessQuery(dbPath)
	if err != nil {
		log.Printf("could not build fitness telemetry query filename=%q db=%q err=%v", filename, dbPath, err)
		http.Error(w, "could not inspect telemetry database", http.StatusInternalServerError)
		return
	}
	if query == "" {
		log.Printf("legacy telemetry database lacks required generation_fitness columns filename=%q db=%q", filename, dbPath)
		writeEmptyJSON(w)
		return
	}
	writeSQLiteJSON(w, dbPath, query, "fitness telemetry query", "could not query telemetry database")
}

func (s *server) handleRunGeneticConvergence(w http.ResponseWriter, filename string) {
	filename = strings.Trim(filename, "/")
	dbPath, err := s.databasePath(filename)
	if err != nil {
		log.Printf("invalid genetic convergence telemetry database filename=%q err=%v", filename, err)
		http.Error(w, "invalid telemetry database", http.StatusBadRequest)
		return
	}

	hasTable, err := tableExists(dbPath, "genetic_convergence_telemetry")
	if err != nil {
		log.Printf("could not inspect genetic convergence telemetry table filename=%q db=%q err=%v", filename, dbPath, err)
		http.Error(w, "could not inspect telemetry database", http.StatusInternalServerError)
		return
	}
	if !hasTable {
		log.Printf("legacy telemetry database lacks genetic_convergence_telemetry table filename=%q db=%q", filename, dbPath)
		writeEmptyJSON(w)
		return
	}

	query, err := geneticConvergenceQuery(dbPath)
	if err != nil {
		log.Printf("could not build genetic convergence telemetry query filename=%q db=%q err=%v", filename, dbPath, err)
		http.Error(w, "could not inspect telemetry database", http.StatusInternalServerError)
		return
	}
	if query == "" {
		log.Printf("legacy telemetry database lacks required genetic_convergence_telemetry columns filename=%q db=%q", filename, dbPath)
		writeEmptyJSON(w)
		return
	}

	writeSQLiteJSON(w, dbPath, query, "genetic convergence telemetry query", "could not query genetic convergence telemetry")
}

func fitnessQuery(dbPath string) (string, error) {
	columns, err := generationFitnessColumns(dbPath)
	if err != nil {
		return "", err
	}
	for _, column := range []string{"generation", "fitness_min", "fitness_mean", "fitness_median", "fitness_max"} {
		if !columns[column] {
			return "", nil
		}
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
` + expr("distinct_fitness_count", "0") + ` AS distinct_values,
` + expr("parent_childless_count", "NULL") + ` AS parent_childless_count,
` + expr("parent_one_child_count", "NULL") + ` AS parent_one_child_count,
` + expr("parent_multiple_children_count", "NULL") + ` AS parent_multiple_children_count
FROM generation_fitness
ORDER BY generation;`, nil
}

func geneticConvergenceQuery(dbPath string) (string, error) {
	columns, err := tableColumns(dbPath, "genetic_convergence_telemetry")
	if err != nil {
		return "", err
	}
	requiredColumns := []string{
		"generation",
		"sample_organisms",
		"sample_weights",
		"pair_count",
		"centroid_distance_mean",
		"centroid_distance_min",
		"centroid_distance_max",
		"pairwise_distance_mean",
		"pairwise_distance_min",
		"pairwise_distance_max",
		"elapsed_ms",
	}
	for _, column := range requiredColumns {
		if !columns[column] {
			return "", nil
		}
	}

	return `SELECT generation AS generation,
sample_organisms AS sample_organisms,
sample_weights AS sample_weights,
pair_count AS pair_count,
centroid_distance_mean AS centroid_distance_mean,
centroid_distance_min AS centroid_distance_min,
centroid_distance_max AS centroid_distance_max,
pairwise_distance_mean AS pairwise_distance_mean,
pairwise_distance_min AS pairwise_distance_min,
pairwise_distance_max AS pairwise_distance_max,
elapsed_ms AS elapsed_ms
FROM genetic_convergence_telemetry
ORDER BY generation;`, nil
}

func tableExists(dbPath string, tableName string) (bool, error) {
	query := "SELECT name FROM sqlite_master WHERE type='table' AND name='" + tableName + "';"
	output, err := runSQLiteJSON(dbPath, query, "table existence check for "+tableName)
	if err != nil {
		return false, err
	}

	var rows []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(output, &rows); err != nil {
		log.Printf("could not decode sqlite3 table existence JSON db=%q table=%q err=%v output=%q", dbPath, tableName, err,
			strings.TrimSpace(string(output)))
		return false, err
	}
	return len(rows) > 0, nil
}

func writeSQLiteJSON(w http.ResponseWriter, dbPath string, query string, context string, errorMessage string) {
	output, err := runSQLiteJSON(dbPath, query, context)
	if err != nil {
		http.Error(w, errorMessage, http.StatusInternalServerError)
		return
	}
	if len(strings.TrimSpace(string(output))) == 0 {
		output = []byte("[]\n")
	}

	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(output)
}

func writeEmptyJSON(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte("[]\n"))
}

func generationFitnessColumns(dbPath string) (map[string]bool, error) {
	return tableColumns(dbPath, "generation_fitness")
}

func tableColumns(dbPath string, tableName string) (map[string]bool, error) {
	output, err := runSQLiteJSON(dbPath, "PRAGMA table_info("+tableName+");", tableName+" column inspection")
	if err != nil {
		return nil, err
	}

	var rows []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(output, &rows); err != nil {
		log.Printf("could not decode sqlite3 table column JSON db=%q table=%q err=%v output=%q", dbPath, tableName, err,
			strings.TrimSpace(string(output)))
		return nil, err
	}

	columns := make(map[string]bool, len(rows))
	for _, row := range rows {
		columns[row.Name] = true
	}
	return columns, nil
}

func runSQLiteJSON(dbPath string, query string, context string) ([]byte, error) {
	const attempts = 3

	var lastErr error
	var lastStdout string
	var lastStderr string
	for attempt := 1; attempt <= attempts; attempt++ {
		command := exec.Command("sqlite3", "-readonly", "-cmd", ".timeout 5000", "-json", dbPath, query)
		var stdout bytes.Buffer
		var stderr bytes.Buffer
		command.Stdout = &stdout
		command.Stderr = &stderr

		err := command.Run()
		if err == nil {
			if attempt > 1 {
				log.Printf("sqlite3 succeeded after retry context=%q db=%q attempt=%d", context, dbPath, attempt)
			}
			if len(strings.TrimSpace(stdout.String())) == 0 {
				return []byte("[]\n"), nil
			}
			return stdout.Bytes(), nil
		}

		lastErr = err
		lastStdout = strings.TrimSpace(stdout.String())
		lastStderr = strings.TrimSpace(stderr.String())
		log.Printf("sqlite3 failed context=%q db=%q attempt=%d/%d err=%v stderr=%q stdout=%q query=%q", context, dbPath,
			attempt, attempts, err, lastStderr, lastStdout, compactQuery(query))

		if !strings.Contains(strings.ToLower(lastStderr), "database is locked") {
			break
		}
		time.Sleep(time.Duration(attempt) * 100 * time.Millisecond)
	}

	return nil, lastErr
}

func compactQuery(query string) string {
	return strings.Join(strings.Fields(query), " ")
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
