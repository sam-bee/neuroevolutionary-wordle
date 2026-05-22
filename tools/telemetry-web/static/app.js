const runSelect = document.getElementById("run-select");
const refreshButton = document.getElementById("refresh-button");
const statusBox = document.getElementById("status");
const chartCanvas = document.getElementById("fitness-chart");
const stddevChartCanvas = document.getElementById("stddev-chart");
const distinctValuesChartCanvas = document.getElementById("distinct-values-chart");
const populationChartCanvas = document.getElementById("population-chart");
const trainingDataChartCanvas = document.getElementById("training-data-chart");
const seriesControls = document.getElementById("series-controls");

let fitnessChart = null;
let stddevChart = null;
let distinctValuesChart = null;
let populationChart = null;
let trainingDataChart = null;
const fitnessSeriesDefinitions = [
  { key: "min", label: "min fitness", color: "#7aa2f7" },
  { key: "mean", label: "mean fitness", color: "#9ece6a" },
  { key: "median", label: "median fitness", color: "#bb9af7" },
  { key: "p90", label: "p90 fitness", color: "#e0af68" },
  { key: "p99", label: "p99 fitness", color: "#ff9e64" },
  { key: "max", label: "max fitness", color: "#f7768e" },
];
const fitnessSeriesVisibility = new Map(fitnessSeriesDefinitions.map((definition) => [definition.key, true]));

function setStatus(message) {
  statusBox.textContent = message;
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json();
}

async function loadRuns() {
  const selectedFilename = runSelect.value;
  const runs = await fetchJson("/api/runs");
  runSelect.innerHTML = "";
  let selectedRunStillExists = false;

  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.filename;
    option.textContent = run.filename;
    if (run.filename === selectedFilename) {
      option.selected = true;
      selectedRunStillExists = true;
    }
    runSelect.appendChild(option);
  }

  if (runs.length === 0) {
    setStatus("No telemetry SQLite files found.");
    updateCharts([]);
    return;
  }

  if (!selectedRunStillExists) {
    runSelect.value = runs[0].filename;
  }
  setStatus(`Loaded ${runs.length} telemetry run${runs.length === 1 ? "" : "s"}.`);
  await loadSelectedRun();
}

async function loadSelectedRun() {
  const filename = runSelect.value;
  if (!filename) {
    return;
  }

  const rows = await fetchJson(`/api/runs/${encodeURIComponent(filename)}/fitness`);
  setStatus(`${filename}: ${rows.length} generation${rows.length === 1 ? "" : "s"}.`);
  updateCharts(rows);
}

function updateCharts(rows) {
  const labels = rows.map((row) => row.generation);
  const fitnessDatasets = fitnessSeriesDefinitions.map((definition) =>
    dataset(
      definition.label,
      rows.map((row) => row[definition.key]),
      definition.color,
      !fitnessSeriesVisibility.get(definition.key),
    ),
  );

  fitnessChart = updateOrCreateChart(fitnessChart, chartCanvas, labels, fitnessDatasets, "Fitness");
  stddevChart = updateOrCreateChart(
    stddevChart,
    stddevChartCanvas,
    labels,
    [dataset("stddev fitness", rows.map((row) => row.stddev), "#73daca")],
    "Stddev Fitness",
  );
  distinctValuesChart = updateOrCreateChart(
    distinctValuesChart,
    distinctValuesChartCanvas,
    labels,
    [dataset("distinct fitness values", rows.map((row) => row.distinct_values), "#c0caf5")],
    "Distinct Values",
  );
  populationChart = updateOrCreateChart(
    populationChart,
    populationChartCanvas,
    labels,
    [dataset("population size", rows.map((row) => row.population_size), "#7dcfff")],
    "Population Size",
  );
  trainingDataChart = updateOrCreateChart(
    trainingDataChart,
    trainingDataChartCanvas,
    labels,
    [dataset("training words", rows.map((row) => row.training_word_count), "#a9b1d6")],
    "Training Words",
  );
}

function dataset(label, data, color, hidden = false) {
  return {
    label,
    data,
    hidden,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.15,
  };
}

function renderSeriesControls() {
  seriesControls.innerHTML = "";
  for (const definition of fitnessSeriesDefinitions) {
    const label = document.createElement("label");
    label.className = "series-toggle";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = fitnessSeriesVisibility.get(definition.key);
    checkbox.addEventListener("change", () => {
      fitnessSeriesVisibility.set(definition.key, checkbox.checked);
      if (!fitnessChart) {
        return;
      }
      const datasetIndex = fitnessSeriesDefinitions.findIndex((item) => item.key === definition.key);
      fitnessChart.setDatasetVisibility(datasetIndex, checkbox.checked);
      fitnessChart.update();
    });

    const swatch = document.createElement("span");
    swatch.className = "series-swatch";
    swatch.style.backgroundColor = definition.color;

    const text = document.createElement("span");
    text.textContent = definition.label;

    label.append(checkbox, swatch, text);
    seriesControls.appendChild(label);
  }
}

function updateOrCreateChart(existingChart, canvas, labels, datasets, yAxisTitle) {
  if (existingChart) {
    existingChart.data.labels = labels;
    existingChart.data.datasets = datasets;
    existingChart.update();
    return existingChart;
  }

  return new Chart(canvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#d7dde5" } },
      },
      scales: {
        x: {
          title: { display: true, text: "Generation", color: "#d7dde5" },
          ticks: { color: "#9aa6b5" },
          grid: { color: "#2a3340" },
        },
        y: {
          title: { display: true, text: yAxisTitle, color: "#d7dde5" },
          ticks: { color: "#9aa6b5" },
          grid: { color: "#2a3340" },
        },
      },
    },
  });
}

refreshButton.addEventListener("click", () => {
  loadRuns().catch((error) => setStatus(error.message));
});

runSelect.addEventListener("change", () => {
  loadSelectedRun().catch((error) => setStatus(error.message));
});

renderSeriesControls();
loadRuns().catch((error) => setStatus(error.message));
