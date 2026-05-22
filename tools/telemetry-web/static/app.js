const runSelect = document.getElementById("run-select");
const refreshButton = document.getElementById("refresh-button");
const statusBox = document.getElementById("status");
const chartCanvas = document.getElementById("fitness-chart");

let chart = null;

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
  const runs = await fetchJson("/api/runs");
  runSelect.innerHTML = "";

  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.filename;
    option.textContent = run.filename;
    runSelect.appendChild(option);
  }

  if (runs.length === 0) {
    setStatus("No telemetry SQLite files found.");
    updateChart([]);
    return;
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
  updateChart(rows);
}

function updateChart(rows) {
  const labels = rows.map((row) => row.generation);
  const datasets = [
    dataset("min", rows.map((row) => row.min), "#5267d6"),
    dataset("max", rows.map((row) => row.max), "#d65252"),
    dataset("mean", rows.map((row) => row.mean), "#2d8a54"),
    dataset("median", rows.map((row) => row.median), "#8a5ab8"),
  ];

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets = datasets;
    chart.update();
    return;
  }

  chart = new Chart(chartCanvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Generation" } },
        y: { title: { display: true, text: "Fitness" } },
      },
    },
  });
}

function dataset(label, data, color) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.15,
  };
}

refreshButton.addEventListener("click", () => {
  loadRuns().catch((error) => setStatus(error.message));
});

runSelect.addEventListener("change", () => {
  loadSelectedRun().catch((error) => setStatus(error.message));
});

loadRuns().catch((error) => setStatus(error.message));
