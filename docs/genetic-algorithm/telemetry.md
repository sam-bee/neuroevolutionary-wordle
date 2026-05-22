# GA Telemetry

`run_genetic_algorithm` can write one SQLite row per evaluated generation. Telemetry is disabled unless a telemetry
flag is passed.

## Run the GA with telemetry

Use an explicit database path:

```sh
./build/run_genetic_algorithm --generations 4 --telemetry-path telemetry/runs/ga-telemetry-test.sqlite
```

Or ask the runner to create a datetime-named file:

```sh
./build/run_genetic_algorithm --generations 4 --telemetry-dir telemetry/runs
```

`--telemetry-dir telemetry/runs` creates a file named like:

```text
telemetry/runs/ga-telemetry-2026-05-22T01-37-04.sqlite
```

The schema is intentionally small:

```sql
CREATE TABLE IF NOT EXISTS generation_fitness (
    generation INTEGER PRIMARY KEY,
    population_size INTEGER NOT NULL DEFAULT 0,
    training_word_count INTEGER NOT NULL DEFAULT 0,
    fitness_min REAL NOT NULL,
    fitness_mean REAL NOT NULL,
    fitness_median REAL NOT NULL,
    fitness_p90 REAL NOT NULL DEFAULT 0,
    fitness_p99 REAL NOT NULL DEFAULT 0,
    fitness_max REAL NOT NULL,
    fitness_stddev REAL NOT NULL DEFAULT 0,
    distinct_fitness_count INTEGER NOT NULL DEFAULT 0,
    logged_at TEXT NOT NULL
);
```

Writes use `INSERT OR REPLACE`, so resuming into an existing telemetry file updates an already-present generation row
instead of duplicating it. The database is opened in WAL mode so the local visualiser can read while the GA is still
running.

## Run the visualiser

Build and start the web service with Docker Compose:

```sh
docker compose up --build telemetry-web
```

Open:

```text
http://127.0.0.1:8090
```

The Compose service mounts `telemetry/runs` into the container and only serves `.sqlite` files from that directory.
The mount is writable because SQLite WAL readers may need sidecar shared-memory files while the GA is writing.

## JavaScript dependency rule

Do not run `npm`, `npx`, `pnpm`, or `yarn` on bare metal for this dashboard. The web container installs its browser
charting asset inside the container image and the runtime UI does not fetch scripts from a CDN.
