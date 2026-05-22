FROM golang:1.22-bookworm

ARG DOCKERCOMPOSE_UID=1001
ARG DOCKERCOMPOSE_GID=1001

ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKERCOMPOSE_UID=${DOCKERCOMPOSE_UID}
ENV DOCKERCOMPOSE_GID=${DOCKERCOMPOSE_GID}
ENV DOCKERCOMPOSE_USER=developer
ENV DOCKERCOMPOSE_GROUP=developer

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libjs-chart.js \
    sqlite3 \
 && groupadd --gid ${DOCKERCOMPOSE_GID} ${DOCKERCOMPOSE_GROUP} \
 && useradd --uid ${DOCKERCOMPOSE_UID} --gid ${DOCKERCOMPOSE_GID} --create-home --shell /bin/bash ${DOCKERCOMPOSE_USER} \
 && mkdir -p /workspace /app/static \
 && chown -R ${DOCKERCOMPOSE_UID}:${DOCKERCOMPOSE_GID} /workspace /app /home/${DOCKERCOMPOSE_USER} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tools/telemetry-web

COPY tools/telemetry-web/go.mod ./
COPY tools/telemetry-web/main.go ./
COPY tools/telemetry-web/static /app/static

RUN chart_js_path="$(find /usr/share/javascript -path '*chart.js*' -type f \( -name 'chart.umd.js' -o -name 'Chart.js' -o -name 'Chart.min.js' -o -name 'chart.js' \) | head -n 1)" \
 && test -n "${chart_js_path}" \
 && cp "${chart_js_path}" /app/static/chart.umd.js \
 && go build -o /app/telemetry-web .

USER ${DOCKERCOMPOSE_USER}

ENV TELEMETRY_DIR=/telemetry/runs
ENV STATIC_DIR=/app/static
ENV LISTEN_ADDRESS=0.0.0.0:8080

CMD ["/app/telemetry-web"]
