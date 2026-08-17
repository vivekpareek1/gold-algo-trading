#!/bin/bash
# Gold Trading System — Automated Problem Detector
#
# Runs periodically via cron (see setup below). Checks for genuinely
# broken conditions and appends a clear ALERT line to alerts.log ONLY
# when something is actually wrong — the file stays empty during normal
# operation, so "is anything wrong?" is a one-command check:
#
#   cat ~/live_feed/gold_trading_system/logs/alerts.log
#
# If that file has content, something needs attention — paste it to
# Claude along with the output of status_check.sh for diagnosis.

ALERTS_LOG="$HOME/live_feed/gold_trading_system/logs/alerts.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

alert() {
    echo "[$TIMESTAMP] ALERT: $1" >> "$ALERTS_LOG"
}

# 1. Is the service even running?
if ! systemctl is-active --quiet gold-dashboard; then
    alert "gold-dashboard service is NOT running"
    exit 0
fi

# 2. Can we reach the health endpoint at all?
HEALTH_FILE=$(mktemp)
curl -s --max-time 10 http://localhost:8001/health > "$HEALTH_FILE"
if [ ! -s "$HEALTH_FILE" ]; then
    alert "Health endpoint did not respond — service may be hung"
    rm -f "$HEALTH_FILE"
    exit 0
fi

# 3. Parse key fields and check for real problems (reads from the temp
# file rather than a shell variable, avoiding any quoting/escaping risk)
python3 - "$HEALTH_FILE" "$ALERTS_LOG" "$TIMESTAMP" << 'PYEOF'
import json, sys

health_path, alerts_path, timestamp = sys.argv[1], sys.argv[2], sys.argv[3]

with open(health_path) as f:
    raw = f.read()

try:
    health = json.loads(raw)
except Exception as e:
    with open(alerts_path, "a") as af:
        af.write(f"[{timestamp}] ALERT: could not parse health response: {e}\n")
    sys.exit(0)

alerts = []

if health.get("volume_feed_broken"):
    alerts.append("Volume feed is broken (no volume in ticks) — strategy will diverge from backtest")

if health.get("feed_stale"):
    age = health.get("seconds_since_last_tick")
    alerts.append(f"Feed is STALE — no ticks for {age}s while market should be open")

if health.get("data_feed") == "PAPER_SIMULATED":
    alerts.append("Running on SIMULATED data, not real Angel One feed — check angel_one_feed.py / credentials")

if alerts:
    with open(alerts_path, "a") as af:
        for a in alerts:
            af.write(f"[{timestamp}] ALERT: {a}\n")
PYEOF

rm -f "$HEALTH_FILE"
