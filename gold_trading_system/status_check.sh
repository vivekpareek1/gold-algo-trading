#!/bin/bash
# Gold Trading System — One-Command Status Check
# Run this any time you want the full picture. Paste the output to Claude
# if anything looks wrong — this single report has everything needed to
# diagnose most issues without running anything else first.

echo "========================================"
echo "GOLD TRADING SYSTEM — STATUS REPORT"
echo "Generated: $(date)"
echo "========================================"
echo ""

echo "--- Service ---"
sudo systemctl status gold-dashboard --no-pager | head -5
echo ""

echo "--- Health ---"
curl -s http://localhost:8001/health
echo ""
echo ""

echo "--- Current Signal ---"
curl -s http://localhost:8001/api/signal
echo ""
echo ""

echo "--- Performance ---"
curl -s http://localhost:8001/api/performance
echo ""
echo ""

echo "--- Candle Count ---"
curl -s http://localhost:8001/api/candles | python3 -c "import json,sys; d=json.load(sys.stdin); print('Candles in history:', len(d['candles']))" 2>/dev/null
echo ""

echo "--- Recent Trades (last 5) ---"
curl -s http://localhost:8001/api/trades | python3 -c "
import json, sys
d = json.load(sys.stdin)
trades = d['trades'][-5:]
if not trades:
    print('No trades yet this session.')
else:
    for t in trades:
        print(f\"{t['direction']} @ {t['entry_price']} -> {t['exit_price']} | net: {t.get('net_pnl_inr', 'N/A')} | {t['exit_reason']}\")
" 2>/dev/null
echo ""

echo "--- Recent Errors/Warnings in Logs (last 20 lines matching) ---"
grep -iE "REJECTED|WARNING|ERROR|Feed disconnected|reconnecting|Login failed" ~/live_feed/gold_trading_system/logs/dashboard.log 2>/dev/null | tail -20
echo "(if nothing printed above, no recent errors/warnings found)"
echo ""

echo "--- Last 5 Log Lines ---"
tail -5 ~/live_feed/gold_trading_system/logs/dashboard.log
echo ""

echo "========================================"
echo "END OF REPORT"
echo "========================================"
