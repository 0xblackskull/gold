import requests, json

# ── 1. Spot meta ─────────────────────────────
r = requests.post("https://api.hyperliquid.xyz/info",
                  json={"type": "spotMeta"}, timeout=10)
data     = r.json()
universe = data.get("universe", [])
tokens   = data.get("tokens", [])

print("=== Spot markets with GOLD/XAU ===")
found = False
for m in universe:
    name = m.get("name", "")
    if "GOLD" in name.upper() or "XAU" in name.upper():
        print(f"  FOUND: {json.dumps(m)}")
        found = True
if not found:
    print("  (none)")

print("\n=== Tokens with GOLD/XAU ===")
found = False
for t in tokens:
    name = t.get("name", "")
    if "GOLD" in name.upper() or "XAU" in name.upper():
        print(f"  FOUND: {json.dumps(t)}")
        found = True
if not found:
    print("  (none)")

print("\n=== First 20 spot markets ===")
for m in universe[:20]:
    print(f"  {m}")

# ── 2. Try Tradfi / external perp endpoints ──
print("\n=== Trying external perp meta ===")
for t in ["perpMetaAndAssetCtxs", "metaAndAssetCtxs", "allMids"]:
    r2 = requests.post("https://api.hyperliquid.xyz/info",
                       json={"type": t}, timeout=10)
    print(f"  {t}: status={r2.status_code}  body={r2.text[:120]}")

# ── 3. Try candle with @index format ─────────
print("\n=== Trying candle fetch with @index names ===")
import time
now_ms   = int(time.time() * 1000)
start_ms = now_ms - 900_000 * 10   # last 10 bars

for name in ["GOLD", "XAU", "@GOLD", "GOLD-USDC", "XAUUSD"]:
    r3 = requests.post("https://api.hyperliquid.xyz/info",
                       json={"type": "candleSnapshot",
                             "req": {"coin": name, "interval": "15m",
                                     "startTime": start_ms, "endTime": now_ms}},
                       timeout=10)
    preview = r3.text[:80].replace("\n", " ")
    print(f"  coin='{name}': status={r3.status_code}  body={preview}")

# ── 4. allMids — shows all active markets ────
print("\n=== allMids (all active prices) ===")
r4 = requests.post("https://api.hyperliquid.xyz/info",
                   json={"type": "allMids"}, timeout=10)
if r4.status_code == 200:
    mids = r4.json()
    gold_keys = {k: v for k, v in mids.items()
                 if "GOLD" in k.upper() or "XAU" in k.upper()}
    print(f"  Gold-related: {gold_keys}")
    print(f"  Total markets in allMids: {len(mids)}")
else:
    print(f"  status={r4.status_code}")
