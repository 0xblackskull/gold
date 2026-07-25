# Save as find_gold.py and run it
import requests, json

# Get all external/spot meta
for endpoint in ["spotMeta", "spotMetaAndAssetCtxs"]:
    r = requests.post("https://api.hyperliquid.xyz/info",
                      json={"type": endpoint}, timeout=10)
    print(f"{endpoint}: {r.status_code} {r.text[:200]}")
    print()

# Also try perpDeploy meta
r = requests.post("https://api.hyperliquid.xyz/info",
                  json={"type": "perpDeployAuction"}, timeout=10)
print(f"perpDeployAuction: {r.status_code} {r.text[:300]}")
