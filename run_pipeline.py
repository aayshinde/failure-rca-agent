"""Run every offline step in order (works the same on Windows, macOS and Linux).

    python run_pipeline.py            # simulate -> train risk model -> train anomaly model -> index docs
    python run_pipeline.py --small    # 30 machines x 120 days, for a quick first try

Then evaluate the agents:
    python -m src.evaluate --variants rules                    # offline, free
    python -m src.evaluate --variants react graph              # needs OPENAI_API_KEY in .env
"""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--small", action="store_true")
args = ap.parse_args()

sim = ["--machines", "30", "--days", "120"] if args.small else []
steps = [
    ("Simulate fleet data + docs + eval questions", ["-m", "src.simulate", *sim]),
    ("Train XGBoost failure-risk model", ["-m", "src.train_risk"]),
    ("Train PyTorch anomaly detector", ["-m", "src.train_anomaly"]),
    ("Embed documentation into the vector index", ["-m", "src.retrieval", "build"]),
]
for i, (name, cmd) in enumerate(steps, 1):
    print(f"\n=== [{i}/{len(steps)}] {name} ===", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, *cmd]).returncode != 0:
        sys.exit(f"Step failed: {name}")
    print(f"--- done in {time.time() - t0:.0f}s")
print("\nAll steps finished. Next: python -m src.evaluate --variants rules")
