"""Create an F1 robustness figure from results.json; not a substitute for statistical analysis."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--results", default="results.json")
p.add_argument("--out", default="f1_missingness.png")
args = p.parse_args()
results = json.loads(Path(args.results).read_text(encoding="utf-8"))
values = defaultdict(lambda: defaultdict(list))
for row in results["missingness"]["rows"]:
    values[row["method"]][row["missing_rate"]].append(row["f1"])
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, rate_values in values.items():
    x = sorted(rate_values)
    means = np.array([np.mean(rate_values[r]) for r in x])
    errors = np.array([np.std(rate_values[r]) for r in x])
    ax.plot(np.array(x) * 100, means, marker="o", label=name)
    ax.fill_between(np.array(x) * 100, means - errors, means + errors, alpha=.10)
ax.set(xlabel="Additional missing brand/description values (%)",
       ylabel="F1 on the fixed test pairs", title="Pairwise matching robustness")
ax.set_ylim(0, 1)
ax.grid(alpha=.25)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(args.out, dpi=180)
print(f"Saved {args.out}")
