from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class GenerationMetrics:
    generation: int
    best_fitness: float
    mean_fitness: float
    fitness_std: float
    best_completion_ratio: float
    best_mean_response: float
    best_energy: float
    jobs_in_snapshot: int
    satellites_in_snapshot: int
    stagnation_generations: int
    world_reset: int


class MetricsCollector:
    def __init__(self, output_dir: str, prefix: str, flush_every: int = 1):
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Metrics collection requires numpy and matplotlib. "
                "Install dependencies with `pip install numpy matplotlib` or run with --disable-metrics."
            ) from exc

        self.np = np
        self.plt = plt

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prefix = prefix
        self.flush_every = max(1, flush_every)
        self.fieldnames = [field.name for field in fields(GenerationMetrics)]
        self.history: Dict[str, List[float]] = {name: [] for name in self.fieldnames}

        self.csv_path = self.output_dir / f"{self.prefix}_metrics.csv"
        self.npz_path = self.output_dir / f"{self.prefix}_metrics.npz"
        self.plot_path = self.output_dir / f"{self.prefix}_metrics.png"

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.fieldnames)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def record(self, metrics: GenerationMetrics) -> None:
        row = asdict(metrics)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        for field in self.fieldnames:
            self.history[field].append(float(row[field]))

        if len(self.history["generation"]) % self.flush_every == 0:
            self.flush_numpy()

    def flush_numpy(self) -> None:
        self.np.savez(self.npz_path, **self._to_numpy())

    def build_graphs(self) -> None:
        arrays = self._to_numpy()
        generations = arrays["generation"]
        if generations.size == 0:
            return

        fig, axes = self.plt.subplots(4, 1, figsize=(12, 14), sharex=True)

        best_fitness = arrays["best_fitness"]
        mean_fitness = arrays["mean_fitness"]
        fitness_std = arrays["fitness_std"]

        axes[0].plot(generations, best_fitness, color="tab:blue", linewidth=2.0, label="Best fitness")
        axes[0].plot(generations, mean_fitness, color="tab:cyan", linewidth=1.5, label="Mean fitness")
        axes[0].fill_between(
            generations,
            mean_fitness - fitness_std,
            mean_fitness + fitness_std,
            color="tab:cyan",
            alpha=0.20,
            label="Mean +/- std",
        )
        axes[0].set_ylabel("Fitness")
        axes[0].set_title("Fitness trend")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(
            generations,
            arrays["best_completion_ratio"],
            color="tab:green",
            linewidth=2.0,
            label="Completion ratio",
        )
        axes[1].set_ylabel("Ratio")
        axes[1].set_title("Best individual completion ratio")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")

        axes[2].plot(
            generations,
            arrays["best_mean_response"],
            color="tab:orange",
            linewidth=2.0,
            label="Mean response (min)",
        )
        axes[2].set_ylabel("Minutes")
        axes[2].set_title("Best individual response time")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="best")

        axes[3].plot(
            generations,
            arrays["best_energy"],
            color="tab:red",
            linewidth=2.0,
            label="Energy (Wmin)",
        )
        axes[3].set_xlabel("Generation")
        axes[3].set_ylabel("Wmin")
        axes[3].set_title("Best individual energy consumption")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(loc="best")

        reset_generations = generations[arrays["world_reset"] > 0.5]
        for axis in axes:
            for generation in reset_generations:
                axis.axvline(generation, color="0.5", linestyle="--", alpha=0.35)

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        self.plt.close(fig)

    def close(self) -> None:
        self.flush_numpy()
        self.build_graphs()
        self._csv_file.close()

    def _to_numpy(self):
        return {
            field: self.np.asarray(values, dtype=self.np.float64)
            for field, values in self.history.items()
        }
