"""Configuration loading for the Cropilot finetune trainer.

A single YAML file describes every finetune job. Values are resolved with the
precedence ``defaults < group < job`` and exposed as immutable ``JobConfig``
objects, one per job, ready to be handed to ``CropilotTrainer``.

See ``jobs.example.yaml`` for the expected file structure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Default API URL used when neither the group nor the job overrides it.
DEFAULT_API_URL = "https://api.cropilot.trinera.cloud/"


class ConfigError(Exception):
    """Raised when the jobs config file is malformed or fails validation."""


@dataclass(frozen=True)
class JobConfig:
    """Fully resolved configuration for a single finetune job."""

    name: str
    api_url: str
    api_key: str
    group: str
    base_model: str
    val_split: float
    train_position: bool
    train_rotation: bool
    title_ids: list[str]
    # Hyperparameter overrides; imgsz, max_det and patience stay hardcoded in
    # the trainer and are intentionally not represented here.
    position: dict = field(default_factory=dict)
    rotation: dict = field(default_factory=dict)

    def comet_parameters(self) -> dict:
        """Flat dict of everything worth logging to Comet.ml as parameters."""
        params = {
            "group": self.group,
            "base_model": self.base_model,
            "val_split": self.val_split,
            "train_position": self.train_position,
            "train_rotation": self.train_rotation,
            "num_title_ids": len(self.title_ids),
        }
        params.update({f"position.{k}": v for k, v in self.position.items()})
        params.update({f"rotation.{k}": v for k, v in self.rotation.items()})
        return params


def _merge(base: dict, override: dict) -> dict:
    """Shallow-merge ``override`` onto a copy of ``base``."""
    merged = copy.deepcopy(base) if base else {}
    if override:
        merged.update(override)
    return merged


def _resolve_job(raw: dict, defaults: dict, groups: dict) -> JobConfig:
    """Resolve a single raw job dict into a JobConfig (defaults < group < job)."""
    name = raw.get("name")
    if not name:
        raise ConfigError("Every job must have a 'name'.")

    group_name = raw.get("group")
    if not group_name:
        raise ConfigError(f"Job '{name}' is missing 'group'.")
    if group_name not in groups:
        raise ConfigError(
            f"Job '{name}' references unknown group '{group_name}'. "
            f"Known groups: {', '.join(sorted(groups)) or '(none)'}."
        )
    group = groups[group_name] or {}

    api_key = group.get("api_key")
    if not api_key:
        raise ConfigError(f"Group '{group_name}' is missing 'api_key'.")

    # api_url and base_model: job overrides group overrides defaults.
    api_url = raw.get("api_url") or group.get("api_url") or defaults.get(
        "api_url", DEFAULT_API_URL
    )
    base_model = raw.get("base_model") or defaults.get("base_model")
    if not base_model:
        raise ConfigError(f"Job '{name}' has no 'base_model' (and no default).")

    val_split = raw.get("val_split", defaults.get("val_split", 0.2))

    # train_position / train_rotation must both be stated explicitly per job.
    for flag in ("train_position", "train_rotation"):
        if flag not in raw:
            raise ConfigError(
                f"Job '{name}' must set '{flag}' explicitly (true/false)."
            )
    train_position = bool(raw["train_position"])
    train_rotation = bool(raw["train_rotation"])
    if not (train_position or train_rotation):
        raise ConfigError(
            f"Job '{name}' has both train_position and train_rotation false; "
            "nothing to train."
        )

    title_ids = raw.get("title_ids") or []
    if not title_ids:
        raise ConfigError(f"Job '{name}' has an empty 'title_ids' list.")
    if len(set(title_ids)) != len(title_ids):
        raise ConfigError(f"Job '{name}' has duplicate title_ids.")

    position = _merge(defaults.get("position", {}), raw.get("position", {}))
    rotation = _merge(defaults.get("rotation", {}), raw.get("rotation", {}))

    return JobConfig(
        name=name,
        api_url=api_url,
        api_key=api_key,
        group=group_name,
        base_model=base_model,
        val_split=float(val_split),
        train_position=train_position,
        train_rotation=train_rotation,
        title_ids=list(title_ids),
        position=position,
        rotation=rotation,
    )


def load_jobs(path: str | Path) -> list[JobConfig]:
    """Load and resolve all jobs from a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("defaults", {}) or {}
    groups = data.get("groups", {}) or {}
    raw_jobs = data.get("jobs", []) or []
    if not raw_jobs:
        raise ConfigError(f"No jobs defined in {path}.")

    jobs = [_resolve_job(raw, defaults, groups) for raw in raw_jobs]

    names = [j.name for j in jobs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConfigError(f"Duplicate job names: {', '.join(sorted(duplicates))}.")

    return jobs


def select_jobs(
    all_jobs: list[JobConfig], names: list[str] | None, run_all: bool
) -> list[JobConfig]:
    """Pick the jobs to run based on --job / --all selection."""
    if run_all:
        return all_jobs
    if not names:
        raise ConfigError("Select jobs with --job <name> (repeatable) or --all.")

    by_name = {j.name: j for j in all_jobs}
    selected = []
    for name in names:
        if name not in by_name:
            raise ConfigError(
                f"Unknown job '{name}'. Available: {', '.join(by_name)}."
            )
        selected.append(by_name[name])
    return selected
