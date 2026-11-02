"""mode=sim for ``run_toss_cycles``: the kinematic TidyBot of
:mod:`prpl_tidybot.sim_robot`, its scripted operator and its video
(:mod:`prpl_tidybot.sim_video`), built from ``cfg.sim``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from prpl_tidybot.real_env import RealTidyBotEnv
from prpl_tidybot.sim_robot import (
    SimCubeTargetSource,
    SimInterface,
    SimOperator,
    SimWorld,
)
from prpl_tidybot.sim_video import SimMarkers, render_video, safety_summary


class SimRecorder:
    """Stands in for the ceiling-camera recorder: the step label goes on the sim's
    record, and closing writes ``sim.mp4`` and ``sim_safety.json``."""

    def __init__(self, world: SimWorld, cfg: DictConfig, log_dir: Path | None) -> None:
        self.world = world
        self._cfg = cfg
        self._log_dir = log_dir

    @property
    def label(self) -> str:
        """The step last asked about."""
        return self.world.label

    @label.setter
    def label(self, value: str) -> None:
        self.world.label = value.strip().rstrip("?")
        self.world.sublabel = ""

    def close(self) -> None:
        """Write the safety summary and the video."""
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        markers = SimMarkers.from_config(self._cfg)
        summary = safety_summary(self.world, markers)
        (self._log_dir / "sim_safety.json").write_text(
            json.dumps(summary, indent=1), encoding="utf-8"
        )
        video = self._cfg.sim.get("video")
        if video is not None and bool(video.get("enabled", True)):
            render_video(
                self.world,
                markers,
                self._log_dir / "sim.mp4",
                fps=int(video.get("fps", 30)),
                slowdown=float(video.get("swing_slowdown", 4.0)),
                width=int(video.get("width", 640)),
                height=int(video.get("height", 480)),
            )


class TossSim:
    """Everything ``run_toss_cycles`` needs for mode=sim, from ``cfg.sim``."""

    def __init__(
        self,
        cfg: DictConfig,
        log_dir: Path | None,
        prompt_fn: Any,
        interactive: bool,
    ) -> None:
        sim = cfg.sim
        cube = [float(v) for v in sim.cube]
        b = cfg.bin
        sim_bin = sim.get("bin")
        self.world = SimWorld(
            base_pose=[float(v) for v in sim.start_base_pose],
            cube=(cube[0], cube[1], math.radians(cube[2])),
            landing_offset=[float(v) for v in sim.landing_offset],
            object_half=float(sim.get("object_half", 0.025)),
            object_kind=str(sim.get("object", "cube")),
            bin_pose=(
                None
                if sim_bin is None
                else (
                    float(sim_bin[0]),
                    float(sim_bin[1]),
                    math.radians(float(sim_bin[2])),
                )
            ),
            bin_model=(
                float(b.length),
                float(b.width),
                float(b.height),
                float(b.get("base_length") or b.length),
                float(b.get("base_width") or b.width),
                float(b.wall),
            ),
        )
        self.interface = SimInterface(self.world)
        self.real_env = RealTidyBotEnv(self.interface, control_period=0.0)
        self.cube_source = SimCubeTargetSource(self.world)
        self.operator = SimOperator(
            self.world, sim.put_back, prompt_fn if interactive else None
        )
        self.recorder = SimRecorder(self.world, cfg, log_dir)
