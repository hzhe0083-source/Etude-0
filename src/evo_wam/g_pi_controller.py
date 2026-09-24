"""Pure offline subgoal/chunk control; callbacks perform model prediction only."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping

import torch
from torch import Tensor

from .g_pi_data import EventRules, EventState, GripperEvent, gripper_event_step, initial_event_state
from .goal_interface import validate_goal_poses, validate_gripper


@dataclass(frozen=True)
class GoalThresholds:
    z: float = .05
    position_m: float = .01
    rotation_deg: float = 5.
    gripper: float = .05

    def __post_init__(self):
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (self.z, self.position_m, self.rotation_deg, self.gripper)):
            raise ValueError("goal thresholds must be finite nonnegative distances")


def _goal(value: Mapping) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(value, Mapping) or any(name not in value for name in ("z", "goal_poses", "goal_gripper")):
        raise ValueError("goal requires z, goal_poses and goal_gripper")
    z, poses, gripper = (value[name] for name in ("z", "goal_poses", "goal_gripper"))
    if (not isinstance(z, Tensor) or z.ndim != 3 or min(z.shape) < 1
            or not z.is_floating_point() or not torch.isfinite(z).all()):
        raise ValueError("goal z must be finite floating [B,K_z,d_z]")
    validate_goal_poses(poses)
    validate_gripper(gripper, poses.shape[:2])
    if z.shape[0] != poses.shape[0]:
        raise ValueError("z and goal poses must share the batch")
    return z, poses, gripper


def goal_reached(goal: Mapping, current: Mapping, thresholds: GoalThresholds = GoalThresholds()) -> bool:
    """Require every effector and all four distances to meet their thresholds."""
    z, poses, gripper = _goal(goal)
    current_z, current_poses, current_gripper = _goal(current)
    if z.shape != current_z.shape or poses.shape != current_poses.shape:
        raise ValueError("current and predicted goals must have matching shapes")
    with torch.no_grad():
        z_distance = (z.double() - current_z.to(z.device).double()).norm(dim=-1).max()
        pose = poses.double()
        measured = current_poses.to(poses.device).double()
        translation = (pose[..., :3, 3] - measured[..., :3, 3]).norm(dim=-1).max()
        trace = (pose[..., :3, :3] * measured[..., :3, :3]).sum(dim=(-2, -1))
        angle = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1))).max()
        grip = (gripper.double() - current_gripper.to(gripper.device).double()).abs().max()
        return bool(z_distance <= thresholds.z and translation <= thresholds.position_m
                    and angle <= thresholds.rotation_deg and grip <= thresholds.gripper)


@dataclass(frozen=True)
class ControllerState:
    demonstration: Tensor | None = None
    task_start_time: float | None = None
    history: tuple[Tensor, ...] = ()
    history_times: tuple[float, ...] = ()
    event_state: EventState | None = None
    goal: Mapping | None = None
    chunk: Tensor | None = None
    cursor: int = 0
    stopped: bool = False


@dataclass(frozen=True)
class ControllerStep:
    state: ControllerState
    action: Tensor | None           # [1,A], one control step
    events: tuple[GripperEvent, ...]
    interrupted: bool
    refreshed: bool
    stopped: bool


def controller_step(previous: ControllerState, *, frame: Tensor, time: float,
                    state: Tensor, language: Tensor, current_goal: Mapping,
                    g_predict: Callable, pi_predict: Callable,
                    new_demo: Tensor | None = None,
                    event_rules: EventRules = EventRules(),
                    thresholds: GoalThresholds = GoalThresholds()) -> ControllerStep:
    """Advance one measured control step without mutating previous or issuing IO.

    G receives (demo, robot_history, state); pi receives
    (robot_history, state, language, goal). The caller executes the returned
    action before supplying the next measured frame. G may cache its demo.
    """
    if (not isinstance(frame, Tensor) or frame.ndim != 5 or frame.shape[0] != 1
            or frame.shape[2] != 1 or min(frame.shape) < 1 or not frame.is_floating_point()
            or not torch.isfinite(frame).all()):
        raise ValueError("controller frame must be finite floating [1,C,1,H,W]")
    if type(time) not in (int, float) or not math.isfinite(time):
        raise ValueError("controller time must be finite seconds")
    _, poses, gripper = _goal(current_goal)
    if poses.shape[0] != 1:
        raise ValueError("controller supports one robot task at a time")
    if (not isinstance(state, Tensor) or state.ndim != 2 or state.shape[0] != 1
            or state.shape[1] < 1 or not state.is_floating_point() or not torch.isfinite(state).all()):
        raise ValueError("controller state must be finite floating [1,S]")
    current = previous
    if new_demo is not None:
        if (not isinstance(new_demo, Tensor) or new_demo.ndim != 5 or new_demo.shape[0] != 1
                or min(new_demo.shape) < 1 or not new_demo.is_floating_point()
                or not torch.isfinite(new_demo).all()):
            raise ValueError("new demonstration must be finite floating [1,C,F,H,W]")
        current = ControllerState(demonstration=new_demo.detach().clone(), task_start_time=float(time),
                                  event_state=initial_event_state(poses.shape[1]))
    if current.demonstration is None or current.task_start_time is None:
        raise ValueError("a new demonstration is required to start a task")
    elapsed = float(time) - current.task_start_time
    if elapsed < 0 or (current.history_times and elapsed <= current.history_times[-1]):
        raise ValueError("controller times must strictly increase within the current task")
    if current.history and current.history[-1].shape != frame.shape:
        raise ValueError("robot frame shape must stay constant within a task")
    if current.demonstration.shape[1] != frame.shape[1]:
        raise ValueError("demonstration and robot frame channels must match")
    if current.stopped:
        return ControllerStep(current, None, (), False, False, True)
    if current.event_state is None:
        raise ValueError("controller event state is missing")
    event_state, events = gripper_event_step(current.event_state, gripper[0], elapsed, event_rules)
    history = current.history + (frame.detach().clone(),)
    history_times = current.history_times + (elapsed,)
    robot_history = torch.cat(history, dim=2)
    reached = current.goal is not None and goal_reached(current.goal, current_goal, thresholds)
    refreshed = current.goal is None or bool(events) or reached
    interrupted = bool((events or reached) and current.chunk is not None
                       and current.cursor < current.chunk.shape[-1])
    goal, chunk, cursor = current.goal, current.chunk, current.cursor
    if refreshed:
        goal = g_predict(current.demonstration, robot_history, state)
        _goal(goal)
        chunk, cursor = None, 0
        if goal_reached(goal, current_goal, thresholds):
            stopped = replace(current, history=history, history_times=history_times,
                              event_state=event_state, goal=goal, chunk=None, cursor=0, stopped=True)
            return ControllerStep(stopped, None, events, interrupted, True, True)
    if chunk is None or cursor >= chunk.shape[-1]:
        prediction = pi_predict(robot_history, state, language, goal)
        if (not isinstance(prediction, Tensor) or prediction.ndim != 5 or prediction.shape[0] != 1
                or prediction.shape[-1] != 1 or min(prediction.shape) < 1
                or not prediction.is_floating_point() or not torch.isfinite(prediction).all()):
            raise ValueError("pi must return a finite floating action chunk [1,A,F,N,1]")
        chunk = prediction.detach().clone().flatten(2)
        cursor = 0
    action = chunk[:, :, cursor].clone()
    following = replace(current, history=history, history_times=history_times, event_state=event_state,
                        goal=goal, chunk=chunk, cursor=cursor + 1, stopped=False)
    return ControllerStep(following, action, events, interrupted, refreshed, False)
