from dataclasses import replace
import unittest

import torch

from evo_wam.g_pi_controller import (ControllerState, GoalThresholds, controller_step, goal_reached)
from evo_wam.g_pi_data import EventRules


def goal(position=0., gripper=0., z=0.):
    poses = torch.eye(4)[None, None].clone()
    poses[..., 0, 3] = position
    return {"z": torch.full((1, 2, 4), float(z)), "goal_poses": poses,
            "goal_gripper": torch.tensor([[float(gripper)]])}


class GPiControllerTest(unittest.TestCase):
    def setUp(self):
        self.g_calls, self.pi_calls = [], []
        self.targets = [goal(1.), goal(2.), goal(3.)]

    def g_predict(self, demo, history, state):
        self.g_calls.append((demo.clone(), history.clone(), state.clone()))
        return self.targets.pop(0)

    def pi_predict(self, history, state, language, target):
        self.pi_calls.append((history.clone(), state.clone(), language.clone(), target))
        return torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2, 1) + len(self.pi_calls) * 10

    def step(self, previous, time, *, current=None, new_demo=None, new_frame=True, **extra):
        frame = torch.full((1, 2, 1, 2, 2), float(time)) if new_frame else None
        available = extra.pop("frame_available_time", float(time) if new_frame else None)
        return controller_step(previous, frame=frame, frame_available_time=available,
                               time=float(time), state=torch.zeros(1, 4), language=torch.zeros(1, 3, 8),
                               current_goal=goal() if current is None else current,
                               g_predict=self.g_predict, pi_predict=self.pi_predict,
                               new_demo=new_demo, **extra)

    def start(self, *, time=10., **extra):
        return self.step(ControllerState(), time, new_demo=torch.ones(1, 2, 3, 2, 2), **extra)

    def test_chunk_once_then_pi_refreshes_without_g(self):
        result = self.start()
        first = result.state
        actions = [result.action.item()]
        for time in (11, 12, 13, 14):
            result = self.step(result.state, time)
            actions.append(result.action.item())
        self.assertEqual(actions, [10., 11., 12., 13., 20.])
        self.assertEqual(len(self.g_calls), 1)
        self.assertEqual(len(self.pi_calls), 2)
        self.assertEqual(first.cursor, 1)
        self.assertEqual(first.history_times, (0.,))
        self.assertEqual(result.state.history_times, (0., 1., 2., 3., 4.))
        self.assertEqual(result.state.last_control_time, 4.)
        self.assertEqual(result.state.history[0].shape[2], 1)

    def test_event_interrupts_pending_actions_and_switches_goal(self):
        result = self.start()
        result = self.step(result.state, 11)
        result = self.step(result.state, 12, current=goal(gripper=1.))
        self.assertFalse(result.interrupted)
        result = self.step(result.state, 13, current=goal(gripper=1.))
        self.assertTrue(result.interrupted)
        self.assertTrue(result.refreshed)
        self.assertEqual(result.events[0].kind, "open")
        self.assertEqual(result.events[0].time, 3.)
        self.assertEqual(result.action.item(), 20.)
        self.assertEqual(len(self.g_calls), 2)
        self.assertEqual(result.state.goal["goal_poses"][0, 0, 0, 3].item(), 2.)

    def test_goal_reached_interrupts_and_calls_g(self):
        result = self.start()
        result = self.step(result.state, 11, current=goal(1.))
        self.assertTrue(result.interrupted)
        self.assertTrue(result.refreshed)
        self.assertFalse(result.stopped)
        self.assertEqual(result.action.item(), 20.)
        self.assertEqual(len(self.g_calls), 2)

    def test_new_goal_matching_current_stops_without_pi(self):
        self.targets = [goal()]
        result = self.start()
        self.assertTrue(result.stopped)
        self.assertIsNone(result.action)
        self.assertEqual(len(self.pi_calls), 0)
        following = self.step(result.state, 11)
        self.assertTrue(following.stopped)
        self.assertEqual(len(self.g_calls), 1)

    def test_reached_goal_then_g_returns_current_and_stops(self):
        self.targets = [goal(1.), goal(1.)]
        result = self.start()
        result = self.step(result.state, 11, current=goal(1.))
        self.assertTrue(result.stopped)
        self.assertTrue(result.interrupted)
        self.assertIsNone(result.state.chunk)
        self.assertEqual(len(self.pi_calls), 1)

    def test_new_demo_resets_history_clock_chunk_and_event_detector(self):
        result = self.start()
        result = self.step(result.state, 11)
        result = self.step(result.state, 12, current=goal(gripper=1.))
        old = result.state
        result = self.step(old, 100, current=goal(gripper=1.), new_demo=torch.zeros(1, 2, 5, 2, 2))
        self.assertEqual(result.state.task_start_time, 100.)
        self.assertEqual(result.state.history_times, (0.,))
        self.assertEqual(result.state.last_control_time, 0.)
        self.assertEqual(len(result.state.history), 1)
        self.assertEqual(result.events, ())
        self.assertEqual(result.state.cursor, 1)
        self.assertEqual(self.g_calls[-1][1].shape[2], 1)
        self.assertEqual(self.g_calls[-1][0].shape[2], 5)
        self.assertEqual(old.history_times, (0., 1., 2.))

    def test_stopped_task_can_be_reset(self):
        self.targets = [goal(), goal(2.)]
        result = self.start()
        self.assertTrue(result.stopped)
        result = self.step(result.state, 20, new_demo=torch.zeros(1, 2, 2, 2, 2))
        self.assertFalse(result.stopped)
        self.assertEqual(result.state.history_times, (0.,))

    def test_stop_requires_z_pose_rotation_and_gripper(self):
        current = goal()
        self.assertTrue(goal_reached(goal(), current))
        for target in (goal(z=1.), goal(position=.1), goal(gripper=.2)):
            self.assertFalse(goal_reached(target, current))
        rotated = goal()
        rotated["goal_poses"][..., :3, :3] = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        self.assertFalse(goal_reached(rotated, current))
        self.assertTrue(goal_reached(goal(position=.1), current, GoalThresholds(position_m=.2)))

    def test_rejects_invalid_observations_and_times(self):
        with self.assertRaisesRegex(ValueError, "new demonstration"):
            self.step(ControllerState(), 0)
        result = self.start()
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.step(result.state, 10)
        for value in (-1., float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GoalThresholds(z=value)
        invalid = replace(result.state, event_state=None)
        with self.assertRaisesRegex(ValueError, "event state"):
            self.step(invalid, 11)

    def test_z_stop_distance_is_independent_of_feature_dimension(self):
        current, target = goal(), goal()
        current["z"] = torch.nn.functional.normalize(torch.ones(1, 8, 3072), dim=-1)
        target["z"] = -current["z"]
        self.assertFalse(goal_reached(target, current))
        target["z"] = current["z"].clone()
        self.assertTrue(goal_reached(target, current))

    def test_event_between_latents_interrupts_without_duplicate_image(self):
        result = self.start()
        result = self.step(result.state, 11, new_frame=False)
        result = self.step(result.state, 12, current=goal(gripper=1.), new_frame=False)
        result = self.step(result.state, 13, current=goal(gripper=1.), new_frame=False)
        self.assertTrue(result.interrupted)
        self.assertTrue(result.refreshed)
        self.assertEqual(result.events[0].kind, "open")
        self.assertEqual(result.events[0].time, 3.)
        self.assertEqual(result.action.item(), 20.)
        self.assertEqual(result.state.history_times, (0.,))
        self.assertEqual(result.state.last_control_time, 3.)
        torch.testing.assert_close(self.g_calls[0][1], self.g_calls[1][1], rtol=0, atol=0)
        self.assertEqual(self.pi_calls[-1][0].shape[2], 1)

    def test_latents_arrive_every_four_control_steps(self):
        result = self.start()
        actions = [result.action.item()]
        for index in range(1, 9):
            result = self.step(result.state, 10 + index, new_frame=index % 4 == 0)
            actions.append(result.action.item())
        self.assertEqual(result.state.history_times, (0., 4., 8.))
        self.assertEqual(result.state.last_control_time, 8.)
        self.assertEqual([call[0].shape[2] for call in self.pi_calls], [1, 2, 3])
        self.assertEqual(actions, [10., 11., 12., 13., 20., 21., 22., 23., 30.])

    def test_available_history_can_lag_control_time(self):
        result = self.start()
        result = self.step(result.state, 11, new_frame=False)
        result = self.step(result.state, 12, frame_available_time=11.5)
        self.assertEqual(result.state.history_times, (0., 1.5))
        self.assertEqual(result.state.last_control_time, 2.)
        result = self.step(result.state, 13, new_frame=False, current=goal(1.))
        self.assertTrue(result.interrupted)
        self.assertEqual(self.g_calls[-1][1].shape[2], 2)
        self.assertEqual(result.state.history_times, (0., 1.5))

    def test_first_latent_cannot_be_omitted_or_shifted(self):
        with self.assertRaisesRegex(ValueError, "initial latent"):
            self.start(new_frame=False)
        with self.assertRaisesRegex(ValueError, "initial latent"):
            self.start(frame_available_time=9.)
        with self.assertRaisesRegex(ValueError, "initial latent"):
            self.start(frame_available_time=10.1)
        result = self.start()
        with self.assertRaisesRegex(ValueError, "initial latent"):
            self.step(result.state, 20, new_frame=False, new_demo=torch.ones(1, 2, 2, 2, 2))

    def test_rejects_future_duplicate_and_unspecified_frame_availability(self):
        result = self.start()
        for available in (9., 12.):
            with self.subTest(available=available), self.assertRaisesRegex(ValueError, "future"):
                self.step(result.state, 11, frame_available_time=available)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.step(result.state, 11, frame_available_time=10.)
        for available in (None, float("nan"), True):
            with self.subTest(available=available), self.assertRaisesRegex(ValueError, "explicit"):
                self.step(result.state, 11, frame_available_time=available)
        with self.assertRaisesRegex(ValueError, "omitted"):
            self.step(result.state, 11, new_frame=False, frame_available_time=11.)

    def test_control_clock_is_strict_even_without_new_latents(self):
        result = self.start()
        result = self.step(result.state, 11, new_frame=False)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.step(result.state, 11, new_frame=False)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.step(result.state, 10.5, new_frame=False)
        self.targets = [goal()]
        stopped = self.start(time=20)
        stopped = self.step(stopped.state, 21, new_frame=False)
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.step(stopped.state, 21, new_frame=False)

    def test_new_demo_resets_between_latent_arrivals(self):
        result = self.start()
        result = self.step(result.state, 11, new_frame=False)
        result = self.step(result.state, 12, new_frame=False, current=goal(gripper=1.))
        previous = result.state
        result = self.step(previous, 100, current=goal(gripper=1.), new_demo=torch.zeros(1, 2, 5, 2, 2))
        self.assertEqual(result.state.task_start_time, 100.)
        self.assertEqual(result.state.last_control_time, 0.)
        self.assertEqual(result.state.history_times, (0.,))
        self.assertEqual(result.events, ())
        self.assertEqual(result.state.cursor, 1)
        self.assertEqual(self.g_calls[-1][1].shape[2], 1)
        self.assertEqual(previous.last_control_time, 2.)

    def test_reached_without_new_latent_can_stop(self):
        self.targets = [goal(1.), goal(1.)]
        result = self.start()
        result = self.step(result.state, 11, current=goal(1.), new_frame=False)
        self.assertTrue(result.interrupted)
        self.assertTrue(result.stopped)
        self.assertIsNone(result.action)
        self.assertEqual(result.state.history_times, (0.,))
        self.assertEqual(result.state.last_control_time, 1.)


if __name__ == "__main__":
    unittest.main()
