"""The toss swing records what the arm did (measured joints and velocities from the
1 kHz loop's own feedback), next to the plan, so release velocity and tracking error
can be read back after a miss."""

# pylint: disable=protected-access,too-few-public-methods

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from prpl_tidybot.third_party import kinova
from prpl_tidybot.toss import summarize_swing

_LAG = 5  # cycles the fake arm's measured position trails the command by
_DOF = 7


class _List(list):
    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def add(self):
        """As a protobuf repeated field's ``add``."""
        self.append(self._factory())
        return self[-1]


def _command():
    command = SimpleNamespace(frame_id=0)
    command.actuators = _List(SimpleNamespace)
    command.interconnect = SimpleNamespace(
        gripper_command=SimpleNamespace(motor_cmd=_List(SimpleNamespace))
    )
    return command


class _LaggingCyclic:
    """Feedback that trails the command by ``_LAG`` cycles, with a velocity unrelated
    to the plan (a constant 33 deg/s), and one dropped frame at cycle 3."""

    def __init__(self, start, cycles):
        self.history = [list(start)]
        self.calls = 0
        self.cycles = cycles

    @property
    def in_loop(self):
        """Between the first loop cycle's Refresh and the last one's (call 1 is the
        Refresh before the loop)."""
        return 2 <= self.calls <= self.cycles

    def _feedback(self):
        q = self.history[max(0, len(self.history) - 1 - _LAG)]
        return SimpleNamespace(
            actuators=[SimpleNamespace(position=v, velocity=33.0) for v in q],
            interconnect=SimpleNamespace(
                gripper_feedback=SimpleNamespace(motor=[SimpleNamespace(position=0.5)])
            ),
        )

    def RefreshFeedback(self):  # pylint: disable=invalid-name
        """The feedback frame read before the loop."""
        return self._feedback()

    def Refresh(self, command, *_args):  # pylint: disable=invalid-name
        """Send one command; the reply carries the lagging feedback."""
        self.calls += 1
        self.history.append([a.position for a in command.actuators])
        if self.calls == 2 + 3:
            raise RuntimeError("the arm moved but its reply was lost")
        return self._feedback()


class _NumpyWatch:
    """Stands in for ``np`` inside kinova and records any use during the loop."""

    def __init__(self, cyclic):
        self.cyclic = cyclic
        self.used_in_loop = []

    def __getattr__(self, name):
        if self.cyclic.in_loop:
            self.used_in_loop.append(name)
        return getattr(np, name)


@pytest.fixture(name="lagging_arm")
def _lagging_arm(monkeypatch):
    trajectory = np.stack(
        [np.linspace(0.0, 30.0 * (j + 1), 60, dtype=np.float32) for j in range(_DOF)],
        axis=1,
    )
    cyclic = _LaggingCyclic(trajectory[0], len(trajectory))
    arm = kinova.TorqueControlledArm.__new__(kinova.TorqueControlledArm)
    arm.cyclic_running = False
    arm.actuator_count = _DOF
    arm.send_options = None
    arm.base_cyclic = cyclic
    arm.base = SimpleNamespace(SetServoingMode=lambda _mode: None)
    monkeypatch.setattr(
        kinova,
        "BaseCyclic_pb2",
        SimpleNamespace(Command=_command),
        raising=False,
    )
    monkeypatch.setattr(
        kinova,
        "Base_pb2",
        SimpleNamespace(
            ServoingModeInformation=SimpleNamespace,
            LOW_LEVEL_SERVOING=1,
            SINGLE_LEVEL_SERVOING=2,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        kinova, "ActuatorCyclic_pb2", SimpleNamespace(SERVO_ENABLE=1), raising=False
    )
    return arm, cyclic, trajectory


def test_the_swing_loop_records_the_arms_feedback_not_the_plan(
    lagging_arm, monkeypatch
):
    """The loop stores each cycle's feedback (lagging the command) with no numpy call
    or file I/O per cycle, and a failed Refresh as NaN."""
    arm, cyclic, trajectory = lagging_arm
    watch = _NumpyWatch(cyclic)
    monkeypatch.setattr(kinova, "np", watch)
    real_open = builtins.open

    def no_open_in_loop(*args, **kwargs):
        assert not cyclic.in_loop, "the swing loop opened a file"
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_open_in_loop)

    trace = arm.run_position_trajectory(trajectory, gripper_release_step=20)

    assert not watch.used_in_loop, "the loop used numpy (allocation) per cycle"
    n = len(trajectory)
    assert trace["q_deg"].shape == (n, _DOF)
    assert trace["dq_deg_s"].shape == (n, _DOF)
    assert trace["t"].shape == (n,)
    assert trace["release_cycle"] == 20
    # A failed Refresh is recorded as NaN, not as the previous frame.
    assert np.isnan(trace["q_deg"][3]).all() and np.isnan(trace["t"][3])
    assert trace["failed_cycles"] == 1
    # Measured = the command from _LAG cycles earlier (the fake arm's lag).
    np.testing.assert_allclose(trace["q_deg"][30], trajectory[30 - _LAG], atol=1e-4)
    np.testing.assert_allclose(trace["dq_deg_s"][30], 33.0)
    np.testing.assert_array_equal(trace["commanded_deg"], trajectory)


def test_the_summary_is_computed_from_the_measured_trace(lagging_arm):
    """Peak speed, release joints and tracking error come from the feedback."""
    arm, _cyclic, trajectory = lagging_arm
    trace = arm.run_position_trajectory(trajectory, gripper_release_step=20)
    summary = summarize_swing(trace)
    assert summary["measured_max_vel_deg_s"] == pytest.approx(33.0)
    assert summary["measured_release_joint_vel_deg_s"] == pytest.approx([33.0] * _DOF)
    assert summary["measured_release_joints_deg"] == pytest.approx(
        [float(v) for v in trajectory[20 - _LAG]], abs=1e-4
    )
    # The joint that moves furthest (210 deg over 59 cycles) trails by _LAG cycles.
    assert summary["max_tracking_error_deg"] == pytest.approx(
        210.0 / 59 * _LAG, rel=1e-3
    )
    assert summary["measured_release_cycle"] == 20
    assert summary["failed_cycles"] == 1
