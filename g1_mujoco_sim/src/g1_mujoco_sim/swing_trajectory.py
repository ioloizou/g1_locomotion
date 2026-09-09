"""Swing-foot trajectory, evaluated as a function of gait phase.

Parameterising by phase rather than by an independent timer is what keeps the
foot in step with the gait: the planner owns the clock, and this only answers
"where should the foot be at phase p".  It also means a small change in the
landing target between ticks moves the foot by a correspondingly small amount
instead of restarting the trajectory.
"""

import numpy as np


class SwingTrajectory(object):
    """Quintic in the horizontal plane, sixth-order in the vertical."""

    def __init__(self, swing_height=0.06, touchdown_velocity=-0.05):
        self.swing_height = float(swing_height)
        self.touchdown_velocity = float(touchdown_velocity)
        self.start = np.zeros(3)
        self.end = np.zeros(3)
        self._z_coeff = np.zeros(7)
        self._solve_z()

    def set_endpoints(self, start, end):
        """Set liftoff and landing positions, both world frame."""
        self.start = np.asarray(start, dtype=float).copy()
        self.end = np.asarray(end, dtype=float).copy()
        self._solve_z()

    def _solve_z(self):
        # Six constraints plus the apex: position, velocity and acceleration at
        # both ends, and the apex height at mid-swing.  The small negative
        # touchdown velocity makes the foot land rather than skim, which the
        # contact detection and the stance task both prefer.
        apex = max(self.start[2], self.end[2]) + self.swing_height
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0],                                   # p(0)
            [0, 1, 0, 0, 0, 0, 0],                                   # v(0)
            [0, 0, 2, 0, 0, 0, 0],                                   # a(0)
            [1, .5, .5 ** 2, .5 ** 3, .5 ** 4, .5 ** 5, .5 ** 6],    # p(0.5)
            [1, 1, 1, 1, 1, 1, 1],                                   # p(1)
            [0, 1, 2, 3, 4, 5, 6],                                   # v(1)
            [0, 0, 2, 6, 12, 20, 30],                                # a(1)
        ])
        b = np.array([self.start[2], 0.0, 0.0, apex, self.end[2],
                      self.touchdown_velocity, 0.0])
        self._z_coeff = np.linalg.solve(A, b)

    @staticmethod
    def _quintic(phase):
        """Minimum-jerk blend and its derivative: 0 to 1 with zero end slopes."""
        p = float(np.clip(phase, 0.0, 1.0))
        s = 10 * p ** 3 - 15 * p ** 4 + 6 * p ** 5
        ds = 30 * p ** 2 - 60 * p ** 3 + 30 * p ** 4
        return s, ds

    def position(self, phase):
        p = float(np.clip(phase, 0.0, 1.0))
        s, _ = self._quintic(p)
        xy = (1.0 - s) * self.start[:2] + s * self.end[:2]
        powers = np.array([1, p, p ** 2, p ** 3, p ** 4, p ** 5, p ** 6])
        return np.array([xy[0], xy[1], float(self._z_coeff @ powers)])

    def velocity(self, phase, duration):
        """Time derivative, given the total swing duration in seconds."""
        p = float(np.clip(phase, 0.0, 1.0))
        _, ds = self._quintic(p)
        xy = ds * (self.end[:2] - self.start[:2]) / duration
        powers = np.array([0, 1, 2 * p, 3 * p ** 2, 4 * p ** 3, 5 * p ** 4,
                           6 * p ** 5])
        return np.array([xy[0], xy[1], float(self._z_coeff @ powers) / duration])

    def acceleration(self, phase, duration):
        p = float(np.clip(phase, 0.0, 1.0))
        d2s = 60 * p - 180 * p ** 2 + 120 * p ** 3
        xy = d2s * (self.end[:2] - self.start[:2]) / duration ** 2
        powers = np.array([0, 0, 2, 6 * p, 12 * p ** 2, 20 * p ** 3, 30 * p ** 4])
        return np.array([xy[0], xy[1], float(self._z_coeff @ powers) / duration ** 2])

    def sample(self, count=50):
        """Positions along the whole swing, for visualisation."""
        return np.array([self.position(p) for p in np.linspace(0.0, 1.0, count)])
