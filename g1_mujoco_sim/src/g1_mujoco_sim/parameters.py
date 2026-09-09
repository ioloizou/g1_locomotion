"""Typed access to the locomotion parameters.

The same YAML feeds this and the C++ MPC node, so friction, gravity, gait
timing and foot geometry cannot drift apart between the planner and the plant.
Lookups are strict: a missing key raises instead of silently falling back to a
default that disagrees with what the other node is using.

Parameters normally come from the ROS parameter server, but can also be read
straight from the YAML file so the controller can be exercised offline without
a master running.
"""

import os

import numpy as np


class _Section(object):
    """Attribute-style view over one namespace of the configuration."""

    def __init__(self, name, lookup):
        self._name = name
        self._lookup = lookup

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return self._lookup(self._name, key)


class Parameters(object):
    """All controller parameters, grouped as in the YAML."""

    SECTIONS = ("robot", "physics", "simulation", "gait", "footstep", "mpc", "wbid")

    def __init__(self, source=None):
        """`source` is a nested dict; when omitted the ROS parameter server is used."""
        self._source = source
        for name in self.SECTIONS:
            setattr(self, name, _Section(name, self._get))

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, "r") as handle:
            return cls(yaml.safe_load(handle))

    @classmethod
    def default_yaml_path(cls):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, os.pardir, os.pardir, "config",
                            "g1_locomotion.yaml")

    def _get(self, section, key):
        if self._source is not None:
            try:
                return self._source[section][key]
            except KeyError:
                raise KeyError("missing parameter '%s/%s'" % (section, key))

        import rospy
        name = "%s/%s" % (section, key)
        if not rospy.has_param(name):
            raise KeyError(
                "required parameter '%s' is missing. Did the launch file load "
                "config/g1_locomotion.yaml?" % name)
        return rospy.get_param(name)

    @property
    def inertia(self):
        """Centroidal composite inertia as a 3x3 array."""
        return np.array(self.robot.inertia, dtype=float)

    @property
    def contact_frames(self):
        """Contact points in the canonical L-heel, L-toe, R-heel, R-toe order."""
        return list(self.robot.contact_frames)

    @property
    def foot_frames(self):
        """Foot frames, ordered left then right."""
        return list(self.robot.foot_frames)

    @property
    def gravity_vector(self):
        return np.array([0.0, 0.0, float(self.physics.gravity)])
