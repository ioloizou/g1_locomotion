"""RViz publishers for the single-rigid-body plan and the full-body state.

Every publisher is created once in the constructor. The previous version built
a fresh ``rospy.Publisher`` inside each publish call, which re-registers with
the master on every invocation; at simulation rate that cost far more than the
control computation it was meant to visualise.
"""

import numpy as np
import rospy
import tf

from geometry_msgs.msg import Point, TransformStamped, WrenchStamped
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from xbot2_interface import pyxbot2_interface as xbi


class Visualizer(object):
    def __init__(self, urdf, contact_frames, publish_period=0.02):
        self.model = xbi.ModelInterface2(urdf)
        self.contact_frames = list(contact_frames)
        self.publish_period = float(publish_period)
        self._last_publish = -1.0

        self.broadcaster = tf.TransformBroadcaster()

        self.joint_state = JointState()
        self.joint_state.name = self.model.getJointNames()[1:]
        self.joint_state_pub = rospy.Publisher("joint_states", JointState,
                                               queue_size=1)

        self.force_pubs = [
            rospy.Publisher("force_" + frame, WrenchStamped, queue_size=1)
            for frame in self.contact_frames
        ]
        self.srbd_pub = rospy.Publisher("srbd_marker", Marker, queue_size=1)
        self.contacts_pub = rospy.Publisher("contacts", MarkerArray, queue_size=1)
        self.com_trajectory_pub = rospy.Publisher("com_trajectory", Marker,
                                                  queue_size=1)
        self.swing_trajectory_pub = rospy.Publisher("swing_trajectory", Marker,
                                                    queue_size=1)
        self.footstep_pub = rospy.Publisher("footstep_targets", MarkerArray,
                                            queue_size=1)

    def due(self, sim_time):
        """Throttle: RViz does not need markers at the physics rate."""
        if sim_time - self._last_publish < self.publish_period:
            return False
        self._last_publish = sim_time
        return True

    # ------------------------------------------------------------------ state

    def publish_joint_state(self, stamp, q):
        self.joint_state.header.stamp = stamp
        self.joint_state.position = q[7:]
        self.joint_state_pub.publish(self.joint_state)

        transform = TransformStamped()
        transform.header.frame_id = "world"
        transform.child_frame_id = "pelvis"
        transform.header.stamp = stamp
        transform.transform.translation.x = q[0]
        transform.transform.translation.y = q[1]
        transform.transform.translation.z = q[2]
        transform.transform.rotation.x = q[3]
        transform.transform.rotation.y = q[4]
        transform.transform.rotation.z = q[5]
        transform.transform.rotation.w = q[6]
        self.broadcaster.sendTransformMessage(transform)

    def publish_contact_forces(self, stamp, forces):
        for i, frame in enumerate(self.contact_frames):
            msg = WrenchStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = frame
            msg.wrench.force.x = forces[i][0]
            msg.wrench.force.y = forces[i][1]
            msg.wrench.force.z = forces[i][2]
            self.force_pubs[i].publish(msg)

    # ------------------------------------------------------------------- plan

    def publish_srbd(self, stamp, inertia, srbd_msg):
        if len(srbd_msg.states_horizon) < 2:
            return
        state = srbd_msg.states_horizon[1]

        transform = TransformStamped()
        transform.header.frame_id = "world"
        transform.child_frame_id = "srbd_com"
        transform.header.stamp = stamp
        transform.transform.translation.x = state.position.x
        transform.transform.translation.y = state.position.y
        transform.transform.translation.z = state.position.z
        q = tf.transformations.quaternion_from_euler(
            state.orientation.x, state.orientation.y, state.orientation.z)
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        self.broadcaster.sendTransformMessage(transform)

        marker = Marker()
        marker.header.frame_id = "srbd_com"
        marker.header.stamp = stamp
        marker.ns = "SRBD"
        marker.id = 200
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        trace = inertia[0, 0] + inertia[1, 1] + inertia[2, 2]
        marker.scale.x = 0.5 * (inertia[2, 2] + inertia[1, 1]) / trace
        marker.scale.y = 0.5 * (inertia[2, 2] + inertia[0, 0]) / trace
        marker.scale.z = 0.5 * (inertia[0, 0] + inertia[1, 1]) / trace
        marker.color.a = 0.6
        marker.color.r = marker.color.g = marker.color.b = 0.7
        self.srbd_pub.publish(marker)

    def publish_contacts(self, stamp, srbd_msg):
        markers = MarkerArray()
        for i, contact in enumerate(srbd_msg.contacts):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "contacts"
            marker.id = 1000 + i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = contact.position.x
            marker.pose.position.y = contact.position.y
            marker.pose.position.z = contact.position.z
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.04
            marker.color.a = 0.8
            marker.color.r = 0.0 if contact.active else 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0 if contact.active else 0.0
            markers.markers.append(marker)
        self.contacts_pub.publish(markers)

    def publish_footstep_targets(self, stamp, srbd_msg):
        markers = MarkerArray()
        for i, foot in enumerate(srbd_msg.foot_plan):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "footsteps"
            marker.id = 1100 + i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = foot.landing_position
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.10
            marker.scale.z = 0.01
            marker.color.a = 0.35 if foot.in_contact else 0.9
            marker.color.r = 1.0
            marker.color.g = 1.0 if foot.in_contact else 0.5
            marker.color.b = 0.0
            markers.markers.append(marker)
        self.footstep_pub.publish(markers)

    def publish_com_trajectory(self, stamp, states):
        marker = self._line_strip(stamp, "com_trajectory", 1200, (0.0, 0.0, 1.0))
        for state in states:
            marker.points.append(
                Point(state.position.x, state.position.y, state.position.z))
        self.com_trajectory_pub.publish(marker)

    def publish_swing_trajectory(self, stamp, points):
        marker = self._line_strip(stamp, "swing_trajectory", 1300, (0.0, 1.0, 0.0))
        for point in np.asarray(points):
            marker.points.append(Point(point[0], point[1], point[2]))
        self.swing_trajectory_pub.publish(marker)

    @staticmethod
    def _line_strip(stamp, namespace, marker_id, color):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.005
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        return marker
