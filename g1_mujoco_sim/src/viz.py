from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped
import rospy
import tf
from xbot2_interface import pyxbot2_interface as xbi

class RvizSrbdFullBody:
    def __init__(self, urdf, contact_frames):
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=1)
        self.joint_state_msg = JointState()
        self.model = xbi.ModelInterface2(urdf)
        self.joint_state_msg.name = self.model.getJointNames()[1::]

        self.broadcaster = tf.TransformBroadcaster()

        # For floating base of full body
        self.w_T_b = TransformStamped()
        self.w_T_b.header.frame_id = "world"
        self.w_T_b.child_frame_id = "pelvis"

        # For single rigid body
        self.w_T_com = TransformStamped()
        self.w_T_com.header.frame_id = "world"
        self.w_T_com.child_frame_id = "srbd_com"

        for i in range(len(contact_frames)):
            self.w_T_c = TransformStamped()
            self.w_T_c.header.frame_id = "world"
            self.w_T_c.child_frame_id ="srbd_" + contact_frames[i]
        
        self.force_msg = list()
        self.fpubs = list()
        for contact_frame in contact_frames:
            self.force_msg.append(WrenchStamped())
            self.force_msg[-1].header.frame_id = contact_frame
            self.force_msg[-1].wrench.torque.x = 0.
            self.force_msg[-1].wrench.torque.y = 0.
            self.force_msg[-1].wrench.torque.z = 0.
            self.fpubs.append(rospy.Publisher(contact_frame, WrenchStamped, queue_size=1))

    def publishJointState(self, t, q):
        
        t = rospy.Time(t)
        self.joint_state_msg.position = q[7::]
        self.joint_state_msg.header.stamp = t
        pub = rospy.Publisher('joint_states', JointState, queue_size=1)

        w_T_b = TransformStamped()
        w_T_b.header.frame_id = "world"
        w_T_b.child_frame_id = "pelvis"
        w_T_b.header.stamp = t
        w_T_b.transform.translation.x = q[0]
        w_T_b.transform.translation.y = q[1]
        w_T_b.transform.translation.z = q[2]
        w_T_b.transform.rotation.x = q[3]
        w_T_b.transform.rotation.y = q[4]
        w_T_b.transform.rotation.z = q[5]
        w_T_b.transform.rotation.w = q[6]

        self.broadcaster.sendTransformMessage(w_T_b)
        pub.publish(self.joint_state_msg)

    def publishContactForce(self, t, f, frame):
        f_msg = WrenchStamped()
        f_msg.header.stamp = t
        f_msg.header.frame_id = frame
        f_msg.wrench.force.x = f[0]
        f_msg.wrench.force.y = f[1]
        f_msg.wrench.force.z = f[2]
        f_msg.wrench.torque.x = f_msg.wrench.torque.y = f_msg.wrench.torque.z = 0.
        pub = rospy.Publisher('force_' + frame, WrenchStamped, queue_size=10).publish(f_msg)

    def publishSRBDViewer(self, I, srbd_msg, t, contact_frames):

        w_T_com = TransformStamped()
        w_T_com.header.frame_id = "world"
        w_T_com.child_frame_id = "srbd_com"
        w_T_com.header.stamp = t

        # Check if states_horizon has elements before accessing
        if len(srbd_msg.states_horizon) > 1:
            # Use index 1 as before
            state = srbd_msg.states_horizon[1]
        else:
            # Handle empty states_horizon case - return early
            rospy.logwarn("Empty states_horizon in srbd_msg, skipping visualization")
            return

        w_T_com.transform.translation.x = state.position.x
        w_T_com.transform.translation.y = state.position.y
        w_T_com.transform.translation.z = state.position.z

        # Get quaternion from euler angles
        q = tf.transformations.quaternion_from_euler(state.orientation.x, state.orientation.y, state.orientation.z)
        w_T_com.transform.rotation.x = q[0]
        w_T_com.transform.rotation.y = q[1]
        w_T_com.transform.rotation.z = q[2]
        w_T_com.transform.rotation.w = q[3]

        self.broadcaster.sendTransformMessage(w_T_com)

        marker = Marker()
        marker.header.frame_id = "srbd_com"
        marker.header.stamp = t
        marker.ns = "SRBD"
        marker.id = 200
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = marker.pose.position.y = marker.pose.position.z = 0.
        marker.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.
        marker.pose.orientation.w = 1.
        a = I[0,0] + I[1,1] + I[2,2]
        marker.scale.x = 0.5*(I[2,2] + I[1,1])/a
        marker.scale.y = 0.5*(I[2,2] + I[0,0])/a
        marker.scale.z = 0.5*(I[0,0] + I[1,1])/a
        marker.color.a = 0.6
        marker.color.r = marker.color.g = marker.color.b = 0.7

        pub = rospy.Publisher('srbd_marker', Marker, queue_size=10).publish(marker)

    def publishPointTrj(self, points, t, name, color = [0., 0., 1.]):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = t
        marker.ns = "SRBD"
        marker.id = 1000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Set the identity quaternion once before iterating over points
        marker.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.
        marker.pose.orientation.w = 1.

        for i in range(len(points)):
            p = Point()
            p.x = points[i].position.x
            p.y = points[i].position.y
            p.z = points[i].position.z
            marker.points.append(p)

        marker.color.a = 1.
        marker.scale.x = 0.005
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        pub = rospy.Publisher(name + "_trj", Marker, queue_size=10).publish(marker)
        
    def publishContactFrames(self, t, srbd_msg, contact_frames):
        marker_array = MarkerArray()

        # Check if states_horizon has elements before accessing
        if len(srbd_msg.states_horizon) == 0:
            # Handle empty states_horizon case - return early
            rospy.logwarn("Empty states_horizon in srbd_msg, skipping visualization")
            return

        for i, contact_frame in enumerate(contact_frames):
            w_T_c = TransformStamped()

            try:
                w_T_c.transform.translation.x = srbd_msg.contacts[i].position.x
                w_T_c.transform.translation.y = srbd_msg.contacts[i].position.y
                w_T_c.transform.translation.z = srbd_msg.contacts[i].position.z
                w_T_c.transform.rotation.x = 0.
                w_T_c.transform.rotation.y = 0.
                w_T_c.transform.rotation.z = 0.
                w_T_c.transform.rotation.w = 1.
                w_T_c.header.frame_id = "world"
                w_T_c.child_frame_id = "marker_" + contact_frame
                w_T_c.header.stamp = t
                # rospy.loginfo("foot position: %f, %f, %f", srbd_msg.contacts[i].position.x, srbd_msg.contacts[i].position.y, srbd_msg.contacts[i].position.z)
                self.broadcaster.sendTransformMessage(w_T_c)
            except Exception as e:
                print(e)

            m = Marker()
            m.header.frame_id = "marker_" + contact_frame
            m.header.stamp = t
            m.ns = "SRBD"
            m.id = 1000 + i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.
            m.pose.orientation.x = m.pose.orientation.y = m.pose.orientation.z = 0.
            m.pose.orientation.w = 1.
            m.scale.x = m.scale.y = m.scale.z = 0.04
            m.color.a = 0.8
            m.color.r = m.color.g = 0.0
            m.color.b = 1.0
            marker_array.markers.append(m)

        pub2 = rospy.Publisher('contacts', MarkerArray, queue_size=10).publish(marker_array)

    def publishLandingPosition(self, t, srbd_msg):

        # Check if states_horizon has elements before accessing
        if len(srbd_msg.states_horizon) == 0:
            # Handle empty states_horizon case - return early
            rospy.logwarn("Empty states_horizon in srbd_msg, skipping visualization")
            return

        w_T_c = TransformStamped()
        try:
            w_T_c.transform.translation.x = srbd_msg.landing_position.x
            w_T_c.transform.translation.y = srbd_msg.landing_position.y
            w_T_c.transform.translation.z = srbd_msg.landing_position.z
            w_T_c.transform.rotation.x = 0.
            w_T_c.transform.rotation.y = 0.
            w_T_c.transform.rotation.z = 0.
            w_T_c.transform.rotation.w = 1.
            w_T_c.header.frame_id = "world"
            w_T_c.child_frame_id = "marker_landing_position"
            w_T_c.header.stamp = t
            self.broadcaster.sendTransformMessage(w_T_c)
        except Exception as e:
            print(e)

        m = Marker()
        m.header.frame_id = "marker_landing_position"
        m.header.stamp = t
        m.ns = "SRBD"
        m.id = 1100 + 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.
        m.pose.orientation.x = m.pose.orientation.y = m.pose.orientation.z = 0.
        m.pose.orientation.w = 1.
        m.scale.x = m.scale.y = m.scale.z = 0.06
        m.color.a = 0.8
        m.color.r = m.color.g = 1.0
        m.color.b = 0.0

        pub3 = rospy.Publisher('Landing_position', Marker, queue_size=10).publish(m)