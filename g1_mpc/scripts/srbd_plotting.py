import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Patch

class SRBDVisualizer:
    def __init__(self, is_static=False):
        """
        Initializes the SRBD humanoid visualizer with two separate figures:
          - fig3d: 3D humanoid view.
          - fig2d: 2D top view of planned & adapted footsteps and COM.
        Parameters:
          is_static: Boolean flag. If True, uses a fixed coordinate frame.
        """
        self.is_static = is_static
        # Create separate figures.
        self.fig3d = plt.figure("3D Humanoid Visualization", figsize=(8, 6))
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        # self.fig2d = plt.figure("2D Top View", figsize=(8, 6))
        # self.ax2d = self.fig2d.add_subplot(111)
        
        # Global visual parameters.
        self.body_size = (0.2, 0.45, 1.0)  # width, depth, height
        # This are visual are not the real values, just for visualization
        self.foot_length = 0.4
        self.foot_width = 0.2
        self.com_history = []

    @staticmethod
    def euler_rotation_matrix(roll, pitch, yaw):
        """
        Returns the rotation matrix for the given Euler angles.
        Rotation order: roll (x), pitch (y), yaw (z).
        """
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
    
        Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
    
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0, 0, 1]])
    
        return Rz @ Ry @ Rx

    def get_torso_faces(self, pos, size, euler_angles=(0, 0, 0)):
        """
        Returns the Poly3DCollection representing a cuboid for the torso.
        """
        dx, dy, dz = size
        x, y, z = pos - np.array([dx/2, dy/2, dz/2])
    
        vertices = np.array([
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz]
        ])
    
        pivot = np.array([x + dx/2.0, y + dy/2.0, z + dz/2.0])
        R = self.euler_rotation_matrix(*euler_angles)
    
        rotated_vertices = []
        for v in vertices:
            rotated = R @ (v - pivot) + pivot
            rotated_vertices.append(rotated)
        rotated_vertices = np.array(rotated_vertices)
    
        faces = [
            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],
            [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],
            [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],
            [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],
            [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],
            [rotated_vertices[3], rotated_vertices[0], rotated_vertices[4], rotated_vertices[7]]
        ]
    
        cuboid = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.7)
        return cuboid

    def foot_vertices(self, center, angle):
        """
        Returns 3D vertices for a foot polygon given its center and rotation.
        """
        cx, cy = center
        hl = self.foot_length / 2.0
        hw = self.foot_width / 2.0
        local_corners = np.array([
            [-hl, -hw],
            [ hl, -hw],
            [ hl,  hw],
            [-hl,  hw]
        ])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        world_corners = np.dot(local_corners, R.T) + np.array([cx, cy])
        return [[pt[0], pt[1], 0] for pt in world_corners]

    def set_left_foot_pose(self, x, y, angle):
        """
        Returns the vertices of the left foot based on the given center.
        """
        center = np.array([x, y])
        return self.foot_vertices(center, angle)

    def set_right_foot_pose(self, x, y, angle):
        """
        Returns the vertices of the right foot based on the given center.
        """
        center = np.array([x, y])
        return self.foot_vertices(center, angle)

    def draw_torso(self, torso_pos, torso_euler):
        """
        Draws the torso of the humanoid.
        """
        cuboid = self.get_torso_faces(np.array(torso_pos), self.body_size, torso_euler)
        self.ax3d.add_collection3d(cuboid)

    def draw_left_foot(self, left_foot_center, left_foot_angle):
        """
        Draws the left foot of the humanoid.
        """
        left_foot = self.set_left_foot_pose(left_foot_center[0], left_foot_center[1], left_foot_angle)
        left_foot_poly = Poly3DCollection([left_foot], facecolors='brown',
                                          linewidths=1, edgecolors='k', alpha=0.8)
        self.ax3d.add_collection3d(left_foot_poly)

    def draw_right_foot(self, right_foot_center, right_foot_angle):
        """
        Draws the right foot of the humanoid.
        """
        right_foot = self.set_right_foot_pose(right_foot_center[0], right_foot_center[1], right_foot_angle)
        right_foot_poly = Poly3DCollection([right_foot], facecolors='brown',
                                           linewidths=1, edgecolors='k', alpha=0.8)
        self.ax3d.add_collection3d(right_foot_poly)

    def draw_ground_reaction_force(self, pos, force, color='black'):
        """
        Draws an arrow representing the ground reaction force.
        """
        self.ax3d.quiver(pos[0], pos[1], pos[2],
                         force[0], force[1], force[2],
                         color=color, length=0.01)

    def draw_left_grf(self, left_foot_center, left_grf):
        """
        Draws the left foot ground reaction force.
        """
        pos = (left_foot_center[0], left_foot_center[1], 0)
        self.draw_ground_reaction_force(pos, left_grf, color='red')

    def draw_right_grf(self, right_foot_center, right_grf):
        """
        Draws the right foot ground reaction force.
        """
        pos = (right_foot_center[0], right_foot_center[1], 0)
        self.draw_ground_reaction_force(pos, right_grf, color='blue')

    def draw_adapted_left_footstep(self, adapted_left_steps):
        """
        Draws the adapted left footsteps in green.
        
        Parameters:
          adapted_left_steps: List of dictionaries where each contains:
            'foot_center': Tuple (x, y) for the foot center.
            'foot_angle': Rotation angle in radians.
        """
        for step in adapted_left_steps:
            foot_center = step['foot_center']
            foot_angle = step['foot_angle']
            foot = self.foot_vertices(foot_center, foot_angle)
            adapted_left_poly = Poly3DCollection([foot], facecolors='green',
                                                 linewidths=1, edgecolors='k', alpha=0.8)
            self.ax3d.add_collection3d(adapted_left_poly)

    def draw_adapted_right_footstep(self, adapted_right_steps):
        """
        Draws the adapted right footsteps in magenta.
        
        Parameters:
          adapted_right_steps: List of dictionaries where each contains:
            'foot_center': Tuple (x, y) for the foot center.
            'foot_angle': Rotation angle in radians.
        """
        for step in adapted_right_steps:
            foot_center = step['foot_center']
            foot_angle = step['foot_angle']
            foot = self.foot_vertices(foot_center, foot_angle)
            adapted_right_poly = Poly3DCollection([foot], facecolors='magenta',
                                                  linewidths=1, edgecolors='k', alpha=0.8)
            self.ax3d.add_collection3d(adapted_right_poly)

    def update_and_plot_humanoid(self, pose):
        """
        Updates the 3D figure with torso, feet, ground reaction forces,
        and adapted footsteps.

        Expects pose keys:
          'torso_pos', 'torso_euler',
          'left_foot_center', 'left_foot_angle',
          'right_foot_center', 'right_foot_angle',
          Optionally: 'left_grf' and 'right_grf'
          'adapted_left_footstep', 'adapted_right_footstep'
          
        This version uses a constant fixed offset for toe/heel positions (ignoring any foot angle)
        and splits the provided GRF equally between toe and heel if available.
        """
        self.ax3d.clear()
        # Set axis limits.
        if self.is_static:
            margin = 3
            self.ax3d.set_xlim(-margin, margin)
            self.ax3d.set_ylim(-margin, margin)
        else:
            x, y, _ = pose['torso_pos']
            margin = 3
            self.ax3d.set_xlim(x - margin, x + margin)
            self.ax3d.set_ylim(y - margin, y + margin)
        self.ax3d.set_zlim(0, 3)
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title('3D Humanoid SRBD Visualization')
    
        # Draw torso.
        self.draw_torso(pose['torso_pos'], pose['torso_euler'])
    
        # Draw feet.
        if 'left_foot_center' in pose:
            self.draw_left_foot(pose['left_foot_center'], pose['left_foot_angle'])
        if 'right_foot_center' in pose:
            self.draw_right_foot(pose['right_foot_center'], pose['right_foot_angle'])
    
        # Fixed offset for toe and heel positions.
        foot_offset = self.foot_length/2  
    
        # Draw ground reaction forces for left foot.
        if 'left_foot_center' in pose:
            left_center = np.array([pose['left_foot_center'][0], pose['left_foot_center'][1], 0])
            # Split the force equally between toe and heel (adjust as needed).
            left_toe_pos = left_center + np.array([foot_offset, 0, 0])
            left_toe_force = np.array(pose['left_grf_toe'])
            left_heel_pos = left_center - np.array([foot_offset, 0, 0])
            left_heel_force = np.array(pose['left_grf_heel'])
            self.draw_ground_reaction_force(left_toe_pos, left_toe_force, color='red')
            self.draw_ground_reaction_force(left_heel_pos, left_heel_force, color='red')
          
        # Draw ground reaction forces for right foot.
        if 'right_foot_center' in pose:
            right_center = np.array([pose['right_foot_center'][0], pose['right_foot_center'][1], 0])
            right_toe_pos = right_center + np.array([foot_offset, 0, 0])
            right_toe_force = np.array(pose['right_grf_toe'])
            right_heel_pos = right_center - np.array([foot_offset, 0, 0])
            right_heel_force = np.array(pose['right_grf_heel'])
            self.draw_ground_reaction_force(right_toe_pos, right_toe_force, color='blue')
            self.draw_ground_reaction_force(right_heel_pos, right_heel_force, color='blue')
    
        # Draw adapted footsteps if provided.
        if 'adapted_left_footstep' in pose:
            self.draw_adapted_left_footstep(pose['adapted_left_footstep'])
        if 'adapted_right_footstep' in pose:
            self.draw_adapted_right_footstep(pose['adapted_right_footstep'])

    def update_top_view(self, gait_phases, torso_pos):
        """
        Updates the 2D top view with planned & adapted footsteps and COM.
        """
        self.ax2d.clear()
        self.ax2d.set_xlim(-1, 5)
        self.ax2d.set_ylim(-1, 1)
        self.ax2d.set_xlabel("X (m)")
        self.ax2d.set_ylabel("Y (m)")
        self.ax2d.set_title("Top View: Planned & Adapted Footsteps and Torso COM")
        self.ax2d.grid(True)

        def add_foot_rect(center, foot_angle, color):
            l = self.foot_length
            w = self.foot_width
            lower_left = (center[0] - l/2.0, center[1] - w/2.0)
            angle_deg = np.degrees(foot_angle)
            rect = Rectangle(lower_left, l, w, angle=angle_deg,
                             edgecolor='black', facecolor=color, alpha=0.6)
            self.ax2d.add_patch(rect)

        for phase in gait_phases:
            if phase["support_leg"] in ["both", "left"]:
                add_foot_rect(phase["left_foot"][:2], 0, 'blue')
                if "adapted_left_footstep" in phase:
                    for step in phase["adapted_left_footstep"]:
                        add_foot_rect(step["foot_center"][:2], step.get("foot_angle", 0), 'green')
            if phase["support_leg"] in ["both", "right"]:
                add_foot_rect(phase["right_foot"][:2], 0, 'red')
                if "adapted_right_footstep" in phase:
                    for step in phase["adapted_right_footstep"]:
                        add_foot_rect(step["foot_center"][:2], step.get("foot_angle", 0), 'green')
        
        com_x, com_y, _ = torso_pos
        self.ax2d.scatter(com_x, com_y, color='magenta', marker='*', s=50, label='COM')
        self.com_history.append((com_x, com_y))
        hx, hy = zip(*self.com_history)
        self.ax2d.plot(hx, hy, color='magenta', label='COM History')
        
        handles = [
            Patch(facecolor='blue', edgecolor='black', label='Planned Left Foot'),
            Patch(facecolor='red', edgecolor='black', label='Planned Right Foot'),
            Patch(facecolor='green', edgecolor='black', label='Adapted Foot'),
            plt.Line2D([], [], marker='*', linestyle='None', color='magenta', markersize=7, label='Current COM'),
            plt.Line2D([], [], color='magenta', label='COM History')
        ]
        self.ax2d.legend(handles=handles, loc='upper right')
