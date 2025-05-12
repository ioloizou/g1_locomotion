import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class SwingTrajectory:
    """
    Class to calculate a swing trajectory on z using a sixth-order polynomial.
    Is defined by desired points at the start, middle, and end of the trajectory.
    With zeros velocity and acceleration at the start and end points.
    
    For x, y position, a sin based speed up is used to cover 80% of the distance in the first half of the cycle.
    The remaining 20% is covered linearly in the second half of the cycle. 
    """
    def __init__(self):
        # Initial positions
        self.p_z_start = 0.0
        self.p_z_middle = 0.0
        self.p_z_final = 0.0

        self.p_x_start = 0.0
        self.p_x_final = 0.0
        self.p_y_start = 0.0
        self.p_y_final = 0.0
        # Coefficients for the polynomial
        self.coeff = np.zeros(7)

    def set_positions_xy(self, p_x_start, p_x_final, p_y_start, p_y_final):
        self.p_x_start = p_x_start
        self.p_x_final = p_x_final
        self.p_y_start = p_y_start
        self.p_y_final = p_y_final

    def set_positions_z(self, p_z_start, p_z_middle, p_z_final):
        self.p_z_start = p_z_start
        self.p_z_final = p_z_final
        self.p_z_middle = p_z_middle

    def calculate_coeff(self):
        # Coefficients for sixth-order polynomial
        A = np.array([[1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0],
                    [1, 0.5, 0.5**2, 0.5**3, 0.5**4, 0.5**5, 0.5**6],
                    [1 ,1 ,1, 1, 1, 1, 1],
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 0, 2, 6, 12, 20, 30]])

        # Constants
        # Added a bit of downward final velocity on z to make foot landing more vertical and has a bit of impact with the ground
        b = np.array([self.p_z_start, 0., 0., self.p_z_middle, self.p_z_final, -0.2, 0.])

        self.coeff = np.linalg.solve(A, b)
    
    def calculate_position_xy(self, cycle_progress):
        # Calculate x and y position with a sin based speed up which
        # allows to cover some % of the distance in the first half of the cycle

        percent_in_first_half = 0.85
        if cycle_progress <= 0.5:
            phase = percent_in_first_half * np.sin(np.pi * cycle_progress)
        else:
            # In the second half of the cycle, we want to cover the remaining linearly   
            phase = percent_in_first_half + (cycle_progress - 0.5) * (1-percent_in_first_half) * 2
        
        p_x_curr = (1 - phase) * self.p_x_start + phase * self.p_x_final
        p_y_curr = (1 - phase) * self.p_y_start + phase * self.p_y_final
        return p_x_curr, p_y_curr

    def calculate_trajectory_xy(self):
        position_xy_trajectory = []
        for t in np.linspace(0, 1, 100):
            p_x, p_y = self.calculate_position_xy(t)
            position_xy_trajectory.append((p_x, p_y))
        return position_xy_trajectory

    def calculate_position_z(self, t):
        t_vec = np.array([1, t, t**2, t**3, t**4, t**5, t**6])
        position = np.dot(self.coeff, t_vec)
        return position
    
    def calculate_velocity_z(self, t):
        t_vec = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5])
        velocity = np.dot(self.coeff, t_vec)
        return velocity

    def calculate_acceleration_z(self, t):
        t_vec = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4])
        acceleration = np.dot(self.coeff, t_vec)
        return acceleration

    def calculate_all_trajectories_z(self):
        # Calculate position, velocity, and acceleration
        position_z_trajectory = []
        velocity_z_trajectory = []
        acceleration_z_trajectory = []
        for t in np.linspace(0, 1, 100):
            position_z = self.calculate_position_z(t)
            velocity_z = self.calculate_velocity_z(t)
            acceleration_z = self.calculate_acceleration_z(t)
            position_z_trajectory.append(position_z)
            velocity_z_trajectory.append(velocity_z)
            acceleration_z_trajectory.append(acceleration_z)
        
        return position_z_trajectory, velocity_z_trajectory, acceleration_z_trajectory
    
    def reset(self):
        self.p_z_start = 0.0
        self.p_z_middle = 0.0
        self.p_z_final = 0.0
        self.p_x_start = 0.0
        self.p_x_final = 0.0
        self.p_y_start = 0.0
        self.p_y_final = 0.0
        self.coeff = np.zeros(7)
    
    def plot_trajectory(self):

        # compute trajectories
        position_z_trajectory, velocity_z_trajectory, acceleration_z_trajectory = self.calculate_all_trajectories_z()
        position_xy_trajectory = self.calculate_trajectory_xy()
        t_vals = np.linspace(0, 1, len(position_z_trajectory))

        # create a figure with one 3D subplot and two 2D subplots
        fig = plt.figure(figsize=(8, 12))

        # 3D position plot
        ax1 = fig.add_subplot(3, 1, 1, projection='3d')
        xs, ys = zip(*position_xy_trajectory)
        ax1.plot(xs, ys, position_z_trajectory, label='swing path')
        ax1.set_title('3D Swing Position Trajectory')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.grid(True)

        # Z-velocity vs time
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t_vals, velocity_z_trajectory, color='orange')
        ax2.set_title('Z Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True)

        # Z-acceleration vs time
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t_vals, acceleration_z_trajectory, color='green')
        ax3.set_title('Z Acceleration')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

trajectory = SwingTrajectory()
trajectory.set_positions_xy(0.0, 0.1, 0.0, 0.1)  # Start and end positions in x and y
trajectory.set_positions_z(0.0, 0.1, 0.0)  # Start, middle, and end positions in z
trajectory.calculate_coeff() # Calculate coefficients for z trajectory
position_z_at_t0 = trajectory.calculate_position_z(0.5)
print(position_z_at_t0)
position_x_at_t0, position_y_at_t0 = trajectory.calculate_position_xy(0.5)
print(position_x_at_t0, position_y_at_t0)


