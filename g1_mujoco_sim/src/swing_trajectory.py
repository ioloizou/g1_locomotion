import numpy as np
import matplotlib.pyplot as plt

class SwingTrajectory:
    def __init__(self, p_start, p_final, p_middle):
        self.p_start = p_start
        self.p_final = p_final
        self.p_middle = p_middle

        # Coefficients for the polynomial
        self.calculate_coeff()

    def calculate_coeff(self):
        # Coefficients for sixth-order polynomial
        A = np.array([[1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0],
                    [1, 0.5, 0.5**2, 0.5**3, 0.5**4, 0.5**5, 0.5**6],
                    [1 ,1 ,1, 1, 1, 1, 1],
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 0, 2, 6, 12, 20, 30]])
        print(A.shape)

        # Constants
        b = np.array([self.p_start, 0, 0, self.p_middle, self.p_final, 0, 0])
        print(b.shape)

        self.coeff = np.linalg.solve(A, b)
        print(self.coeff)

    def calculate_position(self, t):
        t_vec = np.array([1, t, t**2, t**3, t**4, t**5, t**6])
        position = np.dot(self.coeff, t_vec)
        return position
    
    def calculate_velocity(self, t):
        t_vec = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5])
        velocity = np.dot(self.coeff, t_vec)
        return velocity

    def calculate_acceleration(self, t):
        t_vec = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4])
        acceleration = np.dot(self.coeff, t_vec)
        return acceleration
    
    def calculate_all_trajectories(self):
        # Calculate position, velocity, and acceleration
        position_trajectory = []
        velocity_trajectory = []
        acceleration_trajectory = []
        for t in np.linspace(0, 1, 100):
            position = self.calculate_position(t)
            velocity = self.calculate_velocity(t)
            acceleration = self.calculate_acceleration(t)
            position_trajectory.append(position)
            velocity_trajectory.append(velocity)
            acceleration_trajectory.append(acceleration)
        
        return position_trajectory, velocity_trajectory, acceleration_trajectory

    def plot_trajectory(self):        

        position_trajectory, velocity_trajectory, acceleration_trajectory = self.calculate_all_trajectories()
        
        # create three subplots in a single column
        t_vals = np.linspace(0, 1, 100)
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(t_vals, position_trajectory)
        axs[0].set_ylabel('Position (m)')
        axs[0].set_title('Position')
        axs[0].grid(True)

        axs[1].plot(t_vals, velocity_trajectory)
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].set_title('Velocity')
        axs[1].grid(True)

        axs[2].plot(t_vals, acceleration_trajectory)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Acceleration (m/s²)')
        axs[2].set_title('Acceleration')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

trajectory = SwingTrajectory(0, 0, 0.05)    
trajectory.plot_trajectory()

