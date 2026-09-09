import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from srbd_plotting import SRBDVisualizer
from gait_planner import GaitPlanner
import mpc
import time

def get_poses(t, gait_phases):
    phase = None
    for ph in gait_phases:
        if ph["start_time"] <= t < ph["end_time"]:
            phase = ph
            break
    if phase is None:
        phase = gait_phases[-1]

    pose = {
        'torso_pos': (t * 0.25, 0.05 * np.sin(4*t), 1. + 0.1 * np.sin(t)),
        'torso_euler': (0.1 * np.sin(t), 0.1 * np.sin(t), 0.1 * np.sin(t)),
        'left_grf': (1 * np.sin(t), 1 * abs(np.sin(t)), 30 * abs(np.sin(t))),
        'right_grf': (2 * np.sin(t), 3 * abs(np.sin(t)), 20 * abs(np.sin(t)))
    }
    if phase["support_leg"] in ["both", "left"]:
        pose["left_foot_center"] = phase["left_foot"]
        pose["left_foot_angle"] = 0
        if "adapted_left_footstep" not in phase:
            adapted_offset = np.random.rand(2) * 0.1
            adapted_left = phase["left_foot"] + adapted_offset
            phase["adapted_left_footstep"] = [{"foot_center": adapted_left.tolist(), "foot_angle": 0}]
        pose["adapted_left_footstep"] = phase["adapted_left_footstep"]

    if phase["support_leg"] in ["both", "right"]:
        pose["right_foot_center"] = phase["right_foot"]
        pose["right_foot_angle"] = 0
        if "adapted_right_footstep" not in phase:
            adapted_offset = np.random.rand(2) * 0.1
            adapted_right = phase["right_foot"] + adapted_offset
            phase["adapted_right_footstep"] = [{"foot_center": adapted_right.tolist(), "foot_angle": 0}]
        pose["adapted_right_footstep"] = phase["adapted_right_footstep"]

    # print("Time: ", t)
    # if "left_foot_center" in pose:
    #     print("Left foot (stance): ", pose["left_foot_center"])
    # else:
    #     print("Left foot: Swing")
    # if "right_foot_center" in pose:
    #     print("Right foot (stance): ", pose["right_foot_center"])
    # else:
    #     print("Right foot: Swing")
    return pose

def plot():
    # Initialize the gait planner.
    planner = GaitPlanner(0.5, 0.8, 0.3, 0.2, 10)
    gait_phases = planner.plan_gait()

    visualizer = SRBDVisualizer(is_static=False)
    
    # Create a time array from 0 to 8.5 s for animation.
    t_frames = np.linspace(0, 8.5, 200)

    def animate(t):
        pose = get_poses(t, gait_phases)
        visualizer.update_and_plot_humanoid(pose)
        return []

    ani = animation.FuncAnimation(visualizer.fig3d, animate, frames=t_frames, 
                                  interval=100, blit=False, repeat=False)
    plt.show()

def dummy_standing_quad(SRBD_mpc):
    # Dummy foot positions.
    foot_positions = np.array([0.2, 0.2, 0., 0.2, -0.2, 0., -0.2, 0.2, 0., -0.2, -0.2, 0.])*4.
    c_horizon = np.tile(foot_positions, (SRBD_mpc.HORIZON_LENGTH, 1))
    contact_horizon = np.ones((SRBD_mpc.HORIZON_LENGTH, 4))
    return c_horizon, contact_horizon

def plot_all_trajectories(com_hist, orient_hist, vel_hist, orient_deriv_hist, dt):
    n_points = com_hist.shape[0]
    time_array = np.arange(n_points) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Center of Mass Trajectory
    ax = axs[0, 0]
    ax.plot(time_array, com_hist[:, 0], label="X")
    ax.plot(time_array, com_hist[:, 1], label="Y")
    ax.plot(time_array, com_hist[:, 2], label="Z")
    ax.set_title("Center of Mass Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.grid(True)
    ax.legend()

    # Orientation Trajectory
    ax = axs[0, 1]
    ax.plot(time_array, orient_hist[:, 0], label="Roll")
    ax.plot(time_array, orient_hist[:, 1], label="Pitch")
    ax.plot(time_array, orient_hist[:, 2], label="Yaw")
    ax.set_title("SRBD Orientation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()

    # COM Velocity Trajectory
    ax = axs[1, 0]
    ax.plot(time_array, vel_hist[:, 0], label="Vx")
    ax.plot(time_array, vel_hist[:, 1], label="Vy")
    ax.plot(time_array, vel_hist[:, 2], label="Vz")
    ax.set_title("COM Velocity Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True)
    ax.legend()

    # Orientation Derivative Trajectory
    ax = axs[1, 1]
    ax.plot(time_array, orient_deriv_hist[:, 0], label="Roll Rate")
    ax.plot(time_array, orient_deriv_hist[:, 1], label="Pitch Rate")
    ax.plot(time_array, orient_deriv_hist[:, 2], label="Yaw Rate")
    ax.set_title("Orientation Derivative Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (rad/s)")
    ax.grid(True)
    ax.legend()

    plt.show()
    
def plot_ground_reaction_forces(grf_hist, dt):
    grf_hist = np.array(grf_hist)
    time_array = np.arange(grf_hist.shape[0]) * dt
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Left Heel (Back) GRF: columns 0-2
    axs[0, 0].plot(time_array, grf_hist[:, 0], label='Fx Heel (Left)')
    axs[0, 0].plot(time_array, grf_hist[:, 1], label='Fy Heel (Left)')
    axs[0, 0].plot(time_array, grf_hist[:, 2], label='Fz Heel (Left)')
    axs[0, 0].set_title('Left Heel Ground Reaction Forces')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Force (N)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Left Toe (Front) GRF: columns 3-5
    axs[0, 1].plot(time_array, grf_hist[:, 3], label='Fx Toe (Left)')
    axs[0, 1].plot(time_array, grf_hist[:, 4], label='Fy Toe (Left)')
    axs[0, 1].plot(time_array, grf_hist[:, 5], label='Fz Toe (Left)')
    axs[0, 1].set_title('Left Toe Ground Reaction Forces')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Force (N)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Right Heel (Back) GRF: columns 6-8
    axs[1, 0].plot(time_array, grf_hist[:, 6], label='Fx Heel (Right)')
    axs[1, 0].plot(time_array, grf_hist[:, 7], label='Fy Heel (Right)')
    axs[1, 0].plot(time_array, grf_hist[:, 8], label='Fz Heel (Right)')
    axs[1, 0].set_title('Right Heel Ground Reaction Forces')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Force (N)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Right Toe (Front) GRF: columns 9-11
    axs[1, 1].plot(time_array, grf_hist[:, 9], label='Fx Toe (Right)')
    axs[1, 1].plot(time_array, grf_hist[:, 10], label='Fy Toe (Right)')
    axs[1, 1].plot(time_array, grf_hist[:, 11], label='Fz Toe (Right)')
    axs[1, 1].set_title('Right Toe Ground Reaction Forces')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Force (N)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def testing_double_support(SRBD_mpc, current_time, total_duration):
    for i in range(SRBD_mpc.HORIZON_LENGTH):
              t_ref = current_time + i * SRBD_mpc.dt
              # Use the simulation total_duration (2 seconds) to divide into four segments.
              if t_ref < total_duration / 4:
                # First quarter: shift COM to the right (positive Y offset)
                ref_com = np.array([0.0, 0.1, 1.0])
              elif t_ref < total_duration / 2:
                # Second quarter: return COM to the center
                ref_com = np.array([0.0, 0.0, 1.0])
              elif t_ref < 3 * total_duration / 4:
                # Third quarter: move COM a bit down (reduce Z)
                ref_com = np.array([0.0, 0.0, 0.7])
              else:
                # Last quarter: move COM a bit forward (positive X offset)
                ref_com = np.array([0.01, 0.0, 1.0])
              
              SRBD_mpc.x_ref_hor[i, 3:6] = ref_com
    
    return SRBD_mpc.x_ref_hor

def main():
    planner = GaitPlanner(1000., 0.25, 0.1, 0.2, 100)
    gait_phases = planner.plan_gait()
    SRBD_mpc = mpc.MPC()
    SRBD_mpc.init_matrices()

    # Setup reference horizon.
    SRBD_mpc.x_ref_hor = np.zeros((SRBD_mpc.HORIZON_LENGTH, SRBD_mpc.NUM_STATES))
    SRBD_mpc.x_ref_hor[:, -1] = SRBD_mpc.g
    SRBD_mpc.x_ref_hor[:, 5] = 1.     # constant 1 m z position

    current_time = 0.0
    total_duration = 4.
    dt = SRBD_mpc.dt

    com_hist = []
    orient_hist = []
    vel_hist = []
    orient_deriv_hist = []
    grf_hist = []

    # Run the MPC loop.
    while current_time < total_duration:
        loop_start = time.time()
        print("Index:", int(current_time / dt))
        if current_time == 0.0:
            SRBD_mpc.x0 = SRBD_mpc.x_ref_hor[0].copy()
        else:
            SRBD_mpc.x0 = SRBD_mpc.x_opt[2].copy()
            
            testing_double_support(SRBD_mpc, current_time, total_duration)

            # #Simple moving forward reference
            # for i in range(SRBD_mpc.HORIZON_LENGTH):
            #     t_ref = current_time + i * dt
            #     phase = None
            #     for ph in gait_phases:
            #         if ph["start_time"] <= t_ref < ph["end_time"]:
            #             phase = ph
            #             break
            #     if phase is None:
            #         phase = gait_phases[-1]

            #     # Choose the reference foot position based on the support leg.
            #     if phase["support_leg"] == "left":
            #         foot = np.array(phase["left_foot"])
            #     elif phase["support_leg"] == "right":
            #         foot = np.array(phase["right_foot"])
            #     else:  # "both": average the positions
            #         foot = (np.array(phase["left_foot"]) + np.array(phase["right_foot"])) / 2

                # # Set the COM reference near the selected foot.
                # SRBD_mpc.x_ref_hor[i, 3] = foot[0] - 0.01
                # SRBD_mpc.x_ref_hor[i, 5] = 1.0  # Maintain constant z position

                # print("COM ref for horizon {}: {}".format(i, SRBD_mpc.x_ref_hor[i, 3:6]))
            
            

        c_horizon = []
        contact_horizon = []
        foot_offset = 0.08  # offset for front toe and back heel positions
        for i in range(SRBD_mpc.HORIZON_LENGTH):
            t_i = current_time + i * dt
            phase = None
            for ph in gait_phases:
                if ph["start_time"] <= t_i < ph["end_time"]:
                    phase = ph
                    break
            if phase is None:
                phase = gait_phases[-1]

            left = np.hstack((np.array(phase["left_foot"]), 0))
            right = np.hstack((np.array(phase["right_foot"]), 0))
            # Compute GRF positions for back heel and front toe of each foot.
            left_heel = left - np.array([foot_offset, 0, 0])
            left_toe = left + np.array([foot_offset, 0, 0])
            right_heel = right - np.array([foot_offset, 0, 0])
            right_toe = right + np.array([foot_offset, 0, 0])
            # Store GRF positions for both feet (back heel and front toe)
            foot_vec = np.concatenate((left_heel, left_toe, right_heel, right_toe))
            c_horizon.append(foot_vec)
            # Define contact: 1 if foot is in stance, 0 if in swing. Duplicate contact for front toe and back heel.
            left_contact = 1 if phase["support_leg"] in ["both", "left"] else 0
            right_contact = 1 if phase["support_leg"] in ["both", "right"] else 0
            contact_horizon.append([left_contact, left_contact, right_contact, right_contact])
        

        p_com_horizon = SRBD_mpc.x_ref_hor[:, 3:6].copy()
        # Perform MPC calculations.
        SRBD_mpc.extract_psi()
        SRBD_mpc.rotation_matrix_T()
        SRBD_mpc.set_Q()
        SRBD_mpc.set_R()
        SRBD_mpc.calculate_A_continuous()
        SRBD_mpc.calculate_A_discrete()
        SRBD_mpc.calculate_B_continuous(c_horizon, p_com_horizon)
        SRBD_mpc.calculate_B_discrete()
        SRBD_mpc.calculate_Aqp()
        SRBD_mpc.calculate_Bqp()
        SRBD_mpc.calculate_Ac()
        SRBD_mpc.calculate_bounds(contact_horizon)
        SRBD_mpc.calculate_hessian()
        SRBD_mpc.calculate_gradient()
        SRBD_mpc.solve_qp()

        if current_time == 0.0:
            SRBD_mpc.x_opt[0, :] = np.squeeze(SRBD_mpc.x0.copy())
        else:
            SRBD_mpc.x_opt[0, :] = np.squeeze(SRBD_mpc.x_opt[1, :].copy())
        SRBD_mpc.compute_rollout()

        # Record MPC states.
        com = SRBD_mpc.x_opt[0, 3:6].copy()
        com_hist.append(com)
        orientation = SRBD_mpc.x_opt[0, 0:3].copy()
        orient_hist.append(orientation)
        velocity = SRBD_mpc.x_opt[0, 9:12].copy()
        vel_hist.append(velocity)
        orient_deriv = SRBD_mpc.x_opt[0, 6:9].copy()
        orient_deriv_hist.append(orient_deriv)
        grf_hist.append(SRBD_mpc.u_opt[0].copy())

        print("COM:", com)
        loop_end = time.time()
        print("Loop iteration completed in: {:.6f} seconds".format(loop_end - loop_start))
        current_time += dt

    com_hist = np.array(com_hist)
    orient_hist = np.array(orient_hist)
    vel_hist = np.array(vel_hist)
    orient_deriv_hist = np.array(orient_deriv_hist)
    grf_hist = np.array(grf_hist)

    # Now animate the MPC solution.
    visualizer = SRBDVisualizer(is_static=True)
    # Create a time array matching the MPC simulation duration.
    t_frames = np.linspace(0, total_duration, len(com_hist))
    SLOWDOWN_FACTOR = 1.0

    def animate(t):
        # Determine the closest index for the simulation history.
        index = int(round(t / dt))
        if index >= len(com_hist):
            index = len(com_hist) - 1

        # Build a pose based on the MPC state.
        pose = {}
        pose['torso_pos'] = (com_hist[index][0], com_hist[index][1], com_hist[index][2])
        pose['torso_euler'] = (orient_hist[index][0], orient_hist[index][1], orient_hist[index][2])
        
        # Extract heel and toe GRF for each foot.
        pose['left_grf_heel'] = (grf_hist[index][0], grf_hist[index][1], grf_hist[index][2])
        pose['left_grf_toe']  = (grf_hist[index][3], grf_hist[index][4], grf_hist[index][5])
        pose['right_grf_heel'] = (grf_hist[index][6], grf_hist[index][7], grf_hist[index][8])
        pose['right_grf_toe']  = (grf_hist[index][9], grf_hist[index][10], grf_hist[index][11])

        # Use the gait phases for the feet positions. PROBLEM I SHOULD NOT USE IT LIKE THAT
        feet_pose = get_poses(t, gait_phases)
        if "left_foot_center" in feet_pose:
            pose["left_foot_center"] = feet_pose["left_foot_center"]
            pose["left_foot_angle"] = feet_pose["left_foot_angle"]
        if "right_foot_center" in feet_pose:
            pose["right_foot_center"] = feet_pose["right_foot_center"]
            pose["right_foot_angle"] = feet_pose["right_foot_angle"]
        visualizer.update_and_plot_humanoid(pose)
        return []

    ani = animation.FuncAnimation(visualizer.fig3d, animate, frames=t_frames, 
                                  interval=100*SLOWDOWN_FACTOR, blit=False, repeat=False)
    plt.show()

    plot_all_trajectories(com_hist, orient_hist, vel_hist, orient_deriv_hist, dt)
    plot_ground_reaction_forces(grf_hist, dt)
if __name__ == '__main__':
    main()
