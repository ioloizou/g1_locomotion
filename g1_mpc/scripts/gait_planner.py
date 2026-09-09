import numpy as np
import matplotlib.pyplot as plt

class GaitPlanner:
    def __init__(self, double_support_time, single_support_time, step_length, step_width, num_steps):
        """
        Parameters:
          double_support_time: Duration (seconds) of the initial double-support phase.
          single_support_time: Duration (seconds) of each single-support phase.
          step_length: Forward distance (meters) each swing foot travels.
          step_width: Lateral distance (meters) between the feet.
          num_steps: Total number of single-support steps to plan.
        """
        self.double_support_time = double_support_time
        self.single_support_time = single_support_time
        self.step_length = step_length
        self.step_width = step_width
        self.num_steps = num_steps

        # Initialize foot positions.
        self.left_foot_pos = np.array([0.0, -step_width])
        self.right_foot_pos = np.array([0.0, step_width])

        self.current_time = 0.0
        self.phases = []  # List to store the planned phases

    def plan_gait(self):
        self.phases = []
        # First add a double support phase.
        self.phases.append({
            "start_time": self.current_time,
            "end_time": self.current_time + self.double_support_time,
            "left_foot": self.left_foot_pos.copy(),
            "right_foot": self.right_foot_pos.copy(),
            "support_leg": "both"
        })
        self.current_time += self.double_support_time

        # Now add single support phases, alternating support legs.
        for i in range(self.num_steps):
            support_leg = "left" if i % 2 == 0 else "right"
            # Append the phase using the current foot positions.
            self.phases.append({
                "start_time": self.current_time,
                "end_time": self.current_time + self.single_support_time,
                "left_foot": self.left_foot_pos.copy(),
                "right_foot": self.right_foot_pos.copy(),
                "support_leg": support_leg
            })
            self.current_time += self.single_support_time

            # After the first step the step increment is doubled.
            step_increment = self.step_length if i == 0 else 2 * self.step_length

            # Update the swing foot position based on the step_increment.
            if support_leg == "left":
                # Left is support; update right foot.
                self.right_foot_pos += np.array([step_increment, 0.0])
            elif support_leg == "right":
                # Right is support; update left foot.
                self.left_foot_pos += np.array([step_increment, 0.0])
        return self.phases

    def plot_gait(self):
        import matplotlib.patches as mpatches
        fig, ax = plt.subplots(figsize=(15, 2))
        
        # Define y positions and bar height for the two foot rows.
        left_y = 10
        right_y = 5
        height = 2
        
        # Determine overall time span.
        start_time = min(phase["start_time"] for phase in self.phases)
        end_time = max(phase["end_time"] for phase in self.phases)
        
        # Plot each phase as a horizontal bar on both rows.
        for phase in self.phases:
            t0 = phase["start_time"]
            duration = phase["end_time"] - phase["start_time"]
            support_leg = phase["support_leg"]
            
            if support_leg == "left":
                left_color = "grey"   # left foot in contact
                right_color = "white" # right foot swinging
            elif support_leg == "right":
                left_color = "white"  # left foot swinging
                right_color = "grey"  # right foot in contact
            elif support_leg == "both":
                left_color = right_color = "grey"
            else:
                left_color = right_color = "white"
            
            ax.broken_barh([(t0, duration)], (left_y, height), facecolors=left_color, edgecolors="black")
            ax.broken_barh([(t0, duration)], (right_y, height), facecolors=right_color, edgecolors="black")

        ax.set_xlabel("Time (s)")
        ax.set_xlim(start_time, end_time)
        ax.set_yticks([left_y + height/2, right_y + height/2])
        ax.set_yticklabels(["Left Foot", "Right Foot"])
        ax.set_title("Gait Phases")
        ax.grid(True)
        
        grey_patch = mpatches.Patch(color="grey", label="Support")
        white_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Swing")
        ax.legend(handles=[grey_patch, white_patch])
        
        plt.show()

if __name__ == "__main__":
    # Parameters for the gait
    double_support_time = 0.2   # seconds
    single_support_time = 0.8   # seconds per step
    step_length = 0.3           # meters forward per step
    step_width = 0.2            # lateral separation between the feet
    num_steps = 10              # number of single support steps to plan

    planner = GaitPlanner(double_support_time, single_support_time, step_length, step_width, num_steps)
    planner.plan_gait()
    planner.plot_gait()
