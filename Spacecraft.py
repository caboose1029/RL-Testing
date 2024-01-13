import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Spacecraft:
    def __init__(self, inertia, initial_orientation, initial_angular_velocity):
        """
        Initializes the spacecraft with its physical properties.
        inertia: 3x3 inertia matrix.
        initial_orientation: Initial orientation (roll, pitch, yaw) or quaternion.
        initial_angular_velocity: Initial angular velocity vector.
        """
        self.inertia = inertia
        self.orientation = initial_orientation
        self.angular_velocity = initial_angular_velocity

    def rotational_kinematics(self, state, t):
        """
        Computes the spacecraft's rotational kinematics for torque-free motion.
        state: Current state of the spacecraft [orientation, angular_velocity].
        t: Time variable.
        return: Derivative of the state.
        """
        # Unpack the state
        orientation, angular_velocity = state[:3], state[3:]

        # Compute the derivative of orientation, for simplicity using angular velocity directly
        # In more complex simulations, quaternion or rotation matrix approaches are used
        orientation_dot = angular_velocity

        # Torque-free motion: angular momentum is conserved
        # ω_dot = I^-1 * (I * ω x ω)
        inertia_inv = np.linalg.inv(self.inertia)
        angular_velocity_dot = inertia_inv @ (np.cross(self.inertia @ angular_velocity, angular_velocity))

        # Combine the derivatives
        state_dot = np.concatenate([orientation_dot, angular_velocity_dot])
        return state_dot

    def simulate_motion(self, total_time, time_step):
        """
        Simulates the spacecraft motion over a given time period.
        total_time: Total time for the simulation.
        time_step: Time step for the simulation.
        return: Time vector and state history.
        """
        time = np.arange(0, total_time, time_step)
        initial_state = np.concatenate([self.orientation, self.angular_velocity])
        state_history = odeint(self.rotational_kinematics, initial_state, time)
        return time, state_history

# Example usage
# Defining a simple spacecraft
inertia = np.diag([10, 15, 20])  # Simple diagonal inertia matrix
initial_orientation = np.array([0, 0, 0])  # Assuming initial orientation is zero
initial_angular_velocity = np.array([0.1, 0.1, 0])  # Some initial angular velocity

# Creating the spacecraft
spacecraft = Spacecraft(inertia, initial_orientation, initial_angular_velocity)

# Simulating the motion for 60 seconds with a time step of 0.1 seconds
time, state_history = spacecraft.simulate_motion(600, 0.1)

# Plotting the orientation over time
plt.figure(figsize=(12, 6))
plt.plot(time, state_history[:, :3])
plt.title('Spacecraft Orientation Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Orientation (rad)')
plt.legend(['Roll', 'Pitch', 'Yaw'])
plt.grid(True)
plt.show()
