import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Orientation import Quaternion

class Spacecraft:
    def __init__(self, inertia, initial_q, initial_w):
        """
        Initializes the spacecraft with its physical properties.
        inertia: 3x3 inertia matrix.
        initial_orientation: Initial orientation quaternion.
        initial_angular_velocity: Initial angular velocity vector.
        """
        self.inertia = inertia
        self.q = initial_q
        self.w = initial_w
        

    def rotational_kinematics(self, state, t):
        """
        Computes the spacecraft's rotational kinematics for torque-free motion.
        state: Current state of the spacecraft [orientation, angular_velocity].
        t: Time variable.
        return: Derivative of the state.
        """
        # Unpack the state
        q, w = Quaternion(state[:4]), state[4:7]

        # Compute the derivative of orientation using quaternions
        q_dot = Quaternion.ode(q, w)

        # Torque-free motion: angular momentum is conserved
        # ω_dot = I^-1 * (I * ω x ω)
        inertia_T = np.linalg.inv(self.inertia)
        w_dot = inertia_T @ (np.cross(self.inertia @ w, w))

        # Combine the derivatives
        state_dot = np.append(q_dot, w_dot)
        return state_dot

    def simulate_motion(self, total_time, time_step):
        """
        Simulates the spacecraft motion over a given time period.
        total_time: Total time for the simulation.
        time_step: Time step for the simulation.
        return: Time vector and state history.
        """
        time = np.arange(0, total_time, time_step)
        initial_state = np.append(self.q, self.w)
        state_history = odeint(self.rotational_kinematics, initial_state, time)
        return time, state_history



