import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Orientation import Quaternion

class Spacecraft:
    def __init__(self, inertia, initial_q, initial_w):
        """
        Spacecraft Class
        ----------------
        Initializes the spacecraft with its physical properties.

        Parameters
        ----------
        inertia : Array-like 
                3x3 inertia matrix.
        initial_q : Quaternion class object
                Initial orientation quaternion.
        initial_w : Array-like
                3x3 Initial angular velocity vector.
        """
        self.inertia = inertia
        self.q = initial_q
        self.w = initial_w
        

    def quaternion_kinematics(self, state, t):
        """
        Coupled differential equations for quaternion kinematics of a spacecraft.
        
        Parameters
        ----------
        state : Array-like
            Current state of the spacecraft [orientation, angular_velocity].
        t : Any
            Time variable.
        
        Return
        ------
        Derivative of the state.
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

        Parameters
        ----------
        total_time : Any
                Total time for the simulation.
        time_step : Any
                Time step for the simulation.
        
        Returns
        -------
        Time vector and state history.
        """
        time = np.arange(0, total_time, time_step)
        initial_state = np.append(self.q, self.w)
        state_history = odeint(self.quaternion_kinematics, initial_state, time)
        return time, state_history



