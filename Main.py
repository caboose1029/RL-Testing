import numpy as np
from Spacecraft import Spacecraft
from Visualization import Plot
from Orientation import Quaternion

# Example usage
# Defining a simple spacecraft
inertia = np.diag([10, 15, 20])  # Simple diagonal inertia matrix
initial_q = Quaternion(np.array([1, 0, 0, 0]))  # Assuming initial orientation is zero
initial_w = np.array([0.1, 0.1, 0])  # Some initial angular velocity

# Creating the spacecraft
spacecraft = Spacecraft(inertia, initial_q, initial_w)

# Simulating the motion for 60 seconds with a time step of 0.1 seconds
time, state_history = spacecraft.simulate_motion(60, 0.1)

rpy = Plot(time, state_history)

Plot.plotRPY(rpy)

