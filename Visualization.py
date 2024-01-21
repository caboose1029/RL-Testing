import matplotlib.pyplot as plt
from Orientation import Quaternion
import numpy as np

class Plot:
    def __init__(self, time, state_history):
        self.time = time
        self.state_history = state_history

    def plotRPY(self):
    # Plotting the orientation over time
        euler_angles = np.array([Quaternion.to_euler(Quaternion(state[:4])) for state in self.state_history])
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, euler_angles)
        plt.title('Spacecraft Orientation Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Euler Angles (rad)')
        plt.legend(['Phi', 'Theta', 'Psy'])
        plt.grid(True)
        plt.show()