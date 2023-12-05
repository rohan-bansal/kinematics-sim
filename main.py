import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from lib.models import KinematicBicycleModel, CurvilinearKinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose
from lib.controllers import PIDController, PurePursuitController

dt = 0.01   
L = 2 # 2m wheelbase

t_data = np.arange(0, 1, dt)
waypoints = [
    Pose(x=0, y=0, heading=0, velocity=20), 
    Pose(x=10, y=10, heading=0, velocity=20), 
    Pose(x=20, y=20, heading=0, velocity=20),
    Pose(x=30, y=30, heading=0, velocity=20)
]

path = CubicHermiteSpline(waypoints)
cBicycleModel = CurvilinearKinematicBicycleModel(path, L)

x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
model_x_data = []
model_y_data = []

plt.title("Kinematic Bicycle Motion Model")
plt.axis('equal')

# s, delta, vx, e_y, e_psi
state = np.array([0, 0, 5., 0, 0])

# steer
control = [0.001]

# draw predetermined path to follow on graph
for i in range(t_data.shape[0]):
    x_data[i], y_data[i] = path.getPosition(t_data[i])

try:
    while True:

        # clear plot
        plt.clf()
        plt.axis('equal')
        plt.plot(x_data, y_data, label="desired trajectory")
        plt.plot(model_x_data, model_y_data, label="model trajectory")

        # calculate A, b, d matrices with state and control
        A, b, d = cBicycleModel.linearize(state, control, dt)

        # define Q and R matrices
        Q = np.array([[0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0], 
                      [0, 0, 0, 0, 0], 
                      [0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 1]])
        R = np.array([[1]])

        # solve DARE
        x = la.solve_discrete_are(A, b, Q, R)

        # calculate K gain
        k = -(R + b.T @ x @ b)**-1 @ b.T @ x @ A

        # calculate control
        steer = (k @ state)[0]
        print(steer)
        control[0] = steer
        
        # limit control outputs
        if (control[0] > np.pi/4):
            control[0] = np.pi/4
        elif (control[0] < -np.pi/4):
            control[0] = -np.pi/4
        
        # update state with new control input
        state = cBicycleModel.propagate(state, control, dt)

        # update model position on graph
        cBicycleModel.updatePosition(state)
        model_x_data.append(cBicycleModel.x)
        model_y_data.append(cBicycleModel.y)


        plt.pause(dt)


except KeyboardInterrupt:
    pass



