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
cBicycleModel.vx = 5.

controller = PurePursuitController(cBicycleModel, path)

x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

model_x_data = []
model_y_data = []

plt.title("Kinematic Bicycle Motion Model")
plt.axis('equal')

steer = 0.0001

for i in range(t_data.shape[0]):

    x_data[i], y_data[i] = path.getPosition(t_data[i])

try:
    while True:

            # clear plot and draw new
        plt.clf()
        plt.axis('equal')
        plt.plot(x_data, y_data, label="desired trajectory")
        plt.plot(model_x_data, model_y_data, label="model trajectory")

        model_x_data.append(cBicycleModel.x)
        model_y_data.append(cBicycleModel.y)

        cBicycleModel.updateState()


        # steer = controller.step(plt)
        state = cBicycleModel.state.copy()
        print(state)

        # new_state = [state[0], 0, state[2], 0, 0]
        A, b, d = cBicycleModel.linearize(state, np.array([steer]), dt)
        print(A, b, d)


        # x = A @ state + b @ [steer] + d

        # print(A, b, d)

        Q = np.array([[1, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0], 
                      [0, 0, 1, 0, 0], 
                      [0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 1]])
        R = np.array([[100]])

        x = la.solve_discrete_are(A, b, Q, R)
        k = -(R + b.T @ x @ b)**-1 @ b.T @ x @ A

        steer = k @ cBicycleModel.state
        

        if (steer[0] > np.pi/4):
            steer[0] = np.pi/4
        elif (steer[0] < -np.pi/4):
            steer[0] = -np.pi/4
        

        print(steer)

        cBicycleModel.step(delta=steer[0], dt=dt)


        plt.pause(0.01)


except KeyboardInterrupt:
    pass



