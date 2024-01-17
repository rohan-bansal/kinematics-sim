import matplotlib.pyplot as plt
import numpy as np

from lib.models import KinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose
from lib.controllers import PIDController, PurePursuitController

dt = 0.01   
L = 2 # 2m wheelbase

bicycleModel = KinematicBicycleModel(0, 0, L)
# bicycleModel.v = 5

t_data = np.arange(0, 1, dt)
waypoints = [
    Pose(x=0, y=0, heading=np.pi/2, velocity=20), 
    Pose(x=10, y=10, heading=0, velocity=20), 
    Pose(x=20, y=20, heading=0, velocity=20),
    Pose(x=30, y=30, heading=0, velocity=20)
]
path = CubicHermiteSpline(waypoints)
controller = PurePursuitController(bicycleModel, path)
PID = PIDController(1, 0.0, 0.0, 5)

x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

model_x_data = []
model_y_data = []

plt.title("Kinematic Bicycle Motion Model")
plt.axis('equal')

for i in range(t_data.shape[0]):

    x_data[i], y_data[i] = path.getPosition(t_data[i])

try:
    while True:

        # clear plot and draw new
        plt.clf()
        plt.axis('equal')
        plt.plot(x_data, y_data, label="desired trajectory")
        plt.plot(model_x_data, model_y_data, label="model trajectory")

        model_x_data.append(bicycleModel.get_state()[0])
        model_y_data.append(bicycleModel.get_state()[1])

        steer_angle = controller.step(bicycleModel.x, bicycleModel.y, bicycleModel.v, plt)
        bicycleModel.delta = steer_angle

        PIDoutput = PID.step(bicycleModel.get_state()[3], dt)
        bicycleModel.v = PIDoutput

        bicycleModel.step(dt=dt)

        plt.pause(0.01)
except KeyboardInterrupt:
    pass
