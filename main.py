import matplotlib.pyplot as plt
import numpy as np

from models import KinematicBicycleModel

dt = 0.01
v = np.pi
w = np.arctan(2/10)
L = 2 # 2m wheelbase

bicycleModel = KinematicBicycleModel(0, 0, L)
bicycleModel.delta = np.arctan(2/10)

t_data = np.arange(0, 20, dt)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    print(bicycleModel.get_state())
    x_data[i] = bicycleModel.get_state()[0]
    y_data[i] = bicycleModel.get_state()[1]
    bicycleModel.step(v=v, dt=dt)

plt.axis('equal')
plt.plot(x_data, y_data)

plt.show()