import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy
import time
import cvxpy as cp

from lib.models import CurvilinearKinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose
from lib.controllers import PIDController

np.random.seed(0)

################## PARAMETERS ##################
dt = 0.1   
ds = 0.01
L = 2

################## PATH DATA ##################
t_data = np.arange(0, 1, ds)
waypoints = [
    Pose(x=0, y=0, heading=0, velocity=20), 
    Pose(x=10, y=10, heading=0, velocity=20), 
    Pose(x=20, y=20, heading=0, velocity=20),
    Pose(x=30, y=30, heading=0, velocity=20)
]

################## CONTROLLERS ##################
path = CubicHermiteSpline(waypoints)
cBicycleModel = CurvilinearKinematicBicycleModel(path, L)
pidController = PIDController(0.5, 0.001, 0.01, 5)

################## PLOTTING ##################
model_x_data = []
model_y_data = []

################## PATH GENERATION ##################
trajectory_ref = np.array([path.getPosition(t) for t in t_data])

################## MPC ##################

predHorizon = 10
Q_mpc = np.diag([1, 1, 1])
R_mpc = np.diag([1])

min_delta, max_delta = -0.5, 0.5
min_acc, max_acc = -2.0, 2.0

def initMPC(state):
    initial_state = np.array(state)

    return initial_state

def mpcStep(measured_state):

    A = np.array([[1, 0, 0],
                [measured_state[2] / L * dt, 1, 0],
                [1/2 * measured_state[2] * dt, measured_state[2] * dt, 1]])
    B = np.array([[dt],
                [1/2 * dt],
                [0]])

    x_ref = np.array([measured_state[1], measured_state[-1], measured_state[-2]])

    x = cp.Variable((3, predHorizon + 1))
    u = cp.Variable((1, predHorizon))

    cost = 0.0
    constraints = []

    for t in range(predHorizon):
        cost += cp.quad_form(u[:, t], R_mpc)

        if t != 0:
            cost += cp.quad_form(x[:, t] - x_ref, Q_mpc)

        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

    constraints += [min_delta <= u, u <= max_delta]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    return u.value
    
################## MAIN LOOP ##################
    
def main():

    # s, delta, vx, e_y, e_psi
    state = np.array([0, 0, 0.01, 0, 0])

    fig, axs = plt.subplots(3, 3)

    i = 0

    try:
        while True:

            i += 1

            axs[0, 0].cla()
            axs[0, 0].set_title("vehicle trajectory")
            axs[0, 0].axis('equal')
            axs[0, 0].plot(trajectory_ref[:, 0], trajectory_ref[:, 1], label="desired trajectory")
            axs[0, 0].plot(model_x_data, model_y_data, label="model trajectory")

            x, y = cBicycleModel.get_cartesian_position(state)
            model_x_data.append(x)
            model_y_data.append(y)
            
            acc = pidController.step(state[2])
        
            control_delta = mpcStep(state)
            
            print(control_delta)

            control = [control_delta[0][0], acc]
            
            state = cBicycleModel.propagate(state, control, dt)

            plt.pause(dt)


    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()