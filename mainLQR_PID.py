import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy
import time

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
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
model_x_data = []
model_y_data = []

################## PARTICLE FILTER ##################
N = 10000

def generate_uniform_particles():
    # Kp, Ki, Kd, velocity setpoint, R, Q1, Q2, Q3
    particles = np.empty((N, 8))
    particles[:, 0] = np.random.uniform(0, 1, size=N) # Kp
    particles[:, 1] = np.random.uniform(0, 1, size=N) # Ki
    particles[:, 2] = np.random.uniform(0, 1, size=N) # Kd
    particles[:, 3] = np.random.uniform(0, 15, size=N) # velocity setpoint
    particles[:, 4] = np.random.uniform(0, 1, size=N) # R
    particles[:, 5] = np.random.uniform(0, 1, size=N) # Q1
    particles[:, 6] = np.random.uniform(0, 1, size=N) # Q2
    particles[:, 7] = np.random.uniform(0, 1, size=N) # Q3

    # particles = np.empty((N, 4))
    # particles[:, 0] = np.random.uniform(0, 1, size=N) # R
    # particles[:, 1] = np.random.uniform(0, 1, size=N) # Q1
    # particles[:, 2] = np.random.uniform(0, 1, size=N) # Q2
    # particles[:, 3] = np.random.uniform(0, 1, size=N) # Q3

    return particles

def pfStep(old_state, new_state, particles, weights):

    predictedStates = np.empty((N, 8))
    predictedStates = np.apply_along_axis(simulateNextStep, 1, particles, old_state)

    # update weights
    posterior = scipy.stats.multivariate_normal(new_state, 0.005).pdf(predictedStates)
    weights = weights * posterior

    # normalize weights
    weights += 1.e-300
    weights /= sum(weights)

    # # resample
    # if 1. / sum(weights**2) < N:
    #     indexes = systematic_resample(weights)
    #     particles[:] = particles[indexes]
    #     weights[:] = weights[indexes]
    #     weights.fill(1.0 / N)

    # estimate mean and variance
    mean = np.average(particles, weights=weights, axis=0)
    var = np.average((particles - mean)**2, weights=weights, axis=0)

    return mean, var, weights

def simulateNextStep(particle, cur_state):

    t1 = time.time()

    acc = pidController.simStep(particle[0], particle[1], particle[2], particle[3], cur_state[2])
    # acc = pidController.simStep(0.5, 0.001, 0.01, 5, cur_state[2])

    t2 = time.time()

    A0 = np.array([[1, 0, 0],
        [cur_state[2] / L * dt, 1, 0],
        [1/2 * cur_state[2] * dt, cur_state[2] * dt, 1]])
    B0 = np.array([[dt],
                [1/2 * dt],
                [0]])
    Q0 = np.zeros((3, 3))
    R0 = np.zeros((1, 1))
    
    np.fill_diagonal(Q0, [particle[1], particle[2], particle[3]])
    np.fill_diagonal(R0, [particle[0]])

    t3 = time.time()

    P0 = la.solve_discrete_are(A0, B0, Q0, R0)
    K0 = -np.linalg.inv(R0 + B0.T @ P0 @ B0) @ B0.T @ P0 @ A0

    t4 = time.time()

    x0 = np.array([cur_state[1], cur_state[-1], cur_state[-2]]).reshape((-1, 1))
    delta_dot = K0 @ x0

    t5 = time.time()

    control = np.array([delta_dot[0, 0], acc])

    pred_state = cBicycleModel.propagate(cur_state, control, dt)

    t6 = time.time()

    # print(t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, "total: ", t6 - t1, " seconds")

    return pred_state


################## PATH GENERATION ##################
for i in range(t_data.shape[0]):
    x_data[i], y_data[i] = path.getPosition(t_data[i])
    
################## MAIN LOOP ##################
    
def main():

    particles = generate_uniform_particles()
    weights = np.ones(N) / N

    # s, delta, vx, e_y, e_psi
    state = np.array([0, 0, 0.01, 0, 0])
    control = np.array([0.0, 0])

    fig, axs = plt.subplots(3, 3)

    i = 0

    try:
        while True:

            i += 1

            axs[0, 0].cla()
            axs[0, 0].set_title("vehicle trajectory")
            axs[0, 0].axis('equal')
            axs[0, 0].plot(x_data, y_data, label="desired trajectory")
            axs[0, 0].plot(model_x_data, model_y_data, label="model trajectory")

            x, y = cBicycleModel.get_cartesian_position(state)
            model_x_data.append(x)
            model_y_data.append(y)
            
            t1 = time.time()

            acc = pidController.step(state[2])
            
            t2 = time.time()

            A0 = np.array([[1, 0, 0],
                [state[2] / L * dt, 1, 0],
                [1/2 * state[2] * dt, state[2] * dt, 1]])
            B0 = np.array([[dt],
                        [1/2 * dt],
                        [0]])
            
            # A0, B0, d = cBicycleModel.linearize(state, control, dt)

            Q0 = np.eye(3)
            R0 = np.eye(1)
            P0 = la.solve_discrete_are(A0, B0, Q0, R0)
            K0 = -np.linalg.inv(R0 + B0.T @ P0 @ B0) @ B0.T @ P0 @ A0

            x0 = np.array([state[1], state[-1], state[-2]]).reshape((-1, 1))
            delta_dot = K0 @ x0

            t3 = time.time()

            control = np.array([delta_dot[0, 0], acc])
            
            print(control)
            old_state = state.copy()
            state = cBicycleModel.propagate(state, control, dt)
            new_state = state.copy()

            t4 = time.time()

            # mean, var, weights = pfStep(old_state, new_state, particles, weights)

            t5 = time.time()

            # print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)

            # print("STATE: ", state)

            # axs[0, 1].set_title("Kp")
            # axs[0, 1].plot(i, mean[0], 'ro')
            # axs[0, 2].set_title("Ki")
            # axs[0, 2].plot(i, mean[1], 'ro')
            # axs[1, 0].set_title("Kd")
            # axs[1, 0].plot(i, mean[2], 'ro')
            # axs[2, 0].set_title("target velocity")
            # axs[2, 0].plot(i, mean[3], 'ro')
            # axs[1, 1].set_title("R")
            # axs[1, 1].plot(i, mean[4], 'ro')
            # axs[1, 2].set_title("Q1")
            # axs[1, 2].plot(i, mean[5], 'ro')
            # axs[2, 1].set_title("Q2")
            # axs[2, 1].plot(i, mean[6], 'ro')
            # axs[2, 2].set_title("Q3")
            # axs[2, 2].plot(i, mean[7], 'ro')

            plt.pause(dt)


    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()