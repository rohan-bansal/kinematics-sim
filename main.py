import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from lib.models import KinematicBicycleModel, CurvilinearKinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose
from lib.controllers import PIDController, PurePursuitController
from lib.filter import Particle, generate_uniform_particles

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
pidController = PIDController(0.5, 0.001, 0.01, 5)

x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
model_x_data = []
model_y_data = []

plt.title("Kinematic Bicycle Motion Model")
plt.axis('equal')

# s, delta, vx, e_y, e_psi
state = np.array([0, 0, 0.01, 0, 0])

# draw predetermined path to follow on graph
for i in range(t_data.shape[0]):
    x_data[i], y_data[i] = path.getPosition(t_data[i])

def pfInit(parameterRange, numParticles):
    return generate_uniform_particles(parameterRange, numParticles)

def pfStep(measurement, particles):

    numParticles = len(particles)

    # predict particle next value using model (PID for velocity)
    particlesPredictedVals = [pidController.testStep(particle.parameter, dt) for particle in particles]
    
    # update weights
    for i in range(numParticles):

        # based on gaussian distribution
        likelihood = 1 / ((2 * 3.14159) ** 0.5) * 2.71828 ** (-0.5 * (abs(measurement - particlesPredictedVals[i])) ** 2)
        particles[i].weight = likelihood

        # particles[i].weight = np.exp(-0.5 * (measurement - particlesPredictedVals[i])**2)
        # particles[i].weight = (measurement - particlesPredictedVals[i]) ** 2 + 1e-300 # add small number to avoid divide by zero

    # normalize weights
    weightSum = np.sum([particle.weight for particle in particles])
    for i in range(numParticles):
        particles[i].weight /= weightSum

    # fig, ax = plt.subplots()
    # ax.hist([particle.parameter for particle in particles], weights=[particle.weight for particle in particles])
    # fig.savefig("test.png")
    # plt.close(fig)

    # resample particles


    # estimate mean and variance
    mean = np.average([particle.parameter for particle in particles], weights=[particle.weight for particle in particles])
    variance = np.average([(particle.parameter - mean)**2 for particle in particles], weights=[particle.weight for particle in particles])

    return mean, variance


try:

    particles = pfInit([2, 15], 100)

    while True:

        # clear plot
        plt.clf()
        plt.axis('equal')
        plt.plot(x_data, y_data, label="desired trajectory")
        plt.plot(model_x_data, model_y_data, label="model trajectory")

        # run pf on measured velocity param of state
        output = pfStep(state[2], particles)
        print("MEASURED", state[2] / dt, "FILTER OUTPUT", output)
        # print("FILTER OUTPUT", output)


        
        acc = pidController.step(state[2], dt)
        # print("acc", acc)


        A0 = np.array([[1, 0, 0],
               [state[2] / L * dt, 1, 0],
               [1/2 * state[2] * dt, state[2] * dt, 1]])
        B0 = np.array([[dt],
                    [1/2 * dt],
                    [0]])
        Q0 = np.eye(3)
        R0 = np.eye(1)
        P0 = la.solve_discrete_are(A0, B0, Q0, R0)
        K0 = -np.linalg.inv(R0 + B0.T @ P0 @ B0) @ B0.T @ P0 @ A0


        x0 = np.array([state[1], state[-1], state[-2]]).reshape((-1, 1))
        delta_dot = K0 @ x0
        control = np.array([delta_dot[0, 0], acc])
        
        # update state with new control input
        state = cBicycleModel.propagate(state, control, dt)
        print(state)


        # update model position on graph
        x, y = cBicycleModel.get_cartesian_position(state)
        model_x_data.append(x)
        model_y_data.append(y)


        plt.pause(dt)


except KeyboardInterrupt:
    pass





        # calculate A, b, d matrices with state and control
        # A, b, d = cBicycleModel.linearize(state, control, dt)

        # define Q and R matrices
        # Q = np.array([[0, 0, 0, 0, 0],
        #               [0, 1, 0, 0, 0],
        #               [0, 0, 0, 0, 0],
        #               [0, 0, 0, 1, 0],
        #               [0, 0, 0, 0, 1]])
        # R = np.array([[1000]])

        # # solve DARE
        # x = la.solve_discrete_are(A, b, Q, R)
        #
        # # calculate K gain
        # k = -(R + b.T @ x @ b)**-1 @ b.T @ x @ A

        # calculate control
        # steer = (-k @ state)[0]
        # control[0] = steer
        #
        # # limit control outputs
        # if (control[0] > np.pi/4):
        #     control[0] = np.pi/4
        # elif (control[0] < -np.pi/4):
        #     control[0] = -np.pi/4

        # y = A @ state + b @ control + d
        # print("y", y)