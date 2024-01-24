import matplotlib.pyplot as plt
import numpy as np
import scipy
from filterpy.monte_carlo import systematic_resample

from lib.models import KinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose
from lib.controllers import PIDController, PurePursuitController
from lib.filter import Particle, generate_uniform_particles

#################### PARAMETERS ####################
dt = 0.01   
L = 2 # 2m wheelbase

#################### PATH DATA ####################
t_data = np.arange(0, 1, dt)
waypoints = [
    Pose(x=0, y=0, heading=np.pi/2, velocity=20), 
    Pose(x=10, y=10, heading=0, velocity=20), 
    Pose(x=20, y=20, heading=0, velocity=20),
    Pose(x=30, y=30, heading=0, velocity=20)
]

#################### CONTROLLERS ####################
bicycleModel = KinematicBicycleModel(L)
path = CubicHermiteSpline(waypoints)
controller = PurePursuitController(bicycleModel, path)
PID = PIDController(0.9, 0, 0, 5)

# x, y, theta, v, delta
state = [0, 0, np.pi/2, 0, 0]

#################### PLOTTING ####################
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
model_x_data = []
model_y_data = []

plt.title("Kinematic Bicycle Motion Model")
plt.axis('equal')

#################### PARTICLE FILTER ####################
N = 1000

def generate_uniform_particles():
    # Kp, Ki, Kd, target velocity, lookahead scaling factor
    particles = np.empty((N, 5))
    particles[:, 0] = np.random.uniform(0, 1, size=N) # Kp
    particles[:, 1] = np.random.uniform(0, 1, size=N) # Ki
    particles[:, 2] = np.random.uniform(0, 1, size=N) # Kd
    particles[:, 3] = np.random.uniform(0, 15, size=N) # target velocity
    particles[:, 4] = np.random.uniform(0, 1, size=N) # lookahead scaling factor

    return particles

def pfStep(measurement, particles, weights):

    predictedStates = np.empty((N, 5))
    predictedStates = np.apply_along_axis(simulateNextStep, 1, particles, state)

    # update weights
    # for i in range(len(predictedStates)):

        # dst = np.linalg.norm(predictedStates - measurement, axis=1)
    weights *= scipy.stats.multivariate_normal(measurement, 0.1).pdf(predictedStates)

    # normalize weights
    weights += 1.e-300
    weights /= sum(weights)

    # resample
    if 1. / sum(weights**2) < N:
        indexes = systematic_resample(weights)
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights.fill(1.0 / N)


    # estimate mean and variance
    mean = np.average(particles, weights=weights, axis=0)
    var  = np.average((particles - mean)**2, weights=weights, axis=0)

    return mean, var

def simulateNextStep(particle, cur_state):


    steer_angle = controller.simStep(cur_state[0], cur_state[1], cur_state[3], particle[4])
    PIDoutput = PID.simStep(particle[0], particle[1], particle[2], particle[3], cur_state[3], dt)
    pred_state = bicycleModel.step(cur_state, a=PIDoutput, delta=steer_angle, dt=dt)

    return pred_state

particles = generate_uniform_particles()
weights = np.ones(N) / N

#################### PATH GENERATION ####################
for i in range(t_data.shape[0]):

    x_data[i], y_data[i] = path.getPosition(t_data[i])


#################### MAIN LOOP ####################
try:
    while True:

        plt.clf()
        plt.axis('equal')
        plt.plot(x_data, y_data, label="desired trajectory")
        plt.plot(model_x_data, model_y_data, label="model trajectory")

        model_x_data.append(state[0])
        model_y_data.append(state[1])

        steer_angle = controller.step(state[0], state[1], state[3], plt)
        bicycleModel.delta = steer_angle

        PIDoutput = PID.step(state[3], dt)

        state = bicycleModel.step(state, a=PIDoutput, delta=steer_angle, dt=dt)

        mean, var = pfStep(state, particles, weights)
        print("mean", mean, "var", var)

        plt.pause(0.01)

except KeyboardInterrupt:
    pass
