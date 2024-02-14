import matplotlib.pyplot as plt
import numpy as np
import time
import osqp
from scipy import sparse
import scipy


from lib.models import CurvilinearKinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose

np.random.seed(0)

class Main():

    def __init__(self):

        ################## PARAMETERS ##################

        # time step
        self.dt = 0.05
        # path resolution step
        self.ds = 0.01
        # wheelbase
        self.L = 2

        ################## PATH DATA / GENERATION / PLOTTING ##################

        t_data = np.arange(0, 1, self.ds)
        waypoints = [
            Pose(x=0, y=0, heading=0, velocity=50),
            Pose(x=0, y=0, heading=0, velocity=20),
            Pose(x=10, y=10, heading=0, velocity=20), 
            Pose(x=20, y=20, heading=0, velocity=20),
            Pose(x=30, y=30, heading=0, velocity=20)
        ]

        self.path = CubicHermiteSpline(waypoints)

        self.model_x_data = []
        self.model_y_data = []

        self.trajectory_ref = np.array([self.path.getPosition(t) for t in t_data])

        fig, self.axs = plt.subplots(3, 3)

        ################## CONTROLLERS ##################

        self.cBicycleModel = CurvilinearKinematicBicycleModel(self.path, self.L)

        ################## MPC ##################

        self.predHorizon = 20
        self.Q_mpc = np.diag([1, 1, 1, 1])
        self.R_mpc = np.diag([1, 1])
        self.min_delta, self.max_delta = -np.pi, np.pi
        self.Q_mpc = np.diag([0.01, 1.0, 1.0, 1.0])
        self.R_mpc = np.diag([0.01, 1])
        self.min_delta, self.max_delta = -np.pi/2, np.pi/2
        self.min_acc, self.max_acc = -2, 2

        ################## STATE ##################

        self.step_i = 0
        self.target_velocity = 20

        ################## PARTICLE FILTER ##################

        self.N = 1000

    def generate_uniform_particles(self):
        # Velocity Setpoint, R1, R2, Q1, Q2, Q3, Q4
        particles = np.empty((self.N, 7))
        particles[:, 0] = np.random.uniform(0, 30, size=self.N) # Velocity Setpoint
        particles[:, 1] = np.random.uniform(0, 1, size=self.N) # R1
        particles[:, 2] = np.random.uniform(0, 1, size=self.N) # R2
        particles[:, 3] = np.random.uniform(0, 1, size=self.N) # Q1
        particles[:, 4] = np.random.uniform(0, 1, size=self.N) # Q2
        particles[:, 5] = np.random.uniform(0, 1, size=self.N) # Q3
        particles[:, 6] = np.random.uniform(0, 1, size=self.N) # Q4

        return particles
    
    def pf(self, old_state, new_state, particles, weights):

        predictedStates = np.empty((self.N, 7))
        predictedStates = np.apply_along_axis(self.sim_step, 1, particles, old_state)

        # update weights
        posterior = scipy.stats.multivariate_normal(new_state, 0.005).pdf(predictedStates)
        weights = weights * posterior

        # normalize weights
        weights += 1.e-300
        weights /= sum(weights)

        # estimate mean and variance
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean)**2, weights=weights, axis=0)

        return mean, var, weights
    
    def sim_step(self, particle, cur_state):

        delta, acc = self.mpc_osqp(
            cur_state, 
            particle[0], 
            Q=np.diag([particle[3], particle[4], particle[5], particle[6]]), 
            R=np.diag([particle[1], particle[2]])
        )

        control = [delta, acc]

        return self.cBicycleModel.propagate(cur_state, control, self.dt)

    
    def mpc_osqp(self, measured_state, target_vel, Q=None, R=None):

        N = self.predHorizon
        nx, nu = [4, 2]

        umin = np.array([self.min_delta, self.min_acc])
        umax = np.array([self.max_delta, self.max_acc])
        xmin = np.array([-1*np.pi, -1*np.inf, -1*np.inf, 0])
        xmax = np.array([np.pi, np.inf, np.inf, np.inf])
        x0 = np.hstack((measured_state[1], measured_state[4], measured_state[3], measured_state[2]))
        xr = np.array([0, 0, 0, target_vel])

        if Q is None:
            Q = self.Q_mpc
        if R is None:
            R = self.R_mpc

        QN = Q

        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')

        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -1 * Q.dot(xr)), -1 * QN.dot(xr),
                    np.zeros(N*nu)])

        Ad = np.array([[1, 0, 0, 0],
                    [measured_state[2] / self.L * self.dt, 1, 0, 0],
                    [1/2 * measured_state[2] * self.dt, measured_state[2] * self.dt, 1, 0],
                    [0, 0, 0, 1]])
        Bd = np.array([[self.dt, 0],
                    [1/2 * self.dt, 0],
                    [0, 0],
                    [0, self.dt]])

        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])


        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

        res = prob.solve()

        us = res.x[(N+1)*nx:]

        delta_opt, acc_opt = us[:nu]

        return delta_opt, acc_opt


    
    def loop(self, state, pf_particles, pf_weights):

        self.step_i += 1

        self.axs[0, 0].cla()
        self.axs[0, 0].set_title("vehicle trajectory")
        self.axs[0, 0].axis('equal')
        self.axs[0, 0].plot(self.trajectory_ref[:, 0], self.trajectory_ref[:, 1], label="desired trajectory")
        self.axs[0, 0].plot(self.model_x_data, self.model_y_data, label="model trajectory")

        x, y, end = self.cBicycleModel.get_cartesian_position(state)
        self.model_x_data.append(x)
        self.model_y_data.append(y)

        print(state)
        delta, acc = self.mpc_osqp(state, self.target_velocity)

        control = [delta, acc] 

        old_state = state.copy()
        state = self.cBicycleModel.propagate(state, control, self.dt)
        new_state = state.copy()

        mean, var, weights = self.pf(old_state, new_state, pf_particles, pf_weights)

        self.axs[0, 1].set_title("target velocity")
        self.axs[0, 1].plot(self.step_i, mean[0], 'ro')
        self.axs[0, 2].set_title("R1")
        self.axs[0, 2].plot(self.step_i, mean[1], 'ro')
        self.axs[1, 0].set_title("R2")
        self.axs[1, 0].plot(self.step_i, mean[2], 'ro')
        self.axs[2, 0].set_title("Q1")
        self.axs[2, 0].plot(self.step_i, mean[3], 'ro')
        self.axs[1, 1].set_title("Q2")
        self.axs[1, 1].plot(self.step_i, mean[4], 'ro')
        self.axs[1, 2].set_title("Q3")
        self.axs[1, 2].plot(self.step_i, mean[5], 'ro')
        self.axs[2, 1].set_title("Q4")
        self.axs[2, 1].plot(self.step_i, mean[6], 'ro')
        
        plt.pause(self.dt)

        return state, end
        

if __name__ == "__main__":

    #s, delta, vx, e_y, e_psi
    state = np.array([0, 0, 0.01, 0, 0])
    end = False
    
    mainFunc = Main()

    particles = mainFunc.generate_uniform_particles()
    weights = np.ones(mainFunc.N) / mainFunc.N

    try:
        while True:
            if not end:
                state, end = mainFunc.loop(state, particles, weights)
            
    except KeyboardInterrupt:
        pass
