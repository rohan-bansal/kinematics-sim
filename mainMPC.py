import matplotlib.pyplot as plt
import numpy as np
import time
import osqp
from scipy import sparse
import scipy
from filterpy.monte_carlo import systematic_resample

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

        ################## MPC ##################

        self.predHorizon = 20
        self.Q_mpc = np.diag([0.01, 0.7, 0.8, 0.9])
        self.R_mpc = np.diag([0.01, 0.9])
        self.min_delta_dot, self.max_delta_dot = -np.pi/2, np.pi/2
        self.min_acc, self.max_acc = -2, 2
        self.min_control = np.array([self.min_delta_dot, self.min_acc])
        self.max_control = np.array([self.max_delta_dot, self.max_acc])
        self.min_delta, self.max_delta = -np.pi / 4, np.pi / 4
        self.min_e_psi, self.max_e_psi = -np.inf, np.inf
        self.min_e_y, self.max_e_y = -np.inf, np.inf
        self.min_v, self.max_v = 0, np.inf
        self.min_state = np.array([self.min_delta, self.min_e_psi, self.min_e_y, self.min_v])
        self.max_state = np.array([self.max_delta, self.max_e_psi, self.max_e_y, self.max_v])

        ################## MODEL ##################

        self.cBicycleModel = CurvilinearKinematicBicycleModel(self.path, self.L, (self.min_control, self.max_control), (self.min_state, self.max_state))

        ################## STATE ##################

        self.step_i = 0
        self.target_velocity = 5.0

        ################## PARTICLE FILTER ##################

        self.N = 1000
        self.particle_dynamics_noise_cov = np.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

    def generate_uniform_particles(self):
        # Velocity Setpoint, R1, R2, Q1, Q2, Q3, Q4
        eps = 1e-4
        particles = np.empty((self.N, 7))
        particles[:, 0] = np.random.uniform(1, 20, size=self.N) # Velocity Setpoint
        particles[:, 1] = np.random.uniform(eps, 1, size=self.N) # R1
        particles[:, 2] = np.random.uniform(eps, 1, size=self.N) # R2
        particles[:, 3] = np.random.uniform(eps, 1, size=self.N) # Q1
        particles[:, 4] = np.random.uniform(eps, 1, size=self.N) # Q2
        particles[:, 5] = np.random.uniform(eps, 1, size=self.N) # Q3
        particles[:, 6] = np.random.uniform(eps, 1, size=self.N) # Q4

        return particles

    def pf(self, old_state, new_state, particles, weights):

        predictedStates = np.empty((self.N, 5))

        prob = self.mpc_pf_init(
            old_state,
            particles[0, 0],
            sparse.block_diag([particles[0, 3], particles[0, 4], particles[0, 5], particles[0, 6]]),
            sparse.block_diag([particles[0, 1], particles[0, 2]]))

        t0 = time.time()
        for i in range(self.N):
            predictedStates[i, :] = self.sim_step(prob, particles[i, :], old_state)
        print(time.time() - t0)

        # update weights
        posterior = scipy.stats.multivariate_normal(new_state, 0.005).pdf(predictedStates)
        weights = weights * posterior

        # normalize weights
        weights += 1.e-300
        weights /= sum(weights)

        N_eff = np.sum(weights)**2 / np.sum(weights**2)
        print(N_eff)
        if N_eff < 100:
            idxs = systematic_resample(weights)
            weights = weights[idxs] / np.sum(weights[idxs])
            particles = particles[idxs]
        noise = np.random.multivariate_normal(np.zeros(particles.shape[1]), self.particle_dynamics_noise_cov, particles.shape[0])
        particles += noise
        particles = np.maximum(particles, 1e-6)
        assert (particles > 0).all()

        # estimate mean and variance
        mean = np.average(particles, weights=weights, axis=0)
        var = np.average((particles - mean)**2, weights=weights, axis=0)

        return mean, var, weights, particles
    
    def sim_step(self, prob, particle, cur_state):

        t0 = time.time()
        delta, acc = self.mpc_pf(
            prob,
            particle[0],
            Q=sparse.block_diag([particle[3], particle[4], particle[5], particle[6]]),
            R=sparse.block_diag([particle[1], particle[2]])
        )
        t1 = time.time()

        control = [delta, acc]
        x_plus = self.cBicycleModel.propagate(cur_state, control, self.dt)
        t2 = time.time()
        # print(t1 - t0, t2 - t1)
        return x_plus

    def mpc_pf_init(self, measured_state, target_vel, Q, R):

        N = self.predHorizon
        nx, nu = [4, 2]

        umin = self.min_control
        umax = self.max_control
        xmin = self.min_state
        xmax = self.max_state
        x0 = np.hstack((measured_state[1], measured_state[4], measured_state[3], measured_state[2]))
        xr = np.array([0, 0, 0, target_vel])

        QN = Q

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

        Qs = sparse.kron(sparse.eye(N), Q)
        Rs = sparse.kron(sparse.eye(N), R, format='coo')
        P = sparse.block_diag([Qs, QN, Rs], format='csc')

        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])

        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq

        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        
        return prob
    
    def mpc_pf(self, prob, target_vel, Q, R):
        # t0 = time.time()
        N = self.predHorizon
        nx, nu = [4, 2]

        xr = np.array([0, 0, 0, target_vel])
        QN = Q

        Qr = -1 * Q.dot(xr)
        Qrs = np.tile(Qr, N)
        q = np.hstack([Qrs, -1 * QN.dot(xr), np.zeros(N*nu)])

        # Qs = sparse.kron(sparse.eye(N), Q)
        # Rs = sparse.kron(sparse.eye(N), R, format='coo')
        # P = sparse.block_diag([Qs, QN, Rs], format='csc')
        # P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
        #         sparse.kron(sparse.eye(N), R)], format='csc')
        # P = sparse.triu(P).data
        Q_diag = Q.data
        Qs = np.tile(Q_diag, N)
        R_diag = R.data
        Rs = np.tile(R_diag, N)
        P = np.hstack((Qs, QN.data, Rs))
        # P = sparse.triu(P).data
        # t1 = time.time()
        prob.update(q=q, Px=P)
        # prob.update_settings(warm_start=True)
        # t2 = time.time()
        res = prob.solve()
        # t3 = time.time()
        # print(t1 - t0, t2 - t1, t3 - t2)

        us = res.x[(N+1)*nx:]
        delta_opt, acc_opt = us[:nu]

        return delta_opt, acc_opt

    def mpc_osqp(self, measured_state, target_vel, Q=None, R=None):

        N = self.predHorizon
        nx, nu = [4, 2]

        umin = self.min_control
        umax = self.max_control
        xmin = self.min_state
        xmax = self.max_state
        x0 = np.hstack((measured_state[1], measured_state[4], measured_state[3], measured_state[2]))
        xr = np.array([0, 0, 0, target_vel])

        if Q is None:
            Q = self.Q_mpc
        if R is None:
            R = self.R_mpc

        QN = Q

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

        delta, acc = self.mpc_osqp(state, self.target_velocity)

        control = [delta, acc] 

        old_state = state.copy()
        state = self.cBicycleModel.propagate(state, control, self.dt)
        new_state = state.copy()

        mean, var, pf_weights, pf_particles = self.pf(old_state, new_state, pf_particles, pf_weights)

        plot_weights = True

        self.axs[0, 1].set_title("target velocity")
        if plot_weights:
            if self.step_i % 5 == 0:
                self.axs[0, 1].scatter([self.step_i]*self.N, pf_particles[:, 0], marker='o', c='r', alpha=weights)
                self.axs[0, 2].scatter([self.step_i]*self.N, pf_particles[:, 1], marker='o', c='r', alpha=weights)
                self.axs[1, 0].scatter([self.step_i] * self.N, pf_particles[:, 2], marker='o', c='r', alpha=weights)
                self.axs[2, 0].scatter([self.step_i] * self.N, pf_particles[:, 3], marker='o', c='r', alpha=weights)
                self.axs[1, 1].scatter([self.step_i] * self.N, pf_particles[:, 4], marker='o', c='r', alpha=weights)
                self.axs[1, 2].scatter([self.step_i] * self.N, pf_particles[:, 5], marker='o', c='r', alpha=weights)
                self.axs[2, 1].scatter([self.step_i] * self.N, pf_particles[:, 6], marker='o', c='r', alpha=weights)
        else:
            self.axs[0, 1].plot(self.step_i, mean[0], 'ro')
            self.axs[0, 2].plot(self.step_i, mean[1], 'ro')
            self.axs[1, 0].plot(self.step_i, mean[2], 'ro')
            self.axs[2, 0].plot(self.step_i, mean[3], 'ro')
            self.axs[1, 1].plot(self.step_i, mean[4], 'ro')
            self.axs[1, 2].plot(self.step_i, mean[5], 'ro')
            self.axs[2, 1].plot(self.step_i, mean[6], 'ro')
        if self.step_i % 5 == 0 or plot_weights is False:
            self.axs[0, 1].plot(self.step_i, self.target_velocity, 'b*')
            self.axs[0, 2].set_title("R1")
            self.axs[0, 2].plot(self.step_i, self.R_mpc[0, 0], 'b*')
            self.axs[1, 0].set_title("R2")
            self.axs[1, 0].plot(self.step_i, self.R_mpc[1, 1], 'b*')
            self.axs[2, 0].set_title("Q1")
            self.axs[2, 0].plot(self.step_i, self.Q_mpc[0, 0], 'b*')
            self.axs[1, 1].set_title("Q2")
            self.axs[1, 1].plot(self.step_i, self.Q_mpc[1, 1], 'b*')
            self.axs[1, 2].set_title("Q3")
            self.axs[1, 2].plot(self.step_i, self.Q_mpc[2, 2], 'b*')
            self.axs[2, 1].set_title("Q4")
            self.axs[2, 1].plot(self.step_i, self.Q_mpc[3, 3], 'b*')
        
        plt.pause(self.dt)

        return state, pf_particles, pf_weights, end
        

if __name__ == "__main__":

    #s, delta, vx, e_y, e_psi
    state = np.array([0, 0, 2.0, 0, 0])
    end = False
    
    mainFunc = Main()

    particles = mainFunc.generate_uniform_particles()
    weights = np.ones(mainFunc.N) / mainFunc.N

    try:
        while True:
            if not end:
                state, particles, weights, end = mainFunc.loop(state, particles, weights)
            
    except KeyboardInterrupt:
        pass
