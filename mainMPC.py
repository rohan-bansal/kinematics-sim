import matplotlib.pyplot as plt
import numpy as np
import time
import cvxpy as cp
import osqp
from scipy import sparse


from lib.models import CurvilinearKinematicBicycleModel
from lib.path import CubicHermiteSpline, Pose
from lib.controllers import PIDController

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
        self.min_delta, self.max_delta = -np.pi/4, np.pi/4
        self.min_acc, self.max_acc = -2, 2

        ################## STATE ##################

        self.step_i = 0

    def mpc_cvxpy(self, measured_state, target_vel):
        x_0 = np.hstack((measured_state[1], measured_state[4], measured_state[3], measured_state[2]))
        A = np.array([[1, 0, 0, 0],
                    [measured_state[2] / self.L * self.dt, 1, 0, 0],
                    [1/2 * measured_state[2] * self.dt, measured_state[2] * self.dt, 1, 0],
                    [0, 0, 0, 1]])
        B = np.array([[self.dt, 0],
                    [1/2 * self.dt, 0],
                    [0, 0],
                    [0, self.dt]])
        
        x_ref = np.array([0, 0, 0, target_vel])

        x = cp.Variable((4, self.predHorizon + 1))
        u = cp.Variable((2, self.predHorizon))

        cost = 0.0
        constraints = []

        constraints += [x[:, 0] == x_0]
        for t in range(self.predHorizon):
            cost += cp.quad_form(u[:, t], self.R_mpc)

            if t != 0:
                cost += cp.quad_form(x[:, t] - x_ref, self.Q_mpc)

            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
                        
        constraints += [self.min_delta <= u[0, :], u[0, :] <= self.max_delta]
        constraints += [self.min_acc <= u[1, :], u[1, :] <= self.max_acc]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        return u.value
    
    def mpc_osqp(self, measured_state, target_vel):

        N = 20

        umin = np.array([self.min_delta, self.min_acc])
        umax = np.array([self.max_delta, self.max_acc])
        x0 = np.array([0, 0, 0, 0])
        xr = np.array([0, 0, 0, target_vel])

        Q = sparse.eye(4)
        R = sparse.eye(2) * 0.01

        nx, nu = [4, 2]
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
        lineq = np.hstack([np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([lineq])
        u = np.hstack([ueq, uineq])


        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, warm_start=True)

        res = prob.solve()

        print(res.x)

        u = res.x

        delta_opt, acc_opt = u[:2]
        print(delta_opt, acc_opt)

        return delta_opt, acc_opt

    
    def loop(self, state):

        self.step_i += 1

        self.axs[0, 0].cla()
        self.axs[0, 0].set_title("vehicle trajectory")
        self.axs[0, 0].axis('equal')
        self.axs[0, 0].plot(self.trajectory_ref[:, 0], self.trajectory_ref[:, 1], label="desired trajectory")
        self.axs[0, 0].plot(self.model_x_data, self.model_y_data, label="model trajectory")

        x, y, end = self.cBicycleModel.get_cartesian_position(state)
        self.model_x_data.append(x)
        self.model_y_data.append(y)
        
        delta, acc = self.mpc_osqp(state, 5)
        # self.mpc_osqp(state, 5)
        control = [delta, acc] 
        state = self.cBicycleModel.propagate(state, control, self.dt)

        plt.pause(self.dt)

        return state, end
        

if __name__ == "__main__":

    #s, delta, vx, e_y, e_psi
    state = np.array([0, 0, 0.01, 0, 0])
    end = False
    
    mainFunc = Main()
    try:
        while True:
            if not end:
                state, end = mainFunc.loop(state)
            
    except KeyboardInterrupt:
        pass
