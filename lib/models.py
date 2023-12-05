import numpy as np

from lib.path import CubicHermiteSpline, Pose

# OLD CLASS THAT IS NOT CURVILINEAR: SEE FURTHER BELOW
class KinematicBicycleModel:

    # state initialization: x, y, vehicle yaw, velocity
    # parameters: x, y, wheelbase 
    def __init__(self, x, y, L):
        self.v = 0
        self.x = x
        self.y = y
        self.theta = 0
        self.beta = 0
        self.delta = 0
        self.L = L
        self.Lf = L/2

        self.max_w = np.pi/4

        self.state = [self.x, self.y, self.theta, self.v, self.delta]
    
    def step(self, a=0, w=0, dt=0.01):

        if w > self.max_w:
            w = self.max_w
        elif w < -self.max_w:
            w = -self.max_w

        self.v += a * dt
        self.w = w

        theta_update = (self.v / self.L) * (np.cos(self.beta) * np.tan(self.delta))
        x_update = self.v * np.cos(self.theta + self.beta)
        y_update = self.v * np.sin(self.theta + self.beta)

        self.beta = np.arctan(self.Lf * np.tan(self.delta) / self.L)

        self.x += x_update * dt
        self.y += y_update * dt
        self.theta += theta_update * dt

        self.delta += self.w * dt

        self.state = [self.x, self.y, self.theta, self.v, self.delta]

        return self.state
    
    def get_state(self):
        return self.state




class CurvilinearKinematicBicycleModel:

    # parameters: wheelbase 
    def __init__(self, path, L):

        self.x = 0
        self.y = 0
        self.path = path

        self.L = L
        self.Lf = L/2
        self.Lr = L/2

    def linearize(self, nominal_state, nominal_ctrl, dt):

        nominal_state = np.array(nominal_state).copy()
        nominal_ctrl = np.array(nominal_ctrl).copy()

        n = nominal_state.shape[0]
        m = nominal_ctrl.shape[0]

        epsilon = 1e-2
        # A = df/dx
        A = np.zeros((n, n), dtype=float)
        # find A
        for i in range(n):
            # d x / d x_i, ith row in A
            x_l = nominal_state.copy()
            x_l[i] -= epsilon
            x_post_l = self.propagate(x_l, nominal_ctrl, dt)

            x_r = nominal_state.copy()
            x_r[i] += epsilon
            x_post_r = self.propagate(x_r, nominal_ctrl, dt)

            A[:, i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

        # B = df/du
        B = np.zeros((n, m), dtype=float)
        # find B
        for i in range(m):
            # d x / d u_i, ith row in B
            x0 = nominal_state.copy()
            u_l = nominal_ctrl.copy()
            u_l[i] -= epsilon
            x_post_l = self.propagate(x0, u_l, dt)
            x_post_l = x_post_l.copy()

            x0 = nominal_state.copy()
            u_r = nominal_ctrl.copy()
            u_r[i] += epsilon
            x_post_r = self.propagate(x0, u_r, dt)
            x_post_r = x_post_r.copy()

            B[:, i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        x_post = self.propagate(x0, u0, dt)
        # d = x_k+1 - Ak*x_k - Bk*u_k
        x0 = nominal_state.copy()
        u0 = nominal_ctrl.copy()
        d = x_post.flatten() - A @ x0 - B @ u0
        return A, B, d

    def updatePosition(self, state):
        
        t = self.path.getTFromLength(state[0])
        pose = self.path.getPoseAt(t)

        dx, dy = self.path.getVelocity(t)
        tan_angle = np.arctan2(dy, dx)
        self.x = pose.x - state[4] * np.sin(tan_angle)
        self.y = pose.y + state[4] * np.cos(tan_angle)

    # step function but isolated from the system - uses a given state, control, and dt.
    def propagate(self, state, control, dt=0.01):

        copied_state = state.copy()
        copied_control = control.copy()

        s, delta, vx, e_y, e_psi = copied_state
        new_delta = copied_control[0]

        delta_dot = (new_delta - delta) / dt

        t = self.path.getTFromLength(s)
        rho = self.path.getCurvature(t)

        s_dot =  1 / (1 - e_y * rho) * (vx - vx * delta * e_psi * self.Lr / (self.Lf + self.Lr))
        e_psi_dot = vx * delta / (self.Lf + self.Lr) - rho * vx + delta_dot * self.Lr / (self.Lf + self.Lr)
        e_y_dot = vx * delta * self.Lr / (self.Lf + self.Lr) + vx * e_psi

        s += s_dot * dt

        delta += delta_dot * dt

        e_psi += e_psi_dot * dt
        e_y += e_y_dot * dt

        return np.array([s, delta, vx, e_y, e_psi])
