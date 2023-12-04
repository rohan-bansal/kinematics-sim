import numpy as np

from lib.path import CubicHermiteSpline, Pose

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

"""
Cartesian state: x, y, theta (yaw), v, delta (steer angle)
Curvilinear: s (distance along curve), d (lateral displacement), delta (steer angle), heading error (difference between heading angle and angle of tangent line at s)

"""

def angle_between_heading_and_tangent(heading_angle, tangent_vector):
    # Convert heading angle to unit vector
    heading = (np.cos(heading_angle), np.sin(heading_angle))

    # Normalize tangent vector
    tangent_magnitude = np.sqrt(tangent_vector[0]**2 + tangent_vector[1]**2)
    tangent = (tangent_vector[0] / tangent_magnitude, tangent_vector[1] / tangent_magnitude)

    # Calculate dot product
    dot_product = heading[0] * tangent[0] + heading[1] * tangent[1]

    # Calculate angle in radians
    angle_radians = np.arccos(dot_product)


    return angle_radians

class CurvilinearKinematicBicycleModel:

    # parameters: wheelbase 
    def __init__(self, path, L):

        self.x = 0
        self.y = 0
        self.path = path

        # longitudinal speed (along the curve)
        self.vx = 0

        # lateral speed (perpendicular to the curve)
        self.vy = 0

        self.L = L
        self.Lf = L/2
        self.Lr = L/2

        self.s = 0
        self.delta = 0
        self.theta = 0
        self.e_y = 0
        self.e_psi = 0

        self.state = np.array([self.s, self.delta, self.vx, self.e_y, self.e_psi])

    def updateState(self):
        self.state = np.array([self.s, self.delta, self.vx, self.e_y, self.e_psi])

    def linearize(self, nominal_state, nominal_ctrl, dt):
        print("nominal ctrl", nominal_ctrl)

        nominal_state = np.array(nominal_state).copy()
        nominal_ctrl = np.array(nominal_ctrl).copy()
        print("copied nominal ctrl", nominal_ctrl)
        epsilon = 1e-2
        # A = df/dx
        A = np.zeros((5, 5), dtype=float)
        # find A
        for i in range(5):
            # d x / d x_i, ith row in A
            x_l = nominal_state.copy()
            x_l[i] -= epsilon
            x_post_l = self.propagate(x_l, nominal_ctrl, dt)
            print("x_l", x_l)
            print("x_post_l", x_post_l)
            x_r = nominal_state.copy()
            x_r[i] += epsilon
            x_post_r = self.propagate(x_r, nominal_ctrl, dt)
            print("x_r", x_r)
            print("x_post_r", x_post_r)
            A[:, i] += (x_post_r.flatten() - x_post_l.flatten()) / (2 * epsilon)

        # B = df/du
        B = np.zeros((5, 1), dtype=float)
        # find B
        for i in range(1):
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

    
    # delta_dot = change in steer angle with respect to time
    # delta = steer angle
    # theta = heading
    # e_psi = heading error from tangent line
    # e_psi_dot = change in heading error with respect to time
    # s = distance along curve
    # s_dot = change in distance along curve with respect to time
    # e_y = lateral error from line tangent to closest point on curve
    # e_y_dot = change in lateral error with respect to time
    # rho = curvature of curve at closest point

    def step(self, delta=0, dt=0.01):

        delta_dot = (delta - self.delta) / dt

        t = self.path.getTFromLength(self.s)
        pose = self.path.getPoseAt(t)

        dx, dy = self.path.getVelocity(t)
        rho = self.path.getCurvature(t)


        s_dot =  1 / (1 - self.e_y * rho) * (self.vx - self.vx * self.delta * self.e_psi * self.Lr / (self.Lf + self.Lr))
        e_psi_dot = self.vx * self.delta / (self.Lf + self.Lr) - rho * self.vx + delta_dot * self.Lr / (self.Lf + self.Lr)
        e_y_dot = self.vx * self.delta * self.Lr / (self.Lf + self.Lr) + self.vx * self.e_psi

        self.s += s_dot * dt

        self.delta += delta_dot * dt

        self.e_psi += e_psi_dot * dt
        self.e_y += e_y_dot * dt

        self.theta += e_psi_dot * dt

        tan_angle = np.arctan2(dy, dx)
        self.x = pose.x - self.e_y * np.sin(tan_angle)
        self.y = pose.y + self.e_y * np.cos(tan_angle)

        self.state = np.array([self.s, self.delta, self.vx, self.e_y, self.e_psi])

        return self.state

    # step function but isolated from the system - uses a given state, control, and dt.
    def propagate(self, state, control, dt=0.01):

        s, delta, vx, e_y, e_psi = state
        new_delta = control

        print(new_delta[0], delta)

        delta_dot = (new_delta[0] - delta) / dt

        print(delta)

        t = self.path.getTFromLength(s)
        rho = self.path.getCurvature(t)


        s_dot =  1 / (1 - e_y * rho) * (vx - vx * delta * e_psi * self.Lr / (self.Lf + self.Lr))
        e_psi_dot = vx * delta / (self.Lf + self.Lr) - rho * vx + delta_dot * self.Lr / (self.Lf + self.Lr)
        e_y_dot = vx * delta * self.Lr / (self.Lf + self.Lr) + vx * e_psi

        s += s_dot * dt

        delta += delta_dot * dt

        e_psi += e_psi_dot * dt
        e_y += e_y_dot * dt

        state = np.array([s, delta, vx, e_y, e_psi])


        return state
    
    def getState(self):
        return self.state