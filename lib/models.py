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
    def __init__(self, L):

        self.x = 0
        self.y = 0

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

        self.state = [self.s, self.delta, self.e_y, self.e_psi]

    
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

    def step(self, path: CubicHermiteSpline, delta=0, dt=0.01):

        delta_dot = (delta - self.delta) / dt

        # print(delta, self.delta, delta_dot)

        self.vy = (self.vx * self.delta * self.Lr) / (self.Lf + self.Lr)

        # lateral error from path
        t, dist, _, _ = path.closestPointOnCurve((self.x, self.y))

        # gets slope of tangent at t
        dx, dy = path.getVelocity(t)

        # angle between heading and tangent line
        self.e_psi = np.arctan2(dy, dx) - self.theta
        # self.e_psi = angle_between_heading_and_tangent(self.theta, (dx, dy))
        print(self.e_psi)

        self.e_y = dist
        rho = path.getCurvature(t)

        s_dot =  1 / (1 - self.e_y * rho) * (self.vx - self.vx * self.delta * self.e_psi * self.Lr / (self.Lf + self.Lr))
        e_psi_dot = self.vx * self.delta / (self.Lf + self.Lr) - rho * self.vx + delta_dot * self.Lr / (self.Lf + self.Lr)
        e_y_dot = self.vx * self.delta * self.Lr / (self.Lf + self.Lr) + self.vx * self.theta

        # theta_dot = (self.vx * self.delta) / (self.Lf + self.Lr)

        self.s += s_dot * dt

        # change delta
        self.delta = delta

        # change theta
        self.theta += e_psi_dot * dt

        # change x, y by lateral error dot, s_dot, and theta
        self.x += s_dot * np.cos(self.theta) - e_y_dot * np.sin(self.theta)
        self.y += s_dot * np.sin(self.theta) + e_y_dot * np.cos(self.theta)


        return self.state
