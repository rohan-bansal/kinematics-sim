import numpy as np

class KinematicBicycleModel:

    # state initialization: x, y, vehicle yaw, velocity
    # parameters: x, y, vehicle yaw, wheelbase 
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

        self.state = [self.x, self.y, self.theta, self.v]
    
    # control input: velocity, steering angle rate
    def step(self, v=0, w=0, delta=0, dt=0.01):

        if w > self.max_w:
            w = self.max_w
        elif w < -self.max_w:
            w = -self.max_w

        self.v = v
        self.w = w

        theta_update = (self.v / self.L) * (np.cos(self.beta) * np.tan(self.delta))
        x_update = self.v * np.cos(self.theta + self.beta)
        y_update = self.v * np.sin(self.theta + self.beta)

        self.beta = np.arctan(self.Lf * np.tan(self.delta) / self.L)

        self.x += x_update * dt
        self.y += y_update * dt
        self.theta += theta_update * dt
        self.delta += self.w * dt

        self.state = [self.x, self.y, self.theta, self.v]


        return self.state
    
    def get_state(self):
        return self.state

