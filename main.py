import numpy

# write a class that implements the kinematic bicycle model (linearized) using numpy

class KinematicBicycleModel:

    # state initialization: x, y, vehicle yaw, velocity
    # parameters: x, y, vehicle yaw, wheelbase 
    def __init__(self, x, y, theta, L):
        self.v = 0
        self.x = x
        self.y = y
        self.theta = theta
        self.L = L
        self.state = [self.x, self.y, self.theta, self.v]
    
    # control input: velocity, steering angle
    def step(self, v, w):
        self.v = v
        self.w = w

        self.x += self.v * numpy.cos(self.theta)
        self.y += self.v * numpy.sin(self.theta)

        self.theta += self.v * numpy.tan(self.w) / self.L
        self.state = [self.x, self.y, self.theta, self.v]
        return self.state
    
    def get_state(self):
        return self.state

