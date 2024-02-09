import numpy as np
from lib.path import Pose, CubicHermiteSpline
from lib.models import KinematicBicycleModel, CurvilinearKinematicBicycleModel


class PIDController:
    
        def __init__(self, Kp, Ki, Kd, setpoint):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
    
            self.setpoint = setpoint
    
            self.integral = 0
            self.prev_error = 0
        
        def step(self, measurement):
            error = self.setpoint - measurement
            self.integral += error
            derivative = (error - self.prev_error)
            output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
            self.prev_error = error
            return output
        
        def simStep(self, Kp, Ki, Kd, setpoint, measurement):
            error = setpoint - measurement
            integral = self.integral
            integral += error
            derivative = (error - self.prev_error)
            output = Kp * error + Ki * self.integral + Kd * derivative
            return output
        
        def setSetpoint(self, setpoint):
            self.setpoint = setpoint


class PurePursuitController:
    
    def __init__(self, model: CurvilinearKinematicBicycleModel, path: CubicHermiteSpline, LOOKAHEAD_CONSTANT=None):
        self.model = model
        self.path = path

        self.LOOKAHEAD_CONSTANT = 0.35 if LOOKAHEAD_CONSTANT is None else LOOKAHEAD_CONSTANT

        self.min_LD = 1
        self.max_LD = 20
        self.lookahead_distance = 0

    def getLookaheadPoint(self, distanceTraveled, lookahead_distance):
        path_length = self.path.getLengthToT(1)
        if distanceTraveled + lookahead_distance > path_length:
            pathEndPose = self.path.getPoseAt(1)
            distLeft = distanceTraveled + lookahead_distance - path_length

            return (pathEndPose.x + distLeft * np.cos(pathEndPose.heading), pathEndPose.y + distLeft * np.sin(pathEndPose.heading))
        else:
            return self.path.getPosition(self.path.getTFromLength(distanceTraveled + lookahead_distance))
        
    # return center and radius
    def getTangentCircle(self, currentPoint, lookaheadPoint):
        distance = np.sqrt((currentPoint[0] - lookaheadPoint[0])**2 + (currentPoint[1] - lookaheadPoint[1])**2)
        radius = distance / 2

        center_x = currentPoint[0]
        center_y = currentPoint[1]

        return (center_x, center_y, radius)


    # t between 0 and 1
    def step(self, x, y, v, plt=None, dt=0.01):
        
        self.lookahead_distance = np.clip(self.LOOKAHEAD_CONSTANT * v, self.min_LD, self.max_LD)

        closest_t, lat_error, closest_x, closest_y = self.path.closestPointOnCurve((x, y))


        distanceTraveled = self.path.getLengthToT(closest_t)
        lookaheadPoint = self.getLookaheadPoint(distanceTraveled, self.lookahead_distance)

        if plt is not None:
            plt.plot(closest_x, closest_y, 'ro')

        alpha = np.arctan2(lookaheadPoint[1], lookaheadPoint[0]) - np.arctan2(y, x)
        steer = np.arctan((2 * self.model.L * np.sin(alpha)) / self.lookahead_distance)

        return steer
    
    def simStep(self, state, LOOKAHEAD_CONSTANT, dt=0.01):

        x = state[0]
        y = state[1]
        v = state[3]
        
        lookahead_distance = np.clip(LOOKAHEAD_CONSTANT * v, self.min_LD, self.max_LD)

        closest_t, lat_error, closest_x, closest_y = self.path.closestPointOnCurve((x, y))


        distanceTraveled = self.path.getLengthToT(closest_t)
        lookaheadPoint = self.getLookaheadPoint(distanceTraveled, lookahead_distance)

        alpha = np.arctan2(lookaheadPoint[1], lookaheadPoint[0]) - np.arctan2(y, x)
        steer = np.arctan((2 * self.model.L * np.sin(alpha)) / lookahead_distance)

        return steer
