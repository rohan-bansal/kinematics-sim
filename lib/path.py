import numpy as np
import math
from scipy.optimize import newton

class Pose:
    # heading is in radians
    def __init__(self, x=0, y=0, heading=0, velocity=0, acceleration=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity # dy/dx
        self.acceleration = acceleration # d2y/dx2
        
    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", heading: " + str(self.heading) + ", velocity: " + str(self.velocity) + ", acceleration: " + str(self.acceleration)
    

# Cubic Hermite Spline path
# https://www.rose-hulman.edu/~finn/CCLI/Notes/day09.pdf
class CubicHermiteSpline:
    
    def __init__(self, poses: list[Pose]):
        self.spline = []
        
        self.ITERATION_CONSTANT = 100

        self.initFromPoses(poses)

        # for i in range(1000):
        #     self.preComputedLengths[i] = self.getLengthToT(i/1000, preCompute=True)


        self.preComputedLengths = np.zeros(1000)
        self.x_vals = np.zeros(1000)
        self.y_vals = np.zeros(1000)
        
        self.getLengthToT(0, preCompute=True)
        self.closestPointOnCurve([0, 0], preCompute=True)

    def initFromPoses(self, poses: list[Pose]):
        self.spline = []
        for i in range(len(poses)-1):
            self.spline.append(self.createSpline(poses[i], poses[i+1]))

    def createSpline(self, pose1: Pose, pose2: Pose):
        
        x0 = pose1.x
        x1 = pose2.x
        y0 = pose1.y
        y1 = pose2.y

        heading0 = pose1.heading
        heading1 = pose2.heading

        mag0 = pose1.velocity
        mag1 = pose2.velocity

        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        if mag0 == 0:
             mag0 = distance
        if mag1 == 0:
             mag1 = distance

        vx0 = mag0 * np.cos(heading0)
        vx1 = mag1 * np.cos(heading1)
        vy0 = mag0 * np.sin(heading0)
        vy1 = mag1 * np.sin(heading1)

        return [x0, x1, y0, y1, vx0, vx1, vy0, vy1]
    
    # t is a value between 0 and 1
    def getPosition(self, t):
        num_segments = len(self.spline) - 1 if len(self.spline) > 1 else 1
        t_in_segment = t * num_segments
        segment_index = int(t_in_segment)
        t = t_in_segment - segment_index

        x0, x1, y0, y1, vx0, vx1, vy0, vy1 = self.spline[segment_index]

        h0 = 1 - 3 * t * t + 2 * t * t * t
        h1 = t - 2 * t * t + t * t * t
        h2 = -t * t + t * t * t
        h3 = 3 * t * t - 2 * t * t * t

        x = h0 * x0 + h1 * vx0 + h2 * vx1 + h3 * x1
        y = h0 * y0 + h1 * vy0 + h2 * vy1 + h3 * y1

        return (x, y)
    
    def getVelocity(self, t):
        num_segments = len(self.spline) - 1 if len(self.spline) > 1 else 1
        t_in_segment = t * num_segments
        segment_index = int(t_in_segment)
        t = t_in_segment - segment_index

        x0, x1, y0, y1, vx0, vx1, vy0, vy1 = self.spline[segment_index]

        h0 = 6 * t * t - 6 * t
        h1 = 3 * t * t - 4 * t + 1
        h2 = 3 * t * t - 2 * t
        h3 = -6 * t * t + 6 * t

        vx = h0 * x0 + h1 * vx0 + h2 * vx1 + h3 * x1
        vy = h0 * y0 + h1 * vy0 + h2 * vy1 + h3 * y1

        return (vx, vy)
    
    def getAcceleration(self, t):
        num_segments = len(self.spline) - 1 if len(self.spline) > 1 else 1
        t_in_segment = t * num_segments
        segment_index = int(t_in_segment)
        t = t_in_segment - segment_index

        x0, x1, y0, y1, vx0, vx1, vy0, vy1 = self.spline[segment_index]

        h0 = 12 * t - 6
        h1 = 6 * t - 4
        h2 = 6 * t - 2
        h3 = -12 * t + 6

        ax = h0 * x0 + h1 * vx0 + h2 * vx1 + h3 * x1
        ay = h0 * y0 + h1 * vy0 + h2 * vy1 + h3 * y1

        return (ax, ay)
    
    def addPose(self, pose: Pose):

        spline = self.createSpline(self.spline[-1], pose)
        self.spline.append(spline)

        return spline

    def delPose(self, index):
        return self.spline.pop(index)
    
    def getGaussianCoefficients(self):
        return [[0.417959183673469, 0.0000000000000000],
            [0.381830050505119, 0.405845151377397],
            [0.381830050505119, -0.405845151377397],
            [0.279705391489277, -0.741531185599395],
            [0.279705391489277, 0.741531185599395],
            [0.12948496616887, -0.949107912342759],
            [0.12948496616887, 0.949107912342759]]
    
    def getCurvature(self, t):
        vx, vy = self.getVelocity(t)
        ax, ay = self.getAcceleration(t)

        return (vx * ay - vy * ax) / (vx**2 + vy**2)**(3/2)

    def closestPointOnCurve(self, otherPoint, preCompute=False):

        if preCompute:

            for i in range(1000):
                t = i/1000
                x, y = self.getPosition(t)
                self.x_vals[i] = x
                self.y_vals[i] = y

            return ""
        else:

            distances = np.sqrt((self.x_vals - otherPoint[0])**2 + (self.y_vals - otherPoint[1])**2)

            closest_index = np.argmin(distances)
            closest_t = closest_index/1000
            closest_x = self.x_vals[closest_index]
            closest_y = self.y_vals[closest_index]
            min_distance = distances[closest_index]

            return closest_t, min_distance, closest_x, closest_y
        
    def closestPointOnCurveVec(self, otherPoints):

        dx = self.x_vals[:, np.newaxis] - otherPoints[:, 0]  # Broadcast subtraction
        dy = self.y_vals[:, np.newaxis] - otherPoints[:, 1]

        squared_distances = dx**2 + dy**2
        closest_indices = np.argmin(squared_distances, axis=0)  # Find min along columns
        closest_t = closest_indices / 1000

        closest_x = self.x_vals[closest_indices]
        closest_y = self.y_vals[closest_indices]
        min_distances = squared_distances[closest_indices, range(len(closest_indices))]

        return closest_t, min_distances, closest_x, closest_y

    
    def getLengthToT(self, t, preCompute=False):

        if preCompute:
            
            for i in range(1000):

                start = 0
                end = i/1000

                half = (end - start) / 2.0
                avg = (start + end) / 2.0
                length = 0

                for coefficient in self.getGaussianCoefficients():
                    vx, vy = self.getVelocity(avg + half * coefficient[1])
                    mag = np.sqrt(vx**2 + vy**2)
                    length += coefficient[0] * mag

                self.preComputedLengths[i] = length

            return ""

        else:

            start = 0
            end = t


            half = (end - start) / 2.0
            avg = (start + end) / 2.0
            
            return self.preComputedLengths[int(t*100)] * half

    def getTFromLength(self, length):

        initial_t = length / self.getLengthToT(1)

        length_difference = lambda t: self.getLengthToT(t) - length

        t = newton(length_difference, initial_t)

        t = np.min([np.max([t, 0]), 1])

        return t
        
    def getPoseAt(self, t):
        x, y = self.getPosition(t)
        vx, vy = self.getVelocity(t)
        ax, ay = self.getAcceleration(t)
        heading = np.arctan2(vy, vx)
        velocity = np.sqrt(vx**2 + vy**2)
        acceleration = np.sqrt(ax**2 + ay**2)

        return Pose(x, y, heading, velocity, acceleration)
    
