import numpy as np

# Cubic Hermite Spline path
# https://www.rose-hulman.edu/~finn/CCLI/Notes/day09.pdf

class Pose:
      # heading is in radians
      def __init__(self, x=0, y=0, heading=0, velocity=0, acceleration=0):
            self.x = x
            self.y = y
            self.heading = heading
            self.velocity = velocity # dy/dx
            self.acceleration = acceleration # d2y/dx2

class CubicHermiteSpline:
    
    def __init__(self, poses: list[Pose]):
        self.spline = []
        self.initFromPoses(poses)

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
        spline = self.spline[int(len(self.spline) * t)]
        x0, x1, y0, y1, vx0, vx1, vy0, vy1 = spline

        h0 = 12 * t - 6
        h1 = 6 * t - 4
        h2 = 6 * t - 2
        h3 = 6 - 12 * t

        x = h0 * x0 + h1 * vx0 + h2 * vx1 + h3 * x1
        y = h0 * y0 + h1 * vy0 + h2 * vy1 + h3 * y1

        return (x, y)
    
    def addPose(self, pose: Pose):

        spline = self.createSpline(self.spline[-1], pose)
        self.spline.append(spline)

        return spline

    def delPose(self, index):
        return self.spline.pop(index)