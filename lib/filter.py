import numpy as np
from numpy.random import uniform

class Particle():
    
    def __init__(self, parameter, totalParticleNum, weight=0):
        self.parameter = parameter
        if weight == 0:
            self.weight = 1/totalParticleNum
        else:
            self.weight = weight

    def setParameter(self, parameter):
        self.parameter = parameter


def generate_uniform_particles(parameter_range, totalParticleNum):
    particleParams = uniform(parameter_range[0], parameter_range[1], size=totalParticleNum)
    particles = [Particle(particleParams[i], totalParticleNum) for i in range(totalParticleNum)]

    return particles


# particles = generate_uniform_particles([0, 1], 100)
# for particle in particles:
#     print(particle.parameter)
#     print(particle.weight)
#     print()