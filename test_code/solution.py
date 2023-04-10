
import numpy as np
import matplotlib.pyplot as plt
class solution:
    def __init__(self):
        self.startTime=None
        self.endTime=None
        self.executionTime=None
        self.convertTime=None
        self.optimizer=None
        self.objfname=None
        self.best=None
        self.bestIndividual=None
        self.ALLData=None
    def pri(self):
        # print(len(self.convergence))
        # plt.plot(self.convergence)
        # plt.savefig("HHO.jpg")
        # plt.show()
        pass
    def show_time(self):
        print(self.executionTime)
