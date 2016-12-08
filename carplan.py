from ltraj import LTrajAvoid
import numpy as np
import matplotlib.pyplot as plt

class CarPlan(LTrajAvoid):

    def __init__(self, car_a, car_b, num_steps, num_cars, ind_pos=0, ind_spd=1, min_spd=0.0, max_spd=8.94, name="CarPlan", sense=1):
        self.ind_pos = ind_pos
        self.ind_spd = ind_spd
        LTrajAvoid.__init__(self, car_a, car_b, num_steps, name, sense, num_agents=num_cars)
        for cc in range(num_cars):
            self.avar_x[cc][self.Nt][ind_spd].lowBound = min_spd
            self.avar_x[cc][self.Nt][ind_spd].upBound = max_spd
            for kk in range(self.Nt):
                self.avar_x[cc][kk][ind_spd].lowBound = min_spd
                self.avar_x[cc][kk][ind_spd].upBound = max_spd

    def addSpeedRestriction(self, spd, pos_start, pos_finish, car):
        for kk in range(self.Nt):
            self.addUnionConstraint(([self.avar_x[car][kk][self.ind_pos] - pos_start,
                                      self.avar_x[car][kk+1][self.ind_pos] - pos_start],
                                     [pos_finish - self.avar_x[car][kk][self.ind_pos],
                                      pos_finish - self.avar_x[car][kk+1][self.ind_pos]],
                                     [self.avar_x[car][kk][self.ind_spd] - spd,
                                      self.avar_x[car][kk+1][self.ind_spd] - spd]))

    def addCrossingConstraint(self,car_1,start_pos_1,end_pos_1,car_2,start_pos_2,end_pos_2):
        for kk in range(self.Nt):
            self.addUnionConstraint(([self.avar_x[car_1][kk][self.ind_pos] - start_pos_1,
                                      self.avar_x[car_1][kk+1][self.ind_pos] - start_pos_1],
                                     [end_pos_1 - self.avar_x[car_1][kk][self.ind_pos],
                                      end_pos_1 - self.avar_x[car_1][kk+1][self.ind_pos]],
                                     [self.avar_x[car_2][kk][self.ind_pos] - start_pos_2,
                                      self.avar_x[car_2][kk + 1][self.ind_pos] - start_pos_2],
                                     [end_pos_2 - self.avar_x[car_2][kk][self.ind_pos],
                                      end_pos_2 - self.avar_x[car_2][kk + 1][self.ind_pos]]))

    def plotSpeedOverDistance(self):
        for cc in range(self.num_agents):
            plt.plot([x[self.ind_pos].varValue for x in self.avar_x[cc]],
                     [x[self.ind_spd].varValue for x in self.avar_x[cc]])
        plt.grid()
        plt.show()

def car_test():
    dt = 0.5
    car_a = np.array([[1,dt],[0,1]])
    car_b = np.array([[0.5*dt*dt],[dt]])
    cp = CarPlan(car_a, car_b, num_steps=10, num_cars=2)
    cp.setInitialState(np.array([0,0.8,0,3]))
    cp.objective+=-1.0*cp.avar_x[0][-1][cp.ind_pos]
    cp.objective+=-1.0*cp.avar_x[1][-1][cp.ind_pos]
    cp.addInfNormStageCost(np.zeros([1,4]),0.001*np.array([1,0]))
    cp.addInfNormStageCost(np.zeros([1,4]),0.001*np.array([0,1]))
    amax = 0.5*9.81
    #cp.addStageConstraints(np.zeros([4,4]), np.array([[1,0],[-1,0],[0,1],[0,-1]]), [amax,amax,amax,amax])
    #cp.addStageConstraints(np.zeros([2, 2]), np.array([[1], [-1]]), [amax, amax], agent=1)
    cp.addStageConstraints(np.zeros([2, 2]), np.array([[1], [-1]]), [amax, amax], agent='all')
    cp.addSpeedRestriction(5.0,15,20,1)
    cp.addCrossingConstraint(0,15,25,1,10,20)
    #print cp
    #cp.solveByBranchBound()
    cp.solveByMILP(M=1000)
    print cp.objective.value()
    cp.plotStateHistory()
    #cp.plotSpeedOverDistance()
    return(cp)

if __name__=="__main__":
    car_test()
