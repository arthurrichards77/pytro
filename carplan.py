from ltraj import LTrajAvoid
import pulp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

class CarPlan(LTrajAvoid):

    def __init__(self, car_a, car_b, num_steps, num_cars=1, ind_pos=0, ind_spd=1, min_spd=0.0, max_spd=8.94, name="CarPlan", sense=1):
        self.ind_pos = ind_pos # which state is position
        self.ind_spd = ind_spd # which state is velocity
        LTrajAvoid.__init__(self, car_a, car_b, num_steps, name, sense, num_agents=num_cars)
        # provide 'shortcuts' to position and speed variables
        self.pos = [[self.avar_x[cc][kk][ind_pos] for kk in range(0,self.Nt+1)] for cc in range(num_cars)]
        self.spd = [[self.avar_x[cc][kk][ind_spd] for kk in range(0,self.Nt+1)] for cc in range(num_cars)]
        # apply hard speed limits as bounds on decision variables
        for cc in range(num_cars):
            self.avar_x[cc][self.Nt][ind_spd].lowBound = min_spd
            self.avar_x[cc][self.Nt][ind_spd].upBound = max_spd
            for kk in range(self.Nt):
                self.avar_x[cc][kk][ind_spd].lowBound = min_spd
                self.avar_x[cc][kk][ind_spd].upBound = max_spd

    def addSpeedRestriction(self, spd, pos_start, pos_finish, car=0):
        """Limit speed to 'spd' for specified car (0 if omitted) between positions [pos_start,pos_finish].
        Used to accommodate lateral acceleration limits while going round bends."""
        for kk in range(self.Nt):
            self.addUnionConstraint(([self.pos[car][kk] - pos_start,
                                      self.pos[car][kk+1] - pos_start],
                                     [pos_finish - self.pos[car][kk],
                                      pos_finish - self.pos[car][kk+1]],
                                     [self.spd[car][kk] - spd,
                                      self.spd[car][kk+1] - spd]))

    def addCrossingConstraint(self,car_1,start_pos_1,end_pos_1,car_2,start_pos_2,end_pos_2,gap_time = 0.0):
        for kk in range(self.Nt):
            self.addUnionConstraint(([self.pos[car_1][kk]+self.spd[car_1][kk]*gap_time - start_pos_1,
                                      self.pos[car_1][kk+1]+self.spd[car_1][kk+1]*gap_time - start_pos_1],
                                     [end_pos_1 - self.pos[car_1][kk],
                                      end_pos_1 - self.pos[car_1][kk+1]],
                                     [self.pos[car_2][kk]+self.spd[car_2][kk]*gap_time - start_pos_2,
                                      self.pos[car_2][kk+1]+self.spd[car_2][kk+1]*gap_time - start_pos_2],
                                     [end_pos_2 - self.pos[car_2][kk],
                                      end_pos_2 - self.pos[car_2][kk+1]]))

    def addConflictConstraint(self,car_1,car_2,conflict_obst,gap_time = 0.0):
        """Avoid collison between car 1 and car 2, which may be on different routes.  Conflict obstacle is given as (d1lo,d1hi,d2lo,d2hi,D12lo,D12hi) where 
        collision is avoided if any **one** of following is satisfied: d1<d1lo, d1>d1hi, d2<d2lo, d2>d2hi, d2-d1<D12lo, d2-d1>D12hi ."""
        d1lo,d1hi,d2lo,d2hi,D12lo,D12hi = conflict_obst
        for kk in range(self.Nt):
            self.addUnionConstraint(([self.pos[car_1][kk]+self.spd[car_1][kk]*gap_time - d1lo,
                                      self.pos[car_1][kk+1]+self.spd[car_1][kk+1]*gap_time - d1lo],
                                     [d1hi - self.pos[car_1][kk],
                                      d1hi - self.pos[car_1][kk+1]],
                                     [self.pos[car_2][kk]+self.spd[car_2][kk]*gap_time - d2lo,
                                      self.pos[car_2][kk+1]+self.spd[car_2][kk+1]*gap_time - d2lo],
                                     [d2hi - self.pos[car_2][kk],
                                      d2hi - self.pos[car_2][kk+1]],
                                     [self.pos[car_2][kk]+self.spd[car_2][kk]*gap_time - self.pos[car_1][kk] - D12lo,
                                      self.pos[car_2][kk+1]+self.spd[car_2][kk+1]*gap_time - self.pos[car_1][kk+1] - D12lo],
                                     [D12hi - self.pos[car_2][kk] + self.pos[car_1][kk]+self.spd[car_1][kk]*gap_time,
                                      D12hi - self.pos[car_2][kk+1] + self.pos[car_1][kk+1]+self.spd[car_1][kk+1]*gap_time]))

    def plotSpeedOverDistance(self):
        for cc in range(self.num_agents):
            plt.plot([x.varValue for x in self.pos[cc]],
                     [x.varValue for x in self.spd[cc]])
        plt.grid()
        plt.show()

def point_mass_2d_matrices(dt):
    A = np.array([[1,dt],[0,1]])
    B = np.array([[0.5*dt*dt],[dt]])
    return(A,B)

def car_speed_test():
    dt = 0.5
    car_a,car_b = point_mass_2d_matrices(dt)
    cp = CarPlan(car_a, car_b, num_steps=10)
    cp.setInitialState(np.array([0,0.8]))
    cp.objective+=-1.0*cp.pos[0][-1]
    cp.addInfNormStageCost(np.zeros([1,2]),0.001*np.array([1]))
    amax = 0.5*9.81
    cp.addStageConstraints(np.zeros([2, 2]), np.array([[1], [-1]]), [amax, amax], agent='all')
    cp.addSpeedRestriction(5.0,15,20)
    #cp.solveByBranchBound()
    #cp.solveByMILP(M=1000)
    #cp.solveByBranchBound(solver=pulp.GUROBI(msg=0))
    cp.solveByMILP(M=1000,solver=pulp.GUROBI())
    print cp.objective.value()
    print cp.solve_time
    cp.plotStateControlHistory()
    cp.plotSpeedOverDistance()
    return(cp)

def car_cross_test():
    # dynamics
    dt = 0.5
    car_a,car_b = point_mass_2d_matrices(dt)
    cp = CarPlan(car_a, car_b, num_steps=10, num_cars=2)
    cp.setInitialState(np.array([5,0.8,0,0.8]))
    cp.objective+=-1.0*cp.pos[0][-1]
    cp.objective+=-0.5*cp.pos[1][-1]
    cp.addInfNormStageCost(np.zeros([1,2]),0.001*np.array([1]),agent='all')
    amax = 0.5*9.81
    cp.addStageConstraints(np.zeros([2, 2]), np.array([[1], [-1]]), [amax, amax], agent='all')
    # conflict constraints
    cp.addCrossingConstraint(0,15,20,1,10,15,gap_time=0.0)
    # solve it
    #cp.solveByBranchBound()
    #cp.solveByMILP(M=1000)
    #cp.solveByBranchBound(solver=pulp.GUROBI(msg=0))
    cp.solveByMILP(M=1000,solver=pulp.GUROBI())
    # show results
    print cp.objective.value()
    print cp.solve_time
    cp.plotStateControlHistory()
    # plot positions against eachother
    plt.plot([x.varValue for x in cp.pos[0]], [x.varValue for x in cp.pos[1]],'x-')
    plt.plot([15, 20, 20, 15, 15],[10, 10, 15, 15, 10],'r-')
    plt.grid()
    plt.show()
    return(cp)

def car_conf_test():
    # basic car dynamics
    dt = 0.5
    car_a,car_b = point_mass_2d_matrices(dt)
    cp = CarPlan(car_a, car_b, num_steps=10, num_cars=2)
    cp.setInitialState(np.array([5,0.8,0,0.8]))
    # objective weights set priorities
    cp.objective+=-1.0*cp.pos[0][-1]
    cp.objective+=-1.5*cp.pos[1][-1]
    # small weight on running progress
    cp.objective += -0.01*sum(cp.pos[0])
    cp.objective += -0.01*sum(cp.pos[1])
    # and weight on acceleration
    cp.addInfNormStageCost(np.zeros([1,2]),0.001*np.array([1]),agent='all')
    amax = 0.5*9.81
    cp.addStageConstraints(np.zeros([2, 2]), np.array([[1], [-1]]), [amax, amax], agent='all')
    # single crossing constraint - routes cross over
    cp.addConflictConstraint(0,1,(15,20,10,15,-10,0),gap_time=2.0) # should be same as crossing test
    # various ways to solve
    #cp.solveByBranchBound()
    #cp.solveByMILP(M=1000)
    #cp.solveByBranchBound(solver=pulp.GUROBI(msg=0))
    cp.solveByMILP(M=1000,solver=pulp.GUROBI())
    # show results
    print cp.objective.value()
    print cp.solve_time
    cp.plotStateControlHistory()    
    # plot positions against eachother
    plt.plot([x.varValue for x in cp.pos[0]], [x.varValue for x in cp.pos[1]],'x-')
    plt.plot([15, 20, 20, 15, 15],[10, 10, 15, 15, 10],'r-')
    plt.grid()
    plt.show()
    return(cp)

if __name__=="__main__":
    car_cross_test()
