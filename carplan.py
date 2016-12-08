from ltraj import LTrajAvoid
import numpy as np

class CarPlan(LTrajAvoid):

    def __init__(self, car_a, car_b, num_steps, num_cars, ind_pos=0, ind_spd=1, min_spd=0.0, max_spd=8.94, name="CarPlan", sense=1):
        self.ind_pos = ind_pos
        self.ind_spd = ind_spd
        LTrajAvoid.__init__(self, car_a, car_b, num_steps, name, sense, num_agents=num_cars)
	for cc in range(num_cars):
		for kk in range(self.Nt):
		    self.avar_x[cc][kk][ind_spd].lowBound = min_spd
		    self.avar_x[cc][kk][ind_spd].upBound = max_spd
		self.avar_x[cc][self.Nt][ind_spd].lowBound = min_spd	    
		self.avar_x[cc][self.Nt][ind_spd].upBound = max_spd

def car_test():
    dt = 0.2
    car_a = np.array([[1,dt],[0,1]])
    car_b = np.array([[0.5*dt*dt],[dt]])
    cp = CarPlan(car_a, car_b, num_steps=5, num_cars=2)
    cp.setInitialState(np.array([0,0.8,0,0.4]))
    cp.objective+=-1.0*cp.avar_x[0][-1][cp.ind_pos]
    cp.objective+=-1.0*cp.avar_x[1][-1][cp.ind_pos]
    amax = 0.5*9.81
    # cp.addStageConstraints(np.array([[0,0],[0,0]]), np.array([[1],[-1]]), [amax,amax])
    #print cp
    cp.solve()
    cp.plotStateHistory()
    return(cp)

if __name__=="__main__":
    car_test()
