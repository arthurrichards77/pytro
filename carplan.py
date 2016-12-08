from ltraj import LTrajAvoid
import numpy as np

class CarPlan(LTrajAvoid):

    def __init__(self, car_a, car_b, num_steps, num_cars, ind_pos=0, ind_spd=1, name="CarPlan", sense=1):
        self.ind_pos = ind_pos
        self.ind_spd = ind_spd
        sys_a = np.kron(np.eye(num_cars),car_a)
        sys_b = np.kron(np.eye(num_cars),car_b)
        LTrajAvoid.__init__(self, sys_a, sys_b, num_steps, name, sense)

if __name__=="__main__":
    dt = 1.0
    car_a = np.array([[1,dt],[0,1]])
    car_b = np.array([[0.5*dt*dt],[dt]])
    cp = CarPlan(car_a, car_b, num_steps=10, num_cars=1)
    cp.setInitialState(np.array([0,0.8]))
    cp.objective+=1.0*cp.var_x[-1][cp.ind_pos]
    cp.addStageConstraints(np.array([[0,0],[0,0],[0,1],[0,-1]]), np.array([[1],[-1],[0],[0]]), [1,1,1,1])
    #print cp
    cp.solve()
    cp.plotStateHistory()