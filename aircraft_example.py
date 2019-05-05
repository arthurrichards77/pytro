import numpy as np
import pulp
import ltraj

# 2D point mass dynamics
dt = 1.0
A = np.vstack(( np.hstack(( np.eye(2),np.eye(2)*dt )), 
                np.hstack(( np.zeros((2,2)),np.eye(2) )) ))
B = np.vstack((np.eye(2)*0.5*dt*dt,np.eye(2)*dt))

# basic optimizer
num_steps = 10
p = ltraj.LTraj2DAvoid(A,B,num_steps)

# for circle constraints
Nc=19
M = np.transpose(np.vstack((np.cos(2*np.pi*np.array(range(Nc))/Nc),np.sin(2*np.pi*np.array(range(Nc))/Nc))))

# maximum acceleration
max_acc = 0.4
C = np.zeros((Nc,4))
D = M
e = max_acc*np.ones(Nc)
p.addStageConstraints(C,D,e)

# set initial state
p.setInitialState([0.0,0.0,1.0,0.0])

# temp - set fixed terminal state
#term_state = np.array([5.0,5.0,1.0,0.0])
#p.setTerminalState(term_state)

# minimum time objective
term_pos = np.array([15.0,10.0])
# weights
w_time = 1.0
w_dist = 10.0
cost_var = pulp.LpVariable('J')
p.addUnionConstraint(tuple([w_time*kk+w_dist*np.dot(M,(p.var_x[kk][0:2]-term_pos))-cost_var for kk in range(num_steps)]),name="fin")
p.objective+=cost_var

# add acceleration cost
p.add2NormStageCost(np.zeros((2,4)),0.001*np.eye(2),Nc=17)

print(p)
p.solveByMILP(M=1000)
p.plotTraj2D()
p.plotStateControlHistory()
