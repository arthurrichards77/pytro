import numpy as np
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
Nc=20
M = np.transpose(np.vstack((np.cos(2*np.pi*np.array(range(Nc))/Nc),np.sin(2*np.pi*np.array(range(Nc))/Nc))))

# maximum acceleration
max_acc = 0.4
C = np.zeros((Nc,4))
D = M
e = max_acc*np.ones(Nc)
print(C)
print(D)
print(e)
p.addStageConstraints(C,D,e)

# set initial state
p.setInitialState([0.0,0.0,1.0,0.0])

# temp - set fixed terminal state
term_state = np.array([5.0,5.0,1.0,0.0])
p.setTerminalState(term_state)

# add acceleration cost
p.add2NormStageCost(np.zeros((2,4)),np.eye(2))

print(p)
p.solveByMILP()
p.plotTraj2D()
p.plotStateControlHistory()
