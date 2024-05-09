import numpy as np
import pulp
import ltraj

# 2D point mass dynamics
dt = 2.0
A = np.vstack(( np.hstack(( np.eye(2),np.eye(2)*dt )), 
                np.hstack(( np.zeros((2,2)),np.eye(2) )) ))
B = np.vstack((np.eye(2)*0.5*dt*dt,np.eye(2)*dt))

# basic optimizer
num_steps = 30
p = ltraj.LTraj2DAvoid(A,B,num_steps)

# for circle constraints
Nc=12
M = np.transpose(np.vstack((np.cos(2*np.pi*np.array(range(Nc))/Nc),np.sin(2*np.pi*np.array(range(Nc))/Nc))))

# maximum acceleration
max_acc = 0.294/5.0
# max_acc = 0.5*max_acc # reduced turn capability should result in different route
Ca = np.zeros((Nc,4))
Da = M
ea = max_acc*np.ones(Nc)
p.addStageConstraints(Ca,Da,ea)

# maximum speed
max_spd = 0.225
Cv = np.hstack((np.zeros((Nc,2)),M))
Dv = np.zeros((Nc,2))
ev = max_spd*np.ones(Nc)
p.addStageConstraints(Cv,Dv,ev)

# set initial state
p.setInitialState([-5.0,4.0,0.2,0.0])

# minimum time objective
term_pos = np.array([5.0,5.0])
# weights
w_time = 1.0
w_dist = 10.0
cost_var = pulp.LpVariable('J')
p.addUnionConstraint(tuple([w_time*kk+w_dist*np.dot(M,(p.var_x[kk][0:2]-term_pos))-cost_var for kk in range(num_steps)]),name="fin")
p.objective+=cost_var

# add acceleration cost
p.add2NormStageCost(np.zeros((2,4)),0.001*np.eye(2),Nc=17)

# add obstacles
p.addStatic2DObst(-4,-3,3.5,6)
p.addStatic2DObst(0,1,0,5)
p.addStatic2DObst(2,3,3,9)

print(p)
p.solveByMILP(M=100)
p.plotTraj2D()
p.plotStateControlHistory()
