import mylinprog as lp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#lp.test()

# new LP problem
p = lp.lp()

# time steps
Nt = 3
dt = 0.5
amax = 20

# set up vars
x=[p.newvar()]
y=[p.newvar()]
vx=[p.newvar()]
vy=[p.newvar()]
ax=[]
ay=[]
mx=[]
my=[]
sp=[]

for kk in range(Nt):
    # accelerations
    ax.append(p.newvar())
    ay.append(p.newvar())
    # acceleration magnitudes
    mx.append(p.newvar((0,amax)))
    my.append(p.newvar((0,amax)))    
    # positions
    x.append(p.newvar())
    y.append(p.newvar())
    # velocities
    vx.append(p.newvar())
    vy.append(p.newvar())
    # speeds
    sp.append(p.newvar((0,None)))
    
# dynamics constraints
for kk in range(Nt):
    p.addeqcon(x[kk]+dt*vx[kk]+dt*dt*0.5*ax[kk]-x[kk+1],0)
    p.addeqcon(y[kk]+dt*vy[kk]+dt*dt*0.5*ay[kk]-y[kk+1],0)
    p.addeqcon(vx[kk]+dt*ax[kk]-vx[kk+1],0)
    p.addeqcon(vy[kk]+dt*ay[kk]-vy[kk+1],0)
    # grab magitudes
    p.addineq(-mx[kk]-ax[kk],0)
    p.addineq(-my[kk]-ay[kk],0)
    p.addineq(-mx[kk]+ax[kk],0)
    p.addineq(-my[kk]+ay[kk],0)
    # speed
    for tt in range(8):
        theta=np.pi*2.0*(tt/8.0)
        p.addineq(-sp[kk]+np.cos(theta)*vx[kk]+np.sin(theta)*vy[kk],0)

# initial constraints
p.addeqcon(x[0],0)
p.addeqcon(y[0],0)
p.addeqcon(vx[0],0)
p.addeqcon(vy[0],0)

# initial constraints
p.addeqcon(x[Nt],1.5)
p.addeqcon(y[Nt],1)
p.addeqcon(vx[Nt],0)
p.addeqcon(vy[Nt],0)

# objective
p.setobj(sum(mx)+sum(my)+0.00000*sum(sp))

# box obstacle
obs = [0.25, 1.0, 0.45, 1.5]

### plot
##plt.plot(x,y,'b-',[obs[0],obs[0],obs[1],obs[1],obs[0]],[obs[2],obs[3],obs[3],obs[2],obs[2]],'r-')
##plt.show()

# incumbent
incBound = 3000

class subproblem:

    def __init__(self,lp,bound,steps):
        self.lp = lp
        self.bound = bound
        self.steps = steps

    def solve(self):
        return self.lp.solve()

bblist = [subproblem(p,2000,range(Nt))]
    
while len(bblist)>0:
    # depth first - so grab last node
    thisProb = bblist.pop()
    # check if it still needs solving
    if thisProb.bound>incBound:
        # fathomed
        continue
    # solve the thing
    newres=thisProb.solve()
    print newres
    # if it was infeasible
    if thisProb.lp.result.status>0:
        # also fathomed
        continue
    if thisProb.lp.result.fun>incBound:
        # fathomed again
        continue
    # now check for unsatisfied avoidance constraints
    for kk in thisProb.steps:
        # is it in the box?
        if x[kk].result(newres)<obs[0] and x[kk+1].result(newres)<obs[0]:
            continue
        elif x[kk].result(newres)>obs[1] and x[kk+1].result(newres)>obs[1]:
            continue
        elif y[kk].result(newres)<obs[2] and y[kk+1].result(newres)<obs[2]:
            continue
        elif y[kk].result(newres)>obs[3] and y[kk+1].result(newres)>obs[3]:
            continue
        else:
            # I'm inside - branch on the first one found
            # make the new list of steps
            newsteps = thisProb.steps
            newsteps.remove(kk)
            # get current bounds
            xbounds = thisProb.lp.bounds[x[kk].myvar-1]
            ybounds = thisProb.lp.bounds[y[kk].myvar-1]
            # four new subproblems
            p1 = deepcopy(thisProb)
            p1.lp.bounds[x[kk].myvar-1] = (xbounds[0],obs[0])
            p1.lp.bounds[x[kk+1].myvar-1] = (xbounds[0],obs[0])
            p2 = deepcopy(thisProb)
            p2.lp.bounds[y[kk].myvar-1] = (ybounds[0],obs[2])
            p2.lp.bounds[y[kk+1].myvar-1] = (ybounds[0],obs[2])
            p3 = deepcopy(thisProb)
            p3.lp.bounds[x[kk].myvar-1] = (obs[1],xbounds[1])
            p3.lp.bounds[x[kk+1].myvar-1] = (obs[1],xbounds[1])
            p4 = deepcopy(thisProb)
            p4.lp.bounds[y[kk].myvar-1] = (obs[3],ybounds[1])
            p4.lp.bounds[y[kk+1].myvar-1] = (obs[3],ybounds[1])
            # append them to the list
            bblist.append(p1)
            bblist.append(p2)
            bblist.append(p3)
            bblist.append(p4)
            # break out of the for loop checking constraints
            break
    else:
        #if I got through the loop, this is feasible for avoidance
        if newres.fun<incBound:
            # got a new incumbent
            incBound=newres.fun
            incSol=newres
                        
# got to here if it solved
# gather the results
x=[r.result(incSol) for r in x]
y=[r.result(incSol) for r in y]

# plot
plt.plot(x,y,'b-',[obs[0],obs[0],obs[1],obs[1],obs[0]],[obs[2],obs[3],obs[3],obs[2],obs[2]],'r-')
plt.show()
