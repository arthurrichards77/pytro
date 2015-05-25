import mylinprog as lp
import matplotlib.pyplot as plt

#lp.test()

# new LP problem
p = lp.lp()

# time steps
Nt = 10
dt = 0.25
amax = 2

# set up vars
x=[p.newvar((-3.,3.))]
y=[p.newvar((-3.,3.))]
vx=[p.newvar()]
vy=[p.newvar()]
ax=[]
ay=[]
mx=[]
my=[]
for kk in range(Nt):
    # accelerations
    ax.append(p.newvar())
    ay.append(p.newvar())
    # acceleration magnitudes
    mx.append(p.newvar((0,amax)))
    my.append(p.newvar((0,amax)))    
    # positions
    x.append(p.newvar((-3.,3.)))
    y.append(p.newvar((-3.,3.)))
    # velocities
    vx.append(p.newvar())
    vy.append(p.newvar())
    
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
p.setobj(sum(mx)+sum(my))

#print(p.solve())

# gather the results
xopt=[r.result() for r in x]
yopt=[r.result() for r in y]

# box obstacle
obs = [0.75, 1.0, 0.5, 1.0]

# plot
plt.plot(xopt,yopt,'b-',[obs[0],obs[0],obs[1],obs[1],obs[0]],[obs[2],obs[3],obs[3],obs[2],obs[2]],'r-')
plt.show()

# list of time steps to be checked
