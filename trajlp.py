import mylinprog
import matplotlib.pyplot as plt
import numpy as np

class trajlp(object):

    def __init__(self,Nt=10,dt=0.2,amax=2,
                 pstart=[0.0,0.0,0.0,0.0], pgoal=[1.5,1.0,0.0,0.0],
                 xbounds=(-np.inf,np.inf),ybounds=(-np.inf,np.inf)):
        # store parameters
        self.Nt=Nt
        self.dt=dt
        self.amax=amax
        self.pstart=pstart
        self.pgoal=pgoal
        # make a new LP
        lp = mylinprog.lp()
        # set up vars
        x=[lp.newvar(xbounds)]
        y=[lp.newvar(ybounds)]
        vx=[lp.newvar()]
        vy=[lp.newvar()]
        ax=[]
        ay=[]
        mx=[]
        my=[]
        for kk in range(Nt):
            # accelerations
            ax.append(lp.newvar())
            ay.append(lp.newvar())
            # acceleration magnitudes
            mx.append(lp.newvar((0,amax)))
            my.append(lp.newvar((0,amax)))    
            # positions
            x.append(lp.newvar(xbounds))
            y.append(lp.newvar(ybounds))
            # velocities
            vx.append(lp.newvar())
            vy.append(lp.newvar())
            
        # dynamics constraints
        for kk in range(Nt):
            lp.addeqcon(x[kk]+dt*vx[kk]+dt*dt*0.5*ax[kk]-x[kk+1],0)
            lp.addeqcon(y[kk]+dt*vy[kk]+dt*dt*0.5*ay[kk]-y[kk+1],0)
            lp.addeqcon(vx[kk]+dt*ax[kk]-vx[kk+1],0)
            lp.addeqcon(vy[kk]+dt*ay[kk]-vy[kk+1],0)
            # grab magitudes
            lp.addineq(-mx[kk]-ax[kk],0)
            lp.addineq(-my[kk]-ay[kk],0)
            lp.addineq(-mx[kk]+ax[kk],0)
            lp.addineq(-my[kk]+ay[kk],0)

        # initial constraints
        lp.addeqcon(x[0],pstart[0])
        lp.addeqcon(y[0],pstart[1])
        lp.addeqcon(vx[0],pstart[2])
        lp.addeqcon(vy[0],pstart[3])

        # initial constraints
        lp.addeqcon(x[Nt],pgoal[0])
        lp.addeqcon(y[Nt],pgoal[1])
        lp.addeqcon(vx[Nt],pgoal[2])
        lp.addeqcon(vy[Nt],pgoal[3])

        # objective
        lp.setobj(sum(mx)+sum(my))

        # store what I need
        self.x=x
        self.y=y
        self.lp=lp

    def solve(self):
        self.result = self.lp.solve()
        return self.result

    def trajx(self):
        resx = [self.lp.varresult(v) for v in self.x]
        return resx

    def trajy(self):
        resy = [self.lp.varresult(v) for v in self.y]
        return resy

    def traj(self):
        return [self.trajx(),self.trajy()]
                 
def test():
    mytrajlp = trajlp()
    #mytrajlp = trajlp(Nt=20,amax=5,pgoal=[2.0,2.0,0.0,0.5])
    print(mytrajlp.solve())
    mytraj = mytrajlp.traj()
    # plot
    plt.plot(mytraj[0],mytraj[1],'.b-')
    plt.show()

