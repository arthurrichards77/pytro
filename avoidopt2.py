import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import copy

class LpTraj:

    def __init__(self,Nt=5,dt=0.5,amax=2.0,
                 pstart=[0.0,0.0,0.0,0.0], pgoal=[1.5,1.0,0.0,0.5],
                 xbounds=(-10.0,10.0),ybounds=(-10.0,10.0)):
        self.Nt = Nt
        self.dt = dt
        self.amax = amax
        self.pstart = pstart
        self.pgoal = pgoal
        # make a new LP
        self.lp = LpProblem("LpTraj",LpMinimize)
        # set up var lists
        x=[LpVariable("x0",xbounds[0],xbounds[1])]
        y=[LpVariable("y0",ybounds[0],ybounds[1])]
        vx=[LpVariable("vx0")]
        vy=[LpVariable("vy0")]
        ax=[]
        ay=[]
        mx=[]
        my=[]
        dx=[]
        dy=[]
        for kk in range(Nt):
            # accelerations
            ax.append(LpVariable("ax%i" % kk,-amax,amax))
            ay.append(LpVariable("ay%i" % kk,-amax,amax))
            # acceleration magnitudes
            mx.append(LpVariable("mx%i" % kk,0.0,amax))
            my.append(LpVariable("my%i" % kk,0.0,amax))    
            # positions
            x.append(LpVariable("x%i" % (kk+1),xbounds[0],xbounds[1]))
            y.append(LpVariable("y%i" % (kk+1),ybounds[0],ybounds[1]))
            # velocities
            vx.append(LpVariable("vx%i" % (kk+1)))
            vy.append(LpVariable("vy%i" % (kk+1)))
            # distances
            dx.append(LpVariable("dx%i" % kk,0.0,xbounds[1]-xbounds[0]))
            dy.append(LpVariable("dy%i" % kk,0.0,ybounds[1]-ybounds[0]))    
            
        # dynamics constraints
        for kk in range(Nt):
            self.lp += (x[kk]+dt*vx[kk]+dt*dt*0.5*ax[kk]==x[kk+1])
            self.lp += (y[kk]+dt*vy[kk]+dt*dt*0.5*ay[kk]==y[kk+1])
            self.lp += (vx[kk]+dt*ax[kk]==vx[kk+1])
            self.lp += (vy[kk]+dt*ay[kk]==vy[kk+1])
            # grab magitudes
            self.lp += (mx[kk]>=ax[kk])
            self.lp += (mx[kk]>=-ax[kk])
            self.lp += (my[kk]>=ay[kk])
            self.lp += (my[kk]>=-ay[kk])
            # distance magnitudes
            self.lp += (dx[kk]>=x[kk+1]-x[kk])
            self.lp += (dx[kk]>=x[kk]-x[kk+1])
            self.lp += (dy[kk]>=y[kk+1]-y[kk])
            self.lp += (dy[kk]>=y[kk]-y[kk+1])

        # initial constraints
        self.lp+=(x[0]==pstart[0])
        self.lp+=(y[0]==pstart[1])
        self.lp+=(vx[0]==pstart[2])
        self.lp+=(vy[0]==pstart[3])

        # initial constraints
        self.lp+=(x[Nt]==pgoal[0])
        self.lp+=(y[Nt]==pgoal[1])
        self.lp+=(vx[Nt]==pgoal[2])
        self.lp+=(vy[Nt]==pgoal[3])

        # objective
        self.lp += 0.001*(sum(mx)+sum(my))/self.amax + (sum(dx)+sum(dy))

	# store the decision variables
	# only the positions needed for now
	self.x=x
	self.y=y

    def solve(self):
        self.result = self.lp.solve()
        self.xvalue=[xv.varValue for xv in self.x]
        self.yvalue=[yv.varValue for yv in self.y]
        self.objValue = self.lp.objective.value()
        return(self.result)

    def plot(self):
        plt.plot(self.xvalue,self.yvalue,'.b-')
        plt.show()

def lpTest():
    test = LpTraj()
    test.solve()
    print test.objValue
    test.plot()

class ObstStep:

    def __init__(self,obsBox,timeStep):
        self.obsBox = obsBox
        self.timeStep = timeStep

class BbNode:

    def __init__(self,trajlp,bound,obststeps,verbosity=2):
        self.trajlp = trajlp
        self.bound = bound
        self.obststeps = obststeps
        self.verbosity = verbosity
        self.id = ''

    def solve(self):
        self.result = self.trajlp.solve()
        self.bound = self.trajlp.objValue
        if self.verbosity>=2:
            print self.id + (" LP result = %i" % self.result)
        return self.result

    def deepcopy(self):
        newNode = copy.copy(self)
        # need fresh list of steps to check
        newNode.obststeps = copy.copy(self.obststeps)
	# need to do this specially so it doesn't break PuLP
        newNode.trajlp = copy.copy(self.trajlp)
        newNode.trajlp.lp = newNode.trajlp.lp.deepcopy()
        return(newNode)

class AvoidOpt:

    def __init__(self,Nt=5,dt=0.5,amax=2,
                 pstart=[0.0,0.0,0.0,0.0], pgoal=[1.5,1.0,0.0,0.0],
                 xbounds=(-10.0,10.0),ybounds=(-10.0,10.0)):
        self.obststeps = []
        self.Nt = Nt
        self.amax = amax
	# create the completely relaxed LP and make it the only active node
        self.rootlp = LpTraj(Nt,dt,amax,pstart,pgoal,xbounds,ybounds)

    def addStaticObstacle(self,obsBox):
        # needs to be added "for all time steps"
        self.obststeps += [ObstStep(obsBox,kk) for kk in range(self.Nt)]
        # hack to ensure plotting works during testing
        self.obs = obsBox

    def solve(self, maxsolves=1000, verbosity=1):
        # initialize branch and bound tree with single root node
        self.bblist = [BbNode(self.rootlp,-np.inf,self.obststeps,verbosity)]
        inccost=np.inf

        for ii in range(maxsolves):
            if verbosity>=1:
                print("Node %i : inc %f : node list %i" % (ii, inccost, len(self.bblist)))
            if len(self.bblist)<1:
                print "Optimal solution found after %i nodes." % ii
                break
            if verbosity>=2:
                 print("Num active nodes = %i" % len(self.bblist))
            # depth first - so grab last node
            thisNode = self.bblist.pop()
            # check if it still needs solving
            if thisNode.bound>inccost:
                # fathomed
                if verbosity>=2:
                    print("LP bound above incumbent")
                continue
            # solve the thing
            thisNode.solve()
            # if it was infeasible
            if thisNode.result<0:
                # also fathomed
                if verbosity>=2:
                    print("Infeasible")
                continue
            if thisNode.bound>inccost:
                # fathomed again
                if verbosity>=2:
                    print("LP result above incumbent")
                continue
            # now check for unsatisfied avoidance constraints
	    if verbosity>=2:
                print "Checking steps" 	        
            for ob in thisNode.obststeps:
                obs = ob.obsBox
                kk = ob.timeStep
                # is it in the box?
                if thisNode.trajlp.x[kk].varValue<obs[0] and thisNode.trajlp.x[kk+1].varValue<obs[0]:
                    if verbosity>=2:
                        print "Step %i clear left" % kk
                    continue
                elif thisNode.trajlp.x[kk].varValue>obs[1] and thisNode.trajlp.x[kk+1].varValue>obs[1]:
                    if verbosity>=2:
                        print "Step %i clear right" % kk
                    continue
                elif thisNode.trajlp.y[kk].varValue<obs[2] and thisNode.trajlp.y[kk+1].varValue<obs[2]:
                    if verbosity>=2:
                        print "Step %i clear down" % kk
                    continue
                elif thisNode.trajlp.y[kk].varValue>obs[3] and thisNode.trajlp.y[kk+1].varValue>obs[3]:
                    if verbosity>=2:
                        print "Step %i clear up" % kk
                    continue
                else:
                    if verbosity>=2:
                        print("Incursion step %i" % kk)
                    # I'm inside - branch on the first one found
                    # make the new list of steps
                    thisNode.obststeps.remove(ob)
                    # four new subproblems - left
                    p1 = thisNode.deepcopy()
                    p1.trajlp.lp.addConstraint(p1.trajlp.x[kk]<=obs[0])
                    p1.trajlp.lp.addConstraint(p1.trajlp.x[kk+1]<=obs[0])
                    if verbosity>=2:
                        p1.id = p1.id + "%iL" % kk
                    self.bblist.append(p1)
                    # right
                    p2 = thisNode.deepcopy()
                    p2.trajlp.lp.addConstraint(p2.trajlp.x[kk]>=obs[1])
                    p2.trajlp.lp.addConstraint(p2.trajlp.x[kk+1]>=obs[1])
                    if verbosity>=2:
                        p2.id = p2.id + "%iR" % kk
                    self.bblist.append(p2)
                    # down
                    p3 = thisNode.deepcopy()
                    p3.trajlp.lp.addConstraint(p3.trajlp.y[kk]<=obs[2])
                    p3.trajlp.lp.addConstraint(p3.trajlp.y[kk+1]<=obs[2])
                    if verbosity>=2:
                        p3.id = p3.id + "%iD" % kk
                    self.bblist.append(p3)
                    # up
                    p4 = thisNode.deepcopy()
                    p4.trajlp.lp.addConstraint(p4.trajlp.y[kk]>=obs[3])
                    p4.trajlp.lp.addConstraint(p4.trajlp.y[kk+1]>=obs[3])
                    if verbosity>=2:
                        p4.id = p4.id + "%iU" % kk
                    self.bblist.append(p4)
                    # append them to the list
                    # break out of the for loop checking constraints
                    break
            else:
                #if I got through the loop, this is feasible for avoidance
                if verbosity>=2:
                    print("Feasible with cost %f" % thisNode.bound)
                if thisNode.bound<inccost:
                    # got a new incumbent
                    if verbosity>=2:
                        print "New incumbent"
                    inccost=thisNode.bound
                    self.inctrajx=[xv.varValue for xv in thisNode.trajlp.x]
                    self.inctrajy=[yv.varValue for yv in thisNode.trajlp.y]
                # plot
                #plt.plot(self.inctrajx,self.inctrajy,'.b-',
                #         [obs[0],obs[0],obs[1],obs[1],obs[0]],[obs[2],obs[3],obs[3],obs[2],obs[2]],'r-')
                #plt.show()
        else:
            print "Node limit of %i reached" % ii

    def plot(self):
        plt.plot(self.inctrajx,self.inctrajy,'sb-')
        for ob in self.obststeps:
            obs = ob.obsBox
            plt.plot([obs[0],obs[0],obs[1],obs[1],obs[0]],[obs[2],obs[3],obs[3],obs[2],obs[2]],'r-')
        plt.show()

def test():
    opt = AvoidOpt(Nt=6,dt=1.0,pgoal=[2.4,0.5,0.0,-0.5],amax=100)
    testobs = [0.45, 1.0, -0.25, 1.9]
    opt.addStaticObstacle(testobs)
    testobs = [1.3, 1.9, -0.95, 0.1]
    opt.addStaticObstacle(testobs)
    opt.solve(verbosity=0)
    opt.plot()

if __name__=="__main__":
    test()
