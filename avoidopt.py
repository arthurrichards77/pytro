import trajlp as tlp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class bbnode:

    def __init__(self,trajlp,bound,steps):
        self.lp = trajlp
        self.bound = bound
        self.steps = steps
        self.id = ''

    def solve(self):
        print self.id
        print self.steps
        print self.lp.lp.bounds
        self.result = self.lp.solve()
        print self.result
        if self.result.status==0:
            self.trajx = self.lp.trajx()
            self.trajy = self.lp.trajy()
            assert self.result.fun>=self.bound, "Bound went down on solve: %f < %f." % (self.result.fun,self.bound)
            self.bound = self.result.fun
        return self.result

    def restrict(self,step,box):
        cxbounds=self.lp.lp.getbounds(self.lp.x[step])
        cybounds=self.lp.lp.getbounds(self.lp.y[step])
        nxbounds=(np.maximum(cxbounds[0],box[0]),
                  np.minimum(cxbounds[1],box[1]))
        nybounds=(np.maximum(cybounds[0],box[2]),
                  np.minimum(cybounds[1],box[3]))
        self.lp.lp.setbounds(self.lp.x[step],nxbounds)
        self.lp.lp.setbounds(self.lp.y[step],nybounds)

class avoidopt:

    def __init__(self,Nt=10,dt=0.2,amax=2,
                 pstart=[0.0,0.0,0.0,0.0], pgoal=[1.5,1.0,0.0,0.0],
                 xbounds=(-np.inf,np.inf),ybounds=(-np.inf,np.inf),
                 obs = [0.45, 1.0, 0.25, 0.6],
                 maxsolves=20):
        rootlp = tlp.trajlp(Nt,dt,amax,pstart,pgoal,xbounds,ybounds)
        self.bblist = [bbnode(rootlp,-np.inf,range(Nt))]
        inccost=np.inf

        for ii in range(maxsolves):
            if len(self.bblist)<1:
                break
            print("Num active nodes = %i" % len(self.bblist))
            # depth first - so grab last node
            thisProb = self.bblist.pop()
            # check if it still needs solving
            if thisProb.bound>inccost:
                # fathomed
                print("LP bound above incumbent")
                continue
            # solve the thing
            try:
                thisProb.solve()
            except ValueError:
                print("ValueError!!")
                continue
            # if it was infeasible
            if thisProb.result.status>0:
                # also fathomed
                print("Infeasible")
                continue
            if thisProb.result.fun>inccost:
                # fathomed again
                print("LP result above incumbent")
                continue
            # now check for unsatisfied avoidance constraints
            for kk in thisProb.steps:
                # is it in the box?
                if thisProb.trajx[kk]<obs[0] and thisProb.trajx[kk+1]<obs[0]:
                    print "Step %i clear left" % kk
                    continue
                elif thisProb.trajx[kk]>obs[1] and thisProb.trajx[kk+1]>obs[1]:
                    print "Step %i clear right" % kk
                    continue
                elif thisProb.trajy[kk]<obs[2] and thisProb.trajy[kk+1]<obs[2]:
                    print "Step %i clear down" % kk
                    continue
                elif thisProb.trajy[kk]>obs[3] and thisProb.trajy[kk+1]>obs[3]:
                    print "Step %i clear up" % kk
                    continue
                else:
                    print("Incursion step %i" % kk)
                    # I'm inside - branch on the first one found
                    # make the new list of steps
                    thisProb.steps.remove(kk)
                    # four new subproblems - left
                    p1 = deepcopy(thisProb)
                    p1.restrict(kk,[-np.inf,obs[0],-np.inf,np.inf])
                    p1.restrict(kk+1,[-np.inf,obs[0],-np.inf,np.inf])
                    p1.id = p1.id + "%iL" % kk
                    self.bblist.append(p1)
                    # right
                    p2 = deepcopy(thisProb)
                    p2.restrict(kk,[obs[1],np.inf,-np.inf,np.inf])
                    p2.restrict(kk+1,[obs[1],np.inf,-np.inf,np.inf])
                    p2.id = p2.id + "%iR" % kk
                    self.bblist.append(p2)
                    # down
                    p3 = deepcopy(thisProb)
                    p3.restrict(kk,[-np.inf,np.inf,-np.inf,obs[2]])
                    p3.restrict(kk+1,[-np.inf,np.inf,-np.inf,obs[2]])
                    p3.id = p3.id + "%iD" % kk
                    self.bblist.append(p3)
                    # up
                    p4 = deepcopy(thisProb)
                    p4.restrict(kk,[-np.inf,np.inf,obs[3],np.inf])
                    p4.restrict(kk+1,[-np.inf,np.inf,obs[3],np.inf])
                    p4.id = p4.id + "%iU" % kk
                    self.bblist.append(p4)
                    # append them to the list
                    # break out of the for loop checking constraints
                    break
            else:
                #if I got through the loop, this is feasible for avoidance
                print("Feasible with cost %f" % thisProb.result.fun)
                if thisProb.result.fun<inccost:
                    # got a new incumbent
                    inccost=thisProb.result.fun
                    self.inctrajx=thisProb.trajx
                    self.inctrajy=thisProb.trajy
                # plot
                plt.plot(self.inctrajx,self.inctrajy,'.b-',
                         [obs[0],obs[0],obs[1],obs[1],obs[0]],[obs[2],obs[3],obs[3],obs[2],obs[2]],'r-')
                plt.show()

def test():
    testobs = [0.45, 1.01, 0.25, 1.6]
    res = avoidopt(obs=testobs,Nt=4,dt=1.0,xbounds=(-2.0,3.0))
    #plot
    plt.plot(res.inctrajx,res.inctrajy,'sb-',
             [testobs[0],testobs[0],testobs[1],testobs[1],testobs[0]],[testobs[2],testobs[3],testobs[3],testobs[2],testobs[2]],'r-')
    plt.show()
