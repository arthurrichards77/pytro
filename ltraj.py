import numpy as np
import pulp
import matplotlib.pyplot as plt
import copy
import time

class LpProbVectors(pulp.LpProblem):

    def newVarVector(self,name,num_elems):
        v = []
        for ii in range(num_elems):
            newvar = pulp.LpVariable("%s_%i" % (name,ii))
            v.append(newvar)
            self.addVariable(newvar)
        return v
    
    def addVecEqualZeroConstraint(self,vector_expression):
        for ee in vector_expression:
            self.addConstraint(ee==0.0)

    def addVecLessEqZeroConstraint(self,vector_expression):
        for ee in vector_expression:
            self.addConstraint(ee<=0.0)

    def addMaxVarConstraint(self,vector_expression):
        # adds decision variable to grab max(e)
        newvar = pulp.LpVariable("_max%i" % (self.numVariables()+1))
        for ee in vector_expression:
            self.addConstraint(ee-newvar<=0.0)
        return newvar

class LTraj(LpProbVectors):

    def __init__(self,A,B,Nt,name="NoName",sense=1):
        # store dynamics and horizon
        self.A = np.array(A)
        self.B = np.array(B)
        self.Nt = Nt
        # store sizes
        self.num_states = self.A.shape[0]
        self.num_inputs = self.B.shape[1]
        # check size compatibility
        assert self.A.shape[1]==self.A.shape[0], "A must be square"
        assert self.B.shape[0]==self.A.shape[0], "B must have same row count as A"
        # initialize parent Pulp class
        pulp.LpProblem.__init__(self, name, sense)
        # begin with no objective at all
        self+=0.0
	# set up state and input variables
        self.var_x = [self.newVarVector("x(0)",self.num_states)]
        self.var_u = []
        for kk in range(self.Nt):
            self.var_x.append(self.newVarVector("x(%i)" % (kk+1),self.num_states))
            self.var_u.append(self.newVarVector("u(%i)" % kk,self.num_inputs))
            self.addVecEqualZeroConstraint(np.dot(self.A,self.var_x[kk])+np.dot(self.B,self.var_u[kk]) - self.var_x[kk+1])

    def setInitialState(self,x0):
        self.addVecEqualZeroConstraint(self.var_x[0]-np.array(x0))

    def setTerminalState(self,xN):
        self.addVecEqualZeroConstraint(self.var_x[self.Nt]-np.array(xN))

    def addInfNormStageCost(self,E,F):
        # adds sum_k ||Ex(k)+Fu(k)||_inf to cost
        for kk in range(self.Nt):
            newvar=self.addMaxVarConstraint(
                np.hstack((
                    np.dot(np.array(E),self.var_x[kk])+np.dot(np.array(F),self.var_u[kk]),
                    np.dot(-np.array(E),self.var_x[kk])+np.dot(-np.array(F),self.var_u[kk])
                ))
            )
            self.objective += newvar

    def add2NormStageCost(self,E,F,Nc=20):
        # adds sum_k ||Ex(k)+Fu(k)||_2 to cost
        # approximated by Nc linear constraints
        M = np.transpose(np.vstack((np.cos(np.pi*np.array(range(Nc))/Nc),np.sin(np.pi*np.array(range(Nc))/Nc))))
        self.addInfNormStageCost(np.dot(M,E),np.dot(M,F))

    def plotStateHistory(self):
        for ii in range(self.num_states):
            plt.plot([x[ii].varValue for x in self.var_x])
        plt.show()

    def plotTraj2D(self,ind_x=0,ind_y=1):
        plt.plot([x[ind_x].varValue for x in self.var_x],[x[ind_y].varValue for x in self.var_x])
        plt.show()

class LpProbUnionCons(LpProbVectors):

    def __init__(self,name="NoName",sense=1):
        # initialize parent Pulp class
        pulp.LpProblem.__init__(self, name, sense)
        # and set list of union constraints to zero
        self.union_cons = []

    def addUnionConstraint(self,c):
        # c should be a tuple of vector expressions
        # constraint x in union{c[0]<=0, c[1]<=0, ...}
        # elements do not have to be same size
        self.union_cons.append(c)

    def _getNextNode(self):
        # depth first for now
        node = self.node_list.pop()
        return(node)

    def _unionFeasible(self):
        pass

    def _branch(self,branch_node):
        pass

    def solveByBranchBound(self,Nmaxnodes=100):
        # no lower bound yet
        self.lower_bound = -np.inf
        # initialize the node list with root ULP
        self.node_list=[self.deepcopy()]
        # incumbent
        self.incumbent_cost=np.inf
        # loop
        for nn in range(Nmaxnodes):
            this_node = self._getNextNode()
            if this_node.lower_bound >= self.incumbent_cost:
                # fathomed as can't improve
                continue
            this_node.solve()
            if this_node.status < 0:
                # fathomed as infeasible
                continue
            if this_node.objective.value() >= self.incumbent_cost:
                # fathomed as did not improve
                continue
            if this_node._unionFeasible():
                # awesome - this is my new incumbent
                self.incumbent_cost = this_node.objective.value()
                self.incumbent_node = this_node
            else:
                self._branch(this_node)

    def _convertUnionToMILP(self,uc,M):
        # number of regions
        Nr = len(uc)
        # expression for final binary constraint
        bin_con = 0.0
        for ii in range(Nr):
            this_region_exp = uc[ii]
            # new binary dec var
            new_bin = pulp.LpVariable(("_b%i" % (self.numVariables()+1)),cat='Binary')
            # new constraint for each inequality defining the region
            for cc in this_region_exp:
                self.addConstraint(cc-M*new_bin<=0)
            # add the binary to the logic constraint
            bin_con += new_bin
        # add the logical constraint
        self.addConstraint(bin_con <= (Nr-1))

    def _convertToMILP(self,M=100):
        for uc in self.union_cons:
            self._convertUnionToMILP(uc,M)

    def solveByMILP(self,M=100):
        self._convertToMILP(M)
        self.solve()

def unionTest():
    lt = LpProbUnionCons()
    x = pulp.LpVariable("x")
    y = pulp.LpVariable("y")
    lt += -x+y
    r1=[x+1,-2-x,y-2,1-y]
    r2=[x+3,-4-x,y-2,1-y]
    lt.addUnionConstraint((r1,r2))
    lt.solveByMILP()
    return lt

def ltrajTest2():
    A = np.eye(2)
    B = np.eye(2)
    lt = LTraj(A,B,5)
    lt.setInitialState([2.0,3.0])
    lt.setTerminalState([4.0,4.0])
    #lt.addInfNormStageCost(np.zeros((2,2)),np.eye(2))
    lt.add2NormStageCost(np.zeros((2,2)),np.eye(2))
    lt.addConstraint(lt.var_x[2][0]>=5.5)
    #lt.addInfNormStageCost(np.eye(2),np.zeros((2,2)))
    lt.solve()
    lt.plotStateHistory()
    lt.plotTraj2D()
    return lt

class LTrajAvoid(LTraj,LpProbUnionCons):

    def __init__(self,A,B,Nt,name="NoName",sense=1):
        LpProbUnionCons.__init__(self)
        LTraj.__init__(self,A,B,Nt,name,sense)
    
    def addStatic2DObst(self,xmin,xmax,ymin,ymax,ind_x=0,ind_y=1):
        for kk in range(self.Nt):
            rleft = [self.var_x[kk][ind_x]-xmin, self.var_x[kk+1][ind_x]-xmin]
            rright = [xmax-self.var_x[kk][ind_x], xmax-self.var_x[kk+1][ind_x]]
            rbelow = [self.var_x[kk][ind_y]-ymin, self.var_x[kk+1][ind_y]-ymin]
            rabove = [ymax-self.var_x[kk][ind_y], ymax-self.var_x[kk+1][ind_y]]
            self.addUnionConstraint((rleft,rright,rabove,rbelow))

def lavTest():
    A = np.eye(2)
    B = np.eye(2)
    lt = LTrajAvoid(A,B,5)
    lt.setInitialState([2.0,3.0])
    lt.setTerminalState([8.0,4.0])
    #lt.addInfNormStageCost(np.zeros((2,2)),np.eye(2))
    lt.add2NormStageCost(np.zeros((2,2)),np.eye(2))
    lt.addStatic2DObst(2.5,3.5,1.5,4.5)
    lt.addStatic2DObst(5.5,6.5,2.5,7.5)
    #lt.addInfNormStageCost(np.eye(2),np.zeros((2,2)))
    lt.solveByMILP()
    lt.plotTraj2D()
    return lt

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

    def _get_next_node(self):
        # next_node = self.bblist.pop()
        next_node = self.bblist.pop(0)        
        return(next_node)

    def solve(self, solver=None, maxsolves=1000, verbosity=1):
        # initialize branch and bound tree with single root node
        self.bblist = [BbNode(self.rootlp,-np.inf,self.obststeps,verbosity)]
        inccost=np.inf
        # start timer for reporting solution time
        self.starttime = time.time()

        for ii in range(maxsolves):
            if verbosity>=1:
                print("Node %i : inc %f : node list %i" % (ii, inccost, len(self.bblist)))
            if len(self.bblist)<1:
                solveTime = time.time()-self.starttime
                print "Optimal solution found after %i nodes in %f seconds." % (ii, solveTime)
                break
            if verbosity>=2:
                 print("Num active nodes = %i" % len(self.bblist))
            # depth first - so grab last node
            thisNode = self._get_next_node()
            # check if it still needs solving
            if thisNode.bound>inccost:
                # fathomed
                if verbosity>=2:
                    print("LP bound above incumbent")
                continue
            # solve the thing
            thisNode.solve(solver=solver)
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
    try:
        # solve by Gurobi, if installed
        opt.solve(solver=pulp.GUROBI())
    except PulpSolverError:
        print("Could not find Gurobi - trying built-in solver")
        # or solve by PuLP default built-in solver
        opt.solve()
    # all being well, plot the trajectory
    opt.plot()

if __name__=="__main__":
    test()
