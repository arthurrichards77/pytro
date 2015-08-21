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

    def vectorValue(self,vector_expression):
        v = []
        for ee in vector_expression:
            v.append(ee.value())
        return np.array(v)

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

    def _solInUnionRegion(self,r):
        v = self.vectorValue(r)
        if np.max(v)<=0.0:
            flag_reg = True
        else:
            flag_reg = False
        return flag_reg

    def _unionConSatisfied(self,c):
        flag_sat = False
        for r in c:
            if self._solInUnionRegion(r):
                flag_sat = True
                break
        return flag_sat

    def _unionFeasible(self):
        flag_feas = True
        for ci,cc in enumerate(self.union_cons):
            if self._unionConSatisfied(cc):
                continue
            else:
                flag_feas = False
                self.first_violated_union = cc
                self.first_violated_index = ci
                break
        return flag_feas

    def _childNode(self):
        # first just call the existing deepcopy inheritend from LpProblem
        new_node = LpProbUnionCons(name = self.name, sense = self.sense)
        # and copy the lower bound and the union constraints across
        new_node.union_cons = self.union_cons[:]
        new_node.lower_bound = self.lower_bound
        # copy objective
        new_node.objective = self.objective.copy()
        # copy constraints
        new_node.constraints = self.constraints.copy()
        # note not bothering to copy node list, as only relevant for root
        return new_node

    def _branch(self,parent_node):
        parent_node.lower_bound = parent_node.objective.value()
        for rr in parent_node.first_violated_union:
            new_node = parent_node._childNode()
            new_node.union_cons.pop(parent_node.first_violated_index)
            new_node.addVecLessEqZeroConstraint(rr)
            self.node_list.append(new_node)

    def solveByBranchBound(self,Nmaxnodes=1000,**kwargs):
        # no lower bound yet
        self.lower_bound = -np.inf
        # initialize the node list with root ULP
        self.node_list=[self._childNode()]
        # incumbent
        self.incumbent_cost=np.inf
        # loop
        for nn in range(Nmaxnodes):            
            if len(self.node_list)==0:
                # finished - no more nodes
                print "%i : %i : %f : OPTIMAL no more nodes" % (nn,len(self.node_list),self.incumbent_cost)
                # copy result back to parent
                for ii in range(len(self.variables())):
                    self.variables()[ii].varValue = self.incumbent_sol[ii]
                break
            this_node = self._getNextNode()
            if this_node.lower_bound >= self.incumbent_cost:
                # fathomed as can't improve
                print "%i : %i : %f : Fathom before solving bound=%f" % (nn,len(self.node_list),self.incumbent_cost,this_node.lower_bound)
                continue
            this_node.solve(**kwargs)
            if this_node.status < 0:
                # fathomed as infeasible
                print "%i : %i : %f : Fathom infeasible status=%i" % (nn,len(self.node_list),self.incumbent_cost,this_node.status)
                continue
            if this_node.objective.value() >= self.incumbent_cost:
                # fathomed as did not improve
                print "%i : %i : %f : Fathom after solving cost=%f" % (nn,len(self.node_list),self.incumbent_cost,this_node.objective.value())
                continue
            if this_node._unionFeasible():
                # awesome - this is my new incumbent
                self.incumbent_cost = this_node.objective.value()
                self.incumbent_node = this_node
                self.incumbent_sol = [vv.varValue for vv in this_node.variables()]
                print "%i : %i : %f : New incumbent %f" % (nn,len(self.node_list),self.incumbent_cost,this_node.objective.value())                
            else:
                self._branch(this_node)
                print "%i : %i : %f : Branched with bound=%f" % (nn,len(self.node_list),self.incumbent_cost,this_node.objective.value())                
        else:
            print "%i : %i : %f : Node limit reached" % (nn,len(self.node_list),self.incumbent_cost)                

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

    def solveByMILP(self,M=100,**kwargs):
        self._convertToMILP(M)
        self.solve(**kwargs)

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
    lt.addStatic2DObst(5.5,6.5,3.5,7.5)
    return lt

def bbTest():
    lt = lavTest()
    if pulp.GUROBI().available():
        # solve by Gurobi, if installed
        lt.solveByBranchBound(solver=pulp.GUROBI(msg=0))
    else:
        print("Could not find Gurobi - trying built-in solver")
        # or solve by PuLP default built-in solver
        lt.solveByBranchBound()
    lt.plotTraj2D()
    return lt

def milpTest():
    lt = lavTest()
    if pulp.GUROBI().available():
        # solve by Gurobi, if installed
        lt.solveByMILP(solver=pulp.GUROBI())
    else:
        print("Could not find Gurobi - trying built-in solver")
        # or solve by PuLP default built-in solver
        lt.solveByMILP()
    lt.plotTraj2D()
    return lt
