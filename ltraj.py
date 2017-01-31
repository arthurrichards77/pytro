import numpy as np
import pulp
import matplotlib.pyplot as plt
import time

def lpAffExpAsTuple(ee):
    return tuple([(v[0].name,v[1]) for v in ee.items()])

def oppositeLpExpAsTuple(ee):
    return tuple([(v[0].name,-v[1]) for v in ee.items()])

def lpConstraintsOppose(e1,e2):
    return lpAffExpAsTuple(e1)==oppositeLpExpAsTuple(e2)

def pointsOnHemisphere(Nel,Naz):
    M = np.transpose(np.vstack((np.cos(np.pi*np.array(range(Naz))/Naz),np.sin(np.pi*np.array(range(Naz))/Naz))))
    el_range=np.array(range((1-Nel),Nel))*0.5*np.pi/Nel
    el_range=np.reshape(el_range,(2*Nel-1,1))
    M2 = np.kron(np.cos(el_range),M)
    M3 = np.kron(np.sin(el_range),np.ones((Naz,1)))
    M4 = np.hstack((M2,M3))
    M5 = np.vstack((np.array([0,0,1]),M4))
    return(M5)

def testHemisphere(Nel=10,Naz=10):
    M = pointsOnHemisphere(Nel,Naz)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(M[:,0],M[:,1],M[:,2])
    plt.show()

class LpProbVectors(pulp.LpProblem):

    def __init__(self,name="NoName",sense=1):
        # initialize parent Pulp class
        pulp.LpProblem.__init__(self, name, sense)
        self.leq_list = []
        self.contradictory = False

    def newVarVector(self,name,num_elems):
        v = []
        for ii in range(num_elems):
            newvar = pulp.LpVariable("%s_%i" % (name,ii))
            v.append(newvar)
            self.addVariable(newvar)
        return v

    def addVecEqualZeroConstraint(self,vector_expression,name=None):
        for ii,ee in enumerate(vector_expression):
            if name is not None:
                loc_name = "%s_%i" % (name,ii)
            else:
                loc_name = None
            self.addConstraint(ee==0.0,name=loc_name)

    def addLeqZeroConstraint(self,ee):
        # capture this for quick feasibility checking
        # add it as a constraint in usual way
        self.addConstraint(ee<=0.0)
        # extract opposing constants
        opp_consts = [c.constant for c in self.leq_list if lpConstraintsOppose(ee,c)]
        # check if present
        if len(opp_consts)>0:
            max_opp_const = np.max(opp_consts)
            if max_opp_const > -ee.constant:
                self.contradictory = True
        # add new expression to the list
        self.leq_list += [ee]

    def addVecLessEqZeroConstraint(self,vector_expression):
        for ee in vector_expression:
            self.addLeqZeroConstraint(ee)

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

    def __init__(self,A,B,Nt,name="NoName",sense=1,num_agents=1):
        # initialize parent Pulp class
        LpProbVectors.__init__(self, name, sense)
        # begin with no objective at all
        self+=0.0
        # check size compatibility
        assert A.shape[1]==A.shape[0], "A must be square"
        assert B.shape[0]==A.shape[0], "B must have same row count as A"
        # store horizon
        self.Nt = Nt
        self.num_agents = num_agents
        self.num_states = num_agents*A.shape[0]
        self.num_inputs = num_agents*B.shape[1]
        self.var_x = [[] for kk in range(self.Nt+1)]
        self.var_u = [[] for kk in range(self.Nt)]
        self.avar_x = []
        self.avar_u = []
        for aa in range(num_agents):
            # set up state and input variables
            self.avar_x.append([self.newVarVector("x%i(0)" % (aa+1),A.shape[0])])
            self.var_x[0].extend(self.avar_x[aa][0])
            self.avar_u.append([])
            for kk in range(self.Nt):
                self.avar_x[aa].append(self.newVarVector("x%i(%i)" % (aa+1,kk+1),A.shape[0]))
                self.avar_u[aa].append(self.newVarVector("u%i(%i)" % (aa+1,kk),B.shape[1]))
                self.addVecEqualZeroConstraint(np.dot(np.array(A),self.avar_x[aa][kk])+np.dot(np.array(B),self.avar_u[aa][kk]) - self.avar_x[aa][kk+1])
                self.var_x[kk+1].extend(self.avar_x[aa][kk+1])
                self.var_u[kk].extend(self.avar_u[aa][kk])


    def setInitialState(self,x0):
        self.addVecEqualZeroConstraint(self.var_x[0]-np.array(x0),name='xinit')
        self.init_x = x0

    def changeInitState(self,x0):
        assert len(x0)==self.num_states
        self.init_x = x0
        for ii in range(self.num_states):
            self.constraints[("xinit_%i" % ii)].changeRHS(x0[ii])

    def setTerminalState(self,xN):
        self.addVecEqualZeroConstraint(self.var_x[self.Nt]-np.array(xN),name='xterm')
        self.term_x = xN

    def changeTermState(self,xN):
        assert len(xN)==self.num_states
        self.term_x = xN
        for ii in range(self.num_states):
            self.constraints[("xterm_%i" % ii)].changeRHS(xN[ii])

    def addInfNormStageCost(self,E,F):
        # adds sum_k ||Ex(k)+Fu(k)||_inf to cost
        for kk in range(self.Nt):
            newvar=self.addMaxVarConstraint(
                np.hstack((
                    np.dot(np.array(E), self.var_x[kk])+np.dot(np.array(F), self.var_u[kk]),
                    np.dot(-np.array(E), self.var_x[kk])+np.dot(-np.array(F), self.var_u[kk])
                ))
            )
            self.objective += newvar

    def addStageConstraints(self, C, D, e, agent='aug'):
        # adds Cx(k)+Du(k)<=e to constraints
        for kk in range(self.Nt):
            if agent=='aug':
                self.addVecLessEqZeroConstraint(np.dot(np.array(C), self.var_x[kk])+np.dot(np.array(D), self.var_u[kk])-e)
            elif agent in range(self.num_agents):
                self.addVecLessEqZeroConstraint(np.dot(np.array(C), self.avar_x[agent][kk]) + np.dot(np.array(D), self.avar_u[agent][kk]) - e)
            elif agent=='all':
                for aa in range(self.num_agents):
                    self.addVecLessEqZeroConstraint(np.dot(np.array(C), self.avar_x[aa][kk]) + np.dot(np.array(D), self.avar_u[aa][kk]) - e)


    def add2NormStageCost(self,E,F,Nc=20):
        # adds sum_k ||Ex(k)+Fu(k)||_2 to cost
        # where E and F must both have two rows
        # approximated by Nc linear constraints
        M = np.transpose(np.vstack((np.cos(np.pi*np.array(range(Nc))/Nc),np.sin(np.pi*np.array(range(Nc))/Nc))))
        self.addInfNormStageCost(np.dot(M,E),np.dot(M,F))

    def add2Norm3DStageCost(self,E,F,Naz=11,Nel=7):
        # adds sum_k ||Ex(k)+Fu(k)||_2 to cost
        # where E and F must both have THREE rows
        # approximated by 2*Naz*Nel linear constraints
        M = pointsOnHemisphere(Nel,Naz)
        self.addInfNormStageCost(np.dot(M,E),np.dot(M,F))

    def plotStateHistory(self):
        for ii in range(self.num_states):
            plt.plot([x[ii].varValue for x in self.var_x])
        plt.grid()
        plt.show()

    def plotStateControlHistory(self):
        plt.subplot(2,1,1)
        for ii in range(self.num_states):
            plt.plot([x[ii].varValue for x in self.var_x])
        plt.grid()
        plt.subplot(2,1,2)
        for ii in range(self.num_inputs):
            plt.plot([u[ii].varValue for u in self.var_u])
        plt.grid()
        plt.show()

def mutliTest3(num_agents):
    A = np.eye(2)
    B = np.array([[1],[1]])
    lt = LTraj(A,B,5,num_agents=num_agents)
    return lt


class LpProbUnionCons(LpProbVectors):

    def __init__(self,name="NoName",sense=1,presolver=None):
        # initialize parent Pulp class
        LpProbVectors.__init__(self, name, sense)
        # and set list of union constraints to zero
        self.union_cons = []
        self.union_cons_seqs = []
        self.union_cons_names = []
        # identify incompatible combinations early
        self.taboo_list = []
        # store solver early if given
        self.presolver = presolver

    def addUnionConstraint(self,c,seq=100,name=None):
        # c should be a tuple of vector expressions
        # constraint x in union{c[0]<=0, c[1]<=0, ...}
        # elements do not have to be same size
        self.union_cons.append(c)
        self.union_cons_seqs.append(seq)
        if name is None:
            name = ("_union%i" % (1+len(self.union_cons)))
        self.union_cons_names.append(name)        

    def delUnionConstraint(self,ii):
        del self.union_cons[ii]
        del self.union_cons_seqs[ii]
        del self.union_cons_names[ii]

    def delUnionConByName(self,name):
        try:
            print "Deleting union constraint %s." % name
            ii = self.union_cons_names.index(name)
            self.delUnionConstraint(ii)
        except ValueError:
            print "Can't find union constraint with name %s." % name

    def _getNextNode(self,strategy='depth'):
        if strategy=='depth':
            # depth first search   
            node = self.node_list.pop()
        elif strategy=='breadth':
            # breadth first search
            node = self.node_list.pop(0)
        elif strategy=='best_bound':
            bound_list = [n.lower_bound for n in self.node_list]
            best_idx = np.argmin(bound_list)
            node = self.node_list.pop(best_idx)
        elif strategy=='least_infeas':
            infeas_list = [n.infeas for n in self.node_list]
            best_idx = np.argmin(infeas_list)
            node = self.node_list.pop(best_idx)
        elif strategy=='random_hybrid':
            # choose randomly between depth and breadth
            if np.random.uniform()>0.5:
                # depth
                node = self.node_list.pop()
            else:
                # breadth
                node = self.node_list.pop(0)
        else:
            # default back to depth first   
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
        new_node.leq_list = self.leq_list[:]
        new_node.lower_bound = self.lower_bound
        # copy objective
        new_node.objective = self.objective.copy()
        # infeasibility (used later for node selection)
        new_node.infeas = 0
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
            new_node.infeas = np.max(self.vectorValue(rr))
            self.node_list.append(new_node)

    def _sort_unions(self):
        # sort by supplied sequence number or priority
        decorated = [(self.union_cons_seqs[i],i) for i in range(len(self.union_cons))]
        decorated.sort()
        self.union_cons = [self.union_cons[i] for (seq,i) in decorated]
        # and sort the sequence list to avoid silliness
        self.union_cons_seqs = [seq for (seq,i) in decorated]
        # and the name list
        self.union_cons_names = [self.union_cons_names[i] for (seq,i) in decorated]

    def _status_msg(self,msg):
        if self.verbosity>=10:
            if np.mod(self.lp_count,self.verbosity)==0:
                print "%i : %i : %f : %s" % (self.lp_count,len(self.node_list),self.incumbent_cost,msg)
        elif self.verbosity>=1:
            print "%i : %i : %f : %s" % (self.lp_count,len(self.node_list),self.incumbent_cost,msg)

    def solveByBranchBound(self,Nmaxnodes=1000,Nmaxiters=5000,strategy='least_infeas',verbosity=1,**kwargs):
        start_time = time.clock()
        # no lower bound yet
        self.lower_bound = -np.inf
        # initialize the node list with root ULP
        self.node_list=[self._childNode()]
        # incumbent
        self.incumbent_cost=np.inf
        # count number of actual solves
        self.lp_count = 0
        # sort the union constraints for efficiency
        self._sort_unions()
        # store verbosity setting
        self.verbosity = verbosity
        # loop
        for nn in range(Nmaxiters):
            if len(self.node_list)==0:
                # finished - no more nodes
                self.status=1
                self._status_msg("OPTIMAL no more nodes")
                break
            if self.lp_count==Nmaxnodes:
                # finished - no more nodes
                self.status=0
                self._status_msg("Node LP count limit reached")
                break
            this_node = self._getNextNode(strategy)
            if this_node.lower_bound >= self.incumbent_cost:
                # fathomed as can't improve
                self._status_msg("Fathom before solving bound=%f" % (this_node.lower_bound))
                continue
            # check for contradictory constraints
            if this_node.contradictory:
                # fathomed as won't solve
                self._status_msg("Fathom due to incompatible bounds")
                continue
            # solve the LP
            # with unhandled arguments passed to solver
            self.lp_count += 1
            this_node.solve(**kwargs)
            if this_node.status < 0:
                # fathomed as infeasible
                self._status_msg("Fathom infeasible status=%i" % (this_node.status))
                continue
            if this_node.objective.value() >= self.incumbent_cost:
                # fathomed as did not improve
                self._status_msg("Fathom after solving cost=%f" % (this_node.objective.value()))
                continue
            if this_node._unionFeasible():
                # awesome - this is my new incumbent
                self.incumbent_cost = this_node.objective.value()
                self.incumbent_node = this_node
                self.incumbent_sol = [vv.varValue for vv in this_node.variables()]
                self._status_msg("New incumbent %f" % (this_node.objective.value()))
            else:
                self._branch(this_node)
                self._status_msg("Branched with bound=%f" % (this_node.objective.value()))
        else:
            self.status=0
            self._status_msg("Iteration limit reached")
        # if ever found solution
        if self.incumbent_cost<np.inf:
            # copy result back to parent
            for ii in range(len(self.variables())):
                self.variables()[ii].varValue = self.incumbent_sol[ii]
                # stop the clock
        self.solve_time = time.clock() - start_time

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
        start_time = time.clock()
        self._convertToMILP(M)
        self.solve(**kwargs)
        self.solve_time = time.clock() - start_time


class LTrajAvoid(LTraj, LpProbUnionCons):

    def __init__(self, A, B, Nt, name="Trajectory", sense=1, num_agents=1):
        LpProbUnionCons.__init__(self)
        LTraj.__init__(self, A, B, Nt, name, sense, num_agents)


class LTraj2DAvoid(LTrajAvoid):

    def __init__(self, A, B, Nt, ind_x=0, ind_y=1, name="Trajectory", sense=1):
        LTrajAvoid.__init__(self,A,B,Nt,name,sense)
        self.ind_x = ind_x
        self.ind_y = ind_y
        self.boxes = []
        self.box_names = []

    def addStatic2DObst(self,xmin,xmax,ymin,ymax):
        self.boxes += [(xmin,xmax,ymin,ymax)]
        box_name = "box%i" % (len(self.boxes))
	self.box_names += [box_name]
        for kk in range(self.Nt):
            rleft = [self.var_x[kk][self.ind_x]-xmin, self.var_x[kk+1][self.ind_x]-xmin]
            rright = [xmax-self.var_x[kk][self.ind_x], xmax-self.var_x[kk+1][self.ind_x]]
            rbelow = [self.var_x[kk][self.ind_y]-ymin, self.var_x[kk+1][self.ind_y]-ymin]
            rabove = [ymax-self.var_x[kk][self.ind_y], ymax-self.var_x[kk+1][self.ind_y]]
            self.addUnionConstraint((rleft,rright,rabove,rbelow),seq=kk,name="%s_%i" % (box_name,kk))

    def deleteObstByIndex(self,box_index):
        if box_index<len(self.boxes):
            del self.boxes[box_index]
            box_name = self.box_names[box_index]
            del self.box_names[box_index]
            for kk in range(self.Nt):
                con_name="%s_%i" % (box_name,kk)
                self.delUnionConByName(name=con_name)

    def deleteObstByPoint(self,point):
        box_index = -1
        print "deleting box containing (%f,%f)" % (point[0],point[1])
        for ii in range(len(self.boxes)):
            this_box = self.boxes[ii]
            print this_box
            if point[0]>this_box[0] and point[0]<this_box[1]:
                print "OK X"
                if point[1]>this_box[2] and point[1]<this_box[3]:
                    print "OK Y"
                    print "Deleting box %i" % ii
                    self.deleteObstByIndex(ii)
                    box_index = ii
                    # only delete the first one
                    break
        return box_index

    def plotBoxes(self):
        for this_box in self.boxes:
            plt.plot([this_box[0],this_box[0],this_box[1],this_box[1],this_box[0]],[this_box[2],this_box[3],this_box[3],this_box[2],this_box[2]],'r')

    def plotTraj2D(self):
        self.plotBoxes()
        plt.plot([x[self.ind_x].varValue for x in self.var_x],[x[self.ind_y].varValue for x in self.var_x])
        plt.show()

class LTraj3DAvoid(LTrajAvoid):

    def __init__(self,A,B,Nt,ind_x=0,ind_y=1,ind_z=2,name="Trajectory",sense=1):
        LTrajAvoid.__init__(self,A,B,Nt,name,sense)
        self.ind_x = ind_x
        self.ind_y = ind_y
        self.ind_z = ind_z
        self.boxes = []

    def addStatic3DObst(self,xmin,xmax,ymin,ymax,zmin,zmax):
        self.boxes += [(xmin,xmax,ymin,ymax,zmin,zmax)]
        for kk in range(self.Nt):
            rleft = [self.var_x[kk][self.ind_x]-xmin, self.var_x[kk+1][self.ind_x]-xmin]
            rright = [xmax-self.var_x[kk][self.ind_x], xmax-self.var_x[kk+1][self.ind_x]]
            rback = [self.var_x[kk][self.ind_y]-ymin, self.var_x[kk+1][self.ind_y]-ymin]
            rfront = [ymax-self.var_x[kk][self.ind_y], ymax-self.var_x[kk+1][self.ind_y]]
            rbelow = [self.var_x[kk][self.ind_z]-zmin, self.var_x[kk+1][self.ind_z]-zmin]
            rabove = [zmax-self.var_x[kk][self.ind_z], zmax-self.var_x[kk+1][self.ind_z]]
            self.addUnionConstraint((rleft,rright,rback,rfront,rabove,rbelow),seq=(kk*(self.Nt-kk)))

    def addRandomBoxes(self,num_boxes,ctr_range,size_range):
        box_ctrs = np.random.uniform(low=ctr_range[0],high=ctr_range[1],size=(num_boxes,3))
        box_sizes = np.random.uniform(low=size_range[0],high=size_range[1],size=(num_boxes,3))
        for bb in range(num_boxes):
            this_box = (box_ctrs[bb,0]-box_sizes[bb,0],
                        box_ctrs[bb,0]+box_sizes[bb,0],
                        box_ctrs[bb,1]-box_sizes[bb,1],
                        box_ctrs[bb,1]+box_sizes[bb,1],
                        box_ctrs[bb,2]-box_sizes[bb,2],
                        box_ctrs[bb,2]+box_sizes[bb,2])
            assert this_box[1]>this_box[0]
            assert this_box[3]>this_box[2]
            assert this_box[5]>this_box[4]
            self.addStatic3DObst(this_box[0],this_box[1],this_box[2],this_box[3],this_box[4],this_box[5])

    def plotBoxes(self,ax):
        for this_box in self.boxes:
            ax.plot([this_box[0],this_box[0],this_box[1],this_box[1],this_box[0]],
                    [this_box[2],this_box[3],this_box[3],this_box[2],this_box[2]],
                    [this_box[4],this_box[4],this_box[4],this_box[4],this_box[4]],
                    'r')
            ax.plot([this_box[0],this_box[0],this_box[1],this_box[1],this_box[0]],
                    [this_box[2],this_box[3],this_box[3],this_box[2],this_box[2]],
                    [this_box[5],this_box[5],this_box[5],this_box[5],this_box[5]],
                    'r')
            ax.plot([this_box[0],this_box[0]],
                    [this_box[2],this_box[2]],
                    [this_box[4],this_box[5]],
                    'r')
            ax.plot([this_box[1],this_box[1]],
                    [this_box[2],this_box[2]],
                    [this_box[4],this_box[5]],
                    'r')
            ax.plot([this_box[0],this_box[0]],
                    [this_box[3],this_box[3]],
                    [this_box[4],this_box[5]],
                    'r')
            ax.plot([this_box[1],this_box[1]],
                    [this_box[3],this_box[3]],
                    [this_box[4],this_box[5]],
                    'r')

    def plotTraj3D(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.plotBoxes(ax)
        ax.plot([x[self.ind_x].varValue for x in self.var_x],
                [x[self.ind_y].varValue for x in self.var_x],
                [x[self.ind_z].varValue for x in self.var_x])
        if self.term_x:
            ax.plot([self.term_x[self.ind_x]],
                    [self.term_x[self.ind_y]],
                    [self.term_x[self.ind_z]], 'g*')
        if self.init_x:
            ax.plot([self.init_x[self.ind_x]],
                    [self.init_x[self.ind_y]],
                    [self.init_x[self.ind_z]], 'gs')
        plt.show()

class LTr3DShortest(LTraj3DAvoid):

    def __init__(self,Nt,name="Trajectory"):
        A = np.eye(3)
        B = np.eye(3)
        LTraj3DAvoid.__init__(self,A,B,Nt,name=name,sense=1)
        self.add2Norm3DStageCost(np.zeros((3,3)),np.eye(3))

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

def lavTest():
    A = np.eye(2)
    B = np.eye(2)
    lt = LTrajAvoid(A,B,5)
    lt.setInitialState([2.0,3.0])
    lt.setTerminalState([8.0,4.0])
    #lt.addInfNormStageCost(np.zeros((2,2)),np.eye(2))
    lt.add2NormStageCost(np.zeros((2,2)),np.eye(2),Nc=11)
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

def randomTest(num_boxes=3,method='MILP',**kwargs):
    A = np.eye(2)
    B = np.eye(2)
    lt = LTraj2DAvoid(A,B,5)
    lt.setInitialState([0.0,0.0])
    lt.setTerminalState([10.0,10.0])
    lt.add2NormStageCost(np.zeros((2,2)),np.eye(2))
    box_ctrs = np.random.uniform(low=2.0,high=8.0,size=(num_boxes,2))
    box_sizes = np.random.uniform(low=0.1,high=0.75,size=(num_boxes,2))
    plt.cla()
    for bb in range(num_boxes):
        this_box = (box_ctrs[bb,0]-box_sizes[bb,0],
                    box_ctrs[bb,0]+box_sizes[bb,0],
                    box_ctrs[bb,1]-box_sizes[bb,1],
                    box_ctrs[bb,1]+box_sizes[bb,1])
        assert this_box[1]>this_box[0]
        assert this_box[3]>this_box[2]
        lt.addStatic2DObst(this_box[0],this_box[1],this_box[2],this_box[3])
        # solve it
    if method=='MILP':
        lt.solveByMILP(**kwargs)
        lt.plotTraj2D()
    elif method=='BNB':
        lt.solveByBranchBound(**kwargs)
        lt.plotTraj2D()
    return lt

def randomTest3D(num_boxes=3,method='MILP',**kwargs):
    A = np.eye(3)
    B = np.eye(3)
    lt = LTraj3DAvoid(A,B,5)
    lt.setInitialState([0.0,0.0,0.0])
    lt.setTerminalState([10.0,10.0,10.0])
    lt.add2Norm3DStageCost(np.zeros((3,3)),np.eye(3))
    box_ctrs = np.random.uniform(low=2.0,high=8.0,size=(num_boxes,3))
    box_sizes = np.random.uniform(low=0.1,high=0.75,size=(num_boxes,3))
    for bb in range(num_boxes):
        this_box = (box_ctrs[bb,0]-box_sizes[bb,0],
                    box_ctrs[bb,0]+box_sizes[bb,0],
                    box_ctrs[bb,1]-box_sizes[bb,1],
                    box_ctrs[bb,1]+box_sizes[bb,1],
                    box_ctrs[bb,2]-box_sizes[bb,2],
                    box_ctrs[bb,2]+box_sizes[bb,2])
        assert this_box[1]>this_box[0]
        assert this_box[3]>this_box[2]
        assert this_box[5]>this_box[4]
        lt.addStatic3DObst(this_box[0],this_box[1],this_box[2],this_box[3],this_box[4],this_box[5])
        # solve it
    if method=='MILP':
        lt.solveByMILP(**kwargs)
        lt.plotTraj3D()
    elif method=='BNB':
        lt.solveByBranchBound(**kwargs)
        lt.plotTraj3D()
    return lt

def random3DShortest(Nt=5,num_boxes=10,ctr_range=(2.0,8.0),size_range=(0.1,3.0),method='MILP',**kwargs):
    lt = LTr3DShortest(Nt=5)
    lt.setInitialState([0.0,0.0,0.0])
    lt.setTerminalState([10.0,10.0,10.0])
    lt.addRandomBoxes(num_boxes,ctr_range,size_range)
    # solve it
    if method=='MILP':
        lt.solveByMILP(**kwargs)
        lt.plotTraj3D()
    elif method=='BNB':
        lt.solveByBranchBound(**kwargs)
        lt.plotTraj3D()
    return lt

if __name__=="__main__":
    random3DShortest(num_boxes=5,method='BNB')
