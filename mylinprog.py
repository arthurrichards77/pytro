import numpy
from scipy.optimize import linprog
from copy import deepcopy

class linexp(object):
    def __init__(self,varnumarray,varmultarray):
        self.varnums=varnumarray
        self.varmults=varmultarray

    def __repr__(self):
        return "linexp("+self.varmults.__str__()+","+self.varnums.__str__()+")"

    def __add__(self,other):
        addvars = numpy.concatenate((self.varnums, other.varnums))
        addmults = numpy.concatenate((self.varmults, other.varmults))
        return linexp(addvars,addmults)

    def __radd__(self,other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self,other):
        return linexp(self.varnums,self.varmults*other)

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __sub__(self,other):
        return self.__add__(-other)

    def to_array(self,size):
        v=numpy.zeros(size)
        for ii in range(len(self.varnums)):
            v[self.varnums[ii]-1] = v[self.varnums[ii]-1]+self.varmults[ii]
        return v

class decvar(linexp):

    def __init__(self,varnum,problem):
        linexp.__init__(self,numpy.array([varnum]),numpy.array([1.0]))
        self.myvar = varnum
        self.problem = problem

    def result(self,result=None):
        if result==None:
            result=self.problem.result
        return result.x[self.myvar-1]

    @property
    def bounds(self):
        return self.problem.bounds[self.myvar-1]

    @bounds.setter
    def bounds(self,value):
        self.problem.bounds[self.myvar-1]=value
            
class lp(object):
    def __init__(self):
        self.numvars=0
        self.numineqs=0
        self.numeqs=0
        self.bounds=[]

    def newvar(self,bounds=(-numpy.inf,numpy.inf)):
        self.numvars = self.numvars+1
        self.bounds.append(bounds)
        return decvar(self.numvars,self)

    def addeqcon(self,conexp,rhs):
        if self.numeqs>0:
            self.Aeq = numpy.append(self.Aeq,[conexp.to_array(self.numvars)],axis=0)
            self.beq = numpy.append(self.beq,[rhs],axis=0)            
        else:
            self.Aeq = [conexp.to_array(self.numvars)]
            self.beq = [rhs]
        self.numeqs = self.numeqs+1
    
    def addineq(self,conexp,conlim):
        if self.numineqs>0:
            self.A = numpy.append(self.A,[conexp.to_array(self.numvars)],axis=0)
            self.b = numpy.append(self.b,[conlim],axis=0)            
        else:
            self.A = [conexp.to_array(self.numvars)]
            self.b = [conlim]
        self.numineqs = self.numineqs+1
    
    def setobj(self,costexp):
        self.c = costexp.to_array(self.numvars)

    def solve(self):
        # some calcs are done in place, changing the input data
        # so make a copy of myself first to avoid losing problem
        local_self = deepcopy(self)
        if self.numeqs>0:
            self.result=linprog(local_self.c,A_ub=local_self.A,b_ub=local_self.b,A_eq=local_self.Aeq,b_eq=local_self.beq, bounds=local_self.bounds, options=dict(tol=1e-6))
        else:
            self.result=linprog(local_self.c,A_ub=local_self.A,b_ub=local_self.b, bounds=local_self.bounds, options=dict(tol=1e-6))
        return self.result
        
def test():
    p = lp()
    x = p.newvar()
    y = p.newvar()
    p.setobj(y)
    p.addineq(-x-y,-4)
    p.addineq(2*x-y,2)
    p.addineq(x+y,8)
    res=p.solve()
    print(res)
    print("x=%f" % x.result())
    print("y=%f" % y.result())
    p.setobj(-x)
    print(p.solve())
    print("x=%f" % x.result())
    print("y=%f" % y.result())
    p.addeqcon(y-x,1)
    print(p.solve())
    print("x=%f" % x.result())
    print("y=%f" % y.result())
    print("x=%f" % x.result(result=res))
