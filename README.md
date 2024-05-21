# PYTRO - PYthon TRajectory Optimization

## Non-convex optimization using branch-and-bound

The library builds on the [PuLP](https://pypi.org/project/PuLP/) linear programming library to provide trajectory modeling and non-convex constraints.

The latter are expressed as "union" constraints in the form C:={C1,C2,C3...} where at least one of the sets of constraints Ci must all be satisfied.  This handles a variety of avoidance constraints in a trajectory setting.

Problems are solved by converting to Mixed-Integer Linear Programming (MILP) form and solving using PuLP's existing solver capabilities.  An experimental direct branch-and-bound solver is under development, using only LP solvers.

Notebook `pytro_example.ipynb` walks through a simple shortest-path example with obstacle avoidance.

The `aircraft_example.py` file implements the method from [Richards A. G. and How J. P. "Aircraft Trajectory Planning With Collision Avoidance Using Mixed Integer Linear Programming" American Control Conference 2002](http://dx.doi.org/10.1109/ACC.2002.1023918) 
with the additional corner-cutting prevention from [Richards A. G. and Turnbull O. D. N. "Inter-sample avoidance in trajectory optimizers using mixed-integer linear programming" IJRNC 2013](https://doi.org/10.1002/rnc.3101).
See also the reproduction at (https://github.com/auralius/richards-how) using CVXPY 

`carplan.py` is a version of the ideas covered in [Bali C. and Richards A. G. "Merging Vehicles at Junctions using Mixed-Integer Model Predictive Control" ECC 2018](https://doi.org/10.23919/ECC.2018.8550577) on using MILP to separate vehicles moving along fixed tracks with junctions.
