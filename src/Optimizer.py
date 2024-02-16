from z3 import *


class Optimizer:
    """
    This is a general purpose Optimization with a MaxSAT approach that takes into account multiple margins and tries
    to satisfy the strict constraints on a decision point first.

    W: parameters to learn
    T: Boolean variables for training instances. One training instance has M T variables. M is the number of margins.
       For each instance, first constraint is the most lenient and the last constraint is the most strict.
    Soft Constraints: Constraints based on training instances. For each instance there is one constraint.
    Hard Constraints: By default constraint the definitions of each constraint in T. Specific knowledge constraints
                      are also added for specific use cases

    Optimization function is the main MaxSAT algorithm.
    """

    def __init__(self, verbose=False):
        self.solver = SolverFor('NRA')
        self.W = None
        self.T = None
        self.Soft_Constraints = []
        self.Hard_Constraints = []
        self.relaxation = None
        self.out = None
        self.solver.set('timeout',  10000)

        if verbose:
            set_option('verbose', 10)

        set_param('smt.arith.solver', 2)

    def reset(self):
        self.solver.reset()
        self.W = None
        self.T = None
        self.Soft_Constraints = []
        self.Hard_Constraints = []
        self.relaxation = None
        self.out = None

    @staticmethod
    def GetLhsRhs(c):
        split = c.split(' >=')
        return split[0], split[1][-1]

    @staticmethod
    def GetClauseNum(c):
        return int(c.split(',')[0].split('__')[1])

    def OptimizationFuMalik(self, length=None):
        print('starting the check')
        self.solver.add(self.Hard_Constraints)
        if length is None:
            print('please provide the total number of margins')
            sys.exit(0)
        i = 0
        Fs = self.Soft_Constraints.copy()
        while True:
            out = self.solver.check(Fs)
            if out == sat:
                print('found sat')
                self.out = sat
                break
            elif out == unknown:
                print('found unknown')
                self.out = unknown
                break
            else:
                relaxed_variables = []
                core = self.solver.unsat_core()
                for c in core:
                    i += 1
                    Fs.remove(c)
                    Fs.append(Or(c, Bool('r_' + str(i))))
                if len(relaxed_variables) > 0:
                    self.solver.add(Sum([If(r, 1, 0) for r in relaxed_variables]) == self.relaxation)
                if len(core) == 0:
                    print('no solution is possible')
                    self.out = unsat
                    break
