from types import SimpleNamespace
import numpy as np
from scipy import optimize
from Python_scripts.ExchangeEconomyModel import ExchangeEconomyModelClass

class ExchangeEconomyModelQuasiLinearClass(ExchangeEconomyModelClass):

    # implement the quasi-linear model using inheritance from ExchangeEconomyModelClass

    # so change the utility functions to the quasi-linear ones as seen below
    # and the indifference curve methods, by isolating x2 from the utility functions
    # uA(x1A,x2A) = ln(x1A) + alpha*x2A
    # uB(x1B,x2B) = ln(x1B) + beta*x2B

    ######################
    # utility and demand # 
    ######################

    def utility_A(self,x1A,x2A):
        """
        Utility function for agent A.
        """        
        par = self.par
        return np.log(x1A) + par.alpha * x2A
    
    def x2A_indifference(self,uA,x1A):

        par = self.par
        return (uA - np.log(x1A))/par.alpha
    
    def utility_B(self,x1B,x2B):
        """
        Utility function for agent B.
        """
        par = self.par
        return np.log(x1B) + par.beta * x2B
    
    def x2B_indifference(self,uB,x1B):
        par = self.par
        return (uB - np.log(x1B))/par.beta

    # Demand functions updated with the demand functions given in the problem set
    # I suspect that when writing the problem set, they chose p1 as numeraire that is p1=1
    # But in terms of how they wrote the demand functions(taking p1 as an argument),
    # it makes more sense to set p2=1 as numeraire
    def demand_A(self,p1):
        """
        Demand for good 1 and 2 for agent A.
        """
        par = self.par
        # compute income and inner solution
        income = p1*par.w1A+par.w2A
        x1 = 1.0/(par.alpha*p1)
        # check if income is enough for inner solution
        if income > p1*x1:
            return x1, income - p1*x1
        # otherwise corner solution
        else:
            return income/p1, 0

    def demand_B(self,p1):
        """
        Demand for good 1 and 2 for agent B.
        """
        par = self.par 
        income = p1*(1-par.w1A)+(1-par.w2A)
        x1 = 1.0/(par.beta*p1)
        if income > p1*x1:
            return x1, income - p1*x1
        else:
            return income/p1, 0
    
    # See my comments for dictatorA in ExchangeEconomyModel.py
    # The same logic applies for dictatorB below
    def solve_dictator_B(self):
        """ 
        Solve the dictator problem for agent B.
        """
        par = self.par
        sol = self.sol

        # a. objective
        def obj(xB):
            return -self.utility_B(xB[0],xB[1])

        # b. constraint
        uA_w = self.utility_A(par.w1A,par.w2A)
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_A(1-x[0],1-x[1]) - uA_w})
        bounds = ((0,1),(0,1))
        
        # c. optimize
        opt = optimize.minimize(obj,x0=sol.xA,
                                method='SLSQP',
                                bounds=bounds,constraints=constraints)
        
        sol.xB_dictatorB = opt.x
        sol.uB_dictatorB = -opt.fun
        sol.uA_dictatorB = self.utility_A(1-sol.xB_dictatorB[0],1-sol.xB_dictatorB[1])
        sol.xA_dictatorB = np.array([1-sol.xB_dictatorB[0],1-sol.xB_dictatorB[1]])

        print(f'Dictator solution for B:')
        print(f'x1B = {sol.xB_dictatorB[0]:12.8f}')
        print(f'x2B = {sol.xB_dictatorB[1]:12.8f}')
        print(f'Utility = {sol.uB_dictatorB:12.8f}')


            