import sympy as sp 
import numpy as np 
import math 
from scipy.integrate import odeint
import matplotlib.pyplot as plt 


def map_parameters(b): 
    b1,b2,b3,b4,b5 = b 
    assert b3 < 1, "Mapping doesn't exist"
    aph, bt, gm = b1/b4, b5/b4, b2*b3/(1-b3)
    return aph, bt, gm 

def fix_inverse_mapping(b3, b4): 
    def inverse_map(alpha, beta, gamma): 
        return [alpha * b4, ((1 - b3)/b3) * gamma, b3, b4, beta * b4]
    return inverse_map

def map_parameters_inverse_single(alpha, beta, gamma): 
    # Arbitrary choice of inverse mapping
    return fix_inverse_mapping(0.5, 1)(alpha, beta, gamma) 

class Parameters: 
    def __init__(self, parameters):
        if len(parameters) == 3: 
            self.b = map_parameters_inverse_single(*parameters)
            self.abg = parameters
        elif len(parameters) == 5: 
            self.b = parameters
            self.abg = map_parameters(self.b)
        elif len(parameters) != 5: 
            raise ValueError("3 or 5 parameters must be given")

    def steady_state_E3(self):
        b1, b2, b3, b4, b5 = self.b
        alpha, beta, gamma = self.abg
        x1 = gamma 
        x3 = b1/b3 * x1 * (1 - x1)
        eig = b4 * (gamma - (alpha + beta))
        return x1, 0, x3

    def steady_state_E4(self):
        aph, bt, _ = self.abg
        A = 1
        B =  -(aph + aph*bt + bt)
        C = aph*bt
        x1 = (-B + math.sqrt(B**2 - 4*A*C))/(2*A)
        x2 = -((aph*x1*(x1 - 1)) / (x1 + aph*(x1 - 1)))
        return x1, x2, 0

    def steady_state_E5(self): 
        _, b2, _, b4, _ = self.b
        alpha, beta, gamma = self.abg
        x1 = alpha * gamma / (gamma - beta)
        x2 = gamma - x1
        x3 = b4 * (b2 + gamma) * (x1 - alpha * gamma - beta)
        return x1, x2, x3


    def is_x3_dying(self): 
        gamma = self.abg[2]
        return gamma > 1

    def is_x2_dying(self): 
        aph,bt, _ = self.abg
        return aph + bt > 1

    def steady_state_E3_exists(self): 
        return not self.is_x3_dying()

    def is_steady_state_E3_stable_as_2d(self): 
        _, b2, b3, _, _ = self.b
        return b3 * (1 + b2) > 1 - b2

    def steady_state_E4_exists(self): 
        return not self.is_x2_dying()

    def is_E3_steady_state_inner_eigen_positive(self): 
        alpha, beta, gamma = self.abg
        return gamma > alpha + beta

    def is_E4_steady_state_inner_eigen_positive(self): 
        _, _, gamma = self.abg
        return gamma < self.gamma_star()

    def gamma_star(self): 
        alpha, beta, gamma = self.abg
        A = alpha
        B = (1 - alpha)*beta - alpha
        C = -beta**2
        D = B**2 - 4*A*C
        assert -B - math.sqrt(D) < 0  # I want to catch this situation if it happens
        return (-B + math.sqrt(D))/(2*A)  # because I don't expect that it can

    def E5_exists(self): 
        return  not self.is_x2_dying() and \
                self.is_E3_steady_state_inner_eigen_positive() and \
                self.is_E4_steady_state_inner_eigen_positive()

    def iterate_init_conds(self, x1num=2, x2num=2, x3num=2):
        " Iterate inner points inside localizing set"
        # Always skip first x_i = 0
        for x1 in np.linspace(0, 1, x1num + 1, endpoint=False)[1:]:
            for x2 in np.linspace(0, 1 - x1, x2num + 1, endpoint=False)[1:]:
                x3_rhs = self.localizing_set_value() - x1 - x2
                for x3 in np.linspace(0, x3_rhs, x3num + 1, endpoint=False)[1:]:
                    yield (x1, x2, x3)

    
    def localizing_set_value(self): 
        b1, _, b3, _, _ = self.b
        return 1 + b1/(4*b3)



def iterate_params_E5_exists(): 
    for alpha in np.arange(0.1, 1, 0.1): 
        for beta in np.arange(0.1, 1 - alpha, 0.1): 
            gu = gamma_star(alpha, beta)
            for gamma in np.arange(alpha + beta + 0.0001, gu, 0.001): 
                yield (alpha, beta, gamma)


def iterate_params_E3_stable(b2_step, alpha_step=0.1, beta_step=0.1, b3_step=0.1, b4=1):
    """ - Stable in 3d sense,
      - also choose alpha, beta s.t. alpha + beta < 1
       b4 is independent, so its chosen arbitrary, but fixed"""
    for alpha in np.arange(alpha_step, 1, alpha_step):
        for beta in np.arange(beta_step, 1 - alpha, beta_step): 
            for b3 in np.arange(b3_step, 1, b3_step):
                b2_lhs = (1 - b3) / (1 + b3)
                b2_rhs = (1 - b3) * (alpha + beta) / b3
                if b2_lhs >= b2_rhs:
                    continue
                for b2 in np.arange(b2_lhs + b2_step, b2_rhs, b2_step):
                    b1, b5 = alpha * b4, beta * b4
                    yield b1, b2, b3, b4, b5



class SymbolicPredatorPreySystem: 
    def __init__(self): 
        self.parameters = sp.symbols('b1:6')
        self.states = sp.symbols('x1:4')

        b1, b2, b3, b4, b5 = self.parameters
        x1, x2, x3 = self.states
        self.sys = sp.Matrix([
            b1*(x1+x2)*(1-x1) - x1*x3/(b2+x1+x2) - b4*x1*x2, 
            b4*x1*x2 - x2*x3/(b2+x1+x2) - b5*x2 - b1*(x1+x2)*x2, 
            (x1+x2)*x3/(b2+x1+x2) - b3*x3
        ])
        self.jacobian = self.sys.jacobian([x1,x2,x3])

    def lambdify(self, parameters): 
        s = self.sys.subs(zip(self.parameters, parameters))
        return sp.lambdify([self.states], s)



class PredatorPreySystem: 
    def __init__(self, parameters):
        self.params = Parameters(parameters)

        #self.sys = SymbolicPredatorPreySystem()
        s = SymbolicPredatorPreySystem()
        
        self.jacobian = s.jacobian.subs(zip(s.parameters, self.params.b))
        self.states = s.states
        self.dxdt = s.lambdify(self.params.b)

    def integrate(self, x0, T, t0=0): 
        t = np.arange(t0, T, 0.01)
        def f(x,t):  
            return self.dxdt(x).flatten()
        return odeint(f, x0, t, rtol=1e-6)

    def state_eigens(self, state): 
        return self.jacobian.subs(zip(self.states, state)).eigenvals()

    def steady_state_E5_eigens(self): 
        return self.state_eigens(
            state=self.params.steady_state_E5()
        )

    def steady_state_E4_eigens(self): 
        return self.state_eigens(
            state=self.params.steady_state_E4()
        )

    def steady_state_E3_eigens(self): 
        return self.state_eigens(
            state=self.params.steady_state_E3()
        )

    
    def is_limit_state(self, state, x0, T=1000, T_scale=2, eps=0.01, depth=3): 
        " Does trajectory initialy at state 'x0' approach 'state'?" 
        t0 = 0
        for _ in range(depth): 
            sol = self.integrate(x0, T, t0=t0) 
            if max(sol[-1] - state) < eps: 
                return True  
            # Else, integrate some more
            x0 = sol[-1]
            t0 = T
            T *= T_scale
        return False

    def is_global_limit_state(self, state, T=1000, T_scale=2, eps=0.01, depth=3): 
        """ Do all inner trajectories approach 'state'?
          Returns None if true"""
        failed = []
        for i, x0 in enumerate(self.params.iterate_init_conds()): 
            #print(i, x0)
            if not self.is_limit_state(state, x0, T, T_scale, eps, depth): 
                failed.append(x0)
        return None if len(failed) == 0 else failed

def plot(system, x0, T, state=None, title=None): 
    fig = plt.figure(figsize=(7, 7.5))
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

#p = [0.06, 0.1912, 0.5916, 0.6, 0.01]
    #parameters_b = [0.06, 0.2768, 0.5792, 0.6, 0.01]

    sol = system.integrate(x0, T)
    ax.plot(sol[:,0], sol[:,1], sol[:,2], lw=0.5)

    if state: 
        plt.plot(*state, 'ro')

    plt.show()
