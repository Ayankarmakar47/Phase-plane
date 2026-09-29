import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)

#################  Define symbols ###################
x, y = sp.symbols("x y")

################## Get user input #####################
expression1 = input("Enter dxdt (e.g. -y or x - y): ")
expression2 = input("Enter dydt (e.g. x or x + y): ")

######################  Parse input ################
transformations = standard_transformations + (implicit_multiplication_application,)
func1 = parse_expr(expression1, transformations=transformations)
func2 = parse_expr(expression2, transformations=transformations)

####################  Convert SymPy expressions to NumPy functions #####################
f1_num = sp.lambdify((x, y), func1, modules='numpy')
f2_num = sp.lambdify((x, y), func2, modules='numpy')

########################################################
############# CHECK FOR LINEARITY ######################
########################################################

def check_linear(expr, *vars):
    try:
       return sp.Poly(expr, *vars).is_linear
    except sp.PolynomialError:
        return False

print("Is func1 linear?", check_linear(func1, x, y))  
print("Is func2 linear?", check_linear(func2, x, y))  

###########################################################################
############### FINDING THE NATURE OF EQUILIBRIUM POINT ####################
############################################################################




if check_linear(func1, x, y) and check_linear(func2, x, y): 
    A, b = sp.linear_eq_to_matrix([func1, func2], [x, y])

    print("\nMatrix A (Coefficients):")
    sp.pprint(A)

############# CONVERT MATRIX A TO NUMPY ARRAY ################

    A_num = np.array(A, dtype=float)
    ev = np.linalg.eigvals(A_num)

#############################################################
############## CLASSIFICATION OF FIXED POINTS ###############
##############################################################

    if ev[0].imag==0:
        if ev[0].real > 0 and ev[1].real > 0:
                print("The nature of the equilibrium point is a Source (Unstable)")
        if ev[0].real < 0 and ev[1].real < 0:
                print("The nature of the equilibrium point is a Sink (Stable)")
        if ev[0].real * ev[1].real < 0:
                print("The equilibrium point is a Saddle point")
    
    if ev[0].imag!=0:
         if ev[0].real==0:
              print("The Equilobrium point is centre")
         if ev[0].real!=0:
              print("The Equilibrium point is spiral")
              if ev[0].real>0:
                   print("The sprial is unstable")
              else :
                    print("The spiral is stable")


##################################################################
################### PLOTTING THE PHASE PORTRAIT ##################
##################################################################

# 3. Set up the grid
x1 = np.linspace(-2.0, 2.0, 20)
x2 = np.linspace(-2.0, 2.0, 20)
X1, X2 = np.meshgrid(x1, x2)

# 4. Compute vector components numerically across the meshgrid
U = f1_num(X1, X2)
V = f2_num(X1, X2)

'''
# BUG FIX 5: Keep this uncommented to prevent plotting crashes on constant inputs
U = np.broadcast_to(U, X1.shape).astype(float)
V = np.broadcast_to(V, X2.shape).astype(float)
'''
# 5. Normalize arrows to prevent massive overlapping
magnitude = np.hypot(U, V)
magnitude[magnitude == 0] = 1.0  # Prevent division by zero
U_norm = U / magnitude
V_norm = V / magnitude

# 6. Create the plot
fig, ax = plt.subplots(figsize=(8, 7))

ax.quiver(X1, X2, U_norm, V_norm, color='darkblue', 
          scale=30, width=0.003, headwidth=3, headlength=5, pivot='mid')

# 7. Apply formatting
ax.set_title("Phase Portrait", fontsize=14)
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
ax.set_xlim([-2.1, 2.1])
ax.set_ylim([-2.1, 2.1])

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()