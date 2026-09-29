import numpy as np
##import matplotlib.pyplot as plt
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



