# Funcion NLMS
# Estimacion de gradientes de proceso usando expansion de Taylor de segundo grado, datos de proceso y el algoritmo NLMS de identificación
# Entradas:
# func_actual -> valor medido del proceso en el instante k
# func_ant -> valores pasados del proceso
# u_ant -> valores pasados de las variables manipuladas
# theta -> vector de gradientes calculados en la iteracion anterior
# mu -> parametro del NLMS
# rho -> número pequeño que impida la división por cero en NLMS
import numpy as np
from numpy import linalg as LA

def NLMS(u_ant, func_ant, func_actual, theta, mu, rho = 1e-4):

    # Inicializando vectores
    dq  = np.array([0.0, 0.0])
    dFr = np.array([0.0, 0.0])
    phi = np.array([None]*15)

    # Vectores de medidas: mayor indice es el valor anterior más actual
	# En el NLMS los vectores dq y dFr el menor indice es lo más actual		
    dFunc_p = func_actual - func_ant[3]

    dq[0]   = u_ant[6] - u_ant[4]	#u1 - indices 0, 2, 4, 6
    dq[1]   = u_ant[4] - u_ant[2]

    dFr[0]  = u_ant[7] - u_ant[5]   #u2 - indices 1, 3, 5, 7
    dFr[1]  = u_ant[5] - u_ant[3]

    phi[0]  =  dq[0]
    phi[1]  =  dFr[0]
    phi[2]  =  dq[1]
    phi[3]  =  dFr[1]

    phi[4]  =  dq[0]*dq[0]/2
    phi[5]  =  dq[0]*dFr[0]
    phi[6]  =  dFr[0]*dFr[0]/2

    phi[7]  =  dq[0]*dq[1]/2
    phi[8]  =  dq[0]*dFr[1]
    phi[9]  =  dFr[0]*dq[1]	
    phi[10] =  dFr[0]*dFr[1]/2

    phi[11] =  dq[1]*dq[1]/2
    phi[12] =  dq[1]*dFr[1]
    phi[13] =  dFr[1]*dFr[1]/2

    #dfuncdt = (1/6)*(11*(func_actual)-18*func_ant[2]+9*func_ant[1]-2*func_ant[0])
    dfuncdt = (1/6)*(11*(func_ant[3])-18*func_ant[2]+9*func_ant[1]-2*func_ant[0])
    phi[14] = dfuncdt

    eps = dFunc_p - np.dot(phi,theta)  #Diferencia entre funcion medida y calculada (error)

    # NLMS Normalized Least Mean Square
    theta_new = theta + mu*(phi*eps)/(rho+LA.norm(phi)**2)

    return theta_new
