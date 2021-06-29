# Librerías
import matplotlib.pyplot as plt
import numpy as np
import time
# Ficheros propios
from MPC import *
from simulacion import *
from calculo_grad import *
from nlms import *
from rels import *
import MHE as MHE
import DME as DME

def graficar_mod(TIME, LAM1, LAM2):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    axs[0].plot(TIME, LAM1, 'b', label='Lambda 1')
    #axs[0].scatter(TIME[-1], LAM1[-1], c='b')
    #axs[0].text(TIME[-1], LAM1[-1]*1.002, f'{LAM1[-1]:.2f}', fontsize='small')
    axs[0].set_ylim(-50, 50)
    axs[0].legend()
    axs[0].set_title('Lambda 1')
    
    axs[1].plot(TIME, LAM2, 'b', label='Lambda 2')
    #axs[1].scatter(TIME[-1], LAM2[-1,0], c='b')
    #axs[1].text(TIME[-1], LAM2[-1,0]*1.002, f'{LAM2[-1,0]:.2f}', fontsize='small')
    axs[1].set_ylim(-6, 6)
    axs[1].legend()
    axs[1].set_title('Lambda 2')

def graficar_DME(TIME, Jp_DME, J_model_DME, J_modif_DME):
    plt.plot(TIME, Jp_DME,'g', TIME, J_model_DME, 'r', TIME, J_modif_DME,'b')
  
def graficar_costo(TIME, J, U):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

    axs[0].plot(TIME, J, 'b', label='J')
    axs[0].scatter(TIME[-1], J[-1], c='b')
    axs[0].text(TIME[-1], J[-1]*1.002, f'{J[-1]:.2f}', fontsize='small')
    axs[0].set_ylim(-20, 45)
    axs[0].legend()
    axs[0].set_title('Función de costo')
    axs[0].set_ylabel('Costo [€/min]')

    axs[1].plot(TIME, U[:,0], 'b', label='q')
    axs[1].scatter(TIME[-1], U[-1,0], c='b')
    axs[1].text(TIME[-1], U[-1,0]*1.002, f'{U[-1,0]:.2f}', fontsize='small')
    axs[1].set_ylim(0.3, 1.2)
    axs[1].legend()
    axs[1].set_title('Caudal de reactivos')
    axs[1].set_ylabel('Caudal [L/min]')

    axs[2].plot(TIME, U[:,1], 'b', label='Fr')
    axs[2].scatter(TIME[-1], U[-1,1], c='b')
    axs[2].text(TIME[-1], U[-1,1]*1.002, f'{U[-1,1]:.2f}', fontsize='small')
    axs[2].set_ylim(6, 15)
    axs[2].legend()
    axs[2].set_title('Caudal de refrigerante')
    axs[2].set_ylabel('Caudal [L/min]')
    axs[2].set_xlabel('Tiempo [min]')

Nu = 3   # Control horizon (number of sampling periods)
Nd = 2   # Number of measured disturbances
Nx = 4   # Number of states of the model
Nm = 4   # Number of measured process variables
MV = 2   # Number of manipulated variables of the MPC
Ne = 4   # Number of (past and current) samples used in MHE
Ndme = 4  # Number of (past and current) samples used in DME
PV = 2   # Number of process variables with Upper/lower constraints

# PARAMETROS DEL CONTROLADOR
Pred_h = 60.0  # Horizonte de Predicción
tSample = 0.5  # Minutos
nSamples = int(Pred_h/tSample) + 1
Liminfq = 0.3
Limsupq = 1.2
LiminfFr = 6.0
LimsupFr = 15.0
LiminfT = 6.0
LimsupT = 32.0
LiminfCb = 0.0
LimsupCb = 5.0
beta = [2.0, 2.0]  # Penalizacion de cambios

# Precios
pA = 0.2  # (euro/mol)
pB = 18.0  # (euro/mol)
pFr = 3.0  # (euro/mol)

# Parametros MA
conMA = False
K = 1.0  # filtro de los modificadores
opcion_grad = 1             # 1- Exacto, 2- NLMS, 3- RELS , 4 -DME
flagMHE = True

# MHE
beta_xv = 1
beta_x_ant = 1

# NLMS & RELS
theta_J_ant = [0.0]*15
theta_g_ant = [0.0]*15

# NLMS
mu_J = 1.8  # Valores mayores que 0.7, 0.5, 0.4, 0.3, 0.2,0.1, 0.05 dan extremos q = 1.18 y Fr = 15.0
mu_g1 = 1.8   # Valores de 0.6 no cumple restricciones Fr = 6.0 q =1.2

# RELS
alpha = 0.86  # alpha = 0.2 - oscila mucho
I = np.eye(15,15)
sigma_inv_cero = 1/alpha*I
sigma_inv_ant = sigma_inv_cero

# Vectores de medidas actuales y pasadas, parametros
acc = [None]*MV  # Valores actuales de las acciones de control
per = [None]*Nd  # Valores actuales de las perturbaciones medidas
med = [None]*Nm  # current measurements
acc_ant = [None]*(MV*(Ne+1))  # Acciones de control medidas pasadas
per_ant = [None]*(Nd*(Ne+1))  # Perturbaciones medidas pasadas
med_ant = [None]*(Nm*(Ne+1))  # past measured variables
aux = [None]*4  # valores auxiliares
# Función de costo y restricciones actuales
J_y_g = [None]*3
# Past cost function and constraints
J_y_g_ant = [None]*(3*(Ne+1))
J_p_ant = [None]*4
J_g1_ant = [None]*4
u_ant = [None]*(MV*Ne)  # valores pasados del control aplicado
x_Ndme = [None]*Nx		    # Estimated state at t- Ndme
Qdu_ant = [0.0]*Ndme	    # Esfuerzos de control anteriores
du_ant = [0.0]*(MV*Ndme)   # Cambios anteriores de u para Lambda
du_k = [None]*MV		    # Cambios  de u para Lambda en k
# Vector of decision variables computed by the MPC + slacks
# Acciones de control
uq1 = 0.75
uFr1 = 8.51
uq = [0.75]*3
uFr = [8.51]*3
u_new = [uq1, uFr1]
v_new = [0.0]*Nx  # valores estimados de las perturbaciones
Lambda = [0.0, 0.0]                        # Modificadores funcion costo
Gamma = [0.0, 0.0]                         # Modificadores restriccion primer orden
Epsilon = 0.0                            # Modificadores restriccion cero orden
Theta_ant = [0.0]*(MV*Ndme)
j_DME = 0.0
j_m_DME = 0.0
j_modified_DME = 0.0

# INICIALIZACIÓN
Ca = 0.06
Cb = 0.32
T = 25.0
Tc = 22.0
T0 = 24.0 #20.0
Tc0 = 24.0 # 20.0
v_ini = [0.0]*Nx  # Inicializacion vector de perturbaciones
error = [0.0]*Nx
Qdu_k = 0.0

# constraints MVs
lim_manip_low = [Liminfq, LiminfFr]  # Limites inferiores
lim_manip_up = [Limsupq, LimsupFr]   # Limites superiores

# Constraints process variables
lim_con_low = [LiminfT, LiminfCb]   # Limites inferiores  
lim_con_up = [LimsupT, LimsupCb]    # Limites superiores

state = [Ca, Cb, T, Tc]
state_real = state

acc = [uq1, uFr1]
per = [T0, Tc0]
med = [Ca, Cb, T, Tc]

config = [pA, pB, pFr, beta[0], beta[1]]

m_MPC = crear_MPC()
m_SIM = crear_SIM(tSample)
if flagMHE:
    m_MHE = MHE.crear_MHE()
if opcion_grad == 4:
    m_DME = DME.crear_DME()

profiles = np.array([[Ca, Cb, T, Tc]])
TIME = np.array([0.0])                   # Array de tiempo para graficar
Y = np.array([[0.0, 0.0, 0.0, 0.0]])     # Array de estados para graficar
U = np.array([[0.0, 0.0]])               # Array de q y Fr para graficar
J = np.array([0.0])                      # Array de J para graficar
J_modelo = np.array([0.0])               # Array de J del modelo para graficar 
J_modificado = np.array([0.0])           # Array de J modificado para graficar 
LAM1 = np.array([0.0])                   # Array de modificador para graficar
LAM2 = np.array([0.0])                   # Array de modificador para graficar

Jp_DME = np.array([0.0])                 # Array de costo medido para graficar
J_model_DME = np.array([0.0])            # Array de costo del modelo para graficar
J_modif_DME = np.array([0.0])            # Array de costo modificado para graficar

# Borrar
V_MHE = np.array([[0.0, 0.0, 0.0, 0.0]])
STATES_MHE = np.array([[0.0, 0.0, 0.0, 0.0]])
STATES_SIM = np.array([[0.0, 0.0, 0.0, 0.0]])
# ________________Borrar

# Iteracion de MA
k_MA = 0

solver = SolverFactory('ipopt')
solver.options['tol'] = 1e-4
solver.options['linear_solver'] = 'ma57'

t0 = time.time()    # Borrar
for k_sim in range(0, 241): #121 241 481

    # Actualizar vectores
    acc = [value(m_SIM.q), value(m_SIM.Fr)]

    per = [T0, Tc0]
    
    med = state_real # Cuatro estados, en OPC debe hacerse por cada elemento

    # -------------------------------------------
    # uqant, uFrant, T0, Tc0
    aux = [uq1, uFr1, T0, Tc0]

    J_y_g[0] = acc[0]*(pB*med[1] - pA*5) - pFr*acc[1]
    J_y_g[1] = med[2] - LimsupT
    J_y_g[2] = 0.0

    # Actualizar el vector de acciones de control
    for i in range(0, MV):
        for j in range(0, Ne):
            acc_ant[MV*j+i] = acc_ant[MV*(j+1)+i]
        acc_ant[MV*Ne+i] = acc[i]

    # Actualizar el vector de perturbaciones medidas
    for i in range(0, Nd):
        for j in range(0, Ne):
            per_ant[Nd*j+i] = per_ant[Nd*(j+1)+i]
        per_ant[Nd*Ne+i] = per[i]

    # Actualizar vector de medidas
    for i in range(0, Nm):
        for j in range(0, Ne):
            med_ant[Nm*j+i] = med_ant[Nm*(j+1)+i]
        med_ant[Nm*Ne+i] = med[i]

    # Actualizar el valor de la función de costo y las restricciones
    for i in range(0, 3):
        for j in range(0, Ne):
            J_y_g_ant[3*j+i] = J_y_g_ant[3*(j+1)+i]
        J_y_g_ant[3*Ne+i] = J_y_g[i]

    # Actualizar controles anteriores aplicados
    # Revisar que estas iteraciones si estén funcionando bien
    for i in range(0, MV):
        for j in range(0, Ne-1):
            u_ant[MV*j+i] = u_ant[MV*(j+1)+i]
        u_ant[MV*(Ne-1)+i] = u_new[i]

    # Actualizar valores para DME
    # Revisar que estas iteraciones si estén funcionando bien
    for i in range(1, Ndme):
        Qdu_ant[i-1] = Qdu_ant[i]

    Qdu_ant[Ndme-1] = Qdu_k

# Revisar que estas iteraciones si estén funcionando bien
    for i in range(1, MV+1):
        for j in range(1, Ndme):
            du_ant[MV*(j-1)+i-1] = du_ant[MV*j+i-1]

# Revisar que estas iteraciones si estén funcionando bien
    for i in range(1, MV+1):
        du_ant[MV*(Ndme-1)+i-1] = du_k[i-1]
    # -------------------------------------------

    # LLamada al MHE
    # ___________________________________________________________________
    if flagMHE & (k_sim>Ne+1):
        print("\tEjecutando MHE")
        MHE.actualizar_MHE(m_MHE, acc_ant, per_ant, med_ant, beta_xv, beta_x_ant)
        state, v_new, error = MHE.ejecutar_MHE(m_MHE, Ne, tSample)

    else:
        state = med
        v_new = [0.0]*Nx
        error = [0.0]*Nx
    # _________________________________________________________________MHE

    # Calculo de modificadores con MA
    if (conMA == True):

        k_MA = k_MA + 1

        if (opcion_grad == 1):
            print("Calculando grad exactos")
            grad_m, g1_m = grad_m_DD(state, per, aux, v_new, error, config)
            grad_p, g1_p = grad_p_DD(state, per, aux)

            Lambda = filtro_mod([grad_p[0], grad_p[1]], [grad_m[0], grad_m[1]], K, Lambda, k_MA)
            Gamma = filtro_mod([grad_p[2], grad_p[3]], [grad_m[2], grad_m[3]],K,Gamma,k_MA)

            if (k_MA == 1):
                for i in range(0, 2):
                    Epsilon = g1_p - g1_m
            else:
                for i in range(0, 2):
                    Epsilon = Epsilon*(1-K) + K*(g1_p - g1_m)            
        
        elif ((opcion_grad == 2) or (opcion_grad == 3)) & (k_sim > Ne+1):
            
            grad_m, g1_m  = grad_m_DD(state, per, aux, v_new, error, config)

            # Valores más actuales están a finales del vector
            # Para NLMS/RELS sólo necesito los 3 últimos valores
            # El mayor indice tiene el valor más actual
            J_p_ant[0] = J_y_g_ant[0]  # indices son: 0, 3, 6, 9, 12
            J_p_ant[1] = J_y_g_ant[3]
            J_p_ant[2] = J_y_g_ant[6]
            J_p_ant[3] = J_y_g_ant[9]

            J_g1_ant[0] = J_y_g_ant[1]  # indices son: 1, 4, 7, 10, 13
            J_g1_ant[1] = J_y_g_ant[4] 
            J_g1_ant[2] = J_y_g_ant[7]
            J_g1_ant[3] = J_y_g_ant[10]

            if (opcion_grad == 2):
                print("Estimando grad proceso y restricciones por NLMS")
                theta = NLMS(u_ant, J_p_ant, J_y_g[0], theta_J_ant, mu_J)
                theta_J_ant = theta

                theta_g = NLMS(u_ant, J_g1_ant, J_y_g[1], theta_g_ant, mu_g1)
                theta_g_ant = theta_g

            elif (opcion_grad == 3):
                print("Estimando grad proceso por RELS")
                theta,  sigma_inv = RELS(u_ant, J_p_ant, J_y_g[0], theta_J_ant, sigma_inv_ant, alpha)
                theta_J_ant = theta
                sigma_inv_ant = sigma_inv

            grad_p = [theta[0], theta[1]]            
            Lambda = filtro_mod(grad_p, [grad_m[0], grad_m[1]],K,Lambda,k_MA)

            grad_g1 = [theta_g[0],theta_g[1]]
            Gamma = filtro_mod(grad_g1, [grad_m[2], grad_m[3]],K,Gamma,k_MA)

            g1_p = J_y_g[1] - LimsupT

            if (k_MA == 1):
                Epsilon = g1_p - g1_m
            else:
                Epsilon = Epsilon*(1-K) + K*(g1_p - g1_m)   

        elif(opcion_grad == 4) & (k_sim > Ne+1):
            print("Calculando modificadores por DME")
            DME.actualizar_DME(m_DME, acc_ant, per_ant, med_ant, Qdu_ant, du_ant, Theta_ant, v_new, error)            
            Lambda_new, Theta = DME.ejecutar_DME(m_DME, du_k, beta,tSample)
            Theta_ant = Theta

            for i in range(0, 2):
                Lambda[i] = Lambda[i]*(1-K) + K*Lambda_new[i]

            j_DME = value(m_DME.J_proc[tSample])
            j_m_DME = value(m_DME.J_modelo[tSample])
            j_modified_DME = value(m_DME.J_modified[tSample])
    else:
        Lambda = [0.0, 0.0]
        Gamma = [0.0, 0.0]
        Epsilon = 0.0
        j_DME = J_y_g[0]
        j_m_DME = 0.0
        j_modified_DME = 0.0

    # LLamada al controlador
    # ___________________________________________________________________
    actualizar_MPC(m_MPC, uq1, uFr1, per, state, v_new, error, Lambda, Gamma, Epsilon)
    uq, uFr = ejecutar_MPC(m_MPC, tSample)
    uq1 = uq[0]
    uFr1 = uFr[0]
    u_new = [uq1, uFr1]
    # _________________________________________________________controlador

    # Actualizando vectores para DME
    # ___________________________________________________________________
    if(opcion_grad == 4):
        Qdu_k = beta[0]*((uq[0]-u_ant[MV*Ndme-2])**2  + (uq[1]-uq[0])**2 + (uq[2]-uq[1])**2 ) \
             + beta[1]*((uFr[1]-u_ant[MV*Ndme-1])**2 +  (uFr[1]-uFr[0])**2 + (uFr[2]-uFr[1])**2 )
        du_k[0] = (uq[0] -  u_ant[MV*Ndme-2])
        du_k[1] = (uFr[0] - u_ant[MV*Ndme-1])
    # __________________________________________________________DME

    # Llamada al simulador
    # ___________________________________________________________________
    m_SIM.Ca[0.0] = profiles[-1, 0]
    m_SIM.Cb[0.0] = profiles[-1, 1]
    m_SIM.T[0.0]  = profiles[-1, 2]
    m_SIM.Tc[0.0] = profiles[-1, 3]

    m_SIM.T0 = per[0]
    m_SIM.Tc0 = per[1]

    m_SIM.q = uq1
    m_SIM.Fr = uFr1

    sim = Simulator(m_SIM, package='casadi')
    tsim, profiles = sim.simulate(numpoints=11, integrator='idas')

    state_real[0] = profiles[-1, 0]
    state_real[1] = profiles[-1, 1]
    state_real[2] = profiles[-1, 2]
    state_real[3] = profiles[-1, 3]

    # ___________________________________________________________simulador

    # Reporte y gráficas
    print(f'TIME = {k_sim*tSample}')
    print(f'\tq = {(value(m_MPC.q[tSample])):.2f}')
    print(f'\tFr = {(value(m_MPC.Fr[tSample])):.2f}')

    TIME = np.append(TIME, tsim+TIME[-1])
    Y = np.append(Y, profiles, axis=0)
    # Debemos crear un array de q y Fr para graficar
    u1 = np.ones(len(tsim))*m_MPC.q[tSample].value
    u2 = np.ones(len(tsim))*m_MPC.Fr[tSample].value
    U = np.append(U, np.stack((u1, u2), axis=1), axis=0)
    j = np.ones(len(tsim))*J_y_g[0]
    J = np.append(J, j)

    # Graficar valores de modificadores estimados
    mod1 = np.ones(len(tsim))*Lambda[0]
    mod2 = np.ones(len(tsim))*Lambda[1]
    LAM1 = np.append(LAM1,mod1)
    LAM2 = np.append(LAM2,mod2)

    # Graficar valores de costos en DME
    Jp_DME = np.append(Jp_DME,np.ones(len(tsim))*j_DME)
    J_model_DME = np.append(J_model_DME,np.ones(len(tsim))*j_m_DME)
    J_modif_DME = np.append(J_modif_DME,np.ones(len(tsim))*j_modified_DME)        

t1 = time.time()    # Borrar

print(f'Tiempo total: {(t1-t0):.2f} segundos')

TIME = np.delete(TIME, 0)
Y = np.delete(Y, 0, axis=0)
U = np.delete(U, 0, axis=0)
J = np.delete(J, 0)
LAM1 = np.delete(LAM1,0)
LAM2 = np.delete(LAM2,0)
Jp_DME = np.delete(Jp_DME,0)
J_model_DME = np.delete(J_model_DME,0)
J_modif_DME = np.delete(J_modif_DME,0)

graficar_sim(TIME, Y, U)
plt.savefig(f"figuras/Estados_conMA_{conMA}_opcion_{opcion_grad}.pdf")
#plt.show()
graficar_costo(TIME, J, U)
plt.savefig(f"figuras/Costo_conMA_{conMA}_opcion_{opcion_grad}.pdf")
#plt.show()

if(conMA == True):
    graficar_mod(TIME, LAM1, LAM2)
    plt.savefig(f"figuras/Modificadores_opcion_{opcion_grad}.pdf")
    #plt.show()

if (opcion_grad==4):
    graficar_DME(TIME, Jp_DME, J_model_DME, J_modif_DME)
    plt.savefig(f"figuras/Costos_DME.pdf")
    #plt.show()
plt.clf()
