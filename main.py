# Librerías
import matplotlib.pyplot as plt
import numpy as np
import time
# Ficheros propios
from MPC import *
from simulacion import *
from calculo_grad import *
from nlms import *
import MHE as MHE

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
Ndme = 3  # Number of (past and current) samples used in DME
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
LimsupT = 60.0
LiminfCb = 0.0
LimsupCb = 5.0
beta = [2.0, 2.0]  # Penalizacion de cambios

# Precios
pA = 0.2  # (euro/mol)
pB = 18.0  # (euro/mol)
pFr = 3.0  # (euro/mol)

# Parametros MA
conMA = True
K = 0.5  # filtro de los modificadores
opcion_grad = 2             # 1- Exacto, 2- NLMS, 3- DME
flagMHE = True

# NLMS
theta_J_ant = [0.0]*15
mu_J = 1

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
J_p_ant = [None]*3
u_ant = [None]*(MV*Ne)  # valores pasados del control aplicado
x_Ndme = [None]*Nx		    # Estimated state at t- Ndme
Qdu_ant = [0.0]*Ndme	    # Esfuerzos de control anteriores
du_ant = [0.0]*(MV*Ndme)   # Cambios anteriores de u para Lambda
du_k = [None]*MV		    # Cambios  de u para Lambda en k
# Vector of decision variables computed by the MPC + slacks
u_new = [None]*(MV*Nu)
v_new = [None]*Nx  # valores estimados de las perturbaciones
Lambda = [0.0, 0.0]                            # Modificadores funcion costo

# INICIALIZACIÓN
Ca = 0.06
Cb = 0.32
T = 25.0
Tc = 22.0
T0 = 20.0
Tc0 = 20.0
v_ini = [0.0]*Nx  # Inicializacion vector de perturbaciones
error = [0.0]*Nx
Q_du_k = 0.0

# Acciones de control
uq1 = 0.75
uFr1 = 8.51

# ---------BLOCO INIT --------------------
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

profiles = np.array([[Ca, Cb, T, Tc]])
TIME = np.array([0])            # Array de tiempo para graficar
Y = np.array([[0, 0, 0, 0]])    # Array de estados para graficar
U = np.array([[0, 0]])          # Array de q y Fr para graficar
J = np.array([0])               # Array de J para graficar

# Iteracion de MA
k_MA = 0

solver = SolverFactory('ipopt')
solver.options['tol'] = 1e-4
solver.options['linear_solver'] = 'ma57'

t0 = time.time()    # Borrar
for k_sim in range(0, 121):

    # Actualizar vectores
    acc = [value(m_SIM.q), value(m_SIM.Fr)]

    per = [T0, Tc0]
    
    med = state_real # Cuatro estados, en OPC debe hacerse por cada elemento

    # -------------------------------------------
    # uqant, uFrant, T0, Tc0
    aux = [uq1, uFr1, T0, Tc0]

    J_y_g[0] = acc[0]*(pB*med[1] - pA*5) - pFr*acc[1]
    J_y_g[1] = 0
    J_y_g[2] = 0

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

    Qdu_ant[Ndme-1] = Q_du_k

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
        print("\tEjecutnado MHE")
        MHE.actualizar_MHE(m_MHE, acc_ant, per_ant, med_ant)
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
            grad_m = grad_m_DD(med, per, aux, v_new, error, config)
            grad_p = grad_p_DD(med, aux)

            Lambda = filtro_mod(grad_p, grad_m, K, Lambda, k_MA)
        
        elif (opcion_grad == 2) & (k_sim > Ne+1):
            
            grad_m = grad_m_DD(med, per, aux, v_new, error, config)

            print("Estimando grad proceso por NLMS")
            # Valores más actuales están a finales del vector
            # Para NLMS sólo necesito los 3 últimos valores
            # El mayor indice tiene el valor más actual
            J_p_ant[0] = J_y_g_ant[6]  # indices son: 0, 3, 6, 9, 12
            J_p_ant[1] = J_y_g_ant[9]
            J_p_ant[2] = J_y_g_ant[12]

            J_costo_real = acc[0]*(pB*med[1] - pA*5) - pFr*acc[1]

            J_y_g = [J_costo_real, 0.0, 0.0]

            theta = NLMS(u_ant, J_p_ant, J_y_g[0], theta_J_ant, mu_J)
            theta_J_ant = theta
            grad_p = [theta[0], theta[1]]
            Lambda = filtro_mod(grad_p, grad_m,K,Lambda,k_MA)

        elif(opcion_grad == 3):
            print("Calculando mod por DME")
            Q_du_k = 0.0  # beta[1]*((uq[1]-u_ant[MV*(Ndme-1)+1])**2  + SUM( i IN 2,Nu;(uq[i]-uq[i-1])**2 )) \
            # + beta[2]*((uFr[1]-u_ant[MV*(Ndme-1)+2])**2 + SUM( i IN 2,Nu;(uFr[i]-uFr[i-1])**2 ))
            du_k[1] = (uq1 - u_ant[MV*(Ndme-1)+1])
            du_k[2] = (uFr1 - u_ant[MV*(Ndme-1)+2])
    else:
        Lambda = [0.0, 0.0]

    # LLamada al controlador
    # ___________________________________________________________________
    actualizar_MPC(m_MPC, uq1, uFr1, state, v_new, error, Lambda)
    uq1, uFr1 = ejecutar_MPC(m_MPC, tSample)
    u_new = [uq1, uFr1]
    # _________________________________________________________controlador

    # Llamada al simulador
    # ___________________________________________________________________
    m_SIM.Ca[0.0] = profiles[-1, 0]
    m_SIM.Cb[0.0] = profiles[-1, 1]
    m_SIM.T[0.0] = profiles[-1, 2]
    m_SIM.Tc[0.0] = profiles[-1, 3]

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
    u1 = np.ones(len(tsim))*value(m_MPC.q[tSample])
    u2 = np.ones(len(tsim))*value(m_MPC.Fr[tSample])
    U = np.append(U, np.stack((u1, u2), axis=1), axis=0)
    j = np.ones(len(tsim))*J_y_g[0]
    J = np.append(J, j)

t1 = time.time()    # Borrar

print(f'Tiempo total: {(t1-t0):.2f} segundos')

TIME = np.delete(TIME, 0)
Y = np.delete(Y, 0, axis=0)
U = np.delete(U, 0, axis=0)
J = np.delete(J, 0)

graficar_sim(TIME, Y, U)
plt.savefig(f"figuras/Estados_conMA_{conMA}_opcion_{opcion_grad}.pdf")
# plt.show()
graficar_costo(TIME, J, U)
plt.savefig(f"figuras/Costo_conMA_{conMA}_opcion_{opcion_grad}.pdf")
# plt.show()
