import asyncio
from asyncua import ua
from datetime import datetime
from pyomo.environ import *
import logging
import numpy as np
import MPC as MPC
from calculo_grad import *
from rels import *
from nlms import *
import MHE as MHE
import DME as DME

_logger = logging.getLogger("Server")

class Controlador():
    def __init__(s):
        # Configuración MPC
        s.conMA = True
        s.flagMHE = True
        s.opcion_grad = 2             # 1- Exacto, 2- NLMS, 3- DME
        s.tSample = 0.5

        s.K = 1.0
        s.k_MA = 0
        s.mu_J = 0.05
        s.mu_g1 = 0.05

        s.Lambda = [0.0, 0.0]
        s.theta_J_ant = [0.0]*15
        s.theta_g_ant = [0.0]*15
        s.LimsupT = 32.0
        # Iniaciación es
        s.m_MPC = MPC.crear_MPC()
        s.m_MPC.LimsupT = s.LimsupT
        if s.flagMHE:
            s.m_MHE = MHE.crear_MHE()
        if s.opcion_grad == 4:
            s.m_DME = DME.crear_DME()
        # Variables de configuración
        s.tSample = 0.5
        s.Nu = 3   # Control horizon (number of sampling periods)
        s.Nd = 2   # Number of measured disturbances
        s.Nx = 4   # Number of states of the model
        s.Nm = 4   # Number of measured process variables
        s.MV = 2   # Number of manipulated variables of the MPC
        s.Ne = 4   # Number of (past and current) samples used in MHE
        s.Ndme = 3  # Number of (past and current) samples used in DME
        s.PV = 2   # Number of process variables with Upper/lower constraints

        s.Lu_i = [None]*s.MV
        s.Lu_s = [None]*s.MV
        s.Ly_i = [None]*2
        s.Ly_s = [None]*2

        # MHE
        s.state = [None]*s.Nx
        s.v_new = [0.0]*s.Nx
        s.error = [0.0]*s.Nx
        s.beta_x_ant = 1
        s.beta_xv = 1

        s.state_real = [None]*s.Nx
        s.uq1 = 0.75
        s.uFr1 = 8.51
        s.u_new = [s.uq1, s.uFr1]
        
        s.acc = [None]*s.MV
        s.per = [None]*s.Nd
        s.med = [None]*s.Nm
        s.acc_ant = [None]*s.MV*(s.Ne+1)
        s.per_ant = [None]*s.Nd*(s.Ne+1)
        s.med_ant = [None]*s.Nm*(s.Ne+1)
        s.aux = [None]*4
        s.config = [None]*5

        # RELS
        s.alpha = 0.86  # alpha = 0.2 - oscila mucho
        s.I = np.eye(15,15)
        s.sigma_inv_cero = 1/s.alpha*s.I
        s.sigma_inv_ant = s.sigma_inv_cero

        s.J_y_g_ant = [None]*3*(s.Ne+1)
        s.J_y_g = [None]*4
        s.J_g1_ant = [None]*4
        s.J_p_ant = [None]*4
        s.u_ant = [None]*(s.MV*s.Ne)
        s.Qdu_ant = [0.0]*s.Ndme	    # Esfuerzos de control anteriores
        s.Q_du_k = 0.0
        s.du_ant = [0.0]*(s.MV*s.Ndme)   # Cambios anteriores de u para Lambda
        s.du_k = [None]*s.MV		    # Cambios  de u para Lambda en k

        s.Lambda = [0.0, 0.0]                        # Modificadores funcion costo
        s.Gamma = [0.0, 0.0]                         # Modificadores restriccion primer orden
        s.Epsilon = 0.0                            # Modificadores restriccion cero orden
        s.Theta_ant = [0.0]*(s.MV*s.Ndme)
        s.j_DME = 0.0
        s.j_m_DME = 0.0
        s.j_modified_DME = 0.0

    async def escribir_variables(s, server):
        # Los arrays en el espacio de nombres del servidor empiezan en 1, y no en cero como en Python
        # MHE
        for i in range(s.Nx):
            await server.write_attribute_value(server.get_node(f"ns=4;s=state[{i+1}]").nodeid, ua.DataValue(s.state[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=v_new[{i+1}]").nodeid, ua.DataValue(s.v_new[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=error[{i+1}]").nodeid, ua.DataValue(s.error[i]))

            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: state[{i+1}] = {s.state[i]}')
            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: v_new[{i+1}] = {s.v_new[i]}')
            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: error[{i+1}] = {s.error[i]}')
        
        
        for i in range(2):
            # Gradientes
            await server.write_attribute_value(server.get_node(f"ns=4;s=grad_m[{i+1}]").nodeid, ua.DataValue(s.grad_m[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=grad_p[{i+1}]").nodeid, ua.DataValue(s.grad_p[i]))
            # MA
            await server.write_attribute_value(server.get_node(f"ns=4;s=Lambda[{i+1}]").nodeid, ua.DataValue(s.Lambda[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=Gamma[{i+1}]").nodeid, ua.DataValue(s.Gamma[i]))

            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: grad_m[{i+1}] = {s.grad_m[i]}')
            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: grad_p[{i+1}] = {s.grad_p[i]}')
            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: Lambda[{i+1}] = {s.Lambda[i]}')
            _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: Gamma[{i+1}] = {s.Gamma[i]}')
        
        await server.write_attribute_value(server.get_node(f"ns=4;s=Epsilon").nodeid, ua.DataValue(s.Epsilon))
        _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: Epsilon = {s.Epsilon}')

        # Para verificar que está calculando el controlador
        await server.write_attribute_value(server.get_node(f"ns=4;s=MPC_T_end").nodeid, ua.DataValue(s.m_MPC.T[60.0].value+s.m_MPC.error[2].value))
        _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: MPC_T_end = {s.m_MPC.T[60.0].value+s.m_MPC.error[2].value}')

        await server.write_attribute_value(server.get_node(f"ns=4;s=MPC_g1").nodeid, ua.DataValue(value(s.m_MPC.c1)))
        _logger.info(f' [{datetime.now().strftime("%H:%M:%S.%f")[:-3]}]\t Node written: MPC_g1 = {value(s.m_MPC.c1)}')

    async def recibir_variables(s, server):
        # Tipo de control
        s.conMA = await server.get_node("ns=4;s=conMA").read_value()
        s.opcion_grad = await server.get_node("ns=4;s=opciones_grad").read_value()
        # Configuración NLMS
        s.mu_J = await server.get_node("ns=4;s=mu_J").read_value()
        # MHE
        s.beta_x_ant = await server.get_node("ns=4;s=beta_xN").read_value()
        s.beta_xv = await server.get_node("ns=4;s=beta_xv").read_value()
        # Datos de entrada
        s.Lu_i[0] = await server.get_node("ns=4;s=Liminfq").read_value()
        s.Lu_s[0] = await server.get_node("ns=4;s=Limsupq").read_value()
        s.Lu_i[1] = await server.get_node("ns=4;s=LiminfFr").read_value()
        s.Lu_s[1] = await server.get_node("ns=4;s=LimsupFr").read_value()
        s.Ly_i[0] = await server.get_node("ns=4;s=LiminfT").read_value()
        s.Ly_s[0] = await server.get_node("ns=4;s=LimsupT").read_value()
        s.Ly_i[1] = await server.get_node("ns=4;s=LiminfCb").read_value()
        s.Ly_s[1] = await server.get_node("ns=4;s=LimsupCb").read_value()

        s.config[0] = await server.get_node("ns=4;s=pA").read_value()   # pA
        s.config[1] = await server.get_node("ns=4;s=pB").read_value()   # pB
        s.config[2]= await server.get_node("ns=4;s=pFr").read_value()   # pFr
        s.config[3] = await server.get_node("ns=4;s=beta[1]").read_value()   # beta1
        s.config[4] = await server.get_node("ns=4;s=beta[2]").read_value()   # beta2

        s.acc[0] = await server.get_node("ns=4;s=q").read_value()   # q
        s.acc[1] = await server.get_node("ns=4;s=Fr").read_value()  # Fr

        s.per[0] = await server.get_node("ns=4;s=T0").read_value()  # T0
        s.per[1] = await server.get_node("ns=4;s=Tc0").read_value() # Tc0

        s.med[0] = await server.get_node("ns=4;s=Ca").read_value()  # Ca
        s.med[1] = await server.get_node("ns=4;s=Cb").read_value()  # Cb
        s.med[2] = await server.get_node("ns=4;s=T").read_value()   # T
        s.med[3] = await server.get_node("ns=4;s=Tc").read_value()  # Tc

        s.aux[0] = s.acc[0] # uqant
        s.aux[1] = s.acc[1] # uFrant
        s.aux[2] = s.per[0] # T0
        s.aux[3] = s.per[1] # Tc0
        # J_costo_real = q*(pB*Cb - pA*Ca0) - pFr*Fr	
        s.J_y_g[0] = s.acc[0]*(s.config[1]*s.med[1] - s.config[0]*5.0) - s.config[2]*s.acc[1]
        s.J_y_g[1] = s.med[2] - s.LimsupT
        s.J_y_g[2] = 0  # --T - LimsupT

    def actualizar_arrays(s):
        # Actualizar el vector de acciones de control
        for i in range(0, s.MV):
            for j in range(0, s.Ne):
                s.acc_ant[s.MV*j+i] = s.acc_ant[s.MV*(j+1)+i]
            s.acc_ant[s.MV*s.Ne+i] = s.acc[i]

        # Actualizar el vector de perturbaciones medidas
        for i in range(0, s.Nd):
            for j in range(0, s.Ne):
                s.per_ant[s.Nd*j+i] = s.per_ant[s.Nd*(j+1)+i]
            s.per_ant[s.Nd*s.Ne+i] = s.per[i]

        # Actualizar vector de medidas
        for i in range(0, s.Nm):
            for j in range(0, s.Ne):
                s.med_ant[s.Nm*j+i] = s.med_ant[s.Nm*(j+1)+i]
            s.med_ant[s.Nm*s.Ne+i] = s.med[i]

        # Actualizar el valor de la función de costo y las restricciones
        for i in range(0, 3):
            for j in range(0, s.Ne):
                s.J_y_g_ant[3*j+i] = s.J_y_g_ant[3*(j+1)+i]
            s.J_y_g_ant[3*s.Ne+i] = s.J_y_g[i]

        # Actualizar controles anteriores aplicados
        # Revisar que estas iteraciones si estén funcionando bien
        for i in range(0, s.MV):
            for j in range(0, s.Ne-1):
                s.u_ant[s.MV*j+i] = s.u_ant[s.MV*(j+1)+i]
            s.u_ant[s.MV*(s.Ne-1)+i] = s.u_new[i]

        # Actualizar valores para DME
        # Revisar que estas iteraciones si estén funcionando bien
        for i in range(1, s.Ndme):
            s.Qdu_ant[i-1] = s.Qdu_ant[i]

        s.Qdu_ant[s.Ndme-1] = s.Q_du_k

        # Revisar que estas iteraciones si estén funcionando bien
        for i in range(1, s.MV+1):
            for j in range(1, s.Ndme):
                s.du_ant[s.MV*(j-1)+i-1] = s.du_ant[s.MV*j+i-1]

        # Revisar que estas iteraciones si estén funcionando bien
        for i in range(1, s.MV+1):
            s.du_ant[s.MV*(s.Ndme-1)+i-1] = s.du_k[i-1]


    def ejecutar(s):

        # LLamada al MHE
        # ___________________________________________________________________
        if s.flagMHE:
            print("\tEjecutnado MHE")
            MHE.actualizar_MHE(s.m_MHE, s.acc_ant, s.per_ant, s.med_ant, s.beta_xv, s.beta_x_ant)
            s.state, s.v_new, s.error = MHE.ejecutar_MHE(s.m_MHE, s.Ne, s.tSample)

        else:
            s.state = s.med
            s.v_new = [0.0]*s.Nx
            s.error = [0.0]*s.Nx
        # _________________________________________________________________MHE

        # Calculo de modificadores con MA
        if (s.conMA == True):

            s.k_MA = s.k_MA + 1

            if (s.opcion_grad == 1):
                print("Calculando grad exactos")
                s.grad_m, s.g1_m = grad_m_DD(s.state, s.per, s.aux, s.v_new, s.error, s.config, s.LimsupT)
                s.grad_p, s.g1_p = grad_p_DD(s.state, s.per, s.aux, s.LimsupT)

                s.Lambda = filtro_mod([s.grad_p[0], s.grad_p[1]], [s.grad_m[0], s.grad_m[1]], s.K, s.Lambda, s.k_MA)
                s.Gamma = filtro_mod([s.grad_p[2], s.grad_p[3]], [s.grad_m[2], s.grad_m[3]], s.K, s.Gamma, s.k_MA)

                if (s.k_MA == 1):
                    s.Epsilon = s.g1_p - s.g1_m
                else:
                    s.Epsilon = s.Epsilon*(1-s.K) + s.K*(s.g1_p - s.g1_m)            
            
            elif ((s.opcion_grad == 2) or (s.opcion_grad == 3)):
                
                s.grad_m, s.g1_m  = grad_m_DD(s.state, s.per, s.aux, s.v_new, s.error, s.config, s.LimsupT)

                # Valores más actuales están a finales del vector
                # Para NLMS/RELS sólo necesito los 3 últimos valores
                # El mayor indice tiene el valor más actual
                s.J_p_ant[0] = s.J_y_g_ant[0]  # indices son: 0, 3, 6, 9, 12
                s.J_p_ant[1] = s.J_y_g_ant[3]
                s.J_p_ant[2] = s.J_y_g_ant[6]
                s.J_p_ant[3] = s.J_y_g_ant[9]

                s.J_g1_ant[0] = s.J_y_g_ant[1]  # indices son: 1, 4, 7, 10, 13
                s.J_g1_ant[1] = s.J_y_g_ant[4] 
                s.J_g1_ant[2] = s.J_y_g_ant[7]
                s.J_g1_ant[3] = s.J_y_g_ant[10]

                if (s.opcion_grad == 2):
                    print("Estimando grad proceso y restricciones por NLMS")
                    theta = NLMS(s.u_ant, s.J_p_ant, s.J_y_g[0], s.theta_J_ant, s.mu_J)
                    s.theta_J_ant = theta

                    theta_g = NLMS(s.u_ant, s.J_g1_ant, s.J_y_g[1], s.theta_g_ant, s.mu_g1)
                    s.theta_g_ant = theta_g

                elif (s.opcion_grad == 3):
                    print("Estimando grad proceso por RELS")
                    theta,  sigma_inv = RELS(s.u_ant, s.J_p_ant, s.J_y_g[0], s.theta_J_ant, s.sigma_inv_ant, s.alpha)
                    s.theta_J_ant = theta
                    s.sigma_inv_ant = sigma_inv

                s.grad_p = [theta[0], theta[1]]            
                s.Lambda = filtro_mod(s.grad_p, [s.grad_m[0], s.grad_m[1]], s.K, s.Lambda, s.k_MA)

                s.grad_g1 = [theta_g[0],theta_g[1]]
                s.Gamma = filtro_mod(s.grad_g1, [s.grad_m[2], s.grad_m[3]], s.K, s.Gamma, s.k_MA)

                s.g1_p = s.J_y_g[1]

                if (s.k_MA == 1):
                    s.Epsilon = s.g1_p - s.g1_m
                else:
                    s.Epsilon = s.Epsilon*(1-s.K) + s.K*(s.g1_p - s.g1_m)   

            elif(s.opcion_grad == 4):
                print("Calculando modificadores por DME")
                DME.actualizar_DME(s.m_DME, s.acc_ant, s.per_ant, s.med_ant, s.Qdu_ant, s.du_ant, s.Theta_ant, s.v_new, s.error)            
                s.Lambda_new, s.Theta = DME.ejecutar_DME(s.m_DME, s.du_k, [s.config[3], s.config[4]], s.tSample)
                s.Theta_ant = s.Theta

                for i in range(0, 2):
                    s.Lambda[i] = s.Lambda[i]*(1-s.K) + s.K*s.Lambda_new[i]

                s.j_DME = value(s.m_DME.J_proc[s.tSample])
                s.j_m_DME = value(s.m_DME.J_modelo[s.tSample])
                s.j_modified_DME = value(s.m_DME.J_modified[s.tSample])
        else:
            s.Lambda = [0.0, 0.0]
            s.Gamma = [0.0, 0.0]
            s.Epsilon = 0.0
            s.j_DME = s.J_y_g[0]
            s.j_m_DME = 0.0
            s.j_modified_DME = 0.0

        # LLamada al controlador
        # ___________________________________________________________________
        MPC.actualizar_MPC(s.m_MPC, s.uq1, s.uFr1, s.per, s.state, s.v_new, s.error, s.Lambda, s.Gamma, s.Epsilon)
        s.uq, s.uFr = MPC.ejecutar_MPC(s.m_MPC, s.tSample)
        s.uq1 = s.uq[0]
        s.uFr1 = s.uFr[0]
        s.u_new = [s.uq1, s.uFr1]
        # _________________________________________________________controlador


if __name__ == '__main__':
    prueba = Controlador()

    prueba.acc[0] = 0.1
    prueba.acc[1] = 1

    prueba.per[0] = 20
    prueba.per[1] = 20
    
    prueba.med[0] = 2
    prueba.med[1] = 20
    prueba.med[2] = 200
    prueba.med[3] = 2000

    prueba.actualizar_arrays()

    print(prueba.per_ant)