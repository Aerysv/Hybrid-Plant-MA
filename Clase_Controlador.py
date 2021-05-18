import asyncio
from asyncua import ua
from pyomo.environ import *
import numpy as np
import MPC as MPC
import calculo_grad as grd
import nlms as nlms
import MHE as MHE

class Controlador():
    def __init__(s):
        # Configuración MPC
        s.conMA = True
        s.flagMHE = True
        s.opcion_grad = 2             # 1- Exacto, 2- NLMS, 3- DME
        s.tSample = 0.5
        s.K = 0.5
        s.k_MA = 0
        s.Lambda = [0.0, 0.0]
        s.mu_J = 1
        s.theta_J_ant = [0.0]*15
        # Iniaciación es
        s.m_MPC = MPC.crear_MPC()
        if s.flagMHE:
            s.m_MHE = MHE.crear_MHE()
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

        s.state = [None]*s.Nx
        s.v_new = [None]*s.Nx
        s.error = [None]*s.Nx

        s.state_real = [None]*s.Nx
        s.u_new = [None]*s.MV*s.Nu
        
        s.acc = [None]*s.MV
        s.per = [None]*s.Nd
        s.med = [None]*s.Nm
        s.acc_ant = [None]*s.MV*(s.Ne+1)
        s.per_ant = [None]*s.Nd*(s.Ne+1)
        s.med_ant = [None]*s.Nm*(s.Ne+1)
        s.aux = [None]*4
        s.config = [None]*5

        s.uq1 = 0.75
        s.uFr1 = 8.51

        s.J_y_g_ant = [None]*3*(s.Ne+1)
        s.J_y_g = [None]*3
        s.J_p_ant = [None]*3
        s.u_ant = [None]*(s.MV*s.Ne)
        s.Qdu_ant = [0.0]*s.Ndme	    # Esfuerzos de control anteriores
        s.du_ant = [0.0]*(s.MV*s.Ndme)   # Cambios anteriores de u para Lambda
        s.du_k = [None]*s.MV		    # Cambios  de u para Lambda en k

    async def escribir_variables(s, server):
        # Los arrays en el espacio de nombres del servidor empiezan en 1, y no en cero como en Python
        # MHE
        for i in range(s.Nx):
            await server.write_attribute_value(server.get_node(f"ns=4;s=state[{i+1}]").nodeid, ua.DataValue(s.state[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=v_new[{i+1}]").nodeid, ua.DataValue(s.v_new[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=error[{i+1}]").nodeid, ua.DataValue(s.error[i]))
        
        
        for i in range(2):
            # Gradientes
            await server.write_attribute_value(server.get_node(f"ns=4;s=grad_m[{i+1}]").nodeid, ua.DataValue(s.grad_m[i]))
            await server.write_attribute_value(server.get_node(f"ns=4;s=grad_p[{i+1}]").nodeid, ua.DataValue(s.grad_p[i]))
            # MA
            await server.write_attribute_value(server.get_node(f"ns=4;s=Lambda[{i+1}]").nodeid, ua.DataValue(s.Lambda[i]))

    async def recibir_variables(s, server):
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
        s.config[3] = await server.get_node("ns=4;s=beta[1]").get_value()   # beta1
        s.config[4] = await server.get_node("ns=4;s=beta[2]").get_value()   # beta2

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
        s.J_y_g[0] = s.acc[0]*(s.config[1]*s.med[1] - s.config[0]*5) - s.config[2]*s.acc[1]
        s.J_y_g[1] = 0  # -- -T + LiminfT
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
            MHE.actualizar_MHE(s.m_MHE, s.acc_ant, s.per_ant, s.med_ant)
            s.state, s.v_new, s.error = MHE.ejecutar_MHE(s.m_MHE, s.Ne, s.tSample)

        else:
            s.state = s.med
            s.v_new = [0.0]*s.Nx
            s.error = [0.0]*s.Nx
        # _________________________________________________________________MHE

        # Calculo de modificadores con MA
        if (s.conMA == True):

            s.k_MA += 1

            if (s.opcion_grad == 1):
                print("Calculando grad exactos")
                s.grad_m = grd.grad_m_DD(s.med, s.per, s.aux, s.v_new, s.error, s.config)
                s.grad_p = grd.grad_p_DD(s.med, s.aux)

                s.Lambda = grd.filtro_mod(s.grad_p, s.grad_m, s.K, s.Lambda, s.k_MA)
            
            elif s.opcion_grad == 2:
                
                s.grad_m = grd.grad_m_DD(s.med, s.per, s.aux, s.v_new, s.error, s.config)

                print("Estimando grad proceso por NLMS")
                # Valores más actuales están a finales del vector
                # Para NLMS sólo necesito los 3 últimos valores
                # El mayor indice tiene el valor más actual
                s.J_p_ant[0] = s.J_y_g_ant[6]  # indices son: 0, 3, 6, 9, 12
                s.J_p_ant[1] = s.J_y_g_ant[9]
                s.J_p_ant[2] = s.J_y_g_ant[12]

                J_costo_real = s.acc[0]*(s.config[1]*s.med[1] - s.config[0]*5) - s.config[2]*s.acc[1]

                J_y_g = [J_costo_real, 0.0, 0.0]

                theta = nlms.NLMS(s.u_ant, s.J_p_ant, J_y_g[0], s.theta_J_ant, s.mu_J)
                s.theta_J_ant = theta
                s.grad_p = [theta[0], theta[1]]
                s.Lambda = grd.filtro_mod(s.grad_p, s.grad_m, s.K, s.Lambda, s.k_MA)

            elif s.opcion_grad == 3:
                print("Calculando mod por DME")
                Q_du_k = 0.0  # beta[1]*((uq[1]-u_ant[MV*(Ndme-1)+1])**2  + SUM( i IN 2,Nu;(uq[i]-uq[i-1])**2 )) \
                # + beta[2]*((uFr[1]-u_ant[MV*(Ndme-1)+2])**2 + SUM( i IN 2,Nu;(uFr[i]-uFr[i-1])**2 ))
                s.du_k[1] = (s.uq1 - s.u_ant[s.MV*(s.Ndme-1)+1])
                s.du_k[2] = (s.uFr1 - s.u_ant[s.MV*(s.Ndme-1)+2])
        else:
            # Sin MA
            s.Lambda = [0.0, 0.0]

        # LLamada al controlador
        # ___________________________________________________________________
        MPC.actualizar_MPC(s.m_MPC, s.uq1, s.uFr1, s.state, s.v_new, s.error, s.Lambda)
        s.uq1, s.uFr1 = MPC.ejecutar_MPC(s.m_MPC, s.tSample) 
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