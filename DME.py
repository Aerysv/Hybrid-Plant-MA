from main import T
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def crear_DME(MV=2):
    # Parámetros del modelo MALO
    R = 0.00831  # kJ/mol/K
    Ca0 = 5.0     # mol/L
    V = 11.5    # L
    Vc = 1.0      # L
    rho = 1.0     # kg/L
    Cp = 4.184  # kJ/kg/K
    
    k10 = 6.4e+9
    k20 = 4.84e+10
    k30 = 8.86e+11
    Ea1 = 52.1
    Ea2 = 70.0
    Ea3 = 65.0
    dHrxn1 = -48.75
    dHrxn2 = -34.50
    dHrxn3 = -40.25
    alpha = 1.8
 
    # Precios
    pA = 0.2    # €/L
    pB = 18.0     # €/L
    pFr = 3.0     # €/L

    # Sintonía del DME
    LiminfLambda = -2500
    LimsupLambda =  2500
    tEnd = 1.5
    tSample = 0.5  # Minutos
    nSamples = int(tEnd/tSample) + 1

    tiempo = np.round(np.linspace(0, tEnd, nSamples), decimals=6)

    # Declaración del modelo
    m = ConcreteModel(name="ReactorVandeVusse")
    # Declaración del tiempo
    m.t = ContinuousSet(initialize=tiempo)    

    # Variables medidas
    # Acciones de control
    m.q = Param(m.t, default=0.75, mutable=True)
    m.Fr = Param(m.t, default=8.51, mutable=True)
    # Perturbaciones
    m.T0 = Param(m.t, default=20, mutable=True)
    m.Tc0 = Param(m.t, default=20, mutable=True)
    # Estados
    m.Ca_m = Param(m.t, default=0.05, mutable=True)
    m.Cb_m = Param(m.t, default=3, mutable=True)
    m.T_m = Param(m.t, default=30, mutable=True)
    m.Tc_m = Param(m.t, default=20, mutable=True)

    # Parametros de sintonía del DME
    LiminfLambda = -2500
    LimsupLambda =  2500
    tEnd = 1.5
    tSample = 0.5  # Minutos
    nSamples = int(tEnd/tSample) + 1
    beta_theta = [0.1, 0.1]
    beta_cost = 1.0
    m.beta_theta = Param(default=beta_theta, mutable=True)
    m.beta_cost = Param(default=beta_cost, mutable=True)

    # Variables de decisión pasadas
    m.v_ant = Param([0,1,2,3], default=0, mutable=True)
    m.Ca_ant = Param(default=0.05, mutable=True)
    m.Cb_ant = Param(default=3, mutable=True)
    m.T_ant = Param(default=30, mutable=True)
    m.Tc_ant = Param(default=20, mutable=True) 

    # Integración simulación/Planta
    # Condiciones iniciales
    Ca_ini = 0.06
    Cb_ini = 0.32
    T_ini = 25.0
    Tc_ini = 22.0

    # Integración simulación
    m.uqant = Param(initialize=0.75, mutable=True)
    m.uFrant = Param(initialize=8.51, mutable=True)

    # Integración MHE
    m.error = Param([0,1,2,3], initialize={0:0.0, 1:0.0, 2:0.0, 3:0.0}, mutable=True)
    v_new = [0.0, 0.0, 0.0, 0.0]

    # Integración MA
    m.Theta = Param([0,1], initialize={0:0.0, 1:0.0}, mutable=True)

    # Integración con controlador
    m.Qdu = Param([0,1,2], initialize={0:0.0, 1:0.0, 2:0.0}, mutable=True)  # Esfuerzos de control no aplicados al proceso    

    # Declaración de las variables dependientes
    m.Ca = Var(m.t, within=PositiveReals)
    m.Cb = Var(m.t, within=PositiveReals)
    m.T = Var(m.t, within=PositiveReals)
    m.Tc = Var(m.t, within=PositiveReals)

    # Declaración de las variables de decisión
    # Lambdas actuales y pasados
    m.Lambda1 = Var([0, 1, 2], bounds=(LiminfLambda, LimsupLambda), initialize=0.0)    
    m.Lambda2 = Var([0, 1, 2], bounds=(LiminfLambda, LimsupLambda), initialize=0.0)    
    
    # Declaración de las derivadas de las variables
    m.Ca_dot = DerivativeVar(m.Ca, wrt=m.t)
    m.Cb_dot = DerivativeVar(m.Cb, wrt=m.t)
    m.T_dot = DerivativeVar(m.T, wrt=m.t)
    m.Tc_dot = DerivativeVar(m.Tc, wrt=m.t)

    # Condiciones iniciales
    m.Ca[0.0].fix(Ca_ini)
    m.Cb[0.0].fix(Cb_ini)
    m.T[0.0].fix(T_ini)
    m.Tc[0.0].fix(Tc_ini)

    # Ecuaciones del modelo
    def _dCadt(m, t):
        return V*m.Ca_dot[t] == m.q[t]*(Ca0 - m.Ca[t]) + V*(-k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] \
                                                            - 2*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2) + m.v[0]

    def _dCbdt(m, t):
        if m.t == m.t.first():
            return Constraint.Skip
        return V*m.Cb_dot[t] == -m.q[t]*m.Cb[t] + V*(k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] \
                                                     - k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t]) + m.v[1]

    def _dTdt(m, t):
        if m.t == m.t.first():
            return Constraint.Skip
        return rho*Cp*V*m.T_dot[t] == m.q[t]*rho*Cp*(m.T0[t] - m.T[t]) - alpha*m.Fr[t]**0.8*(m.T[t] - m.Tc[t]) + \
                                        V*(-dHrxn1*k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] \
                                           - dHrxn2*k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t] \
                                           - 2*dHrxn3*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2) + m.v[2]

    def _dTcdt(m, t):
        if m.t == m.t.first(): # Para que eso?
            return Constraint.Skip
        return rho*Cp*Vc*m.Tc_dot[t] == m.Fr[t]*rho*Cp*(m.Tc0[t] - m.Tc[t]) + \
                                        + alpha*m.Fr[t]**0.8*(m.T[t] - m.Tc[t]) + m.v[3]

    # Declaración de las ODES
    m.ode_Ca = Constraint(m.t, rule=_dCadt)
    m.ode_Cb = Constraint(m.t, rule=_dCbdt)
    m.ode_T = Constraint(m.t, rule=_dTdt)
    m.ode_Tc = Constraint(m.t, rule=_dTcdt)

    #Definir la función de coste
    def _obj(m):

        tS = tSample

        for i in range(nSamples):
            m.Level[i] = m.Qdu[i*tS] #Qdu[0] + (Qdu[i+1]-Qdu[i])*sigmoide(TIME,i*t_Sample,Sig))

        # Theta*deltau: el mayor indice es el más actual
        for i in range (0, MV):
            for j in range(nSamples):
                m.theta_deltau[i] = m.Theta[i,j]*m.du[i,j] #Theta[i,1]*du[i,1] + (Theta[i,j+1]*du[i,j+1]-Theta[i,j]*du[i,j])*sigmoide(TIME,j*t_Sample,Sig))

        # Past process data
        for i in range(nSamples):
            m.J_proc[i]   = m.q[i*tS]*(pB*m.Cb_m[i*tS] - pA*Ca0) - m.Fr[i*tS]*pFr

        for i in range(nSamples):
            m.J_modelo[i] = m.q[i*tS]*(pB*(m.Cb[i*tS]+m.error[1]) - pA*Ca0) - m.Fr[i*tS]*pFr

        # Valor del costo y resticciones del Modelo + MA
        for i in range(nSamples):
            m.J_modified[i] = m.J_modelo[i] + m.theta_deltau[0] + m.theta_deltau[1] + m.Level[i]	

        # Calculando diferencia al cuadrado entre el proceso y modelo
        Delta_J = 0.0
        for i in range(nSamples):
            Delta_J += (m.J_proc[i] - m.J_modified[i])*(m.J_proc[i] - m.J_modified[i])  

        for i in range(nSamples):
            Delta_theta = m.beta_theta[0]*((m.Theta[1,j]-m.Theta_ant[MV*(j-1)])**2) + m.beta_theta[1]*((m.Theta[2,j]-m.Theta_ant[MV*(j-1)+1])**2) 

        J_DME = m.beta_cost*Delta_J + Delta_theta #INTEGRAL--!!!!

        return J_DME
    
    m.obj = Objective(rule=_obj, sense=minimize)

    # Discretizar con colocación en elementos finitos
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=nSamples, ncp=3, scheme='LAGRANGE-RADAU')
    m = discretizer.reduce_collocation_points(m, var=m.q, ncp=1, contset=m.t)
    m = discretizer.reduce_collocation_points(m, var=m.Fr, ncp=1, contset=m.t)

    return m

def actualizar_DME(m, acc, per, med, MV=2, Nd=2, Nm=4, Ne=4, tSample=0.5):
    t_fe = m.t._fe
    t_MHE = [val for val in m.t]
    # Actualización en los elementos finitos
    for i, t in enumerate(t_fe):
        # Variables manipuladas
        for j in range(0,3):
            if t<(Ne*tSample):
                idx = t_MHE[t_MHE.index(t)+j]
            elif t==Ne*tSample:
                idx = t_MHE[-1]
            # Variables manipuladas
            m.q[idx] = acc[MV*i]
            m.Fr[idx] = acc[MV*i+1]
            # Perturbaciones
            m.T0[idx] = per[Nd*i]
            m.Tc0[idx] = per[Nd*i+1]
            # Estados medidos
            m.Ca_m[idx] = med[Nm*i]
            m.Cb_m[idx] = med[Nm*i+1]
            m.T_m[idx] = med[Nm*i+2]
            m.Tc_m[idx] = med[Nm*i+3]

    # Actualización de las variables de decisión anteriores
    # Vector de perturbaciones
    for i in range(0, Nm):
        m.v_ant[i] = m.v[i].value
    # Estimaciones iniciales anteriores
    m.Ca_ant = m.Ca[m.t.first()].value
    m.Cb_ant = m.Cb[m.t.first()].value
    m.T_ant = m.T[m.t.first()].value
    m.Tc_ant = m.Tc[m.t.first()].value 

def graficar_DME(m_DME):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    axs[0, 0].plot(list(m_DME.t), [value(m_DME.Ca[x]) for x in m_DME.t], 'b', label='Ca')
    axs[0, 0].legend()
    axs[0, 0].set_title('Concentración A')
    axs[0, 0].set_ylabel('Cocentración [mol/L]')

    axs[0, 1].plot(list(m_DME.t), [value(m_DME.Cb[x]) for x in m_DME.t], 'b', label='Cb')
    axs[0, 1].axhline(3)
    axs[0, 1].legend()
    axs[0, 1].set_title('Concentración B')
    axs[0, 1].set_ylabel('Cocentración [mol/L]')

    axs[1, 0].plot(list(m_DME.t), [value(m_DME.T[x]) for x in m_DME.t], 'b', label='T')
    axs[1, 0].axhline(30)
    axs[1, 0].legend()
    axs[1, 0].set_title('Temperatura del Reactor')
    axs[1, 0].set_ylabel('Temperatura [ºC]')

    axs[1, 1].plot(list(m_DME.t), [value(m_DME.Tc[x]) for x in m_DME.t], 'b', label='Tc')
    axs[1, 1].legend()
    axs[1, 1].set_title('Temperatura del serpentín')
    axs[1, 1].set_ylabel('Temperatura [ºC]')

    #axs[2, 0].step(tiempo, [value(m.q[x]) for x in tiempo], 'b', label='T')
    axs[2, 0].step(list(m_DME.t), [value(m_DME.q[x]) for x in m_DME.t], 'b', label='T')
    axs[2, 0].legend()
    axs[2, 0].set_title('Caudal de Reactivos')
    axs[2, 0].set_ylabel('Caudal [L/min]')
    axs[2, 0].set_xlabel('Tiempo [min]')

    #axs[2, 1].step(tiempo, [value(m.Fr[x]) for x in tiempo], 'b', label='Tc')
    axs[2, 1].step(list(m_DME.t), [value(m_DME.Fr[x]) for x in m_DME.t], 'b', label='Tc')
    axs[2, 1].legend()
    axs[2, 1].set_title('Caudal de Refrigerante')
    axs[2, 1].set_ylabel('Caudal [L/min]')
    axs[2, 1].set_xlabel('Tiempo [min]')

    plt.show()


if __name__ == '__main__':
    tSample = 0.5
    m_DME = crear_DME()
    solver = SolverFactory('ipopt')
    results = solver.solve(m_DME)        # Llamada al solver
    print(f'q = {value(m_DME.q[tSample])}')
    print(f'Fr = {value(m_DME.Fr[tSample])}')
    m_DME.Ca[0.0] = 0.07
    m_DME.Cb[0.0] = 3.2
    m_DME.T[0.0] = 30
    m_DME.Tc[0.0] = 14
    m_DME.Tsp = 40
    solver = SolverFactory('ipopt')
    results = solver.solve(m_DME)        # Llamada al solver
    print(f'q = {value(m_DME.q[tSample])}')
    print(f'Fr = {value(m_DME.Fr[tSample])}')
    graficar_DME(m_DME)
    print(results)



