from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def crear_MPC():
    # Parámetros del modelo MALO
    R = 0.00831  # kJ/mol/K
    Ca0 = 5.0     # mol/L
    V = 11.5    # L
    Vc = 1.0      # L
    rho = 1.0     # kg/L
    Cp = 4.184  # kJ/kg/K
    
    k10 = 6.4e+9
    k20 = 4.84e+10
    k30 = 0
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

    # Perturbaciones
    T0 = 20.0
    Tc0 = 20.0

    # Parámetros del controlador
    Liminfq = 0.3
    Limsupq = 1.2

    LiminfFr = 6.0
    LimsupFr = 15.0

    LiminfT = 10.0
    LimsupT = 60.0

    LiminfTc = 10.0
    LimsupTc = 60.0

    LiminfCb = 0.0
    LimsupCb = 5.0

    beta = [2.0, 2.0]

    Pred_h = 60.0  # Horizonte de Predicción
    Nu = 3  # Horizonte de control
    tSample = 0.5  # Minutos
    nSamples = int(Pred_h/tSample) + 1

    tiempo = np.round(np.linspace(0, Pred_h, nSamples), decimals=6)

    # Declaración del modelo
    m = ConcreteModel(name="ReactorVandeVusse")
    # Declaración del tiempo
    m.t = ContinuousSet(initialize=tiempo)

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
    m.error = Param([0,1,2,3], default=0, mutable=True)
    m.v = Param([0,1,2,3], default=0, mutable=True)

    # Integración MA
    m.Lambda = Param([0,1], default=0, mutable=True)
    
    # Declaración de las variables dependientes
    m.Ca = Var(m.t, within=PositiveReals)
    m.Cb = Var(m.t, within=PositiveReals)
    m.T = Var(m.t, within=PositiveReals)
    m.Tc = Var(m.t, within=PositiveReals)

    # Declaración de las variables de decisión
    # Acciones de control
    m.q = Var(m.t, bounds=(Liminfq, Limsupq), initialize=0.75)
    m.Fr = Var(m.t, bounds=(LiminfFr, LimsupFr), initialize=8.51)

    # Definición del horizonte de control
    m.horizonte_q = ConstraintList()
    m.horizonte_Fr = ConstraintList()
    for ti in tiempo:
        if ti > tSample*Nu:
            m.horizonte_q.add(m.q[ti] == m.q[ti-tSample])
            m.horizonte_Fr.add(m.Fr[ti] == m.Fr[ti-tSample])

    # Declaración de las derivadas de las variables
    m.Ca_dot = DerivativeVar(m.Ca, wrt=m.t)
    m.Cb_dot = DerivativeVar(m.Cb, wrt=m.t)
    m.T_dot = DerivativeVar(m.T, wrt=m.t)
    m.Tc_dot = DerivativeVar(m.Tc, wrt=m.t)

    # Condiciones inicilaes
    m.Ca[0.0].fix(Ca_ini)
    m.Cb[0.0].fix(Cb_ini)
    m.T[0.0].fix(T_ini)
    m.Tc[0.0].fix(Tc_ini)

    # Ecuaciones del modelo
    def _dCadt(m, t):
        return V*m.Ca_dot[t] == m.q[t]*(Ca0 - m.Ca[t]) + V*(-k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] - 2*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2) + m.v[0]

    def _dCbdt(m, t):
        return V*m.Cb_dot[t] == -m.q[t]*m.Cb[t] + V*(k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] - k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t]) + m.v[1]

    def _dTdt(m, t):
        return rho*Cp*V*m.T_dot[t] == m.q[t]*rho*Cp*(T0 - m.T[t]) - alpha*m.Fr[t]**0.8*(m.T[t] - m.Tc[t]) + \
                                    V*(-dHrxn1*k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] 
                                        - dHrxn2*k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t]
                                        - 2*dHrxn3*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2) + m.v[2]

    def _dTcdt(m, t):
        return rho*Cp*Vc*m.Tc_dot[t] == m.Fr[t]*rho*Cp*(Tc0 - m.Tc[t]) +alpha*m.Fr[t]**0.8*(m.T[t] - m.Tc[t]) + m.v[3]

    # Declaración de las ODES
    m.ode_Ca = Constraint(m.t, rule=_dCadt)
    m.ode_Cb = Constraint(m.t, rule=_dCbdt)
    m.ode_T = Constraint(m.t, rule=_dTdt)
    m.ode_Tc = Constraint(m.t, rule=_dTcdt)

    # Restricciones de camino
    '''
    m.path_Tlo = Constraint(m.t, rule=lambda m, t: m.T[t] >= LiminfT)
    m.path_Tup = Constraint(m.t, rule=lambda m, t: m.T[t] <= LimsupT)
    m.path_Tclo = Constraint(m.t, rule=lambda m, t: m.Tc[t] >= LiminfTc)
    m.path_Tcup = Constraint(m.t, rule=lambda m, t: m.Tc[t] <= LimsupTc)
    m.path_Cblo = Constraint(m.t, rule=lambda m, t: m.Cb[t] >= LiminfCb)
    m.path_Cbup = Constraint(m.t, rule=lambda m, t: m.Cb[t] <= LimsupCb)
    '''
    #Definir la función de coste
    def _obj(m):

        J_modelo = m.q[Pred_h]*(pB*(m.Cb[Pred_h]+m.error[1]) - pA*Ca0) - m.Fr[Pred_h]*pFr

        J_costo_MA = -(J_modelo + m.Lambda[0]*(m.q[Pred_h] - m.uqant) + m.Lambda[1]*(m.Fr[Pred_h] - m.uFrant))

        tS = tSample

        J_cambios = beta[0]*((m.q[tS] - m.uqant)**2 + (m.q[2*tS]-m.q[tS])**2 + (m.q[3*tS]-m.q[2*tS])**2) + \
                    beta[1]*((m.Fr[tS] - m.uFrant)**2 + (m.Fr[2*tS]-m.Fr[tS])**2 + (m.Fr[3*tS]-m.Fr[2*tS])**2)

        # J_T = sum((m.T[i] - m.Tsp)**2 for i in m.t)
        # J_Cb = sum((m.Cb[i] - 3)**2 for i in m.t)
        # return (m.T[Pred_h]-30)**2 + J_cambios
        # return J_T + J_Cb
        return J_costo_MA + J_cambios

    m.obj = Objective(rule=_obj, sense=minimize)

    # Discretizar con colocación en elementos finitos
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=nSamples-1, ncp=3, scheme='LAGRANGE-RADAU')
    m = discretizer.reduce_collocation_points(m, var=m.q, ncp=1, contset=m.t)
    m = discretizer.reduce_collocation_points(m, var=m.Fr, ncp=1, contset=m.t)

    return m

def actualizar_MPC(m_MPC, uq1, uFr1, state, v, error, Lambda, Nm=4):
    m_MPC.uqant = uq1
    m_MPC.uFrant = uFr1

    m_MPC.Ca[0.0] = state[0]
    m_MPC.Cb[0.0] = state[1]
    m_MPC.T[0.0] = state[2]
    m_MPC.Tc[0.0] = state[3]

    m_MPC.Lambda[0] = Lambda[0]
    m_MPC.Lambda[1] = Lambda[1]

    for i in range(0, Nm):
        m_MPC.v[i] = v[i]
        m_MPC.error[i] = error[i]

def ejecutar_MPC(m_MPC, tSample):
    solver = SolverFactory('ipopt')
    solver.options['tol'] = 1e-4
    solver.options['linear_solver'] = 'ma57'
    results = solver.solve(m_MPC)        # Llamada al solver

    q = [None]*3
    q[0] = m_MPC.q[tSample].value
    q[1] = m_MPC.q[2*tSample].value
    q[2] = m_MPC.q[3*tSample].value
    Fr = [None]*3
    Fr[0] = m_MPC.Fr[tSample].value
    Fr[1] = m_MPC.Fr[2*tSample].value
    Fr[2] = m_MPC.Fr[3*tSample].value

    return q, Fr

def graficar_MPC(m_MPC):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    axs[0, 0].plot(list(m_MPC.t), [value(m_MPC.Ca[x]) for x in m_MPC.t], 'b', label='Ca')
    axs[0, 0].legend()
    axs[0, 0].set_title('Concentración A')
    axs[0, 0].set_ylabel('Cocentración [mol/L]')

    axs[0, 1].plot(list(m_MPC.t), [value(m_MPC.Cb[x]) for x in m_MPC.t], 'b', label='Cb')
    axs[0, 1].axhline(3)
    axs[0, 1].legend()
    axs[0, 1].set_title('Concentración B')
    axs[0, 1].set_ylabel('Cocentración [mol/L]')

    axs[1, 0].plot(list(m_MPC.t), [value(m_MPC.T[x]) for x in m_MPC.t], 'b', label='T')
    axs[1, 0].axhline(30)
    axs[1, 0].legend()
    axs[1, 0].set_title('Temperatura del Reactor')
    axs[1, 0].set_ylabel('Temperatura [ºC]')

    axs[1, 1].plot(list(m_MPC.t), [value(m_MPC.Tc[x]) for x in m_MPC.t], 'b', label='Tc')
    axs[1, 1].legend()
    axs[1, 1].set_title('Temperatura del serpentín')
    axs[1, 1].set_ylabel('Temperatura [ºC]')

    #axs[2, 0].step(tiempo, [value(m.q[x]) for x in tiempo], 'b', label='T')
    axs[2, 0].step(list(m_MPC.t), [value(m_MPC.q[x]) for x in m_MPC.t], 'b', label='T')
    axs[2, 0].legend()
    axs[2, 0].set_title('Caudal de Reactivos')
    axs[2, 0].set_ylabel('Caudal [L/min]')
    axs[2, 0].set_xlabel('Tiempo [min]')

    #axs[2, 1].step(tiempo, [value(m.Fr[x]) for x in tiempo], 'b', label='Tc')
    axs[2, 1].step(list(m_MPC.t), [value(m_MPC.Fr[x]) for x in m_MPC.t], 'b', label='Tc')
    axs[2, 1].legend()
    axs[2, 1].set_title('Caudal de Refrigerante')
    axs[2, 1].set_ylabel('Caudal [L/min]')
    axs[2, 1].set_xlabel('Tiempo [min]')

    plt.show()


if __name__ == '__main__':
    tSample = 0.5
    m_MPC = crear_MPC()
    solver = SolverFactory('ipopt')
    results = solver.solve(m_MPC)        # Llamada al solver
    print(f'q = {value(m_MPC.q[tSample])}')
    print(f'Fr = {value(m_MPC.Fr[tSample])}')
    m_MPC.Ca[0.0] = 0.07
    m_MPC.Cb[0.0] = 3.2
    m_MPC.T[0.0] = 30
    m_MPC.Tc[0.0] = 14
    m_MPC.Tsp = 40
    solver = SolverFactory('ipopt')
    results = solver.solve(m_MPC)        # Llamada al solver
    print(f'q = {value(m_MPC.q[tSample])}')
    print(f'Fr = {value(m_MPC.Fr[tSample])}')
    graficar_MPC(m_MPC)
    print(results)



