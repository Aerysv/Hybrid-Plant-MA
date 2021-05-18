from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def crear_MHE():
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

    # Sintonía
    tEnd = 2.0
    tSample = 0.5  # Minutos
    nSamples = int(tEnd/tSample) + 1
    beta_xv = 1
    beta_x_ant = 1

    tiempo = np.round(np.linspace(0, tEnd, nSamples), decimals=6)

    # Declaración del modelo
    m = ConcreteModel(name="ReactorVandeVusse_MHE")
    # Declaración del tiempo
    m.t = ContinuousSet(initialize=tiempo)

    # Integración simulación/Planta
    # Condiciones iniciales
    Ca_ini = 0.06
    Cb_ini = 0.32
    T_ini = 25.0
    Tc_ini = 22.0

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

    # Parámetros de sintonía
    m.beta_xv = Param(default=beta_xv, mutable=True)
    m.beta_x_ant = Param(default=beta_x_ant, mutable=True)

    # Variables de decisión pasadas
    m.v_ant = Param([0,1,2,3], default=0, mutable=True)
    m.Ca_ant = Param(default=0.05, mutable=True)
    m.Cb_ant = Param(default=3, mutable=True)
    m.T_ant = Param(default=30, mutable=True)
    m.Tc_ant = Param(default=20, mutable=True)

    # Declaración de las variables de decisión
    # Variables de decisión MHE
    m.error = Var([0,1,2,3], initialize=0)
    m.v = Var([0,1,2,3], initialize=0, bounds=(-0.5, 0.5))

    # Declaración de las variables dependientes
    m.Ca = Var(m.t, within=PositiveReals)
    m.Cb = Var(m.t, within=PositiveReals)
    m.T = Var(m.t, within=PositiveReals)
    m.Tc = Var(m.t, within=PositiveReals)

    # Ecuaciones adicionales
    m.error_eqs = ConstraintList()
    m.error_eqs.add(m.error[0] == m.Ca_m[m.t.last()] - m.Ca[m.t.last()])
    m.error_eqs.add(m.error[1] == m.Cb_m[m.t.last()] - m.Cb[m.t.last()])
    m.error_eqs.add(m.error[2] == m.T_m[m.t.last()] - m.T[m.t.last()])
    m.error_eqs.add(m.error[3] == m.Tc_m[m.t.last()] - m.Tc[m.t.last()])

    # Declaración de las derivadas de las variables
    m.Ca_dot = DerivativeVar(m.Ca, wrt=m.t)
    m.Cb_dot = DerivativeVar(m.Cb, wrt=m.t)
    m.T_dot = DerivativeVar(m.T, wrt=m.t)
    m.Tc_dot = DerivativeVar(m.Tc, wrt=m.t)

    # Condiciones inicilaes
    m.Ca[0.0] = Ca_ini
    m.Cb[0.0] = Cb_ini
    m.T[0.0] = T_ini
    m.Tc[0.0] = Tc_ini

    # Ecuaciones del modelo
    def _dCadt(m, t):
        if m.t == m.t.first():
            return Constraint.Skip
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
        if m.t == m.t.first():
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
        
        # Penalización por desviarse de la solución anterior
        J_costo_N = (m.Ca[0.0]/m.Ca_ant - 1)**2 + (m.Cb[0.0]/m.Cb_ant - 1)**2 + \
			 	    + (m.T[0.0]/m.T_ant - 1)**2 + (m.Tc[0.0]/m.Tc_ant - 1)**2
        
        # Penalización desviación de las variables medidas
        J_costo_m = 0
        for i in range(nSamples):
            J_costo_m += (m.Ca[i*tS]/m.Ca_m[i*tS] - 1)**2 + (m.Cb[i*tS]/m.Cb_m[i*tS] - 1)**2 + \
                         + (m.T[i*tS]/m.T_m[i*tS] - 1)**2 + (m.Tc[i*tS]/m.Tc_m[i*tS] - 1)**2

        # Penalización desviación de perturbaciones anteriores
        J_costo_v = 0
        for i in range(4):
            J_costo_v +=  (m.v[i]-m.v_ant[i])**2

        return 100*J_costo_m + m.beta_x_ant*J_costo_N + m.beta_xv*J_costo_v

    m.obj = Objective(rule=_obj, sense=minimize)

    # Discretizar con colocación en elementos finitos
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=nSamples-1, ncp=3, scheme='LAGRANGE-RADAU')

    return m

def actualizar_MHE(m, acc, per, med, beta_xv, beta_x_ant, MV=2, Nd=2, Nm=4, Ne=4, tSample=0.5):
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
    # Pesos de la función de costo
    m.beta_xv = beta_xv
    m.beta_x_ant = beta_x_ant

def ejecutar_MHE(m_MHE, Ne, tSample):
    solver = SolverFactory('ipopt')
    solver.options['tol'] = 1e-4
    solver.options['linear_solver'] = 'ma57'
    results = solver.solve(m_MHE)        # Llamada al solver

    state = [None]*4
    state[0] = m_MHE.Ca[Ne*tSample].value
    state[1] = m_MHE.Cb[Ne*tSample].value
    state[2] = m_MHE.T[Ne*tSample].value
    state[3] = m_MHE.Tc[Ne*tSample].value
    v_new = [m_MHE.v[i].value for i in range(4)]
    error = [m_MHE.error[i].value for i in range(4)]

    return state, v_new, error


def graficar_MHE(m_MHE):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    axs[0, 0].plot(list(m_MHE.t), [value(m_MHE.Ca[x]) for x in m_MHE.t], 'b', label='Ca')
    axs[0, 0].plot(list(m_MHE.t), [value(m_MHE.Ca_m[x]) for x in m_MHE.t], 'r', label='Ca_m')
    axs[0, 0].legend()
    axs[0, 0].set_title('Concentración A')
    axs[0, 0].set_ylabel('Cocentración [mol/L]')

    axs[0, 1].plot(list(m_MHE.t), [value(m_MHE.Cb[x]) for x in m_MHE.t], 'b', label='Cb')
    axs[0, 1].plot(list(m_MHE.t), [value(m_MHE.Cb_m[x]) for x in m_MHE.t], 'r', label='Cb_m')
    axs[0, 1].legend()
    axs[0, 1].set_title('Concentración B')
    axs[0, 1].set_ylabel('Cocentración [mol/L]')

    axs[1, 0].plot(list(m_MHE.t), [value(m_MHE.T[x]) for x in m_MHE.t], 'b', label='T')
    axs[1, 0].plot(list(m_MHE.t), [value(m_MHE.T_m[x]) for x in m_MHE.t], 'r', label='T_m')
    axs[1, 0].legend()
    axs[1, 0].set_title('Temperatura del Reactor')
    axs[1, 0].set_ylabel('Temperatura [ºC]')

    axs[1, 1].plot(list(m_MHE.t), [value(m_MHE.Tc[x]) for x in m_MHE.t], 'b', label='Tc')
    axs[1, 1].plot(list(m_MHE.t), [value(m_MHE.Tc_m[x]) for x in m_MHE.t], 'r', label='Tc_m')
    axs[1, 1].legend()
    axs[1, 1].set_title('Temperatura del serpentín')
    axs[1, 1].set_ylabel('Temperatura [ºC]')

    #axs[2, 0].step(tiempo, [value(m.q[x]) for x in tiempo], 'b', label='T')
    axs[2, 0].step(list(m_MHE.t), [value(m_MHE.q[x]) for x in m_MHE.t], 'b', label='T')
    axs[2, 0].legend()
    axs[2, 0].set_title('Caudal de Reactivos')
    axs[2, 0].set_ylabel('Caudal [L/min]')
    axs[2, 0].set_xlabel('Tiempo [min]')

    #axs[2, 1].step(tiempo, [value(m.Fr[x]) for x in tiempo], 'b', label='Tc')
    axs[2, 1].step(list(m_MHE.t), [value(m_MHE.Fr[x]) for x in m_MHE.t], 'b', label='Tc')
    axs[2, 1].legend()
    axs[2, 1].set_title('Caudal de Refrigerante')
    axs[2, 1].set_ylabel('Caudal [L/min]')
    axs[2, 1].set_xlabel('Tiempo [min]')

    plt.show()

if __name__ == '__main__':
    tSample = 0.5
    m_MHE = crear_MHE()
    solver = SolverFactory('ipopt')
    results = solver.solve(m_MHE)        # Llamada al solver
    print(f'q = {value(m_MHE.q[tSample])}')
    print(f'Fr = {value(m_MHE.Fr[tSample])}')
    m_MHE.Ca[0.0] = 0.07
    m_MHE.Cb[0.0] = 3.2
    m_MHE.T[0.0] = 30
    m_MHE.Tc[0.0] = 14
    solver = SolverFactory('ipopt')
    results = solver.solve(m_MHE)        # Llamada al solver
    print(f'q = {value(m_MHE.q[tSample])}')
    print(f'Fr = {value(m_MHE.Fr[tSample])}')
    #graficar_MHE(m_MHE)
    print(results)
    m_MHE.Ca[tSample] = 0.69
    m_MHE.Ca[2*tSample] = 0.69

    
    acc = [0.1, 1, 0.2, 2, 0.3, 3, 0.4, 4, 0.5, 5]
    per = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    med = [2, 0.1, 1, 1, 4, 0.2, 2, 2, 6, 0.3, 3, 3, 8, 0.4, 4, 4, 10, 0.5, 5, 5]

    actualizar_MHE(m_MHE, acc, per, med)
    for i in m_MHE.t:
        print(f'Ca_m[{i}] = {value(m_MHE.Ca_m[i])}')

    for i in range(0,4):
        print(m_MHE.v_ant[i].value)
    print(m_MHE.Ca[0.0].value)
    print(m_MHE.Ca_ant.value)