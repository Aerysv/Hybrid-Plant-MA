from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def crear_SIM(tEnd):
    # Parámetros del modelo BUENO
    R = 0.00831 # kJ/mol/K
    Ca0 = 5.0   # mol/L
    V = 11.5    # L
    Vc = 1.0    # L
    rho = 1.0   # kg/L
    Cp = 4.184  # kJ/kg/K
  
    k10 = 9.94755854e+10
    k20 = 9.98553963e+11
    k30 = 9.99263023e+12
    Ea1 = 58.9922611
    Ea2 = 77.6157046
    Ea3 = 71.1106314
    dHrxn1 = -21.2199341
    dHrxn2 = -2.68152145
    dHrxn3 = -66.5367189
    alpha = 1.59324467

    # Condiciones iniciales
    Ca_ini = 0.06
    Cb_ini = 0.32
    T_ini = 25.0
    Tc_ini = 22.0

    # Perturbaciones
    T0 = 20.0
    Tc0 = 20.0

    #tEnd = 0.5  # Horizonte de Predicción

    # Declaración del modelo
    m = ConcreteModel(name="ReactorVandeVusse")
    # Declaración del tiempo
    m.t = ContinuousSet(bounds=(0.0, tEnd))

    # Integración simulación/Planta
    # Condiciones iniciales
    m.Ca_ini = Param(initialize=0.06, mutable=True)
    m.Cb_ini = Param(initialize=2, mutable=True)
    m.T_ini = Param(initialize=25, mutable=True)
    m.Tc_ini = Param(initialize=22, mutable=True)

    # Declaración de las variables dependientes
    # Estados
    m.Ca = Var(m.t, within=PositiveReals, initialize=Ca_ini)
    m.Cb = Var(m.t, within=PositiveReals, initialize=Cb_ini)
    m.T = Var(m.t, within=PositiveReals, initialize=T_ini)
    m.Tc = Var(m.t, within=PositiveReals, initialize=Tc_ini)

    # Declaración de las variables de decisión
    # Acciones de control
    m.q = Param(initialize=0.75, mutable=True)
    m.Fr = Param(initialize=8.51, mutable=True)

    # Derivadas de los estados
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
        return V*m.Ca_dot[t] == m.q*(Ca0 - m.Ca[t]) + V*(-k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] 
                                                            - 2*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2)

    def _dCbdt(m, t):
        return V*m.Cb_dot[t] == -m.q*m.Cb[t] + V*(k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] 
                                                    - k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t])

    def _dTdt(m, t):
        return rho*Cp*V*m.T_dot[t] == m.q*rho*Cp*(T0 - m.T[t]) - alpha*m.Fr**0.8*(m.T[t] - m.Tc[t]) + \
                                    V*(-dHrxn1*k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] 
                                        - dHrxn2*k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t]
                                        - 2*dHrxn3*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2)

    def _dTcdt(m, t):
        return rho*Cp*Vc*m.Tc_dot[t] == m.Fr*rho*Cp*(Tc0 - m.Tc[t]) + alpha*m.Fr**0.8*(m.T[t] - m.Tc[t])

    # Ecuaciones diferenciales
    m.ode_Ca = Constraint(m.t, rule=_dCadt)
    m.ode_Cb = Constraint(m.t, rule=_dCbdt)
    m.ode_T = Constraint(m.t, rule=_dTdt)
    m.ode_Tc = Constraint(m.t, rule=_dTcdt)

    return m

def graficar_sim(tsim, profiles, U):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    axs[0, 0].plot(tsim, profiles[:,0], 'b', label='Ca')
    axs[0, 0].scatter(tsim[-1], profiles[-1,0], c='b')
    axs[0, 0].text(tsim[-1], profiles[-1,0]*1.002, f'{profiles[-1,0]:.2f}' , fontsize='small')
    axs[0, 0].legend()
    axs[0, 0].set_title('Concentración A')
    axs[0, 0].set_ylabel('Cocentración [mol/L]')

    axs[0, 1].plot(tsim, profiles[:,1], 'b', label='Cb')
    axs[0, 1].scatter(tsim[-1], profiles[-1,1], c='b')
    axs[0, 1].text(tsim[-1], profiles[-1,1]*1.002, f'{profiles[-1,1]:.2f}', fontsize='small')
    axs[0, 1].legend()
    axs[0, 1].set_title('Concentración B')
    axs[0, 1].set_ylabel('Cocentración [mol/L]')

    axs[1, 0].plot(tsim, profiles[:,2], 'b', label='T')
    axs[1, 0].scatter(tsim[-1], profiles[-1,2], c='b')
    axs[1, 0].text(tsim[-1], profiles[-1,2]*1.002, f'{profiles[-1,2]:.2f}' , fontsize='small')
    axs[1, 0].legend()
    axs[1, 0].set_title('Temperatura del Reactor')
    axs[1, 0].set_ylabel('Temperatura [ºC]')

    axs[1, 1].plot(tsim, profiles[:,3], 'b', label='Tc')
    axs[1, 1].scatter(tsim[-1], profiles[-1,3], c='b')
    axs[1, 1].text(tsim[-1], profiles[-1,3]*1.002, f'{profiles[-1,3]:.2f}', fontsize='small')
    axs[1, 1].legend()
    axs[1, 1].set_title('Temperatura del serpentín')
    axs[1, 1].set_ylabel('Temperatura [ºC]')

    axs[2, 0].plot(tsim, U[:,0], 'b', label='q')
    axs[2, 0].scatter(tsim[-1], U[-1,0], c='b')
    axs[2, 0].text(tsim[-1], U[-1,0]*1.002, f'{U[-1,0]:.2f}', fontsize='small')
    axs[2, 0].legend()
    axs[2, 0].set_title('Caudal de reactivos')
    axs[2, 0].set_ylabel('Caudal [L/min]')
    axs[2, 0].set_xlabel('Tiempo [min]')

    axs[2, 1].plot(tsim, U[:,1], 'b', label='Fr')
    axs[2, 1].scatter(tsim[-1], U[-1,1], c='b')
    axs[2, 1].text(tsim[-1], U[-1,1]*1.002, f'{U[-1,1]:.2f}', fontsize='small')
    axs[2, 1].legend()
    axs[2, 1].set_title('Caudal de refrigerante')
    axs[2, 1].set_ylabel('Caudal [L/min]')
    axs[2, 1].set_xlabel('Tiempo [min]')

def costo_constraint_planta_steady(m_proc, limT):

    # Precios
    pA = 0.2  # (euro/mol)
    pB = 18  # (euro/mol)
    pFr = 3  # (euro/mol)

    # Llamada al simulador
    sim = Simulator(m_proc, package='casadi')
    tsim, profiles = sim.simulate(numpoints=101, integrator='idas')
    J_costo = value(m_proc.q)*(pB*profiles[-1,1] - pA*5.0) - value(m_proc.Fr)*pFr
    g1 = profiles[-1,2] - limT
    return J_costo, g1

if __name__ == "__main__":
    tSample=0.5
    m_sim = crear_SIM(tEnd=0.5)
    TIME = np.array([0])
    Y = np.array([[0, 0, 0, 0]])
    U = np.array([[0, 0]])

    # Llamada al simulador
    sim = Simulator(m_sim, package='casadi')
    tsim, profiles = sim.simulate(numpoints=101, integrator='idas')

    TIME = np.append(TIME, tsim+TIME[-1])
    Y = np.append(Y, profiles, axis=0)
    u1 = np.ones(len(tsim))*value(m_sim.q)
    u2 = np.ones(len(tsim))*value(m_sim.Fr)
    U = np.append(U, np.stack((u1, u2), axis=1), axis=0)

    m_sim.Fr = 0.1
    m_sim.Ca[0.0] = profiles[-1,0]
    m_sim.Cb[0.0] = profiles[-1,1]
    m_sim.T[0.0] = profiles[-1,2]
    m_sim.Tc[0.0] = profiles[-1,3]
    sim = Simulator(m_sim, package='casadi')
    tsim, profiles = sim.simulate(numpoints=101, integrator='idas')

    TIME = np.append(TIME, tsim+TIME[-1])
    Y = np.append(Y, profiles, axis=0)
    u1 = np.ones(len(tsim))*value(m_sim.q)
    u2 = np.ones(len(tsim))*value(m_sim.Fr)
    U = np.append(U, np.stack((u1, u2), axis=1), axis=0)

    TIME = np.delete(TIME, 0)
    Y = np.delete(Y, 0, axis=0)
    U = np.delete(U, 0, axis=0)

    graficar_sim(TIME, Y, U)
    plt.show()



'''
#axs[2, 0].step(tiempo, [value(m.q[x]) for x in tiempo], 'b', label='T')
axs[2, 0].step(list(m.t), [value(m.q[x]) for x in m.t], 'b', label='T')
axs[2, 0].legend()
axs[2, 0].set_title('Caudal de Reactivos')
axs[2, 0].set_ylabel('Caudal [L/min]')
axs[2, 0].set_xlabel('Tiempo [min]')

#axs[2, 1].step(tiempo, [value(m.Fr[x]) for x in tiempo], 'b', label='Tc')
axs[2, 1].step(list(m.t), [value(m.Fr[x]) for x in m.t], 'b', label='Tc')
axs[2, 1].legend()
axs[2, 1].set_title('Caudal de Refrigerante')
axs[2, 1].set_ylabel('Caudal [L/min]')
axs[2, 1].set_xlabel('Tiempo [min]')
'''


