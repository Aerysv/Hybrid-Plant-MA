from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def crear_MODELO():

    # Parámetros del modelo MALO
    R = 0.00831 # kJ/mol/K
    Ca0 = 5.0   # mol/L
    V = 11.5    # L
    Vc = 1.0    # L
    rho = 1.0   # kg/L
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
  
    # Condiciones iniciales
    Ca_ini = 0.06
    Cb_ini = 0.32
    T_ini = 25.0
    Tc_ini = 22.0

    tEnd = 120  # Horizonte de Predicción para garantizar estado estacionario

    # Declaración del modelo
    m = ConcreteModel(name="ModelReactorVandeVusse")
    # Declaración del tiempo
    m.t = ContinuousSet(bounds=(0.0, tEnd))

    # Integración simulación/Planta
    # Perturbaciones
    m.T0 = Param(default=20, mutable=True)
    m.Tc0 = Param(default=20, mutable=True)
    # Condiciones iniciales
    m.Ca_ini = Param(initialize=0.06, mutable=True)
    m.Cb_ini = Param(initialize=2, mutable=True)
    m.T_ini = Param(initialize=25, mutable=True)
    m.Tc_ini = Param(initialize=22, mutable=True)

    # Integración MHE
    m.error = Param([0,1,2,3], default=0, mutable=True)
    m.v = Param([0,1,2,3], default=0, mutable=True)

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
                                                            - 2*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2) + m.v[0]

    def _dCbdt(m, t):
        return V*m.Cb_dot[t] == -m.q*m.Cb[t] + V*(k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] 
                                                    - k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t]) + m.v[1]

    def _dTdt(m, t):
        return rho*Cp*V*m.T_dot[t] == m.q*rho*Cp*(m.T0 - m.T[t]) - alpha*m.Fr**0.8*(m.T[t] - m.Tc[t]) + \
                                    V*(-dHrxn1*k10*exp(-Ea1/(R*(m.T[t]+273.15)))*m.Ca[t] 
                                        - dHrxn2*k20*exp(-Ea2/(R*(m.T[t]+273.15)))*m.Cb[t]
                                        - 2*dHrxn3*k30*exp(-Ea3/(R*(m.T[t]+273.15)))*m.Ca[t]**2)  + m.v[2]

    def _dTcdt(m, t):
        return rho*Cp*Vc*m.Tc_dot[t] == m.Fr*rho*Cp*(m.Tc0 - m.Tc[t]) + alpha*m.Fr**0.8*(m.T[t] - m.Tc[t])  + m.v[3]

    # Ecuaciones diferenciales
    m.ode_Ca = Constraint(m.t, rule=_dCadt)
    m.ode_Cb = Constraint(m.t, rule=_dCbdt)
    m.ode_T = Constraint(m.t, rule=_dTdt)
    m.ode_Tc = Constraint(m.t, rule=_dTcdt)

    return m

def graficar_modelo(tsim, profiles):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    axs[0, 0].plot(tsim, profiles[:,0], 'b', label='Ca')
    axs[0, 0].legend()
    axs[0, 0].set_title('Concentración A')
    axs[0, 0].set_ylabel('Cocentración [mol/L]')

    axs[0, 1].plot(tsim, profiles[:,1], 'b', label='Cb')
    axs[0, 1].axhline(3)
    axs[0, 1].legend()
    axs[0, 1].set_title('Concentración B')
    axs[0, 1].set_ylabel('Cocentración [mol/L]')

    axs[1, 0].plot(tsim, profiles[:,2], 'b', label='T')
    axs[1, 0].axhline(30)
    axs[1, 0].legend()
    axs[1, 0].set_title('Temperatura del Reactor')
    axs[1, 0].set_ylabel('Temperatura [ºC]')

    axs[1, 1].plot(tsim, profiles[:,3], 'b', label='Tc')
    axs[1, 1].legend()
    axs[1, 1].set_title('Temperatura del serpentín')
    axs[1, 1].set_ylabel('Temperatura [ºC]')
    plt.show()

def costo_model_steady(m_model, config):

    # Precios
    pA = config[0]    # (euro/mol)
    pB = config[1]  # (euro/mol)
    pFr = config[2]  # (euro/mol)

    # Llamada al simulador
    sim = Simulator(m_model, package='casadi')
    tsim, profiles = sim.simulate(numpoints=101, integrator='idas')
    Cb = profiles[-1,1]
    q = value(m_model.q)
    Fr = value(m_model.Fr)
    return q*(pB*(Cb + value(m_model.error[1])) - pA*5.0) - pFr*Fr


if __name__ == "__main__":
    tSample=0.5
    m_sim = crear_MODELO()
    TIME = np.array([0])
    Y = np.array([[0, 0, 0, 0]])

    # Llamada al simulador
    sim = Simulator(m_sim, package='casadi')
    tsim, profiles = sim.simulate(numpoints=101, integrator='idas')

    TIME = np.append(TIME, tsim+TIME[-1])
    Y = np.append(Y, profiles, axis=0)

    m_sim.Fr = 0.1
    m_sim.Ca[0.0] = profiles[-1,0]
    m_sim.Cb[0.0] = profiles[-1,1]
    m_sim.T[0.0] = profiles[-1,2]
    m_sim.Tc[0.0] = profiles[-1,3]
    sim = Simulator(m_sim, package='casadi')
    tsim, profiles = sim.simulate(numpoints=101, integrator='idas')
    TIME = np.append(TIME, tsim+TIME[-1])
    Y = np.append(Y, profiles, axis=0)

    TIME = np.delete(TIME, 0)
    Y = np.delete(Y, 0, axis=0)
    print(tsim)
    graficar_modelo(TIME, Y)

    print("hola Modelo")


