from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt

def crear_DME(MV=2,Ndme=4):
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

     # Parametros de sintonía del DME
    LiminfLambda = -2500
    LimsupLambda =  2500
    tEnd = 2.0
    tSample = 0.5  # Minutos
    nSamples = int(tEnd/tSample) + 1
    beta_theta = [0.1, 0.1]
    beta_cost = 1.0

    tiempo = np.round(np.linspace(0, tEnd, nSamples), decimals=6)

    # Vectores para calculos
    theta_deltau = [None]*2

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

    # Sintonia DME
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
    m.error = Param([0,1,2,3], default=0.0, mutable=True)
    m.v = Param([0,1,2,3], default=0.0, mutable=True)
    
    # Integración con controlador
    m.Qdu = Param([0,1,2], default=0.0, mutable=True)  # Esfuerzos de control no aplicados al proceso
    m.du =  Param([0,1],[0,1,2,3], default=0.0, mutable=True) 

    # Declaración de las variables dependientes
    m.Ca = Var(m.t, within=PositiveReals)
    m.Cb = Var(m.t, within=PositiveReals)
    m.T = Var(m.t, within=PositiveReals)
    m.Tc = Var(m.t, within=PositiveReals)

    # Declaración de las variables de decisión
    # Thetas actuales y pasados: 8 thetas
    m.Theta = Var([0, 1],[0, 1, 2, 3], bounds=(LiminfLambda, LimsupLambda), initialize=0.0)    

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

        #for i in range(nSamples):
         #   Level[i] = m.Qdu[i*tS] #Qdu[0] + (Qdu[i+1]-Qdu[i])*sigmoide(TIME,i*t_Sample,Sig))
        J_proc = [None]*nSamples
        J_modelo = [None]*nSamples
        J_modified = [None]*nSamples
        # Theta*deltau: el mayor indice es el más actual
        for i in range (0, MV):
            for j in range(Ndme):
                theta_deltau[i] = m.Theta[i,j]*m.du[i,j] #Theta[i,1]*du[i,1] + (Theta[i,j+1]*du[i,j+1]-Theta[i,j]*du[i,j])*sigmoide(TIME,j*t_Sample,Sig))
        
        # Past process data
        for i in range(nSamples):
            J_proc[i] = m.q[i*tS]*(pB*m.Cb_m[i*tS] - pA*Ca0) - m.Fr[i*tS]*pFr
            J_modelo[i] = m.q[i*tS]*(pB*(m.Cb[i*tS]+m.error[1]) - pA*Ca0) - m.Fr[i*tS]*pFr
            J_modified[i] = J_modelo[i] + theta_deltau[0] + theta_deltau[1] + m.Qdu[i]	

        # Calculando diferencia al cuadrado entre el proceso y modelo
        Delta_J = 0.0
        for i in range(nSamples):
            Delta_J += (J_proc[i] - J_modified[i])*(J_proc[i] - J_modified[i])  

        for i in range(Ndme):
            Delta_theta = m.beta_theta[0]*((m.Theta[1,j]-m.Theta_ant[MV*(j-1)])**2) + m.beta_theta[1]*((m.Theta[2,j]-m.Theta_ant[MV*(j-1)+1])**2) 

        J_DME = m.beta_cost*Delta_J + Delta_theta #INTEGRAL--!!!!

        return J_DME
    
    m.obj = Objective(rule=_obj, sense=minimize)

    # Discretizar con colocación en elementos finitos
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=nSamples-1, ncp=3, scheme='LAGRANGE-RADAU')
    
    return m

def actualizar_DME(m, acc, per, med, Qdu, du, Theta_ant, v, error, MV=2, Nd=2, Nm=4, Ndme=4, tSample=0.5):
    t_fe = m.t._fe
    t_DME = [val for val in m.t]
    # Actualización en los elementos finitos
    for i, t in enumerate(t_fe):
        # Variables manipuladas
        for j in range(0,3):
            if t<(Ndme*tSample):
                idx = t_DME[t_DME.index(t)+j]
            elif t==Ndme*tSample:
                idx = t_DME[-1]
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
    
    # Esfuerzos de control
    m.Qdu = Qdu
    m.du = du

    # Actualización de las variables de decisión anteriores
    # Vector de Thetas
    for j in range(Ndme):
        m.Theta[0,j] = Theta_ant[MV*(j-1)]
        m.Theta[1,j] = Theta_ant[MV*(j-1)+1]

    for i in range(0, Nm):
        m.v[i] = v[i]
        m.error[i] = error[i]

"""     for i in range(0, Nm):
        m.v_ant[i] = m.v[i].value
    # Estimaciones iniciales anteriores
    m.Ca_ant = m.Ca[m.t.first()].value
    m.Cb_ant = m.Cb[m.t.first()].value
    m.T_ant = m.T[m.t.first()].value
    m.Tc_ant = m.Tc[m.t.first()].value
    # Pesos de la función de costo
    m.beta_xv = beta_xv
    m.beta_x_ant = beta_x_ant """

def ejecutar_DME(m_DME, du_k, beta,tSample):
    solver = SolverFactory('ipopt')
    solver.options['tol'] = 1e-4
    solver.options['linear_solver'] = 'ma57'
    results = solver.solve(m_DME)        # Llamada al solver

    Theta = [None]*6
    Theta[0] = m_DME.Theta[0].value
    Theta[1] = m_DME.Theta[1].value
    Theta[2] = m_DME.Theta[2].value
    Theta[3] = m_DME.Theta[3].value
    Theta[4] = m_DME.Theta[4].value
    Theta[5] = m_DME.Theta[5].value
    
    # Calculo de Lambda
    Lambda = [None]*2
    Lambda[0] = Theta[4] + 2*du_k[0]*beta[0]
    Lambda[1] = Theta[5] + 2*du_k[1]*beta[1]

    return Lambda, Theta
 
def graficar_DME(m_DME):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 6))

    # Figura con J_modificado, J_real
    axs[0, 0].plot(list(m_DME.t), [value(m_DME.J_modified[x]) for x in m_DME.t], 'b', label='Ca')
    axs[0, 0].legend()
    axs[0, 0].set_title('Costos')
    axs[0, 0].set_ylabel('Costo modificado')    

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



