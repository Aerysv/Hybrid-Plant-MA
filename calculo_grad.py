# Calculo del gradiente del modelo por Diferencias Finitas
from pyomo.environ import *
from pyomo.dae import *

# Ficheros propios
from modelo import *
from simulacion import *

def grad_m_DD(med, per, aux, v, error, config, limT):

    J_model = [0.0]*3
    g1_model = [0.0]*3
    grad_m = [0.0]*4

    delta = 0.01

    # Llamada al simulador
    m_model = crear_MODELO()
    # Perturbaciones
    m_model.T0 = per[0]
    m_model.Tc0 = per[1]
    # Perturbaciones MHE
    for i in range(4):
        m_model.v[i] = v[i]
        m_model.error[i] = error[i]

    m_model.Ca[0.0] = med[0]
    m_model.Cb[0.0] = med[1]
    m_model.T[0.0] = med[2]
    m_model.Tc[0.0] = med[3]

    m_model.q = aux[0]
    m_model.Fr = aux[1]

    J_model[0], g1_model[0] = costo_constraint_model_steady(m_model, config,limT) # Integra y retorna valor de la funcion de costo

    m_model.Ca[0.0] = med[0]
    m_model.Cb[0.0] = med[1]
    m_model.T[0.0] = med[2]
    m_model.Tc[0.0] = med[3]

    m_model.q = aux[0] + delta
    m_model.Fr = aux[1]

    J_model[1], g1_model[1]  = costo_constraint_model_steady(m_model, config, limT)

    m_model.Ca[0.0] = med[0]
    m_model.Cb[0.0] = med[1]
    m_model.T[0.0] = med[2]
    m_model.Tc[0.0] = med[3]
    
    m_model.q = aux[0] 
    m_model.Fr = aux[1] + delta

    J_model[2], g1_model[2]  = costo_constraint_model_steady(m_model, config,limT)

    grad_m[0] = (J_model[1] - J_model[0])/delta
    grad_m[1] = (J_model[2] - J_model[0])/delta	

    grad_m[2] = (g1_model[1] - g1_model[0])/delta
    grad_m[3] = (g1_model[2] - g1_model[0])/delta

    return grad_m, g1_model[0]


def grad_p_DD(med, per, aux, limT):

    J_planta = [0.0]*3
    g1_planta = [0.0]*3
    grad_p = [0.0]*4

    delta = 0.01

    # Llamada al simulador
    m_Proc = crear_SIM(120)
    m_Proc.Ca[0.0] = med[0]
    m_Proc.Cb[0.0] = med[1]
    m_Proc.T[0.0] = med[2]
    m_Proc.Tc[0.0] = med[3]

    m_Proc.T0 = per[0]
    m_Proc.Tc0 = per[1]

    m_Proc.q = aux[0]
    m_Proc.Fr = aux[1]

    #Integra y retorna valor de la funcion de costo
    J_planta[0], g1_planta[0] = costo_constraint_planta_steady(m_Proc, limT) 

    m_Proc.Ca[0.0] = med[0]
    m_Proc.Cb[0.0] = med[1]
    m_Proc.T[0.0] = med[2]
    m_Proc.Tc[0.0] = med[3]

    m_Proc.q = aux[0] + delta
    m_Proc.Fr = aux[1]

    J_planta[1], g1_planta[1] = costo_constraint_planta_steady(m_Proc, limT)

    m_Proc.Ca[0.0] = med[0]
    m_Proc.Cb[0.0] = med[1]
    m_Proc.T[0.0] = med[2]
    m_Proc.Tc[0.0] = med[3]
    
    m_Proc.q = aux[0] 
    m_Proc.Fr = aux[1] + delta

    J_planta[2], g1_planta[2] = costo_constraint_planta_steady(m_Proc, limT)

    grad_p[0] = (J_planta[1] - J_planta[0])/delta
    grad_p[1] = (J_planta[2] - J_planta[0])/delta

    grad_p[2] = (g1_planta[1] - g1_planta[0])/delta
    grad_p[3] = (g1_planta[2] - g1_planta[0])/delta

    return grad_p, g1_planta[0]

def filtro_mod(grad_p, grad_m, K, Lambda_ant, k_MA):
    
    Lambda_new = [0.0]*2

    if (k_MA == 1):
        for i in range(0, 2):
            Lambda_new[i] = grad_p[i] - grad_m[i]
    else:
        for i in range(0, 2):
            Lambda_new[i] = Lambda_ant[i]*(1-K) + K*(grad_p[i] - grad_m[i])

    return Lambda_new