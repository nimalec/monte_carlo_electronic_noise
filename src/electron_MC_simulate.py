import numpy as np
import scipy as sp 



def evolvek_h(del_t, E_field, k0): ##ok 
    ## Computes k point at next time step due to field drift
    ## Input: timestep [s], field (2D array) [V/m], current k-point [m**(-1)]
    ## Output: k point at next time step [m**(-1)]
    del_k = (qe*del_t/hbar)*E_field 
    return k0 + del_k

def pick_k_h(k0, kmesh, prob_list):  
     ## Computes k point at next time step due to diffusion
    ## Input: current k-point [m**(-1)], kmesh (2D array) [m**(-1)], probability list for scattering mech. [list]
    ## Output: k point at next time step due to diffusion [m**(-1)] 
    idx = [i for i in range(len(prob_list))]
    idx_ran = np.random.choice(idx, p = prob_list)
    kf = np.array([kmesh[0][idx_ran], kmesh[1][idx_ran]]) 
    return kf 

def run_MC(kmesh, mstar, T, phonon_typ, E_field, del_t, Tmax, vs = 10**3, energ_POP = 0.2, Dac = 9.4*(10**(-3)), V = 10**(-30), rho = 6.16):
    ## Performs MC simulation according to drift and diffusion steps 
    ## Input: current k-point [m**(-1)], kmesh (2D array) [m**(-1)], probability list for scattering mech. [list]
    ## Output: k point at next time step due to diffusion [m**(-1)] 

    assert phonon_typ == 1 or phonon_typ == 0, "phonon_typ should be 1 or 0"
    k_list = []
    T_sim = []
    t_sim = 0 
    itr = 0      
    k0 = np.array([kmesh[0][0],kmesh[1][0]])
    ki = np.array([kmesh[0][0],kmesh[1][0]])
    prob_list = scatter_prob_h(k0, kmesh, mstar, T, phonon_typ, vs, energ_POP)

   # if(phonon_typ == 1): 
   #     tau = lambda q: get_lifetime_POP(q, T,   )
   # else: 
    #    tau = lambda q, em: compute_lifetime_AP_h(q, T, Dac, vs, rho, V, em)
    tau = lambda q: compute_lifetime_AP_h(q, T, Dac, vs, rho, V, 1)
    
    while(itr*del_t <= Tmax): 
        ki = evolvek_h(del_t, E_field, ki)
        k0 = pick_k_h(k0, np.array([prob_list[0],prob_list[1]]), prob_list[2]) 
        kf = ki + k0 
        k_list.append(kf.tolist()) 
        tau_i = tau(k0*100)
        if(tau_i < 10**(-11)): 
            t_sim +=  del_t + tau_i  
        else: 
            tau_i = 0 
        T_sim.append(t_sim)
        itr += 1        
        
    kx = [k_list[i][0] for i in range(len(k_list))]
    ky = [k_list[i][1] for i in range(len(k_list))]  
    k_vec = [np.array(kx), np.array(ky)]
    return [np.array(T_sim), np.array(k_vec)] 



