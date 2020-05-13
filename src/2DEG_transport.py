import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def get_fermi_circle(k_rad, nk, nthet):##ok 
    ##Input: radius of 2D Fermi circle (float)[m**(-1)], size of k array (int), size of theta array (int)
    ##Output: grid of k points in Fermi circle [m**(-1)] 
    kx_list = []
    ky_list = []
    k_range = np.linspace(0 ,k_rad, int(nk))
    thet_range = np.linspace(0, 2*pi, int(nthet))  
    for i in range(0,nk):
        for j in range(0,nthet): 
            kx_list.append(k_range[i]*np.cos(thet_range[j]))
            ky_list.append(k_range[i]*np.sin(thet_range[j]))
    return [np.asarray(kx_list), np.asarray(ky_list)]  



def get_bose_distribution(energy, T):## ok 
    ##Computes bose einstein distribution 
    ##Input: energy [J], temperature [K] 
    ##Output: BE distribution (phonon number)
    return (np.exp(energy/(kb_SI*T))-1)**(-1) 


def get_conduction_band(Ec, mstar, k_vec): ##ok 
    ##Input: Conduction band offset (float) [J], effective mass (float) [m0], k_point to evaluate at (array)[m^(-1)]
    ##Output: kmesh [m^(-1)](array), conduction band (array)[J]
    Ek = lambda Ec, mstar, kx, ky: Ec + ((hbar**2)/(2*mstar*m0))*(kx**(2) + ky**(2)) 
    return Ek(Ec, mstar, k_vec[0], k_vec[1]) 




def compute_rate_prob_POP(k0, kf, mstar, energ_ph,T):  
    ##Computes probability of transition from k0 to kf at temperature 
    ##Input: inital k state k0  [m**(-1)], final k state kf  [m**(-1)],  effective mass mstar [m0], energy of polar optical phonon [eV] 
    ##Output: pseudo-probability of transition
    
    E0 = get_conduction_band(0, mstar, k0) 
    Ef = get_conduction_band(0, mstar, kf) 
    delE = (Ef-E0)*(1/qe)
    q = np.linalg.norm(k0-kf)
    nq = get_bose_distribution(energ_ph*qe, T)

    if(abs(delE) < 0.1*energ_ph):
        if(delE > 0): 
            prob = ((q**2)*nq)/(0.2*energ_ph)  
                
        elif(delE < 0):
            prob = ((q**2)*(nq+1))
        else: 
            prob = 0  
    else: 
        prob = 0 
    
    return prob 








def compute_rate_prob_AP(k0, kf, mstar, vs, T):  
    ##Computes probability of transition from k0 to kf at temperature T for acoustic phonon scattering
    ##Input: inital k state k0 [m**(-1)] final k state kf [m**(-1)], effective mass mstar [m0], sound velocity vs [m/s], temperature T[K]
    ##Output: pseudo-probability of transition 
    
    E0 = get_conduction_band(0, mstar, k0) 
    Ef = get_conduction_band(0, mstar, kf) 
    delE = Ef-E0

    q = np.linalg.norm(k0-kf)
    energ_ph  = hbar*q*vs 
    nq = get_bose_distribution(energ_ph, T)
    
    if(delE > 0): 
        prob = (q*nq)/(0.2*energ_ph)  
        
    elif(delE < 0):
        prob = (q*(nq+1))/(0.2*energ_ph) 
    
    else: 
        prob = 0   
 
    return prob 






def compute_lifetime_AP_h(q, T, Dac, vs, rho, V, emm_abs):
    ##Computes lifetime at temeprature T
    ##Input:  effective mass mstar [m0], temperature [K],  
    ##Output: pseudo-probability of transition
    
    assert emm_abs == 1 or emm_abs== 0, "absorption/emission type should be 1 or 0"
    if(emm_abs == 1): 
        ##emission 
        a = 0 
    else: 
        ## absorption
        a = 1 
    Dac = Dac*qe
    rho = rho*(10**(3))
    nq = get_bose_distribution(hbar*q*vs, T)
    mat_elem_sq = (Dac**2)*((hbar*q)/(2*V*rho*vs))
    energ_ph = hbar*q*vs 
    rate = (2*pi/(energ_ph*hbar))*mat_elem_sq*(nq+a) 
    return 1/rate


def scatter_prob_h(k0, kmesh, mstar, T, phonon_typ, vs = 0, energ_POP = 0):
    ##Computes transition probability matrix at temeprature T
    ##Input: inital k state k0  [m**(-1)], kmesh  [m**(-1)],  effective mass mstar [m0], temperature [K], phonon type [1 if acoustic, 0 if POP], energy of polar optical phonon [eV] 
    ##Output: pseudo-probability of transition
    
    assert phonon_typ == 1 or phonon_typ == 0, "phonon_typ should be 1 or 0"
    if(phonon_typ == 1): 
        prob_fun = lambda k0, kf: compute_rate_prob_AP_h(k0, kf, mstar, vs, T)    
    else: 
        prob_fun = lambda k0, kf: compute_rate_prob_POP_h(k0, kf, mstar, energ_POP,T)            

    rate_list = np.array([prob_fun(k0, np.array([kmesh[0][i], kmesh[1][i]])) for i in range(len(kmesh[0]))])
    rate_list_not_nan = [] 
    kx_list_not_nan = [] 
    ky_list_not_nan = [] 
    for i in range(len(rate_list)):
        if(np.isnan(rate_list[i]) == False):
            rate_list_not_nan.append(rate_list[i])
            kx_list_not_nan.append(kmesh[0][i])
            ky_list_not_nan.append(kmesh[1][i])   
    
    return [np.array(kx_list_not_nan), np.array(ky_list_not_nan), np.array(rate_list_not_nan)/np.sum(np.array(rate_list_not_nan))]
 











