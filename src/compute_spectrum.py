import numpy as np
import scipy as sp 

def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def get_power_spectrum(noise_current): 
    auto_corr = estimated_autocorrelation(noise_current)
    ft = np.absolute(sp.fft(auto_corr))
    return ft**(2)
