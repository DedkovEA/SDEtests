import numpy as np
import sdeint
from scipy import linalg

def Iwik_wrapper(dW, h, generator=None):
    n = int(np.ceil(np.sqrt(1/h)))
    print(n)
    return sdeint.Iwik(dW, h, n=n, generator=generator)

def find_errs(multa, step):
    dt = step
    tspan = np.linspace(0.0, 10.0, int(10.0/dt))
    
    dW = sdeint.deltaW(len(tspan)-1, 1, dt)
    W = np.vstack(([0.], np.cumsum(dW, axis=0)))
    x0 = np.array([3.0, 0.0])
    A = np.array([[-0.1, -2.0],
              [ 2.0, -0.1]])
    B = multa*np.array([[0., -1.],
                        [ 1., -1.]])
    x_th = np.array([linalg.expm(A*t) @ linalg.expm(B*w) @ x0 for t,w in zip(tspan, W)])

    def fb(x, t):
        return (A + 0.5*linalg.expm(A*t) @ B**2 @ linalg.expm(-A*t)) @ x

    def Gb(x, t):
        return (linalg.expm(A*t) @ B @ linalg.expm(-A*t) @ x).reshape(2,1)

    x_eu = sdeint.itoEuler(fb, Gb, x0, tspan, dW=dW)
    x_sriIw = sdeint.itoSRI2(fb, Gb, x0, tspan, Imethod=Iwik_wrapper, dW=dW)
    x_sri = sdeint.itoSRI2(fb, Gb, x0, tspan, Imethod=sdeint.Iwik, dW=dW)

    x_impl = []
    x_impl.append(x0)
    x_c = x0
    delt = dt
    for tt, deltw in zip(tspan, dW):
        x_c = linalg.solve((np.identity(2) - (A + 0.5*linalg.expm(A*(tt+delt)) @ B**2 @ linalg.expm(-A*(tt+delt)))*delt - 
                                              linalg.expm(A*(tt+delt)) @ B @ linalg.expm(-A*(tt+delt))*deltw), x_c)  
        x_impl.append(x_c)

    x_impl = np.array(x_impl)

    return (np.linalg.norm(x_eu-x_th, axis=1).max(),
            np.linalg.norm(x_sri-x_th, axis=1).max(),
            np.linalg.norm(x_impl-x_th, axis=1).max(),
            np.linalg.norm(x_sriIw-x_th, axis=1).max()) # eu, sri, impl, sriIw