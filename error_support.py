import numpy as np

L = 48
a_inv = 1.73 # GeV
Lambda = 0.35 # GeV

# from https://doi.org/10.1016/S0550-3213(01)00121-3
mu = 0.77
L1, L2, L3, L4 = 0.53/1000, 0.71/1000, -2.72/1000, 0
L5, L6, L7, L8 = 0.91/1000, 0, -0.32/1000, 0.62/1000
Lscat = 2*L1 +2*L2 + L3 - 2*L4 - 0.5*L5 + 2*L6 + L8
m_eta = 0.547862/1.73 #in lattice units
F0 = 0.0871

# from http://dx.doi.org/10.1103/PhysRevD.96.034510 
c1 = -2.837297
c2 = 6.375183 
c3 = -8.311951 

# from PDG
m_p_pm, m_p_0 = 0.13957061, 0.1349770
m_p_pdg = (2/3)*m_p_pm + (1/3)*m_p_0
m_k_pdg = 0.493677
pdg = {'m_p':m_p_pdg/1.73, 'm_k':m_k_pdg/1.73}

def pn4(c, m_p=0, m_k=0, **kwargs):
    return c[0]*(m_p**4) + c[1]*(m_p**3)*m_k + c[2]*(m_p**2)*(m_k**2) + c[3]*m_p*(m_k**3) + c[4]*(m_k**4)

def pn2(c, m_p=0, m_k=0, **kwargs):
    return c[0]*(m_p**2) + c[1]*m_p*m_k + c[2]*(m_k**2)

def err_pc(val1, val2):
    return round(100*np.abs((val1-val2)/val1),2)

# from https://arxiv.org/abs/1902.08191
def x(m):
    return m*m*np.log((m/mu)**2)/(32*((np.pi*F0)**2)) 

def F_p(m_p=0, m_k=0, **kwargs):
    # NLO formula for pion decay constantwith mass dependence
    return F0*(1-2*x(m_p)-x(m_k))

def F_k(m_p=0, m_k=0, **kwargs):
    # NLO formula for kaon decay constantwith mass dependence
    return F0*(1-(3*x(m_p)/4)-(3*x(m_k)/2)-(3*x(m_eta)/4))

# from https://arxiv.org/abs/2111.09849
def t1(m_p=0, m_k=0, **kwargs):
    term1 = (((m_k+m_p)*(2*m_k-m_p))**0.5)/(m_k-m_p)
    term2 = 2*(m_k-m_p)*(((m_k+m_p)/(2*m_k-m_p))**0.5)/(m_k+2*m_p)
    return term1*np.arctan(term2)

def t2(m_p=0, m_k=0, **kwargs):
    term1 = (((m_k-m_p)*(2*m_k+m_p))**0.5)/(m_k+m_p)
    term2 = 2*(m_k+m_p)*(((m_k-m_p)/(2*m_k+m_p))**0.5)/(m_k-2*m_p)
    return term1*np.arctan(term2)

def a012(m_p=0, m_k=0,  **kwargs):
    X12_t1 = pn4([5,11,-11,0,0],m_p,m_k)*np.log((m_p/mu)**2)/pn2([-4,0,4],m_p,m_k)
    X12_t2 = pn4([0,8,55,-67,-9],m_p,m_k)*np.log((m_k/mu)**2)/pn2([-18,0,18],m_p,m_k)
    X12_t3 = pn4([9,-5,-11,24,-36],m_p,m_k)*np.log((m_eta/mu)**2)/pn2([-36,0,36],m_p,m_k)
    X12_t4 = (43+4*t1(m_p,m_k)-12*t2(m_p,m_k))*pn2([0,1/9,0],m_p,m_k)

    X12 = (X12_t1+X12_t2+X12_t3+X12_t4)/((16*np.pi)**2)

    red = m_p*m_k/(m_p+m_k)
    inner = pn2([L5/2,Lscat,L5/2],m_p,m_k) + X12
    a12 = red*(2+(16*inner/(F_p(m_p=m_p,m_k=m_k)*F_k(m_p=m_p,
            m_k=m_k))))/(8*np.pi*F_p(m_p=m_p,m_k=m_k)*F_k(m_p=m_p,m_k=m_k))
    return a12*m_p

def a032(m_p=0, m_k=0, **kwargs):
    X32_t1 = pn4([-5,22,11,0,0],m_p,m_k)*np.log((m_p/mu)**2)/pn2([-8,0,8],m_p,m_k)
    X32_t2 = pn4([0,16,-55,-134,9],m_p,m_k)*np.log((m_k/mu)**2)/pn2([-36,0,36],m_p,m_k)
    X32_t3 = pn4([-9,-10,11,48,36],m_p,m_k)*np.log((m_eta/mu)**2)/pn2([-72,0,72],m_p,m_k)
    X32_t4 = (43-8*t1(m_p,m_k))*pn2([0,1/9,0],m_p,m_k)

    X32 = (X32_t1+X32_t2+X32_t3+X32_t4)/((16*np.pi)**2)

    red = m_p*m_k/(m_p+m_k)
    inner = pn2([-L5/4,Lscat,-L5/4],m_p,m_k) + X32
    a32 = red*(-1+(16*inner/(F_p(m_p=m_p,m_k=m_k)*F_k(m_p=m_p,
                m_k=m_k))))/(8*np.pi*F_p(m_p=m_p,m_k=m_k)*F_k(m_p=m_p,m_k=m_k))
    return a32*m_p

def a0_O5(m_p=0, m_k=0, DE=0, **kwargs):
    k0 = DE
    k1 = 2*np.pi*(m_p+m_k)/(m_p*m_k*(L**3))
    k2 = k1*c1/L
    k3 = k1*c2/(L**2)
    roots = np.roots([k3,k2,k1,k0])
    a = np.real(roots[np.isreal(roots)][0])
    #print(a,roots)
    return a*m_p

def a0_O6(m_p=0, m_k=0, DE=0, **kwargs):
    k0 = DE
    k1 = 2*np.pi*(m_p+m_k)/(m_p*m_k*(L**3))
    k2 = k1*c1/L
    k3 = k1*c2/(L**2)
    k4 = k1*c3/(L**3)
    roots = np.roots([k4,k3,k2,k1,k0])
    a = np.real(np.min(roots[np.isreal(roots)]))
    #print(a,roots)
    return a*m_p
