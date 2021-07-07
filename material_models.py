import numpy as np

class Mooney_Rivlin(): #Source: Continuummechanics
#Assumptions: Fully incompressible material. Thus paramter D is irrelevant.
    #Strain Energy
    def U(lm, C10, C01, C20): #lm.shape = (n,2)
        lm3 = np.zeros(np.size(lm,0))
        lm3[:] = 1/(lm[:,0] + lm[:,1])
        I1 = lm[:,0]**2 + lm[:,1]**2 + lm3**2
        I2 = 1/lm[:,0]**2 + 1/lm[:,1]**2 + 1/lm3**2
        return C10*(I1-3) + C01*(I2-3) + C20*(I1-3)**2
    #Vector containing cauchy stresses s1 and s2 given lm1 and lm2, assuming s3=0 and J=1
    def s_general(lm, C10 = -0.11578102, C01 = 0.28067402, C20 =  0.00828701): #lm.shape = (n,2)
        lm1 = lm[:,0]
        lm2 = lm[:,1]
        lm3 = 1/(lm1*lm2)
        s1 = 2*(C10*(lm1**2 - lm3**2) - C01*(1/lm1**2 - 1/lm3**2) +
                  2*C20*(lm1**2 - lm3**2)*(lm1**2 + lm2**2 + lm3**2 - 3))
        s2 = 2*(C10*(lm2**2 - lm3**2) - C01*(1/lm2**2 - 1/lm3**2) +
                  2*C20*(lm2**2 - lm3**2)*(lm1**2 + lm2**2 + lm3**2 - 3))
        s = np.zeros(np.shape(lm))
        s[:,0] = s1
        s[:,1] = s2
        return s
    #Cauchy stress in uniaxial loading given stretch in an axis.
    def s_uni(lm, C10 = -0.11578102, C01 = 0.28067402, C20 =  0.00828701): #lm.shape = (n,)
        #Default values belong to Dragonskin 30
        lm1 = lm
        lm2 = lm**(-1/2)
        lm3 = lm**(-1/2)
        return 2*(C10*(lm1**2 - lm3**2) - C01*(1/lm1**2 - 1/lm3**2) +
                  2*C20*(lm1**2 - lm3**2)*(lm1**2 + lm2**2 + lm3**2 - 3))
    #Cauchy stress in equibiaxial loading given stretch in an axis.
    def s_equibi(lm, C10 = 0.1201621, C01 = -0.11675297): #lm.shape = (n,)
        #In Equibiaxial loading S3=0. Therefore, D can be eliminated by taking S1-S3
        lm1 = lm
        #note that lm2 = lm1 in this case.
        lm3 = 1/lm**2
        return 2*(C10*(lm1**2 - lm3**2) - C01*(1/lm1**2 - 1/lm3**2))
    #Uniaxial loading again, but this time also including the C20 parameter


class GOH_nearlyinc():
    #Paper: Propagation of material behavior uncertainty in a nonlinear finite
    #element model of reconstructive surgery
    #This assumes nearly incompressible material
    def __init__(self, lm, mu, k1, k2, kappa, K): #lm.shape = (n,3)
        self.mu = mu
        self.k1 = k1
        self.k2 = k2
        self.kappa = kappa
        self.K = K
        n = np.size(lm,0)
        F = np.zeros([n,3,3])
        self.I1 = np.zeros(n)
        self.I4 = np.zeros(n)
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = lm[:,2]
        self.F = F
        self.J = np.linalg.det(F)
        C = F*F #Since F^T = F
        C_isoch = np.einsum('...,...ij->...ij',self.J**(-2/3), C)
        self.I1 = np.trace(C_isoch, axis1=1, axis2=2)
        self.C_inv = np.linalg.inv(C)
        e_0 = [0, 0, 1] #Fiber direction.
        for i in range(0,3):
            for j in range(0,3):
                self.I4[:] = self.I4[:] + e_0[i]*C_isoch[:,i,j]*e_0[j]
        self.E = kappa*(self.I1-3) + (1-3*kappa)*(self.I4-1)
    #Strain Energy
    def U(self, lm): #lm.shape = (n,3)
        mu = self.mu
        k1 = self.k1
        k2 = self.k2
        K = self.K
        U_iso = mu/2*(self.I1-3)
        U_vol = K/2*((self.J**2-1)/2 - np.log(self.J))
        U_aniso = k1/2/k2*(np.exp(k2*self.E**2) - 1)
        U = U_iso + U_vol + U_aniso
        return U
    def s(self, lm): #General stress
        mu = self.mu
        k1 = self.k1
        k2 = self.k2
        kappa = self.kappa
        K = self.K
        I = np.identity(3)
        # temp = I - 1/3*np.einsum('...,...ij->...ij',self.I1,self.C_inv)
        # S_iso = mu*np.einsum('...,...ij->...ij',self.J**(-2/3), temp)
        #There is a mistake in the paper in the equation above.
        #The J**(-2/3) is supposed to be factored out. See Tepole's notes.
        # S_vol = K/2*np.einsum('...,...ij->...ij',self.J**2 - 1, self.C_inv)
        eiej = np.zeros([3,3])
        S_iso = np.zeros_like(self.C_inv)
        S_vol = np.zeros_like(self.C_inv)
        S_aniso = np.zeros_like(self.C_inv)
        S = np.zeros_like(self.C_inv)
        s = np.zeros_like(self.C_inv)
        F_T = np.transpose(self.F,[0,2,1])
        eiej[2,2] = 1 #eiej is supposed to be "e dyadic e" or e_i*e_j
        for i in range(3):
            for j in range(3):
                S_iso[:,i,j] = mu*self.J[:]**(-2/3)*(I[i,j] - 1/3*self.I1[:]*self.C_inv[:,i,j])
                S_vol[:,i,j] = K/2*(self.J[:]**2-1)*self.C_inv[:,i,j]
                S_aniso[:,i,j] = 2*k1*np.exp(k2*self.E[:]**2)*self.E[:]*(
                    kappa*self.J[:]**(-2/3)*(I[i,j] - 1/3*self.I1[:]*self.C_inv[:,i,j])
                    +
                    (1-3*kappa)*self.J[:]**(-2/3)*(eiej[i,j] - 1/3*self.I4[:]*self.C_inv[:,i,j])
                    )
                S[:,i,j] = S_iso[:,i,j] + S_vol[:,i,j] + S_aniso[:,i,j]
                s[:,i,j] = 1/self.J[:]*self.F[:,i,j]*S[:,i,j]*F_T[:,i,j]
        # S = S_iso + S_vol +S_aniso
        # s = 1/self.J*self.F*S*F_T
        return s


class GOH_fullyinc():
    #Paper: Propagation of material behavior uncertainty in a nonlinear finite
    #element model of reconstructive surgery

    #This assumes fully incompressible material. Therefore bulk modulus, K and 
    #U_vol are 0.
    def __init__(self, lm, mu = 0.04498, k1 = 4.9092, k2 = 76.64134, kappa = 1/3, theta = 0): #lm.shape = (n,2)
        self.mu = mu
        self.k1 = k1
        self.k2 = k2
        self.kappa = kappa
        n = np.size(lm,0)
        F = np.zeros([n,3,3])
        F[:,0,0] = lm[:,0]
        F[:,1,1] = lm[:,1]
        F[:,2,2] = 1/(lm[:,0]*lm[:,1])
        self.C = F*F #Since F^T = F.
        self.F = F
        self.I1 = np.zeros(n)
        self.I1 = self.C.trace(axis1=1, axis2=2) #In this case C_isoch = C because J = 1.
        e_0 = [np.cos(theta), np.sin(theta), 0] #Fiber direction.
        self.I4 = np.zeros(n)
        for i in range(0,3):
            for j in range(0,3):
                self.I4[:] = self.I4[:] + e_0[i]*self.C[:,i,j]*e_0[j]
        self.E = kappa*(self.I1-3) + (1-3*kappa)*(self.I4-1)
        self.e_0 = e_0
    #Strain Energy
    def U(self, lm): #lm.shape = (n,2)
        mu = self.mu
        k1 = self.k1
        k2 = self.k2
        U_iso = mu/2*(self.I1-3)
        #U_vol = 0.0 because we are assuming fully incompressible material
        U_aniso = k1/2/k2*(np.exp(k2*self.E**2) - 1)
        U = U_iso + U_aniso
        return U
    #Partial derivatives of Strain Energy wrt Invariants I1, I4
    def partial(self, lm):
        mu = self.mu
        k1 = self.k1
        k2 = self.k2
        kappa = self.kappa
        E = self.E
        U1 = mu/2 + k1*np.exp(k2*E**2)*E*kappa
        U4 = k1*np.exp(k2*E**2)*E*(1-3*kappa)
        return U1, U4
    def s(self, lm): #Cauchy stress
        mu = self.mu
        k1 = self.k1
        k2 = self.k2
        kappa = self.kappa
        I1 = self.I1
        I4 = self.I4
        E = self.E
        F = self.F
        C = self.C
        C_inv = np.linalg.inv(self.C)
        n = np.size(lm,0)
        I = np.identity(3)
        S_iso = np.zeros([n,3,3])
        S_vol = np.zeros([n,3,3])
        for i in range(0,3):
            for j in range(0,3):
                S_iso[:,i,j] = mu*(I[i,j] - 1/3*I1[:]*C_inv[:,i,j])
        #There is a mistake in the paper in the equation above.
        #The J**(-2/3) is supposed to be factored out. See Prof. Tepole's notes.
        #S_vol = 0.0 because we are assuming fully incompressible material
        eiej = np.outer(self.e_0,self.e_0) #e_i dyadic e_j
        dI1dC = np.zeros(n)
        dI4dC = np.zeros(n)
        S_aniso = np.zeros([n,3,3])
        for i in range(0,3):
            for j in range(0,3):
                dI1dC[:] = I[i,j] - 1/3*I1[:]*C_inv[:,i,j]
                # dI1dC[:] = I[i,j] #Both this and the line above are true when C_isoch = C
                dI4dC[:] = eiej[i,j] - 1/3*I4[:]*C_inv[:,i,j]
                # dI4dC[:] = eiej[i,j] #Both this and the line above are true when C_isoch = C

                S_aniso[:,i,j] = 2*k1*np.exp(k2*E[:]**2)*E[:]*(kappa*dI1dC[:] + (1-3*kappa)*dI4dC[:])
        p = -(S_iso[:,2,2] + S_aniso[:,2,2])*C[:,2,2]
        for i in range(0,3):
            for j in range(0,3):
                S_vol[:,i,j] = p[:]*C_inv[:,i,j]
        S = S_iso + S_aniso + S_vol
        s = F*S*F #Since F^T = F
        return s

    
class HGO():
    #Paper: A generic physics-informed neural network-based constitutive model
    #for soft biological tissues
    def __init__(self, lm, C10, k1, k2, theta):
        self.F = np.zeros([np.size(lm,0),3,3])
        self.I1 = np.zeros(np.size(lm,0))
        self.I2 = np.zeros(np.size(lm,0))
        self.I4 = np.zeros(np.size(lm,0))
        self.I6 = np.zeros(np.size(lm,0))
        self.F[:,0,0] = lm[:,0]
        self.F[:,1,1] = lm[:,1]
        self.F[:,2,2] = lm[:,2]
        self.J = np.linalg.det(self.F)
        C = self.F*self.F #Since F^T = F
        C2 = C*C
        self.C_inv = np.linalg.inv(C)
        self.a_01 = [np.cos(theta), np.sin(theta), 0]
        self.a_02 = [np.cos(theta), -np.sin(theta), 0]
        self.I1 = C.trace(axis1=1, axis2=2)
        self.I2[:] = 1/2*(self.I1[:]**2 - (C2[:,0,0] + C2[:,1,1] + C2[:,2,2]))
        for i in range(0,3):
            for j in range(0,3):
                self.I4[:] = self.I4[:] + self.a_01[i]*self.C[i,j]*self.a_01[j]
                self.I6[:] = self.I6[:] + self.a_02[i]*self.C[i,j]*self.a_02[j]

    def U(self, lm, C10, k1, k2, theta):
        U_iso = C10*(self.I1-3)
        U_aniso = k1/2/k2*(np.exp(k2*(self.I4-1)**2)-1  +  np.exp(k2*(self.I6-1)**2)-1)
        U = U_iso + U_aniso
        return U
    def s(self, lm, C10, k1, k2, theta):
        I = np.identity(3)
        S = np.zeros([np.size(lm,0),3,3])
        W1 = C10
        W4 = k1*(self.I4 - 1)*np.exp(k2*(self.I4 - 1)**2)
        W6 = k1*(self.I6 - 1)*np.exp(k2*(self.I6 - 1)**2)
        p = 1.0 #??? Find out how to calculate this

        for i in range(0,3):
            for j in range(0,3):
                S[:,i,j] = -p*self.C_inv[:,i,j] + 2*W1*I[i,j]
                + 2*W4*self.a_01[i]*self.a_01[j] + 2*W6*self.a_02[i]*self.a_02[j]

        s = 1/self.J*self.F*S*np.transpose(self.F,[0,2,1])
        return s

