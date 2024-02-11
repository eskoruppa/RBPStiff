import sys, os, glob
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .PolyCG.polycg.transform_algebra2group import algebra2group_stiffmat
from .PolyCG.polycg.transform_midstep2triad import midstep2triad
from .PolyCG.polycg.SO3 import so3



class GenStiffness:
    
    def __init__(self, method: str = 'md'):
        self.method = method
        self.load_from_file()
        
    def load_from_file(self):
        if self.method.lower() == 'md':
            path = os.path.join(os.path.dirname(__file__), 'Parametrization/MolecularDynamics')
        elif 'crystal' in self.method.lower():
            path = os.path.join(os.path.dirname(__file__), 'Parametrization/Crystallography')
        else:
            raise ValueError(f'Unknown method "{self.method}".')
        
        bases = 'ATCG'
        self.dimers = {}
        for b1 in bases:
            for b2 in bases:
                seq = b1+b2
                self.dimers[seq] = self.read_dimer(seq,path)
                
            
    def read_dimer(self, seq: str, path: str):
        fnstiff = glob.glob(path+'/Stiffness*'+seq+'*')[0]
        fnequi  = glob.glob(path+'/Equilibrium*'+seq+'*')[0]
        
        equi = np.loadtxt(fnequi)
        stiff = np.loadtxt(fnstiff)
        equi_triad = midstep2triad(equi)
        stiff_group = algebra2group_stiffmat(np.array([equi_triad]),stiff)
        dimer = {
            'seq' : seq,
            'group_gs':   equi_triad,
            'group_stiff':stiff_group,
            'equi': equi,
            'stiff' : stiff
            }
        return dimer
    
    def gen_params(self, seq: str, use_group: str=True):
        N = len(seq)-1
        stiff = np.zeros((6*N,6*N))
        gs    = np.zeros((N,6))
        for i in range(N):
            bp = seq[i:i+2].upper()
            if use_group:
                pstiff = self.dimers[bp]['group_stiff']
                pgs    = self.dimers[bp]['group_gs']
            else:
                pstiff = self.dimers[bp]['equi']
                pgs    = self.dimers[bp]['stiff']
            
            stiff[6*i:6*i+6,6*i:6*i+6] = pstiff
            gs[i] = pgs
        return stiff,gs 
    
def read_nucleosome_triads(fn):
    data = np.loadtxt(fn)
    N = len(data) // 12
    nuctriads = np.zeros((N,4,4))
    for i in range(N):
        tau = np.eye(4)
        pos   = data[i*12:i*12+3] / 10
        triad = data[i*12+3:i*12+12].reshape((3,3))
        triad = so3.euler2rotmat(so3.rotmat2euler(triad))
        tau[:3,:3] = triad
        tau[:3,3]  = pos
        nuctriads[i] = tau
    return nuctriads


    

        
        
        



if __name__ == '__main__':
    
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    
    # genstiff = GenStiffness(method='MD')
    
    # seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(147)])
    # stiff,gs = genstiff.gen_params(seq)

    # print(gs)
    # print(stiff)
    
    triadfn = os.path.join(os.path.dirname(__file__), 'State/Nucleosome.state')
    
    nuctriads = read_nucleosome_triads(triadfn)
    
    N = len(nuctriads)

    for i in range(N-1):
        
        g = np.linalg.inv(nuctriads[i]) @ nuctriads[i+1]
        X = so3.se3_rotmat2euler(g)
        X[:3] *= 180/np.pi
        print(X)