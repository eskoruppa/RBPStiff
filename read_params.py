import sys, os, glob
import numpy as np
from typing import List, Tuple, Callable, Any, Dict
from .SO3 import so3

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
        equi_triad = so3.se3_midstep2triad(equi)                
        stiff_group = so3.se3_algebra2group_stiffmat(equi,stiff,translation_as_midstep=True)        
        dimer = {
            'seq' : seq,
            'group_gs':   equi_triad,
            'group_stiff':stiff_group,
            'equi': equi,
            'stiff' : stiff
            }
        return dimer
    
    def gen_params(self, seq: str, use_group: str=False):
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
    


if __name__ == '__main__':
    
    np.set_printoptions(linewidth=250,precision=3,suppress=True)
    method = sys.argv[1].upper()
    genstiff = GenStiffness(method=method)
    
    seq = 'ATCG'
    
    genstiff.gen_params(seq,use_group=True)
    
    
    triadfn = os.path.join(os.path.dirname(__file__), 'State/Nucleosome.state')
    nuctriads = read_nucleosome_triads(triadfn)
    N = len(nuctriads)

    for i in range(N-1):
        
        g = np.linalg.inv(nuctriads[i]) @ nuctriads[i+1]
        X = so3.se3_rotmat2euler(g)
        X[:3] *= 180/np.pi
        print(X)