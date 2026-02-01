from __future__ import annotations

import glob
import os

import numpy as np
import scipy as sp
from .SO3 import so3


class GenStiffness:
    def __init__(
        self, 
        method: str = "md", 
        stiff_method: str | None = None, 
        gs_method: str | None = None
    ):
        self.method         = method
        self.stiff_method   = stiff_method
        self.gs_method      = gs_method
        self.load_from_file()

    def load_from_file(self):
        if self.method.lower() == "md":
            stiff_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
            gs_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
        elif "crystal" in self.method.lower():
            stiff_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
            gs_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
        elif "hybrid" in self.method.lower():
            stiff_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/MolecularDynamics"
            )
            gs_path = os.path.join(
                os.path.dirname(__file__), "Parametrization/Crystallography"
            )
        else:
            raise ValueError(f'Unknown method "{self.method}".')
        
        if self.stiff_method is not None:
            if self.stiff_method.lower() == 'md':
                stiff_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/MolecularDynamics"
                )
            elif self.stiff_method.lower() == 'crystal':
                stiff_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/Crystallography"
                )
            else:
                raise ValueError(f'Unknown stiff_method "{self.stiff_method}".')
            
        if self.gs_method is not None:
            if self.gs_method.lower() == 'md':
                gs_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/MolecularDynamics"
                )
            elif self.gs_method.lower() == 'crystal':
                gs_path = os.path.join(
                    os.path.dirname(__file__), "Parametrization/Crystallography"
                )
            else:
                raise ValueError(f'Unknown gs_method "{self.gs_method}".')
        
        bases = "ATCG"
        self.dimers = {}
        for b1 in bases:
            for b2 in bases:
                seq = b1 + b2
                self.dimers[seq] = self.read_dimer(seq, stiff_path, gs_path)

    def read_dimer(self, seq: str, stiff_path: str, gs_path: str):
        fnstiff = glob.glob(stiff_path + "/Stiffness*" + seq + "*")[0]
        fnequi = glob.glob(gs_path + "/Equilibrium*" + seq + "*")[0]

        equi = np.loadtxt(fnequi)
        stiff = np.loadtxt(fnstiff)
        equi_triad = so3.se3_midstep2triad(equi)
        stiff_group = so3.se3_algebra2group_stiffmat(
            equi, stiff, translation_as_midstep=True
        )
        dimer = {
            "seq": seq,
            "group_gs": equi_triad,
            "group_stiff": stiff_group,
            "equi": equi,
            "stiff": stiff,
        }
        return dimer

    def gen_params(self, seq: str, use_group: bool = False, sparse: bool = False) -> dict[str, np.ndarray]:
        N = len(seq) - 1
        gs = np.zeros((N, 6))
        blocks = []
        for i in range(N):
            bp = seq[i : i + 2].upper()
            if use_group:
                pstiff = self.dimers[bp]["group_stiff"]
                pgs = self.dimers[bp]["group_gs"]
            else:
                pstiff = self.dimers[bp]["stiff"]
                pgs = self.dimers[bp]["equi"]
            blocks.append(pstiff)
            gs[i] = pgs
        
        if sparse:
            stiff = sp.sparse.block_diag(blocks, format='csr')
        else:
            stiff = np.zeros((6 * N, 6 * N))
            for i,block in enumerate(blocks):
                stiff[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = block
        return {'stiffness':stiff, 'groundstate':gs}
                