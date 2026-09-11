"""
DFT (Teoria do Funcional da Densidade) real via pyscf, sobre as geometrias iniciais
geradas em build_structures.py (RDKit ETKDG + MMFF94).

Nivel de teoria: B3LYP/6-31G(d) -- padrao amplamente usado na literatura de
adsorcao/quimiossorcao para calculo de HOMO/LUMO, gap de energia e indices de
reatividade de corantes organicos (ex.: azul de metileno) e acucares/carboidratos.

Para cada molecula:
  1. Otimizacao de geometria (pyscf.geomopt.berny_solver, gradiente analitico DFT)
  2. SCF final na geometria otimizada -> energia, HOMO/LUMO (eV), gap
  3. Indices de reatividade (aproximacao de Koopmans, teoria DFT conceitual):
       potencial quimico     mu  = (E_HOMO + E_LUMO) / 2
       eletronegatividade    chi = -mu
       dureza quimica        eta = (E_LUMO - E_HOMO) / 2
       maciez (softness)     S   = 1 / (2*eta)
       indice de eletrofilia omega = mu^2 / (2*eta)
  4. Cubos volumetricos (pyscf.tools.cubegen) para visualizacao 3D no dashboard
     (3Dmol.js): potencial eletrostatico molecular (MEP), orbital HOMO, orbital LUMO.
  5. Geometria otimizada final em XYZ.

Custo computacional: e o motivo real de nao fazer isso pra toda a biomassa (celulose
inteira, lignina, hemicelulose simultaneamente) -- aqui usamos 1 molecula-sonda por
sistema (azul de metileno = adsorbato; beta-D-glicopiranose = proxy dos sitios -OH
da celulose, o polissacarideo dominante na fibra da Vagem).
"""
import json
import os
import sys
import time

import numpy as np
from pyscf import dft, gto
from pyscf.geomopt.berny_solver import optimize
from pyscf.tools import cubegen

HARTREE_TO_EV = 27.211386245988

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
BASIS = "6-31g*"
XC = "b3lyp"

SYSTEMS = {
    "methylene_blue": dict(charge=1, spin=0, label="Azul de Metileno (cátion, adsorbato)"),
    "glucose": dict(charge=0, spin=0, label="Biomassa In Natura (glicopiranose)"),
    "glucose_acido": dict(charge=0, spin=0, label="Biomassa Ácido — H₃PO₄ (glicose-6-fosfato)"),
    "glucose_base": dict(charge=-1, spin=0, label="Biomassa Base — NaOH (C6-alcóxido)"),
}


def read_xyz(path):
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0])
    atom_lines = lines[2:2 + n]
    return "\n".join(atom_lines)


def write_xyz(path, mol, comment=""):
    coords = mol.atom_coords(unit="Angstrom")
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    with open(path, "w") as fh:
        fh.write(f"{mol.natm}\n{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            fh.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")


def run_system(name, spec, cube_res=40):
    t0 = time.time()
    print(f"\n=== {name} ({spec['label']}) ===", flush=True)

    xyz_opt_path = os.path.join(ARTIFACTS_DIR, f"{name}_optimized.xyz")
    if os.path.exists(xyz_opt_path):
        # Ja tem geometria otimizada salva de uma corrida anterior (ex.: processo
        # anterior foi interrompido apos a otimizacao, antes dos cubos) -- reusa em
        # vez de reotimizar do zero, que e a etapa mais cara.
        print(f"  Geometria otimizada já existe em disco ({xyz_opt_path}) -- pulando otimização.", flush=True)
        atom_block = read_xyz(xyz_opt_path)
        mol_opt = gto.M(atom=atom_block, basis=BASIS, charge=spec["charge"], spin=spec["spin"], verbose=0)
        t_opt = 0.0
    else:
        xyz_path = os.path.join(ARTIFACTS_DIR, f"{name}_initial.xyz")
        atom_block = read_xyz(xyz_path)
        mol = gto.M(atom=atom_block, basis=BASIS, charge=spec["charge"], spin=spec["spin"], verbose=0)
        print(f"  {mol.natm} átomos, {mol.nao} funções de base, carga={spec['charge']}", flush=True)

        mf = dft.RKS(mol).density_fit()  # aproximacao RI/DF -- acelera SCF/gradiente muito,
        mf.xc = XC                        # com perda de precisao tipicamente <0.1 kcal/mol
        mf.verbose = 0                    # (padrao aceito p/ otimizacao de geometria)
        print("  Otimizando geometria (B3LYP/6-31G*, density fitting)...", flush=True)
        mol_opt = optimize(mf, maxsteps=100)
        t_opt = time.time() - t0
        print(f"  Geometria otimizada em {t_opt:.1f}s", flush=True)
        write_xyz(xyz_opt_path, mol_opt, comment=f"{spec['label']} -- B3LYP/6-31G* otimizado")

    print("  Rodando SCF final...", flush=True)
    mf_final = dft.RKS(mol_opt).density_fit()
    mf_final.xc = XC
    mf_final.verbose = 0
    e_tot = mf_final.kernel()
    print(f"  SCF final concluído ({time.time()-t0:.1f}s desde o início).", flush=True)
    if not mf_final.converged:
        print("  AVISO: SCF final não convergiu de forma robusta -- resultado marcado como incerto.", flush=True)

    mo_energy = mf_final.mo_energy  # Hartree
    nelec_alpha = mf_final.mol.nelec[0]
    homo_idx = nelec_alpha - 1
    lumo_idx = nelec_alpha
    e_homo = float(mo_energy[homo_idx]) * HARTREE_TO_EV
    e_lumo = float(mo_energy[lumo_idx]) * HARTREE_TO_EV
    gap = e_lumo - e_homo

    mu = (e_homo + e_lumo) / 2.0
    chi = -mu
    eta = gap / 2.0
    softness = 1.0 / (2.0 * eta) if eta != 0 else None
    omega = (mu ** 2) / (2.0 * eta) if eta != 0 else None

    dm = mf_final.make_rdm1()
    mep_path = os.path.join(ARTIFACTS_DIR, f"{name}_mep.cube")
    homo_path = os.path.join(ARTIFACTS_DIR, f"{name}_homo.cube")
    lumo_path = os.path.join(ARTIFACTS_DIR, f"{name}_lumo.cube")
    print(f"  Gerando cubo MEP (grade {cube_res}³)...", flush=True)
    cubegen.mep(mol_opt, mep_path, dm, nx=cube_res, ny=cube_res, nz=cube_res)
    print(f"  Gerando cubo HOMO ({time.time()-t0:.1f}s)...", flush=True)
    cubegen.orbital(mol_opt, homo_path, mf_final.mo_coeff[:, homo_idx], nx=cube_res, ny=cube_res, nz=cube_res)
    print(f"  Gerando cubo LUMO ({time.time()-t0:.1f}s)...", flush=True)
    cubegen.orbital(mol_opt, lumo_path, mf_final.mo_coeff[:, lumo_idx], nx=cube_res, ny=cube_res, nz=cube_res)
    print(f"  Cubos prontos ({time.time()-t0:.1f}s).", flush=True)

    t_total = time.time() - t0
    result = dict(
        name=name, label=spec["label"], basis=BASIS, xc=XC,
        charge=spec["charge"], n_atoms=mol_opt.natm, n_basis_functions=int(mol_opt.nao),
        e_total_hartree=float(e_tot), converged=bool(mf_final.converged),
        e_homo_ev=e_homo, e_lumo_ev=e_lumo, gap_ev=gap,
        chemical_potential_ev=mu, electronegativity_ev=chi,
        hardness_ev=eta, softness_ev=softness, electrophilicity_ev=omega,
        optimization_time_s=t_opt, total_time_s=t_total,
        files=dict(
            xyz_optimized=os.path.basename(xyz_opt_path),
            mep_cube=os.path.basename(mep_path),
            homo_cube=os.path.basename(homo_path),
            lumo_cube=os.path.basename(lumo_path),
        ),
    )
    print(f"  E_HOMO={e_homo:.3f} eV  E_LUMO={e_lumo:.3f} eV  gap={gap:.3f} eV  "
          f"(total {t_total:.1f}s)")
    return result


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    summary_path = os.path.join(ARTIFACTS_DIR, "dft_summary.json")

    results = {}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            results = json.load(fh)

    # Passe nomes de sistema como argv para rodar so um subconjunto (ex.: adicionar
    # sistemas novos sem recalcular os que ja estao em dft_summary.json). Sem args,
    # roda todos os SYSTEMS.
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(SYSTEMS.keys())
    for name in targets:
        if name not in SYSTEMS:
            print(f"AVISO: sistema desconhecido '{name}', pulando.")
            continue
        results[name] = run_system(name, SYSTEMS[name])
        with open(summary_path, "w") as fh:
            json.dump(results, fh, indent=2)  # salva incrementalmente (sobrevive a crash)

    print(f"\nResumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
