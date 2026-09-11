"""
Constroi as geometrias 3D iniciais (RDKit: embed ETKDG + pre-otimizacao MMFF94) para
os dois sistemas da analise DFT:

  - Azul de metileno (cation, a especie realmente adsorvida em solucao aquosa --
    o contraion cloreto se dissocia). SMILES oficial: PubChem CID 6099, via API
    PUG-REST (nao de memoria): "CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2.[Cl-]"
    -- usamos so o cation, sem o ".[Cl-]".
  - Beta-D-glicopiranose (monomero da celulose, o polissacarideo dominante na fibra
    da Vagem) representando os sitios funcionais -OH do biossorvente. SMILES oficial:
    PubChem CID 64689, via API PUG-REST: "C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)O)O)O)O)O"

Escopo documentado: usar um monomero de celulose como proxy da biomassa e uma
simplificacao padrao na literatura de DFT de adsorcao (celulose e um polimero --
DFT nao escala pra ele inteiro); lignina/hemicelulose ficam fora deste primeiro
recorte.
"""
import json
import os

from rdkit import Chem
from rdkit.Chem import AllChem

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

MOLECULES = {
    "methylene_blue": dict(
        smiles="CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2",
        charge=1, multiplicity=1,
        source="PubChem CID 6099 (cátion, sem contraíon Cl-)",
    ),
    "glucose": dict(
        smiles="C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)O)O)O)O)O",
        charge=0, multiplicity=1,
        source="PubChem CID 64689 (beta-D-glicopiranose, monômero da celulose) -- proxy 'in natura'",
    ),
    # Tratamento acido: imersao em H3PO4 0,1 mol/L -- mecanismo real de ativacao acida
    # de biomassa e a esterificacao/fosforilacao das hidroxilas de superficie. Molecula:
    # mesma glicopiranose com o -OH do C6 (primario, exociclico) substituido por um
    # ester fosfato -- estrutura real e catalogada (nao inventada).
    "glucose_acido": dict(
        smiles="C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)O)O)O)O)OP(=O)(O)O",
        charge=0, multiplicity=1,
        source="PubChem CID 439427 (beta-D-glicose 6-fosfato) -- proxy tratamento H3PO4",
    ),
    # Tratamento alcalino: NaOH 0,1 mol/L -- mercerizacao/ativacao alcalina aumenta o
    # carater anionico/nucleofilico das hidroxilas de superficie (desprotonacao parcial).
    # Molecula: MESMA glicopiranose acima, com o -OH do C6 desprotonado (alcoxido),
    # para manter o mesmo sitio estrutural comparavel entre os 3 tratamentos.
    "glucose_base": dict(
        smiles="C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)O)O)O)O)[O-]",
        charge=-1, multiplicity=1,
        source="Glicopiranose C6-alcóxido (desprotonação do -OH primário) -- proxy tratamento NaOH",
    ),
}


def build(name, spec):
    mol = Chem.MolFromSmiles(spec["smiles"])
    if mol is None:
        raise ValueError(f"SMILES invalida para {name}: {spec['smiles']}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol, params)
    if cid != 0:
        raise RuntimeError(f"Falha ao gerar conformero 3D para {name}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))

    n_atoms = len(atoms)
    xyz_path = os.path.join(OUT_DIR, f"{name}_initial.xyz")
    with open(xyz_path, "w") as fh:
        fh.write(f"{n_atoms}\n")
        fh.write(f"{name} (RDKit ETKDG + MMFF94), carga={spec['charge']}, fonte={spec['source']}\n")
        for sym, x, y, z in atoms:
            fh.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"{name}: {n_atoms} átomos, carga={spec['charge']} -> {xyz_path}")
    return dict(name=name, n_atoms=n_atoms, xyz_path=xyz_path, **spec)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for name, spec in MOLECULES.items():
        results[name] = build(name, spec)
    with open(os.path.join(OUT_DIR, "structures_meta.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
