#!/usr/bin/env python3
"""
PDB → CCD-style mmCIF (AF3 runfile compatible) with append & aliases
-------------------------------------------------------------------
- Emits **data_comp_<RES>** blocks (exact match to your runfile parser).
- Can **append** to an existing CIF (so multiple components live in one file).
- Can **duplicate** the same geometry under extra residue names (aliases), e.g.
  write both data_comp_CKA *and* data_comp_UNK into the same file.


Examples
--------
# 1) Single block
python pdb2ccd_cif_af3_append_and_alias.py --pdb CKA.pdb --resname CKA --out cka_unk.cif

# 2) Single geometry written under multiple names in one CIF
python pdb2ccd_cif_af3_append_and_alias.py --pdb CKA.pdb --resname CKA \
  --also_resnames UNK --out cka_unk.cif --append

# 3) Prefer bonds from MOL2; add another alias
python pdb2ccd_cif_af3_append_and_alias.py --pdb CKA.pdb --mol CKA.mol2 \
  --resname CKA --also_resnames UNK,FOO --out cka_unk.cif --append


python pdb2ccd_cif_af3_append_and_alias.py \
  --pdb /path/CKA.pdb \
  --mol /path/CKA.mol2 \        # optional but preferred for bond orders
  --resname CKA \               # first block name -> data_comp_CKA
  --also_resnames UNK \         # duplicate the same geometry as data_comp_UNK
  --out /N/slate/mattdemp/Projects/MLProtDesignTools/inputs/cka_unk.cif \
  --append


Your runfile lines like:
  ligands:
    - "CKA,/path/cka_unk.cif"
    - "UNK,/path/cka_unk.cif"
will work because the file contains both data_comp_CKA and data_comp_UNK blocks.
"""

from openbabel import openbabel
import os, sys, argparse, json
from collections import defaultdict

BOND_ORDER_MAP = {
    "1": ("SING", "N"),
    "2": ("DOUB", "N"),
    "3": ("TRIP", "N"),
    "4": ("QUAD", "N"),
    "ar": ("DOUB", "Y"),
    "am": ("AMIDE", "N"),
}

def _first_res(m):
    for r in openbabel.OBResidueIter(m):
        return r
    return None

def _collect(m, target=None):
    if target:
        chosen=None
        for r in openbabel.OBResidueIter(m):
            if r.GetName().strip()==target.strip():
                chosen=r; break
        if chosen is None:
            raise ValueError(f"Residue '{target}' not found in PDB.")
    else:
        chosen=_first_res(m)
        if chosen is None: raise ValueError("No residues found in the PDB.")
    recs=[]
    for a in openbabel.OBMolAtomIter(m):
        if chosen.IsMember(a):
            nm=chosen.GetAtomID(a).strip()
            elem=openbabel.OBElementTable().GetSymbol(a.GetAtomicNum())
            recs.append((a,nm,elem,a.GetFormalCharge(),a.GetX(),a.GetY(),a.GetZ()))
    if not recs: raise ValueError("Selected residue has no atoms.")
    return chosen,recs

def _unique_ids(recs):
    seen=set(); counts=defaultdict(int); out=[]
    for (_a,nm,elem,*_) in recs:
        base=(nm or elem).replace(" ","") or elem
        cand=base
        if cand in seen:
            counts[elem]+=1
            cand=f"{elem}{counts[elem]}"
            while cand in seen:
                counts[elem]+=1; cand=f"{elem}{counts[elem]}"
        else:
            counts[elem]=max(counts[elem],1)
        seen.add(cand); out.append(cand)
    assert len(out)==len(set(out)),"Atom IDs not unique after repair."
    return out

def _mol2_bonds(p):
    with open(p) as f: lines=f.readlines()
    try: start=lines.index("@<TRIPOS>BOND\n")+1
    except ValueError: raise ValueError("MOL2 missing @<TRIPOS>BOND section.")
    bonds=[]
    for ln in lines[start:]:
        ln=ln.strip()
        if not ln or ln.startswith("@<TRIPOS>"): break
        parts=ln.split();
        if len(parts)<4: continue
        _,a1,a2,order=parts[:4]; bonds.append((int(a1),int(a2),order))
    return bonds

def _pdb_bonds(obm):
    out=[]
    for b in openbabel.OBMolBondIter(obm):
        order="ar" if b.IsAromatic() else str(b.GetBondOrder())
        out.append((b.GetBeginAtomIdx(),b.GetEndAtomIdx(),order))
    return out

def _idx2id(atom_ids,recs):
    mp={}
    for aid,(a,*_) in zip(atom_ids,recs): mp[a.GetIdx()]=aid
    return mp

def _map_bonds(bidx,mp):
    out=[]
    for a1,a2,ordr in bidx:
        if a1 not in mp or a2 not in mp: continue
        if ordr not in BOND_ORDER_MAP: raise ValueError(f"Unsupported bond order '{ordr}'.")
        vo,ar=BOND_ORDER_MAP[ordr]; out.append((mp[a1],mp[a2],vo,ar))
    return out

def _formula(m):
    toks=m.GetSpacedFormula().split(); return " ".join(toks[i]+toks[i+1] for i in range(0,len(toks),2))

def _block_text(comp_id, atom_ids, elements, charges, coords, cif_bonds, formula, mw, leaving):
    lines=[]
    lines.append(f"data_comp_{comp_id}")
    lines.append("#")
    lines.append(f"_chem_comp.id {comp_id}")
    lines.append(f"_chem_comp.name '{comp_id}'")
    lines.append("_chem_comp.type non-polymer")
    lines.append(f"_chem_comp.formula '{formula}'")
    lines.append("_chem_comp.mon_nstd_parent_comp_id ?")
    lines.append("_chem_comp.pdbx_synonyms ?")
    lines.append(f"_chem_comp.formula_weight {mw:.2f}")
    lines.append("_pdbx_chem_comp_descriptor.type ?")
    lines.append("_pdbx_chem_comp_descriptor.descriptor ?")
    lines.append("#")
    lines.append("loop_")
    lines.append("_chem_comp_atom.comp_id")
    lines.append("_chem_comp_atom.atom_id")
    lines.append("_chem_comp_atom.type_symbol")
    lines.append("_chem_comp_atom.charge")
    lines.append("_chem_comp_atom.pdbx_leaving_atom_flag")
    lines.append("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
    lines.append("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
    lines.append("_chem_comp_atom.pdbx_model_Cartn_z_ideal")
    for aid,elem,chg,(x,y,z) in zip(atom_ids,elements,charges,coords):
        lines.append(f"{comp_id} {aid} {elem} {chg} {leaving.get(aid,'N')} {x:.3f} {y:.3f} {z:.3f}")
    lines.append("#")
    lines.append("loop_")
    lines.append("_chem_comp_bond.atom_id_1")
    lines.append("_chem_comp_bond.atom_id_2")
    lines.append("_chem_comp_bond.value_order")
    lines.append("_chem_comp_bond.pdbx_aromatic_flag")
    for a1,a2,vo,ar in cif_bonds:
        lines.append(f"{a1} {a2} {vo} {ar}")
    lines.append("#\n")
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description="Make AF3-ready CCD CIF with optional append and alias blocks.")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--mol")
    ap.add_argument("--resname", required=True, help="Primary residue code (e.g., CKA).")
    ap.add_argument("--also_resnames", default="", help="Comma-separated extra residue codes to duplicate the same geometry under (e.g., UNK,FOO).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting.")
    ap.add_argument("--leaving_flags", default=None, help="JSON mapping atom_id→'Y'|'N'. Applies to ALL blocks.")
    args=ap.parse_args()

    obconv=openbabel.OBConversion(); assert obconv.SetInFormat("pdb")
    obmol=openbabel.OBMol();
    if not obconv.ReadFile(obmol,args.pdb): raise RuntimeError(f"Failed to read PDB: {args.pdb}")

    chosen,recs=_collect(obmol,args.resname)
    atom_ids=_unique_ids(recs)
    idxmap=_idx2id(atom_ids,recs)
    elements=[r[2] for r in recs]; charges=[r[3] for r in recs]; coords=[(r[4],r[5],r[6]) for r in recs]

    if args.mol:
        bonds_idx=_mol2_bonds(args.mol)
    else:
        bonds_idx=_pdb_bonds(obmol)
    bonds_idx=[(a1,a2,o) for (a1,a2,o) in bonds_idx if a1 in idxmap and a2 in idxmap]
    cif_bonds=_map_bonds(bonds_idx,idxmap)

    leaving={aid:"N" for aid in atom_ids}
    if args.leaving_flags:
        custom=json.loads(args.leaving_flags)
        for k,v in custom.items():
            if k in leaving: leaving[k] = 'Y' if str(v).upper().startswith('Y') else 'N'

    formula=_formula(obmol); mw=obmol.GetExactMass()

    # Build blocks: primary + aliases
    codes=[args.resname]+[c.strip() for c in args.also_resnames.split(',') if c.strip()]
    blocks=[]
    for code in codes:
        blocks.append(_block_text(code, atom_ids, elements, charges, coords, cif_bonds, formula, mw, leaving))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    mode='a' if (args.append and os.path.exists(args.out)) else 'w'
    with open(args.out, mode) as fh:
        if mode=='a' and not open(args.out,'rb').read().endswith(b"\n"):
            fh.write("\n")
        fh.write("\n".join(blocks))

    print(f"Wrote {len(blocks)} block(s) to {args.out}: {', '.join(f'data_comp_{c}' for c in codes)}")

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n"); sys.exit(1)

