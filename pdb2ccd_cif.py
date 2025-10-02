#!/usr/bin/env python3
"""
PDB → CCD-style mmCIF converter (AF3 runfile compatible)
-------------------------------------------------------
Writes a single CCD component block named exactly **data_comp_<RES>** so it can
be discovered by your run file's `parse_fasta_input` logic which searches for
that specific data block name. Generates unique atom IDs and a bond table that
references those exact IDs. Supports optional MOL2 for explicit bond orders.

Usage
-----
python pdb2ccd_cif_af3_runfile_compatible.py \
  --pdb path/to/ligand.pdb \
  [--mol path/to/ligand.mol2] \
  [--resname CKA] \
  [--out CKA.cif]

python pdb2ccd_cif_af3_runfile_compatible.py \
  --pdb CKA.pdb \
  --mol CKA.mol2 \        # optional but preferred
  --resname CKA \         # must match the ligand code you pass in --ligand
  --out CKA.cif

Notes
-----
- `--resname` **must** match the ligand code you pass before the comma in
  `--ligand LIG,/path/to/LIG.cif` so `_chem_comp.id == LIG` and the block is
  `data_comp_LIG`.
- If `--mol` is omitted, bonds are inferred from the PDB (CONECT) via Open Babel.
- All `_chem_comp_atom.pdbx_leaving_atom_flag` default to 'N'; override with
  `--leaving_flags '{"Cl1":"Y"}'` if needed.
"""

from openbabel import openbabel
import os
import sys
import argparse
import json
from collections import defaultdict

BOND_ORDER_MAP = {
    "1": ("SING", "N"),
    "2": ("DOUB", "N"),
    "3": ("TRIP", "N"),
    "4": ("QUAD", "N"),
    "ar": ("DOUB", "Y"),  # aromatic
    "am": ("AMIDE", "N"),
}

# ----------------------------- helpers ---------------------------------

def _first_residue(mol):
    for r in openbabel.OBResidueIter(mol):
        return r
    return None


def _collect_residue_atoms(mol, target_resname=None):
    if target_resname:
        chosen = None
        for r in openbabel.OBResidueIter(mol):
            if r.GetName().strip() == target_resname.strip():
                chosen = r
                break
        if chosen is None:
            raise ValueError(f"Residue '{target_resname}' not found in PDB.")
    else:
        chosen = _first_residue(mol)
        if chosen is None:
            raise ValueError("No residues found in the PDB.")

    sel = []
    for a in openbabel.OBMolAtomIter(mol):
        if chosen.IsMember(a):
            atom_name = chosen.GetAtomID(a).strip()
            elem = openbabel.OBElementTable().GetSymbol(a.GetAtomicNum())
            charge = a.GetFormalCharge()
            sel.append((a, atom_name, elem, charge, a.GetX(), a.GetY(), a.GetZ()))
    if not sel:
        raise ValueError("Selected residue has no atoms.")
    return chosen, sel


def _unique_atom_ids(records):
    # Prefer given atom names; repair duplicates to Elem# (C1,C2,...)
    counts = defaultdict(int)
    seen = set()
    out = []
    for (_a, name, elem, *_rest) in records:
        base = name.replace(" ", "") if name else elem
        cand = base if base else elem
        if cand in seen or not cand:
            counts[elem] += 1
            cand = f"{elem}{counts[elem]}"
            while cand in seen:
                counts[elem] += 1
                cand = f"{elem}{counts[elem]}"
        else:
            counts[elem] = max(counts[elem], 1)
        seen.add(cand)
        out.append(cand)
    assert len(out) == len(set(out)), "Atom IDs not unique after repair."
    return out


def _read_mol2_bonds(path):
    with open(path, "r") as f:
        lines = f.readlines()
    try:
        start = lines.index("@<TRIPOS>BOND\n") + 1
    except ValueError:
        raise ValueError("MOL2 missing @<TRIPOS>BOND section.")
    bonds = []
    for ln in lines[start:]:
        ln = ln.strip()
        if not ln or ln.startswith("@<TRIPOS>"):
            break
        parts = ln.split()
        if len(parts) < 4:
            continue
        _, a1, a2, order = parts[:4]
        bonds.append((int(a1), int(a2), order))
    return bonds


def _read_pdb_bonds(obmol):
    bonds = []
    for b in openbabel.OBMolBondIter(obmol):
        order = "ar" if b.IsAromatic() else str(b.GetBondOrder())
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order))
    return bonds


def _idx_to_id_map(atom_ids, records):
    mapping = {}
    for aid, (a, *_rest) in zip(atom_ids, records):
        mapping[a.GetIdx()] = aid
    return mapping


def _map_bonds(bonds_idx, idx_to_id):
    out = []
    for a1, a2, order in bonds_idx:
        if order not in BOND_ORDER_MAP:
            raise ValueError(f"Unsupported bond order token '{order}'.")
        if a1 not in idx_to_id or a2 not in idx_to_id:
            # Skip bonds that reference atoms outside the chosen residue
            continue
        value_order, arom = BOND_ORDER_MAP[order]
        out.append((idx_to_id[a1], idx_to_id[a2], value_order, arom))
    return out


def _formula_from_obmol(m):
    toks = m.GetSpacedFormula().split()
    return " ".join(toks[i] + toks[i + 1] for i in range(0, len(toks), 2))

# ------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert ligand PDB (+optional MOL2) to CCD-style mmCIF (data_comp_<RES>).")
    ap.add_argument("--pdb", required=True, help="Ligand PDB containing the residue of interest.")
    ap.add_argument("--mol", default=None, help="Optional MOL2 with explicit bonds matching the same atom order as PDB.")
    ap.add_argument("--resname", default=None, help="Residue code for _chem_comp.id and block name (e.g., CKA). Defaults to first residue name in PDB.")
    ap.add_argument("--out", default=None, help="Output CIF path. Defaults to <resname>.cif next to the PDB.")
    ap.add_argument("--leaving_flags", default=None, help="JSON dict mapping atom_id→'Y'|'N' for pdbx_leaving_atom_flag.")
    args = ap.parse_args()

    obconv = openbabel.OBConversion()
    if not obconv.SetInFormat("pdb"):
        raise RuntimeError("Failed to set PDB input format.")
    obmol = openbabel.OBMol()
    if not obconv.ReadFile(obmol, args.pdb):
        raise RuntimeError(f"Failed to read PDB: {args.pdb}")

    chosen_res, recs = _collect_residue_atoms(obmol, args.resname)
    comp_id = (args.resname or chosen_res.GetName().strip()) or "LIG"

    atom_ids = _unique_atom_ids(recs)
    idx_to_id = _idx_to_id_map(atom_ids, recs)

    elements = [r[2] for r in recs]
    charges = [r[3] for r in recs]
    coords = [(r[4], r[5], r[6]) for r in recs]

    # Bonds
    if args.mol:
        bonds_idx = _read_mol2_bonds(args.mol)
    else:
        bonds_idx = _read_pdb_bonds(obmol)
    bonds_idx = [(a1, a2, o) for (a1, a2, o) in bonds_idx if a1 in idx_to_id and a2 in idx_to_id]
    cif_bonds = _map_bonds(bonds_idx, idx_to_id)

    # Leaving flags
    leaving = {aid: "N" for aid in atom_ids}
    if args.leaving_flags:
        custom = json.loads(args.leaving_flags)
        for k, v in custom.items():
            if k in leaving:
                leaving[k] = "Y" if str(v).upper().startswith("Y") else "N"

    # Compose CIF
    formula = _formula_from_obmol(obmol)
    mw = obmol.GetExactMass()

    lines = []
    lines.append(f"data_comp_{comp_id}")  # *** IMPORTANT: runfile expects this ***
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
    for aid, elem, chg, (x, y, z) in zip(atom_ids, elements, charges, coords):
        lines.append(f"{comp_id} {aid} {elem} {chg} {leaving[aid]} {x:.3f} {y:.3f} {z:.3f}")
    lines.append("#")

    lines.append("loop_")
    lines.append("_chem_comp_bond.atom_id_1")
    lines.append("_chem_comp_bond.atom_id_2")
    lines.append("_chem_comp_bond.value_order")
    lines.append("_chem_comp_bond.pdbx_aromatic_flag")
    for a1, a2, order, arom in cif_bonds:
        lines.append(f"{a1} {a2} {order} {arom}")
    lines.append("#\n")

    out_path = args.out or os.path.join(os.path.dirname(args.pdb), f"{comp_id}.cif")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))

    with open(os.path.splitext(out_path)[0] + "_name_mapping.json", "w") as fh:
        json.dump({comp_id: {aid: aid for aid in atom_ids}}, fh, indent=2)

    print(f"Wrote CCD mmCIF (data_comp_{comp_id}) to: {out_path}")
    print(f"Atoms: {len(atom_ids)} | Bonds: {len(cif_bonds)} | comp_id: {comp_id}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\n[ERROR] {e}\n")
        sys.exit(1)
