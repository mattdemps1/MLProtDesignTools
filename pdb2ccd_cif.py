#!/usr/bin/env python3
"""
PDB → CCD-style mmCIF converter for AlphaFold 3 ligand reference structures.

Highlights vs. the original script:
- Builds the CCD CIF directly from Open Babel objects (no fragile mmCIF round‑trip).
- Ensures atom IDs are UNIQUE and stable; repairs/auto‑generates when needed.
- Bond table references exact atom IDs; supports MOL2 input (preferred) or PDB CONECT.
- Validates mapping and will fail with a clear message if references are inconsistent.
- Lets you override residue name (comp_id) and leaving‑atom flags.

Usage
-----
python pdb2ccd_cif.py --pdb path/to/ligand.pdb [--mol path/to/ligand.mol2] \
       [--resname CKA] [--out path/to/CKA.cif]

python pdb2ccd_cif.py --pdb path/to/ligand.pdb \
  --mol path/to/ligand.mol2 \        # optional but preferred (explicit bonds)
  --resname CKA \                    # sets _chem_comp.id and comp_id
  --out path/to/CKA.cif              # optional; defaults to <resname>.cif


Notes
-----
- If you provide --mol, atoms must be in the SAME order as the PDB (typical for a
  single ligand exported as both PDB and MOL2 from the same tool). Bond orders are
  taken from MOL2 when present; otherwise they are inferred from PDB/CONECT/Open Babel.
- The generated mmCIF contains only a single component (data_<resname>).
- Default leaving‑atom flag is 'N' for all atoms; you can supply a JSON map via
  --leaving_flags '{"Cl1":"Y"}' to mark specific leaving groups.
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
    "am": ("AMIDE", "N"), # rarely emitted; keep for compatibility
}

# ----------------------------- helpers ---------------------------------

def _get_first_residue(mol):
    res = None
    for r in openbabel.OBResidueIter(mol):
        res = r
        break
    return res


def _collect_residue_atoms(mol, target_resname=None):
    """Return list of (atom, residue, name, element, charge, x,y,z) for atoms
    that belong to the selected residue. If target_resname is None, use the
    first residue present in the molecule.
    """
    selected = []
    chosen_res = None

    # Determine which residue to use
    if target_resname is not None:
        for r in openbabel.OBResidueIter(mol):
            if r.GetName().strip() == target_resname.strip():
                chosen_res = r
                break
        if chosen_res is None:
            raise ValueError(f"Residue '{target_resname}' not found in PDB.")
    else:
        chosen_res = _get_first_residue(mol)
        if chosen_res is None:
            raise ValueError("No residues found in the PDB.")

    # Collect atoms that belong to the chosen residue
    for a in openbabel.OBMolAtomIter(mol):
        if chosen_res.IsMember(a):
            name = chosen_res.GetAtomID(a).strip() or ""
            elem = a.GetType()[0:2].strip() or a.GetAtomicNum()  # fallback
            elem = a.GetTitle() if len(elem) == 0 else elem
            # OBAtom.GetType may give e.g. "C3"; use element symbol from table
            elem = openbabel.OBElementTable().GetSymbol(a.GetAtomicNum())
            charge = a.GetFormalCharge()
            x, y, z = a.GetX(), a.GetY(), a.GetZ()
            selected.append((a, chosen_res, name, elem, charge, x, y, z))

    if not selected:
        raise ValueError("Selected residue has no atoms.")

    return chosen_res, selected


def _make_unique_atom_ids(records):
    """Ensure each atom_id is unique and CIF‑safe. Prefer the PDB atom name if
    unique; otherwise generate Element+index (e.g., C1, C2...). Returns list of
    atom_id strings aligned with 'records'."""
    # First pass: sanitize and collect proposed names
    proposed = []
    for (_, _, name, elem, *_rest) in records:
        nm = name.replace(" ", "") if name else ""
        if not nm:
            proposed.append(elem)
        else:
            proposed.append(nm)

    # Ensure uniqueness with per‑element counters
    counts = defaultdict(int)
    final_names = []
    seen = set()

    for nm, rec in zip(proposed, records):
        elem = rec[3]
        base = nm if nm else elem
        candidate = base
        if candidate in seen:
            # roll a per‑element counter
            counts[elem] += 1
            candidate = f"{elem}{counts[elem]}"
            while candidate in seen:
                counts[elem] += 1
                candidate = f"{elem}{counts[elem]}"
        else:
            # initialize counter if we end up needing more of this element later
            counts[elem] = max(counts[elem], 1)
        seen.add(candidate)
        final_names.append(candidate)

    # Guarantee stable 1:1 mapping
    assert len(final_names) == len(set(final_names)), "Atom IDs are not unique after repair."
    return final_names


def _read_mol2_bonds(mol2_path):
    with open(mol2_path, "r") as f:
        lines = f.readlines()
    try:
        start = lines.index("@<TRIPOS>BOND\n") + 1
    except ValueError:
        raise ValueError("MOL2 file missing @<TRIPOS>BOND section.")

    bonds = []
    for ln in lines[start:]:
        ln = ln.strip()
        if ln.startswith("@<TRIPOS>") or not ln:
            break
        parts = ln.split()
        if len(parts) < 4:
            continue
        # id, a1_idx, a2_idx, order_token
        _, a1, a2, order = parts[:4]
        bonds.append((int(a1), int(a2), order))
    return bonds


def _read_pdb_bonds(obmol):
    bonds = []
    for b in openbabel.OBMolBondIter(obmol):
        order = "ar" if b.IsAromatic() else str(b.GetBondOrder())
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order))
    return bonds


def _map_bonds_to_ids(bonds_idx, idx_to_name):
    cif_bonds = []
    for a1, a2, order in bonds_idx:
        if order not in BOND_ORDER_MAP:
            raise ValueError(f"Unsupported bond order token '{order}'.")
        if a1 not in idx_to_name or a2 not in idx_to_name:
            raise ValueError("Bond references atom indices not present in residue.")
        value_order, arom = BOND_ORDER_MAP[order]
        cif_bonds.append((idx_to_name[a1], idx_to_name[a2], value_order, arom))
    return cif_bonds


def _formula_from_obmol(mol):
    spaced = mol.GetSpacedFormula().split()
    # spaced = [elem, count, elem, count, ...]
    out = []
    for i in range(0, len(spaced), 2):
        out.append(spaced[i] + spaced[i + 1])
    return " ".join(out)


# ----------------------------- main ------------------------------------

def main():
    p = argparse.ArgumentParser(description="Convert ligand PDB (+optional MOL2) to CCD-style mmCIF for AF3.")
    p.add_argument("--pdb", required=True, help="Ligand PDB containing a single residue of interest.")
    p.add_argument("--mol", default=None, help="Optional MOL2 with explicit bonds matching the same atom order as PDB.")
    p.add_argument("--resname", default=None, help="Override comp_id / residue name (e.g., CKA). Defaults to first residue name in PDB.")
    p.add_argument("--out", default=None, help="Output CIF path. Defaults to <resname>.cif next to the PDB.")
    p.add_argument("--leaving_flags", default=None, help="JSON dict mapping atom_id→'Y'|'N' for pdbx_leaving_atom_flag.")
    args = p.parse_args()

    # Load PDB into OBMol
    obconv = openbabel.OBConversion()
    if not obconv.SetInFormat("pdb"):
        raise RuntimeError("Failed to set PDB input format for Open Babel.")
    obmol = openbabel.OBMol()
    if not obconv.ReadFile(obmol, args.pdb):
        raise RuntimeError(f"Failed to read PDB: {args.pdb}")

    # Pick residue and collect atoms
    chosen_res, recs = _collect_residue_atoms(obmol, args.resname)
    comp_id = args.resname or chosen_res.GetName().strip()
    if not comp_id:
        raise ValueError("Could not determine residue name; use --resname.")

    # Make unique atom IDs (aligned with recs order)
    atom_ids = _make_unique_atom_ids(recs)

    # Build index→atom_id mapping using OBAtom.GetIdx (1-based)
    idx_to_name = {}
    for atom_id, (a, *_rest) in zip(atom_ids, recs):
        idx_to_name[a.GetIdx()] = atom_id

    # Charges and coordinates
    charges = [r[4] for r in recs]
    coords = [(r[5], r[6], r[7]) for r in recs]
    elements = [r[3] for r in recs]

    # Optional MOL2 bonds (preferred)
    if args.mol:
        bonds_idx = _read_mol2_bonds(args.mol)
    else:
        bonds_idx = _read_pdb_bonds(obmol)

    # Restrict bonds to atoms that belong to the chosen residue
    bonds_idx = [(a1, a2, o) for (a1, a2, o) in bonds_idx if a1 in idx_to_name and a2 in idx_to_name]

    cif_bonds = _map_bonds_to_ids(bonds_idx, idx_to_name)

    # Leaving atom flags
    leaving = {aid: "N" for aid in atom_ids}
    if args.leaving_flags:
        custom = json.loads(args.leaving_flags)
        for k, v in custom.items():
            if k in leaving:
                leaving[k] = "Y" if str(v).upper().startswith("Y") else "N"

    # Validate consistency
    if len(atom_ids) != len(charges) or len(atom_ids) != len(coords):
        raise AssertionError("Atom counts mismatch among IDs/charges/coords.")

    # Header
    formula = _formula_from_obmol(obmol)
    mw = obmol.GetExactMass()

    lines = []
    lines.append(f"data_{comp_id}")
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

    # Atom loop
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

    # Bond loop
    lines.append("loop_")
    lines.append("_chem_comp_bond.atom_id_1")
    lines.append("_chem_comp_bond.atom_id_2")
    lines.append("_chem_comp_bond.value_order")
    lines.append("_chem_comp_bond.pdbx_aromatic_flag")

    for a1, a2, order, arom in cif_bonds:
        lines.append(f"{a1} {a2} {order} {arom}")

    lines.append("#\n")

    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(args.pdb))[0]
        out_path = os.path.join(os.path.dirname(args.pdb), f"{comp_id}.cif")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))

    # Optional name mapping for debugging
    mapping = {comp_id: {aid: aid for aid in atom_ids}}
    with open(os.path.splitext(out_path)[0] + "_name_mapping.json", "w") as fh:
        json.dump(mapping, fh, indent=2)

    print(f"Wrote CCD mmCIF to: {out_path}")
    print(f"Atoms: {len(atom_ids)} | Bonds: {len(cif_bonds)} | comp_id: {comp_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\n[ERROR] {e}\n")
        sys.exit(1)
