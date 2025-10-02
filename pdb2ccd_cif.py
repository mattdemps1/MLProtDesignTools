#!/usr/bin/env python3
"""
PDB → CCD-style mmCIF (AF3 runfile compatible) with append, aliases, and
Open Babel compatibility fallbacks (no OBResidue.IsMember, no OBElementTable).

============================
Quickstart (your paths)
============================
1) Single block (CKA) → writes one CCD block to a new CIF
   python /home/matt-dempsey/Projects/MLProtDesignTools/pdb2ccd_cif.py \
     --pdb /home/matt-dempsey/Projects/MLProtDesignTools/cka_unk.pdb \
     --resname CKA \
     --out /home/matt-dempsey/Projects/MLProtDesignTools/inputs/cka_unk.cif

2) Append a second block (UNK) from the same PDB
   python /home/matt-dempsey/Projects/MLProtDesignTools/pdb2ccd_cif.py \
     --pdb /home/matt-dempsey/Projects/MLProtDesignTools/cka_unk.pdb \
     --resname UNK \
     --out /home/matt-dempsey/Projects/MLProtDesignTools/inputs/cka_unk.cif \
     --append

3) Same geometry under two codes (CKA + UNK) in one go (aliases)
   python /home/matt-dempsey/Projects/MLProtDesignTools/pdb2ccd_cif.py \
     --pdb /home/matt-dempsey/Projects/MLProtDesignTools/cka_unk.pdb \
     --resname CKA \
     --also_resnames UNK \
     --out /home/matt-dempsey/Projects/MLProtDesignTools/inputs/cka_unk.cif

Tips
----
• Use --list-residues to print residue names found in the PDB and exit.
• If your PDB has multiple residues with the same name, disambiguate with
  --chain <A/B/...> and/or --resnum <integer>.
• If you later add a MOL2, pass --mol path/to.mol2 to improve bond orders.
"""

from openbabel import openbabel
import os, sys, argparse, json
from collections import defaultdict, Counter

# Periodic table (index == atomic number); index 0 is empty sentinel
PT = [
    "", "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]

BOND_ORDER_MAP = {
    "1": ("SING", "N"),
    "2": ("DOUB", "N"),
    "3": ("TRIP", "N"),
    "4": ("QUAD", "N"),
    "ar": ("DOUB", "Y"),  # aromatic
    "am": ("AMIDE", "N"),
}

def _element_symbol(Z: int) -> str:
    return PT[Z] if 0 <= Z < len(PT) else "X"

essential_hint = (
    "Hint: If you see \"Residue '<RES>' not found\", list residue names with:\n"
    "  awk '{print $4}' /path/to/your.pdb | sort -u\n"
    "and pass the exact name to --resname. Optionally add --chain and --resnum.\n"
)

def _iter_residues(mol):
    for r in openbabel.OBResidueIter(mol):
        yield r

def _first_residue(mol):
    for r in _iter_residues(mol):
        return r
    return None

def _match_residue(mol, target_resname=None, chain=None, resnum=None):
    """Return the first residue matching filters; None if not found."""
    for r in _iter_residues(mol):
        if target_resname and r.GetName().strip() != str(target_resname).strip():
            continue
        if chain and (r.GetChain() or "").strip() != str(chain).strip():
            continue
        if resnum is not None and r.GetNum() != int(resnum):
            continue
        return r
    return None

def _collect_atoms_of_residue(mol, residue):
    """Collect atoms whose parent residue equals the chosen residue.
    Avoids OBResidue.IsMember (not present in some wheels).
    """
    recs = []
    for a in openbabel.OBMolAtomIter(mol):
        res = a.GetResidue()
        if not res:
            continue
        # match by identity or by (name, num, chain) triple
        same_ptr = (res == residue)
        same_key = (
            res.GetName().strip() == residue.GetName().strip()
            and res.GetNum() == residue.GetNum()
            and (res.GetChain() or "") == (residue.GetChain() or "")
        )
        if same_ptr or same_key:
            nm = res.GetAtomID(a).strip()
            elem = _element_symbol(a.GetAtomicNum())
            recs.append((a, nm, elem, a.GetFormalCharge(), a.GetX(), a.GetY(), a.GetZ()))
    return recs

def _unique_ids(recs):
    seen=set(); counts=defaultdict(int); out=[]
    for (_a,nm,elem,*_) in recs:
        base=(nm or elem).replace(" ","") or elem
        cand=base or elem or "X"
        if cand in seen:
            counts[elem]+=1
            cand=f"{elem}{counts[elem]}"
            while cand in seen:
                counts[elem]+=1; cand=f"{elem}{counts[elem]}"
        else:
            counts[elem]=max(counts[elem],1)
        seen.add(cand); out.append(cand)
    assert len(out)==len(set(out)), "Atom IDs not unique after repair."
    return out

def _mol2_bonds(path):
    with open(path) as f: lines=f.readlines()
    try:
        start = lines.index("@<TRIPOS>BOND\n") + 1
    except ValueError:
        raise ValueError("MOL2 missing @<TRIPOS>BOND section.")
    bonds=[]
    for ln in lines[start:]:
        ln=ln.strip()
        if not ln or ln.startswith("@<TRIPOS>"):
            break
        parts=ln.split()
        if len(parts) < 4:
            continue
        _, a1, a2, order = parts[:4]
        bonds.append((int(a1), int(a2), order))
    return bonds

def _pdb_bonds(obmol):
    out=[]
    for b in openbabel.OBMolBondIter(obmol):
        order = "ar" if b.IsAromatic() else str(b.GetBondOrder())
        out.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order))
    return out

def _idx2id(atom_ids, recs):
    mp={}
    for aid, (a, *_) in zip(atom_ids, recs):
        mp[a.GetIdx()] = aid
    return mp

def _map_bonds(bidx, mp):
    out=[]
    for a1, a2, ordr in bidx:
        if a1 not in mp or a2 not in mp:
            continue
        if ordr not in BOND_ORDER_MAP:
            raise ValueError(f"Unsupported bond order '{ordr}'.")
        vo, ar = BOND_ORDER_MAP[ordr]
        out.append((mp[a1], mp[a2], vo, ar))
    return out

def _formula_from_obmol(m):
    # Prefer Open Babel formula when available; fallback to our own count
    try:
        spaced = m.GetSpacedFormula()
        if spaced:
            toks = spaced.split()
            return " ".join(toks[i] + toks[i+1] for i in range(0, len(toks), 2))
    except Exception:
        pass
    cnt = Counter(_element_symbol(a.GetAtomicNum()) for a in openbabel.OBMolAtomIter(m))
    # Simple Hill-like order: C, H, then alphabetical
    parts=[]
    for sym in (["C","H"] + sorted(k for k in cnt if k not in {"C","H"})):
        if sym in cnt and cnt[sym] > 0:
            parts.append(f"{sym}{cnt[sym]}")
    return " ".join(parts)

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
    for aid, elem, chg, (x, y, z) in zip(atom_ids, elements, charges, coords):
        lines.append(f"{comp_id} {aid} {elem} {chg} {leaving.get(aid,'N')} {x:.3f} {y:.3f} {z:.3f}")
    lines.append("#")
    lines.append("loop_")
    lines.append("_chem_comp_bond.atom_id_1")
    lines.append("_chem_comp_bond.atom_id_2")
    lines.append("_chem_comp_bond.value_order")
    lines.append("_chem_comp_bond.pdbx_aromatic_flag")
    for a1, a2, vo, ar in cif_bonds:
        lines.append(f"{a1} {a2} {vo} {ar}")
    lines.append("#\n")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Make AF3-ready CCD CIF with optional append and alias blocks.")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--mol")
    ap.add_argument("--resname", required=True, help="Primary residue code (e.g., CKA).")
    ap.add_argument("--also_resnames", default="", help="Comma-separated extra residue codes to duplicate the same geometry under (e.g., UNK,FOO).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting.")
    ap.add_argument("--leaving_flags", default=None, help="JSON mapping atom_id→'Y'|'N'. Applies to ALL blocks.")
    ap.add_argument("--list-residues", action="store_true", help="List residue names (and chain/resnum) in the PDB and exit.")
    ap.add_argument("--chain", default=None, help="Optional chain ID to disambiguate residues with same name.")
    ap.add_argument("--resnum", type=int, default=None, help="Optional residue number to disambiguate residues with same name.")
    args = ap.parse_args()

    obconv = openbabel.OBConversion(); assert obconv.SetInFormat("pdb")
    obmol = openbabel.OBMol()
    if not obconv.ReadFile(obmol, args.pdb):
        raise RuntimeError(f"Failed to read PDB: {args.pdb}")

    if args.list_residues:
        seen = []
        for r in _iter_residues(obmol):
            seen.append((r.GetName().strip(), (r.GetChain() or "").strip(), r.GetNum()))
        print("Residues found (name, chain, resnum):")
        for name, ch, num in seen:
            print(f"  {name:>4s}  chain={ch or '-':1s}  resnum={num}")
        sys.exit(0)

    chosen = _match_residue(obmol, args.resname, args.chain, args.resnum)
    if chosen is None:
        sys.stderr.write(f"Residue '{args.resname}' not found with given filters.\n" + essential_hint)
        sys.exit(2)

    recs = _collect_atoms_of_residue(obmol, chosen)
    if not recs:
        raise ValueError("Selected residue has no atoms or couldn't be matched.")

    atom_ids = _unique_ids(recs)
    idxmap = _idx2id(atom_ids, recs)
    elements = [r[2] for r in recs]
    charges  = [r[3] for r in recs]
    coords   = [(r[4], r[5], r[6]) for r in recs]

    if args.mol:
        bonds_idx = _mol2_bonds(args.mol)
    else:
        bonds_idx = _pdb_bonds(obmol)
    bonds_idx = [(a1, a2, o) for (a1, a2, o) in bonds_idx if a1 in idxmap and a2 in idxmap]
    cif_bonds = _map_bonds(bonds_idx, idxmap)

    leaving = {aid: "N" for aid in atom_ids}
    if args.leaving_flags:
        custom = json.loads(args.leaving_flags)
        for k, v in custom.items():
            if k in leaving:
                leaving[k] = 'Y' if str(v).upper().startswith('Y') else 'N'

    formula = _formula_from_obmol(obmol)
    mw = obmol.GetExactMass()

    # Build blocks: primary + aliases
    codes = [args.resname] + [c.strip() for c in args.also_resnames.split(',') if c.strip()]
    blocks = []
    for code in codes:
        blocks.append(_block_text(code, atom_ids, elements, charges, coords, cif_bonds, formula, mw, leaving))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    mode = 'a' if (args.append and os.path.exists(args.out)) else 'w'
    with open(args.out, mode) as fh:
        if mode == 'a':
            try:
                with open(args.out, 'rb') as rb:
                    if not rb.read().endswith(b"\n"):
                        fh.write("\n")
            except Exception:
                pass
        fh.write("\n".join(blocks))

    print(f"Wrote {len(blocks)} block(s) to {args.out}: {', '.join(f'data_comp_{c}' for c in codes)}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n"); sys.exit(1)
