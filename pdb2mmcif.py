from openbabel import openbabel
import os, sys
import argparse
import json

#python /home/matt-dempsey/Projects/MLProtDesignTools/pdb2mmcif.py --pdb path/to/pdb.pdb
#/home/matt-dempsey/Downloads/rh2 denovo cofactor.pdb



bond_orders = {"ar": "DOUB", "2": "DOUB", "1": "SING", "3": "TRIP", "4": "QUAD", "am": "AMIDE"}

OBABEL_AF3_labels = {'_atom_site.type_symbol': '_chem_comp_atom.type_symbol',
                     '_atom_site.label_atom_id': '_chem_comp_atom.atom_id',
                     '_atom_site.label_comp_id': '_chem_comp_atom.comp_id',
                     '_atom_site.Cartn_x': '_chem_comp_atom.pdbx_model_Cartn_x_ideal',
                     '_atom_site.Cartn_y': '_chem_comp_atom.pdbx_model_Cartn_y_ideal',
                     '_atom_site.Cartn_z': '_chem_comp_atom.pdbx_model_Cartn_z_ideal'}
AF3_OBABEL_labels = {v: k for k,v in OBABEL_AF3_labels.items()}

AF3_cif_labels = ['_chem_comp_atom.comp_id',
                 '_chem_comp_atom.atom_id',
                 '_chem_comp_atom.type_symbol',
                 '_chem_comp_atom.charge',
                 '_chem_comp_atom.pdbx_leaving_atom_flag',
                 '_chem_comp_atom.pdbx_model_Cartn_x_ideal',
                 '_chem_comp_atom.pdbx_model_Cartn_y_ideal',
                 '_chem_comp_atom.pdbx_model_Cartn_z_ideal']

"""
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal

_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z

"""

def main(args):
    
    pdbfile = args.pdb
    molfile = args.mol

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "mmcif")
    pdbmol = openbabel.OBMol()
    obConversion.ReadFile(pdbmol, pdbfile)
    outCIF = obConversion.WriteString(pdbmol)
    
    NAME3 = pdbmol.GetResidue(0).GetName()
    print(f"Processing ligand {NAME3}")

    if args.mol is not None:
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("mol2")
        obmol = openbabel.OBMol()
        obConversion.ReadFile(obmol, molfile)

        charges = [a.GetFormalCharge() for a in openbabel.OBMolAtomIter(obmol)]
    else:
        print("Extracted charges from the PDB file - double check that they are correct!")
        charges = [a.GetFormalCharge() for a in openbabel.OBMolAtomIter(pdbmol)]

    # Cleaning up mmCIF string
    outCIF_lines = outCIF.split("\n")
    
    atom_labels = []
    for i,l in enumerate(outCIF_lines):
        if "_atom_site." in l:
            atom_labels.append(l)
            last_atom_site_line = i
    
    molecule_lines = outCIF_lines[last_atom_site_line+1:]
    molecule_lines = [l for l in molecule_lines if len(l) != 0]
    
    molecule_lines_fixed = []
    for i,l in enumerate(molecule_lines):
        if i % 2 == 0:
            molecule_lines_fixed.append(l+molecule_lines[i+1])
    
    molecule_lines_assigned = []
    for l in molecule_lines_fixed:
        lspl = l.strip().split("\t")
        _dct = {}
        for lbl,par in zip(atom_labels, lspl):
            _dct[lbl] = par
        molecule_lines_assigned.append(_dct)
    
    assert len(molecule_lines_assigned) == len(charges)

    atoms = [_dct["_atom_site.label_atom_id"] for _dct in molecule_lines_assigned]
    # O1 is problematic because if an atom is named that, then it will disappear from AF3 output!?
    if "O1" in atoms:
        if "O9" not in atoms:
            new_O1 = "O9"
        else:
            n = 2
            while f"O{n}" in atoms:
                n += 1
                if n > 99:
                    sys.exit("Something is really wrong with O naming???")
            new_O1 = f"O{n}"
        print(f"Atom 'O1' found in molecule - renaming it to {new_O1}")
    
    # Building molecule lines in AF3 format
    molecule_lines_AF3 = []
    name_mapping = {}
    for i,_dct in enumerate(molecule_lines_assigned):
        line = []
        for lbl in AF3_cif_labels:
            if lbl in AF3_OBABEL_labels:
                line.append(_dct[AF3_OBABEL_labels[lbl]].strip())
                if lbl == "_chem_comp_atom.atom_id":
                    if _dct[AF3_OBABEL_labels[lbl]] == "O1":
                        line[-1] = new_O1
                    name_mapping[line[-1]] = _dct[AF3_OBABEL_labels[lbl]]
            elif lbl == '_chem_comp_atom.charge':
                line.append(str(charges[i]))
            elif lbl == '_chem_comp_atom.pdbx_leaving_atom_flag':
                line.append("N")
        molecule_lines_AF3.append(" ".join(line))
    
    
    ### Bonding section
    
    if args.mol is not None:
        print("Reading bonds from molfile")
        molf = open(molfile, "r").readlines()
        num_bonds = int(molf[ molf.index("@<TRIPOS>MOLECULE\n")+2].split()[1])  # assuming n_atoms and n_bonds is always space-separated in molfile header - is this safe?
        bond_start_mol = molf.index("@<TRIPOS>BOND\n")+1
        bond_lines_mol = molf[bond_start_mol:bond_start_mol+num_bonds]
    else:
        print("Reading bonds from PDB file - this may be problematic if hte PDB does not have correct CONECT section.")
        print("Double-check bond orders!")
        
        bonds = [b for b in openbabel.OBMolBondIter(pdbmol)]
        bond_lines_mol = []
        for b in bonds:
            order = b.GetBondOrder()
            if b.IsAromatic():
                order = "ar"
            line = f"{b.GetIdx()+1} {b.GetBeginAtomIdx()} {b.GetEndAtomIdx()} {order}"
            bond_lines_mol.append(line)

    
    # building AF3-style bonding section
    cifbonds = []
    for bond in bond_lines_mol:
        spl = bond.split()
        a1 = atoms[int(spl[1])-1]
        a2 = atoms[int(spl[2])-1]
        order = bond_orders[spl[3]]
        arom = "N"
        if spl[3] == "ar":
            arom = "Y"
        ln = f"{a1} {a2} {order} {arom}"
        cifbonds.append(ln)


    ### Building the molecule header and putting it all together
    formula = pdbmol.GetSpacedFormula().split()
    formula_formatted = []
    for i,x in enumerate(formula):
        if i%2 ==0:
            formula_formatted.append(x+formula[i+1])
    hdr = f"""data_{NAME3}
#
_chem_comp.id {NAME3}
_chem_comp.name 'blabla'
_chem_comp.type non-polymer
_chem_comp.formula '{' '.join(formula_formatted)}'
_chem_comp.mon_nstd_parent_comp_id ?
_chem_comp.pdbx_synonyms ?
_chem_comp.formula_weight {pdbmol.GetExactMass():.2f}
_pdbx_chem_comp_descriptor.type ?
_pdbx_chem_comp_descriptor.descriptor ?
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
"""
    
    hdr += "\n".join(molecule_lines_AF3)
    
    hdr += """
#
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
"""
    
    hdr += "\n".join(cifbonds)
    hdr += "\n#\n"
    
    
    cif_file = pdbfile.replace(".pdb", ".cif")
    # assert not os.path.exists(cif_file)
    with open(cif_file, "w") as file:
        file.write(hdr)
    print(f"Wrote AF3 mmCIF file to {cif_file}")
    
    name_mapping_file = pdbfile.replace(".pdb", "_af3_name_mapping.json")
    with open(name_mapping_file, "w") as file:
        json.dump({NAME3: name_mapping}, file)
    print(f"Wrote mmCIF/PDB name mapping to {name_mapping_file}")
    print("Please edit the CIF file to make sure the `_chem_comp_atom.pdbx_leaving_atom_flag` values are correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True, help="PDB file of a ligand with correct atom names")
    parser.add_argument("--mol", type=str, required=False, help="Optional MOL2 file of a ligand with bonding section and correct charges.\n"
                                                                "Atoms need to be in the same order as in the PDB file.")
    args = parser.parse_args()
    main(args)
