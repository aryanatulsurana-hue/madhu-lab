import os
import pandas as pd
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import ConvexHull

def get_atom_dict(structure):
    atoms = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()
                resnum = res_id[1]
                chain_id = chain.id
                for atom in residue:
                    key = (chain_id, resnum, atom.get_name())
                    atoms[key] = atom
    return atoms

def extract_coords(atom_dict, keys):
    return np.array([atom_dict[k].coord for k in keys if k in atom_dict])

def compute_volume(coords):
    if len(coords) >= 4:
        return ConvexHull(coords).volume
    return 0.0

def compute_rg(coords):
    if len(coords) == 0:
        return 0.0
    com = coords.mean(axis=0)
    return np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1)))

def compare_structures(apo_full_path, apo_transformed_path):
    parser = PDBParser(QUIET=True)
    try:
        apo_full = parser.get_structure("full", apo_full_path)
        apo_site = parser.get_structure("site", apo_transformed_path)
    except FileNotFoundError:
        print(f"[MISSING] One of the files not found:\n  {apo_full_path}\n  {apo_transformed_path}")
        return None

    full_dict = get_atom_dict(apo_full)
    site_dict = get_atom_dict(apo_site)
    common_keys = list(set(site_dict.keys()) & set(full_dict.keys()))

    coords_full = extract_coords(full_dict, common_keys)
    coords_trans = extract_coords(site_dict, common_keys)

    if len(coords_full) < 4 or len(coords_trans) < 4:
        print(f"[SKIP] Not enough atoms for ConvexHull in: {apo_full_path}")
        return None

    vol_full = compute_volume(coords_full)
    vol_trans = compute_volume(coords_trans)
    rg_full = compute_rg(coords_full)
    rg_trans = compute_rg(coords_trans)

    return {
        "volume_full": vol_full,
        "volume_transformed": vol_trans,
        "delta_volume": vol_trans - vol_full,
        "rg_full": rg_full,
        "rg_transformed": rg_trans,
        "delta_rg": rg_trans - rg_full,
        "site_status": "opening" if vol_trans > vol_full else "closing"
    }

# === Main Loop ===
def process_csv(input_csv, apo_folder, transformed_folder, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for i, row in df.iterrows():
        uniprot = row['uniprot_id']
        apo_id = str(row['apo_pdb']).replace('.pdb', '')
        lig = row['lig_name']
        chain = row['lig_chain']
        num = str(row['lig_num'])

        apo_full_path = os.path.join(apo_folder, f"{uniprot}_Apo_{apo_id}.pdb")
        apo_trans_path = os.path.join(transformed_folder, f"{uniprot}_Apo_transformed_{lig}_{chain}{num}.pdb")

        res = compare_structures(apo_full_path, apo_trans_path)
        if res:
            result_row = {
                "uniprot_id": uniprot,
                "apo_pdb_id": apo_id,
                "lig_name": lig,
                "lig_chain": chain,
                "lig_num": num,
                **res
            }
            results.append(result_row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")

# === Example Usage ===
process_csv("merged_output_final_local_1_global_3.csv", "relevant_pdbs/", "transformed_apo_coords/", "output_opening_closing.csv")
