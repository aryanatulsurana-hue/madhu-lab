import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

'''def calculate_rmsd(coords1, coords2):
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

amino_acid_masses = {
    'ALA': 89.09, 'ARG': 174.20, 'ASN': 132.12, 'ASP': 133.10, 'CYS': 121.16,
    'GLN': 146.15, 'GLU': 147.13, 'GLY': 75.07, 'HIS': 155.16, 'ILE': 131.17,
    'LEU': 131.17, 'LYS': 146.19, 'MET': 149.21, 'PHE': 165.19, 'PRO': 115.13,
    'SER': 105.09, 'THR': 119.12, 'TRP': 204.23, 'TYR': 181.19, 'VAL': 117.15
}

sorted_amino_acids = sorted(amino_acid_masses, key=lambda aa: amino_acid_masses[aa])

colors = plt.cm.get_cmap('tab20', len(sorted_amino_acids))

df = pd.read_csv("merged_output_final_local_1_global_3.csv")'''
apo_dir = "relevant_pdbs"
transformed_dir = "transformed_apo_coords_with_ligand_CA"

'''rmsd_dict = {aa: [] for aa in sorted_amino_acids}
imp_rmsd_data = []

for index, row in df.iterrows():
    uniprot_id = row['uniprot_id']
    lig_name = str(row['lig_name'])
    apo_chain = str(row['Apo_Chain'])
    holo_chain = str(row['Holo_Chain'])
    lig_chain = str(row['lig_chain'])
    lig_num = row['lig_num']

    apo_file_candidates = [file for file in os.listdir(apo_dir) if file.startswith(f"{uniprot_id}_Apo_")]
    if not apo_file_candidates:
        print(f"No apo file found for {uniprot_id}")
        continue
    
    apo_file_name = os.path.join(apo_dir, apo_file_candidates[0])
    transformed_file_name = os.path.join(transformed_dir, f"{uniprot_id}_holo_aligned_on_apo_{lig_name}_{lig_chain}{lig_num}.pdb")

    if not os.path.exists(transformed_file_name):
        print(f"Missing transformed file for {uniprot_id}: {transformed_file_name}")
        continue

    with open(apo_file_name, 'r') as apo_file:
        apo_residues = {}
        for line in apo_file:
            if line.startswith("ATOM") and line[12:16].strip() == 'CA':
                residue_name = line[17:20].strip()
                residue_number = line[22:26].strip()
                residue_chain_id = line[21]
                if residue_chain_id == apo_chain:
                    coordinates = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
                    apo_residues[(residue_name, residue_number)] = coordinates

    with open(transformed_file_name, 'r') as transformed_file:
        transformed_residues = {}
        for line in transformed_file:
            if line.startswith("ATOM") and line[12:16].strip() == 'CA':
                residue_name = line[17:20].strip()
                residue_number = line[22:26].strip()
                residue_chain_id = line[21]
                if residue_chain_id == holo_chain:
                    coordinates = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
                    transformed_residues[(residue_name, residue_number)] = coordinates

    for (residue_name, residue_number), apo_coords in apo_residues.items():
        if (residue_name, residue_number) in transformed_residues:
            transformed_coords = transformed_residues[(residue_name, residue_number)]
            rmsd = calculate_rmsd([apo_coords], [transformed_coords])
            
            if rmsd < 6.0 :
                imp_rmsd_data.append({
                    'uniprot_id': uniprot_id,
                    'residue_name': residue_name,
                    'residue_number': residue_number,
                    'lig_name': lig_name,
                    'lig_num': lig_num,
                    'apo_chain': apo_chain,
                    'holo_chain': holo_chain,
                    'rmsd': rmsd,
                    'coords_apo': apo_coords,
                    'coords_transformed': transformed_coords
                })
        

imp_rmsd_df = pd.DataFrame(imp_rmsd_data)
imp_rmsd_df.to_csv('CA_rmsd_less_than_6_cases.csv', index=False)'''

def get_secondary_structure(pdb_file, chain, residue_number):
    secondary_structures = {'HELIX': [], 'SHEET': []}

    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("HELIX"):
                start_res = int(line[21:25].strip())
                end_res = int(line[33:37].strip())
                chain_id = line[19]
                if chain_id == chain:
                    secondary_structures['HELIX'].append((start_res, end_res))

            elif line.startswith("SHEET"):
                start_res = int(line[22:26].strip())
                end_res = int(line[33:37].strip())
                chain_id = line[21]
                if chain_id == chain:
                    secondary_structures['SHEET'].append((start_res, end_res))

    # Determine if the residue belongs to a secondary structure
    for start, end in secondary_structures['HELIX']:
        #print(type(start))
        #print(type(end))
        if start <= residue_number <= end:
            return "HELIX"

    for start, end in secondary_structures['SHEET']:
        if start <= residue_number <= end:
            return "SHEET"

    return "LOOP"  # Default to LOOP if not in HELIX or SHEET
imp_rmsd_data =[]
df = pd.read_csv("CA_rmsd_less_than_6_cases.csv")
for index, row in df.iterrows():
    uniprot_id = row['uniprot_id']
    holo_chain = row['holo_chain']
    residue_number = int(row['residue_number'])
    #print(type(residue_number))
    pdb_file = glob.glob(os.path.join(apo_dir, f"{uniprot_id}_Holo_*.pdb"))
    pdb_file_name = pdb_file[0]
    if os.path.exists(pdb_file_name):
        sec_struct = get_secondary_structure(pdb_file_name, holo_chain, residue_number)
        #print(sec_struct)
    else:
        sec_struct = "Unknown"

    imp_rmsd_data.append({
            'uniprot_id': row['uniprot_id'],
            'residue_name': row['residue_name'],
            'residue_number': row['residue_number'],
            'rmsd': row['rmsd'],
            'secondary_structure': sec_struct,
            "ligand_num":row['lig_num'],
            "ligand_name":row['lig_name']
        })

imp_rmsd_df = pd.DataFrame(imp_rmsd_data)
imp_rmsd_df.to_csv('CA_rmsd_less_than_6_cases_sec_struc.csv', index=False)
'''plt.figure(figsize=(10, 6))
bins = np.arange(0, 7, 1)

bottom = np.zeros(len(bins) - 1)
total_counts = np.zeros(len(bins) - 1)

for aa in sorted_amino_acids:
    if rmsd_dict[aa]:
        counts, _ = np.histogram(rmsd_dict[aa], bins=bins)
        total_counts += counts

for i, aa in enumerate(sorted_amino_acids):
    if rmsd_dict[aa]:
        counts, _ = np.histogram(rmsd_dict[aa], bins=bins)
        norm_counts = counts / total_counts if total_counts.sum() > 0 else counts
        plt.bar(bins[:-1], norm_counts, width=1, edgecolor='black', alpha=0.7, label=aa, bottom=bottom, color=colors(i))
        for j in range(len(counts)):
            if counts[j] > 0:
                plt.text(bins[j], bottom[j] + norm_counts[j] / 2, f'{counts[j]}', ha='center', va='center', fontsize=8)
        bottom += norm_counts

plt.xlabel('RMSD (Å)')
plt.ylabel('Normalized Frequency')
plt.title('Normalized Stacked RMSD Distribution (0-6 Å) by Amino Acid')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 7, 1))
plt.legend(loc='upper left', fontsize='small', ncol=2, bbox_to_anchor=(1, 1))
plt.savefig('filtered_rmsd_stacked_histogram_local_1_global_3.png', bbox_inches='tight')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('imp_CA_rmsd_cases_sec_struc.csv')

# Replace single-letter secondary structure labels with full names
df['secondary_structure'] = df['secondary_structure'].replace({'H': 'HELIX', 'E': 'SHEET', 'C': 'LOOP'})

# Define RMSD bins (starting from 1Å, skipping 0-1Å)
bin_edges = np.arange(1, df['rmsd'].max() + 1, 1)

# Create binned RMSD column
df['rmsd_bin'] = pd.cut(df['rmsd'], bins=bin_edges, right=False)

# Count occurrences per bin per secondary structure
counts = df.groupby(['rmsd_bin', 'secondary_structure']).size().unstack(fill_value=0)

# Print number of cases per bin
print("Number of cases per RMSD bin:")
print(counts.sum(axis=1))

# Normalize each bin (so sum of each bin is 1)
counts_norm = counts.div(counts.sum(axis=1), axis=0)

# Colors for better visibility
colors = {'HELIX': '#ff7f0e', 'SHEET': '#1f77b4', 'LOOP': '#2ca02c'}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(len(counts_norm))

for structure in ['HELIX', 'SHEET', 'LOOP']:
    bar_container = ax.bar(
        counts_norm.index.astype(str), counts_norm[structure],
        label=structure, color=colors[structure], bottom=bottoms, width=0.9
    )
    # Add number of cases on each bar
    for i, rect in enumerate(bar_container):
        total_cases = counts.sum(axis=1).iloc[i]
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_y() + rect.get_height()/2,
                f"{counts.iloc[i][structure]}", ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    bottoms += counts_norm[structure]

# Formatting
ax.set_xlabel("RMSD (Å) Bins")
ax.set_ylabel("Fraction of Residues")
ax.set_title("Normalized Secondary Structure Distribution per RMSD Bin (≥1Å)")
ax.legend(title="Secondary Structure")
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

plt.savefig('secondary structure histogram')'''

