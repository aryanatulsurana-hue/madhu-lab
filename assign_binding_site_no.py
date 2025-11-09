import pandas as pd

# Load the CSV file
df = pd.read_csv("CA_rmsd_less_than_6_cases_sec_struc.csv")

# Create a unique triplet column
df["triplet"] = df.apply(lambda row: (row["ligand_name"], row["uniprot_id"], row["ligand_num"]), axis=1)

# Create a mapping from triplet to binding site number
triplet_to_number = {}
current_number = 1
binding_site_numbers = []

for triplet in df["triplet"]:
    if triplet not in triplet_to_number:
        triplet_to_number[triplet] = current_number
        current_number += 1
    binding_site_numbers.append(triplet_to_number[triplet])

# Assign the binding site number column
df["binding_site_number"] = binding_site_numbers

# Optionally drop the triplet column
df.drop(columns=["triplet"], inplace=True)

# Save to a new CSV file
df.to_csv("with_binding_site_numbers.csv", index=False)

print("Binding site numbers assigned and saved to 'with_binding_site_numbers.csv'")
