import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import random
import selfies as sf

def generate_smiles(molecule, num_randomizations=20):
    """
    Generates restricted randomized SMILES for a given RDKit molecule.
    
    Parameters:
    - molecule: RDKit molecule object
    - num_randomizations: Number of randomized SMILES to generate per molecule
    
    Returns:
    - List of restricted randomized SMILES strings
    """
    smiles_set = set()
    for _ in range(num_randomizations):
        atom_indices = list(range(molecule.GetNumAtoms()))
        random.shuffle(atom_indices)
        
        # Apply restricted randomization by prioritizing sidechains
        rdmolops.AssignAtomChiralTagsFromStructure(molecule)
        rdmolops.AssignStereochemistry(molecule)
        randomized_molecule = Chem.RenumberAtoms(molecule, atom_indices)
        
        # Generate SMILES with RDKit restrictions
        smiles = Chem.MolToSmiles(randomized_molecule, canonical=False, isomericSmiles=True)
        smiles_set.add(smiles)
    return list(smiles_set)

def augment_smiles(df, smiles_column='smiles', num_randomizations=20):
    """
    Augments a dataset of SMILES strings by generating restricted randomized SMILES and their corresponding SELFIES.
    
    Parameters:
    - df: pandas DataFrame containing SMILES strings
    - smiles_column: Name of the column containing SMILES strings
    - num_randomizations: Number of randomized SMILES to generate per molecule
    
    Returns:
    - pandas DataFrame
    """
    augmented_data = []

    # Keep the original rows
    for idx, row in df.iterrows():
        augmented_data.append(row)

    # Add randomized SMILES and their corresponding SELFIES rows
    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            randomized_smiles = generate_smiles(molecule, num_randomizations)
            for r_smiles in randomized_smiles:
                try:
                    r_selfies = sf.encoder(r_smiles)
                    augmented_row = row.copy()
                    augmented_row[smiles_column] = r_smiles
                    augmented_row["SELFIES"] = r_selfies
                    augmented_data.append(augmented_row)
                except Exception as e:
                    print(f"Error processing SMILES {r_smiles}: {e}")
                    continue
    
    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df

def preprocess(augmented_df):
    # Remove molecules containing a period (".") in their SMILES notations
    augmented_df = augmented_df[~augmented_df["SMILES"].str.contains(r"\.")]
    print(
        f"Filtered out molecules with periods in SMILES: {len(augmented_df)} augmented"
    )

    # Remove duplicate entries
    augmented_df = augmented_df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    print("Removed duplicates and grouped targeted dataset by SMILES")

    # Remove molecules with SELFIES strings longer than 100 characters
    augmented_df = augmented_df[augmented_df["SELFIES"].apply(sf.len_selfies) <= 100]
    print(
        f"Filtered out molecules with SELFIES longer than 100 characters: {len(augmented_df)} augmented"
    )
    return augmented_df

targeted_dataset = pd.read_csv(r'C:\Users\Audrey\eif4e-inhibitor-discovery\src\datasets\targeted_dataset.csv')
augmented_dataset = augment_smiles(targeted_dataset, smiles_column='SMILES', num_randomizations=100)
augmented_dataset = preprocess(augmented_dataset)
augmented_dataset.to_csv("augmented_dataset.csv", index=False)