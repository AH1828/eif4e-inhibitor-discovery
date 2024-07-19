import os
import sys
import requests
import pandas as pd
import numpy as np
from rdkit.Chem import Crippen, Descriptors, MolFromSmiles, QED, RDConfig
from selfies import encoder as sf_encoder

# Custom functions
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

# Base URL for the ChEMBL API
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_general_dataset():
    """Fetch all molecular entries and their SMILES notations from the ChEMBL API."""
    url = f"{BASE_URL}/molecule"
    params = {"format": "json", "limit": 1000, "offset": 0}
    molecules = []

    while True:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        molecules.extend(data["molecules"])
        if "next" in data["page_meta"]:
            params["offset"] += params["limit"]
        else:
            break

    smiles_data = [
        (mol["molecule_chembl_id"], mol["molecule_structures"]["canonical_smiles"])
        for mol in molecules
        if "molecule_structures" in mol and mol["molecule_structures"]
    ]

    general_df = pd.DataFrame(smiles_data, columns=["ChEMBL_ID", "SMILES"])
    return general_df


def fetch_targeted_dataset():
    """Fetch all reported eIF4E inhibitors and their SMILES notations and IC50 values from the ChEMBL API."""
    url = f"{BASE_URL}/activity"
    params = {
        "target_chembl_id": "CHEMBL4848",
        "assay_type": "B",
        "endpoint": "IC50",
        "format": "json",
        "limit": 1000,
        "offset": 0,
    }
    activities = []

    while True:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        activities.extend(data["activities"])
        if "next" in data["page_meta"]:
            params["offset"] += params["limit"]
        else:
            break

    targeted_data = [
        (act["molecule_chembl_id"], act["canonical_smiles"], act["value"])
        for act in activities
        if "canonical_smiles" in act and "value" in act
    ]

    targeted_df = pd.DataFrame(targeted_data, columns=["ChEMBL_ID", "SMILES", "IC50"])
    return targeted_df


def preprocess_datasets(general_df, targeted_df):
    """Preprocess the datasets by cleaning and calculating molecular properties."""
    # Remove molecules containing a period (".") in their SMILES notations
    general_df = general_df[~general_df["SMILES"].str.contains(r"\.")]
    targeted_df = targeted_df[~targeted_df["SMILES"].str.contains(r"\.")]

    # Remove duplicate entries
    general_df = general_df.drop_duplicates(subset=["SMILES"])
    targeted_df["IC50"] = pd.to_numeric(targeted_df["IC50"], errors="coerce")
    targeted_df = (
        targeted_df.groupby("SMILES")
        .agg({"ChEMBL_ID": "first", "IC50": "mean"})
        .reset_index()
    )

    # Convert IC50 to pIC50 for the targeted dataset
    targeted_df["pIC50"] = -np.log10(targeted_df["IC50"].astype(float))

    def calc_properties(smiles):
        mol = MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            qed = QED.qed(mol)
            sas = sascorer.calculateScore(mol)
            return mw, logp, qed, sas
        else:
            return np.nan, np.nan, np.nan, np.nan

    # Calculate molecular properties: MW, LogP, QED, SAS
    general_df[["MW", "LogP", "QED", "SAS"]] = general_df["SMILES"].apply(
        lambda x: pd.Series(calc_properties(x))
    )
    targeted_df[["MW", "LogP", "QED", "SAS"]] = targeted_df["SMILES"].apply(
        lambda x: pd.Series(calc_properties(x))
    )

    # Convert SMILES to SELFIES
    general_df["SELFIES"] = general_df["SMILES"].apply(sf_encoder)
    targeted_df["SELFIES"] = targeted_df["SMILES"].apply(sf_encoder)

    # Remove molecules with SELFIES strings longer than 100 characters
    general_df = general_df[general_df["SELFIES"].apply(len) <= 100]
    targeted_df = targeted_df[targeted_df["SELFIES"].apply(len) <= 100]

    return general_df, targeted_df


# Fetch the datasets
general_dataset = fetch_general_dataset()
targeted_dataset = fetch_targeted_dataset()

# Preprocess the datasets
general_dataset, targeted_dataset = preprocess_datasets(
    general_dataset, targeted_dataset
)

# Save datasets to CSV files
general_dataset.to_csv("general_dataset.csv", index=False)
targeted_dataset.to_csv("targeted_dataset.csv", index=False)

print("Datasets have been fetched, processed, and saved successfully.")
