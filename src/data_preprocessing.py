import os
import sys
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter, Retry
from rdkit.Chem import Crippen, Descriptors, MolFromSmiles, QED, RDConfig
from selfies import encoder as sf_encoder
import selfies as sf

# Custom functions
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

# Base URL for the ChEMBL API
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Setup a session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


def fetch_general_dataset():
    """
    Fetch all molecular entries and their SMILES notations from the ChEMBL API.

    The function fetches molecular data from the ChEMBL API in a paginated manner,
    processes the response to extract relevant information, and constructs a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing ChEMBL IDs and SMILES notations of the molecules.
    """
    print("Fetching general dataset from ChEMBL API...")
    url = f"{BASE_URL}/molecule"
    params = {"format": "json", "limit": 1000, "offset": 0}
    molecules = []

    while True:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        molecules.extend(data["molecules"])
        print(f"Fetched {len(data['molecules'])} molecules, offset: {params['offset']}")

        if not data["page_meta"].get("next"):
            break
        params["offset"] += params["limit"]

    smiles_data = [
        (mol["molecule_chembl_id"], mol["molecule_structures"]["canonical_smiles"])
        for mol in molecules
        if mol.get("molecule_structures")
    ]

    print(f"Total molecules fetched: {len(smiles_data)}")
    return pd.DataFrame(smiles_data, columns=["ChEMBL_ID", "SMILES"])


def fetch_targeted_dataset():
    """
    Fetch all reported eIF4E inhibitors, their SMILES notations, and IC50 values from the ChEMBL API.

    The function fetches activity data targeting eIF4E from the ChEMBL API in a paginated manner,
    processes the response to extract relevant information, and constructs a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing ChEMBL IDs, SMILES notations, and IC50 values of the inhibitors.
    """
    print("Fetching targeted dataset (eIF4E inhibitors) from ChEMBL API...")
    url = f"{BASE_URL}/activity"
    params = {
        "target_chembl_id": "CHEMBL4848",
        "assay_type": "B",
        "standard_type": "IC50",
        "format": "json",
        "limit": 1000,
        "offset": 0,
    }
    activities = []

    while True:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        activities.extend(data["activities"])
        print(
            f"Fetched {len(data['activities'])} activities, offset: {params['offset']}"
        )

        if not data["page_meta"].get("next"):
            break
        params["offset"] += params["limit"]

    targeted_data = [
        (act["molecule_chembl_id"], act["canonical_smiles"], act["value"])
        for act in activities
        if "canonical_smiles" in act and "value" in act
    ]

    print(f"Total activities fetched: {len(targeted_data)}")
    return pd.DataFrame(targeted_data, columns=["ChEMBL_ID", "SMILES", "IC50"])


def preprocess_datasets(general_df, targeted_df):
    """
    Preprocess the datasets by cleaning and calculating molecular properties.

    This function removes unwanted molecules, calculates molecular properties,
    converts SMILES to SELFIES, and filters out long SELFIES strings.

    Args:
        general_df (pd.DataFrame): DataFrame containing the general dataset with SMILES notations.
        targeted_df (pd.DataFrame): DataFrame containing the targeted dataset with SMILES notations and IC50 values.

    Returns:
        tuple: Two DataFrames, one for the general dataset and one for the targeted dataset, both preprocessed.
    """
    print("Preprocessing datasets...")

    # Remove molecules containing a period (".") in their SMILES notations
    general_df = general_df[~general_df["SMILES"].str.contains(r"\.")]
    targeted_df = targeted_df[~targeted_df["SMILES"].str.contains(r"\.")]
    print(
        f"Filtered out molecules with periods in SMILES: {len(general_df)} general, {len(targeted_df)} targeted"
    )

    # Remove duplicate entries
    general_df = general_df.drop_duplicates(subset=["SMILES"])
    targeted_df["IC50"] = pd.to_numeric(targeted_df["IC50"], errors="coerce")
    targeted_df = (
        targeted_df.groupby("SMILES")
        .agg({"ChEMBL_ID": "first", "IC50": "mean"})
        .reset_index()
    )
    print("Removed duplicates and grouped targeted dataset by SMILES")

    # Convert IC50 to pIC50 for the targeted dataset
    targeted_df["pIC50"] = -np.log10(targeted_df["IC50"].astype(float))
    print("Converted IC50 to pIC50 for targeted dataset")

    def calc_properties(smiles):
        """
        Calculate molecular properties: molecular weight (MW), LogP, QED, and synthetic accessibility score (SAS).

        Args:
            smiles (str): The SMILES notation of the molecule.

        Returns:
            tuple: A tuple containing MW, LogP, QED, and SAS.
        """
        mol = MolFromSmiles(smiles)
        if mol:
            return (
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                QED.qed(mol),
                sascorer.calculateScore(mol),
            )
        return np.nan, np.nan, np.nan, np.nan

    # Calculate molecular properties: MW, LogP, QED, SAS
    general_df[["MW", "LogP", "QED", "SAS"]] = general_df["SMILES"].apply(
        lambda x: pd.Series(calc_properties(x))
    )
    targeted_df[["MW", "LogP", "QED", "SAS"]] = targeted_df["SMILES"].apply(
        lambda x: pd.Series(calc_properties(x))
    )
    print("Calculated molecular properties for both datasets")

    # Convert SMILES to SELFIES
    general_df["SELFIES"] = general_df["SMILES"].apply(sf_encoder)
    targeted_df["SELFIES"] = targeted_df["SMILES"].apply(sf_encoder)
    print("Converted SMILES to SELFIES")

    # Remove molecules with SELFIES strings longer than 100 characters
    general_df = general_df[general_df["SELFIES"].apply(sf.len_selfies) <= 100]
    targeted_df = targeted_df[targeted_df["SELFIES"].apply(sf.len_selfies) <= 100]
    print(
        f"Filtered out molecules with SELFIES longer than 100 characters: {len(general_df)} general, {len(targeted_df)} targeted"
    )

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
