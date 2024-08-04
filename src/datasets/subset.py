import pandas as pd

def get_random_subsets(df):
    # Get a random subset of 100,000 molecules
    subset_100k = df.sample(n=100000, random_state=42)

    # Get a random subset of 500,000 molecules
    subset_500k = df.sample(n=500000, random_state=42)

    return subset_100k, subset_500k

general_dataset = pd.read_csv(r'C:\Users\Audrey\eif4e-inhibitor-discovery\src\datasets\general_dataset.csv')
subset_100k, subset_500k = get_random_subsets(general_dataset)

# Save the subsets to new CSV files
subset_100k.to_csv("subset_100k.csv", index=False)
subset_500k.to_csv("subset_500k.csv", index=False)

