import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df, stratify_col, test_size=0.2, val_size=0.1):
    # Split into train and temp (test + val) sets
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), stratify=df[stratify_col])

    # Adjust val_size for second split
    adjusted_val_size = val_size / (test_size + val_size)

    # Split temp into test and val sets
    test_df, val_df = train_test_split(temp_df, test_size=adjusted_val_size, stratify=temp_df[stratify_col])

    return train_df, test_df, val_df

# Example usage
# df is your DataFrame
df = pd.read_csv('data.csv')
train_df, test_df, val_df = stratified_split(df, 'category_id', test_size=0.2, val_size=0.1)
