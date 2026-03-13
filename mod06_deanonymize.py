import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    shared_cols = set(anon_df.columns).intersection(set(aux_df.columns))
    quasi_cols = [col for col in shared_cols if col not in ["anon_id", "name", "matched_name"]]

    merged = pd.merge(anon_df, aux_df, on=quasi_cols, how="inner")

    counts = merged.groupby("anon_id").size()
    unique_ids = counts[counts == 1].index

    result = merged[merged["anon_id"].isin(unique_ids)][["anon_id", "name"]].copy()
    result = result.rename(columns={"name": "matched_name"})

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)
