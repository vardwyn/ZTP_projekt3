import pandas as pd
import requests
import zipfile
import io, os
import hashlib

# id archiwum dla poszczególnych lat
gios_archive_url = "https://powietrze.gios.gov.pl/pjp/archives/downloadFile/"
gios_url_ids = {2014: '302', 2019: '322', 2024: '582'}
gios_pm25_file = {2014: '2014_PM2.5_1g.xlsx', 2019: '2019_PM25_1g.xlsx', 2024: '2024_PM25_1g.xlsx'}
gios_archive_sha256 = {
    2014: "8cabcc2118f019d8d1c0998561c01d57eda8c0a4c531cd2158b18522cd1aed27",
    2019: "777bc03c3c6d1ac77bd4353a80b6e064506368d42be19edece60f040c17dba1c",
    2024: "571dfa56866388c2904284ca6029bbf6016af3905b95bcacc5b3b6f6fa2d00e1"
}

# funkcja do ściągania podanego archiwum
def download_gios_archive(year):
    gios_id = gios_url_ids[year]
    gios_hash = gios_archive_sha256[year]
    filename = gios_pm25_file[year]
    
    # Pobranie archiwum ZIP do pamięci
    url = f"{gios_archive_url}{gios_id}"
    response = requests.get(url)
    response.raise_for_status()  # jeśli błąd HTTP, zatrzymaj
    
    actual_hash = hashlib.sha256(response.content).hexdigest()
    if actual_hash != gios_hash.lower():
        raise ValueError(f"SHA256 mismatch: {actual_hash}")

    # Otwórz zip w pamięci
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # znajdź właściwy plik z PM2.5
        if not filename:
            print(f"Błąd: nie znaleziono {filename}.")
        else:
            # wczytaj plik do pandas
            with z.open(filename) as f:
                try:
                    df = pd.read_excel(f, header=None)
                except Exception as e:
                    print(f"Błąd przy wczytywaniu {year}: {e}")
    return df


def reformat_raw_dfs(raw_df, meta_keys=None):
    labels = raw_df.iloc[:, 0]
    data   = raw_df.iloc[:, 1:]
    
    is_metadata = labels.isin(meta_keys)
    is_datapoint = ~is_metadata
    
    stations = data[is_metadata.values].T
    stations.columns = labels[is_metadata].values
    
    measurements = data[is_datapoint.values]
    measurements.index = pd.to_datetime(labels[is_datapoint].values)
    
    station_codes = stations["Kod stacji"].values
    measurements = data[is_datapoint.values].copy()
    measurements.columns = station_codes
    measurements.index = pd.to_datetime(labels[is_datapoint].values)

    return stations, measurements


def download_updated_metadata():
    url = f"https://powietrze.gios.gov.pl/pjp/archives/downloadFile/622"
    response = requests.get(url)
    response.raise_for_status()
        
    actual_hash = hashlib.sha256(response.content).hexdigest()
    if actual_hash != "174290b98ceb780c69769806f7f7a6054015cd78ead8ee65746f3fceba66b2ab":
        raise ValueError(f"SHA256 mismatch: {actual_hash}")

    try:
        updated_metadata_df = pd.read_excel(io.BytesIO(response.content))
    except Exception as e:
        print(f"Błąd przy wczytywaniu metadanych!")

    updated_metadata_df = updated_metadata_df.rename(columns={"Stary Kod stacji \n(o ile inny od aktualnego)": "Stary Kod stacji"})
    updated_metadata_df["Kod stacji"] = updated_metadata_df["Kod stacji"].astype(str).str.strip() # jedna stacja (patrzę na ciebie LuLubsStrzelMOB) ma spację na końcu nazwy

    return updated_metadata_df


def build_station_code_mapping(updated_metadata, verbose=True):

    mapping = {}

    for _, row in updated_metadata.iterrows():
        new_code = str(row["Kod stacji"]).strip()
        if new_code:
            mapping[new_code] = new_code

        old_codes = row.get("Stary Kod stacji")
        if pd.notna(old_codes):
            for old in str(old_codes).split(","):
                old = old.strip()
                if old:
                    mapping[old] = new_code

    if verbose:
        print(
            f"[mapping] {updated_metadata['Kod stacji'].nunique()} current codes, "
            f"{len(mapping) - updated_metadata['Kod stacji'].nunique()} distinct old codes"
        )

    return mapping


def update_station_names_metadata(metadata_df, updated_metadata, code_mapping, label="metadata"):

    df = metadata_df.copy()

    df["Kod stacji"] = df["Kod stacji"].astype(str).str.strip()
    original_codes = df["Kod stacji"].copy()

    mapped_codes = original_codes.map(lambda x: code_mapping.get(x, x))

    in_mapping = original_codes.isin(code_mapping.keys())
    updated_mask   = in_mapping & (mapped_codes != original_codes)
    current_mask   = in_mapping & (mapped_codes == original_codes)
    unmatched_mask = ~in_mapping

    n_updated   = int(updated_mask.sum())
    n_current   = int(current_mask.sum())
    n_unmatched = int(unmatched_mask.sum())
    n_total     = len(df)

    # sanity check
    if n_updated + n_current + n_unmatched != n_total:
        print(
            f"[{label}] WARNING: counts do not add up! "
            f"updated={n_updated}, current={n_current}, unmatched={n_unmatched}, "
            f"total={n_total}"
        )

    print(f"[{label}] total stations (rows): {n_total}")
    print(f"[{label}]   current names (already new): {n_current}")
    print(f"[{label}]   updated from old -> new:       {n_updated}")
    print(f"[{label}]   unmatched (not in mapping):   {n_unmatched}")

    if n_updated > 0:
        changes = (
            pd.DataFrame(
                {
                    "old_code": original_codes[updated_mask],
                    "new_code": mapped_codes[updated_mask],
                }
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"[{label}] updated codes (old -> new):")
        print(changes)

    if n_unmatched > 0:
        unmatched_codes = (
            original_codes[unmatched_mask]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"[{label}] UNMATCHED codes (neither current nor old in updated_metadata):")
        print(unmatched_codes)
        raise ValueError(
            f"[{label}] Found {n_unmatched} unmatched station codes: "
            f"{', '.join(unmatched_codes.astype(str))}"
        )
        
    df["Kod stacji"] = mapped_codes

    df = df.merge(
        updated_metadata[["Kod stacji", "Stary Kod stacji"]],
        on="Kod stacji",
        how="left",
        suffixes=("", "_from_updated"),
    )
    
    missing = set(df["Kod stacji"]) - set(updated_metadata["Kod stacji"])
    assert not missing, (
    f"[{label}] Some station codes in metadata_df are not in updated_metadata: "
    f"{sorted(missing)}"
    )

    return df


def update_station_names_data(data_df, code_mapping, label="measurements"):

    original_cols = pd.Index(data_df.columns)
    new_cols = original_cols.map(lambda x: code_mapping.get(x, x))

    # diagnostics: which columns will change
    changed_mask = original_cols != new_cols
    n_changed = int(changed_mask.sum())
    print(f"[{label}] measurements: {n_changed} station columns renamed")

    if n_changed > 0:
        changes = (
            pd.DataFrame({"old_code": original_cols, "new_code": new_cols})
            .loc[changed_mask]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"[{label}] renamed columns (old -> new):")
        print(changes)

    vc = pd.Series(new_cols).value_counts()
    duplicates = vc[vc > 1]
    if not duplicates.empty:
        print(f"[{label}] WARNING: {len(duplicates)} station codes appear multiple "
              f"times after renaming (likely multiple old codes -> one new code):")
        print(duplicates)

    # apply rename
    data_df = data_df.rename(columns=code_mapping)

    return data_df


EXTRA_COLS = [
    "Typ stacji",
    "Typ obszaru",
    "Rodzaj stacji",
    "Województwo",
    "Miejscowość",
    "WGS84 φ N",
    "WGS84 λ E",
]


def add_station_info(metadata_df, updated_metadata, label="metadata"):

    df = metadata_df.copy()

    df["Kod stacji"] = df["Kod stacji"].astype(str).str.strip()
    upd = updated_metadata.copy()
    upd["Kod stacji"] = upd["Kod stacji"].astype(str).str.strip()

    missing = set(df["Kod stacji"]) - set(upd["Kod stacji"])
    assert not missing, (
        f"[{label}] Some station codes in metadata_df are not in updated_metadata: "
        f"{sorted(missing)}"
    )

    cols_to_use = ["Kod stacji"] + EXTRA_COLS
    extra = upd[cols_to_use]

    df = df.merge(extra, on="Kod stacji", how="left")

    df = df.rename(columns={
        "WGS84 φ N": "Szerokość geograficzna",
        "WGS84 λ E": "Długość geograficzna",
    })

    return df


def combine_metadata_frames(meta_dfs, code_col="Kod stacji"):

    code_sets = []
    for i, df in enumerate(meta_dfs):
        codes = df[code_col].astype(str).str.strip()
        s = set(codes)
        print(f"[meta {i}] unique stations: {len(s)}")
        code_sets.append(s)

    common_codes = set.intersection(*code_sets)
    print(f"[meta] stations present in ALL years: {len(common_codes)}")

    base = meta_dfs[0].copy()
    base[code_col] = base[code_col].astype(str).str.strip()
    combined = base[base[code_col].isin(common_codes)].reset_index(drop=True)

    return combined, common_codes


def combine_data_frames(data_dfs):

    col_sets = []
    for i, df in enumerate(data_dfs):
        cols = set(df.columns.astype(str))
        print(f"[data {i}] station columns: {len(cols)}")
        col_sets.append(cols)

    # intersection
    common_cols = sorted(set.intersection(*col_sets))
    print(f"[data] stations present in ALL years: {len(common_cols)}")

    filtered = [df.loc[:, common_cols].copy() for df in data_dfs]
    combined = pd.concat(filtered).sort_index()

    return combined, common_cols
