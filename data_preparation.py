import pandas as pd
import requests
import zipfile
import io
import hashlib

# funkcja do ściągania podanego archiwum
def download_gios_archive(archive_url, filename, sha256=None):
    """
    Pobiera archiwum ZIP z danymi GIOŚ i wczytuje wskazany plik Excela.

    Parametry:
    - archive_url: pełny URL do archiwum ZIP (np. https://.../downloadFile/XXX)
    - sha256: opcjonalny oczekiwany hash SHA-256 (hex) archiwum ZIP; jeśli None,
      pomija weryfikację
    - filename: nazwa pliku w archiwum ZIP; typowa (zwykle oczekiwana) nazwa to
      <YEAR>_PM2.5_1g.xlsx
    """
    response = requests.get(archive_url)
    response.raise_for_status()  # jeśli błąd HTTP, zatrzymaj

    # Dla nieprawidłowych ścieżek serwer GIOS zwraca 200 OK jako text/html
    # z prośbą o kontakt a administratorem w treści.
    # Jako, że nie jest to 404 ten błąd zostaje propagowany za raise_for_status()
    # Możemy poprzestać na wyjątku z zipfile podnoszący nieprawidłowy zip,
    # ale możemy też rozpoznać ten przypadek jawnie
    content_type = getattr(response, "headers", {}).get("Content-Type")
    if content_type and "text/html" in content_type.lower():
        raise ValueError(f"Nieprawidłowy Content-Type: {content_type}\nOczekiwany: application/zip")

    if sha256: # skip check gdy nie podano hasha
        actual_hash = hashlib.sha256(response.content).hexdigest()
        if actual_hash.lower() != sha256.lower():
            raise ValueError(f"SHA256 nieprawidłowy! Plik: {actual_hash}")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        if filename not in z.namelist():
            raise FileNotFoundError(f"Nie znaleziono pliku {filename} w archiwum.")

        with z.open(filename) as f:
            try:
                df = pd.read_excel(f, header=None)
            except Exception as e:
                raise RuntimeError(
                    f"Błąd przy wczytywaniu pliku {filename} z archiwum."
                ) from e

    return df


def split_raw_df_to_metadata_and_measurements(raw_df, meta_keys=None):
    """
    Dzieli surowy DataFrame z danymi GIOŚ na:
    - metadane stacji (kolumny opisowe),
    - pomiary (wiersze czasowe, kolumny = kody stacji).

    Parametry:
    - raw_df: surowy DataFrame w formacie GIOŚ (metadane w pierwszej kolumnie)
    - meta_keys: lista etykiet wierszy uznawanych za metadane
    """
    labels = raw_df.iloc[:, 0]
    data   = raw_df.iloc[:, 1:]
    
    is_metadata = labels.isin(meta_keys)
    is_datapoint = ~is_metadata
    
    stations = data[is_metadata.values].T
    stations.columns = labels[is_metadata].values
    
    measurements = data[is_datapoint.values]
    measurements.index = pd.to_datetime(
        labels[is_datapoint].values,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    
    if "Kod stacji" in stations.columns:
        station_codes = stations["Kod stacji"].values
    else:
        station_codes = data.columns
    measurements = data[is_datapoint.values].copy()
    measurements.columns = station_codes
    measurements.index = pd.to_datetime(
        labels[is_datapoint].values,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    return stations, measurements


def download_updated_metadata(metadata_url, sha256=None):
    """
    Pobiera aktualne metadane stacji i zwraca je jako DataFrame.

    Parametry:
    - metadata_url: pełny URL do pliku z metadanymi (xlsx)
    - sha256: opcjonalny oczekiwany hash SHA-256 (hex); jeśli None, pomija weryfikację
    """
    response = requests.get(metadata_url)
    response.raise_for_status()

    content_type = getattr(response, "headers", {}).get("Content-Type")
    if content_type and "text/html" in content_type.lower():
        raise ValueError(f"Unexpected Content-Type: {content_type}")

    if sha256:
        actual_hash = hashlib.sha256(response.content).hexdigest()
        if actual_hash.lower() != sha256.lower():
            raise ValueError(f"SHA256 nieprawidłowy! Plik: {actual_hash}")

    try:
        updated_metadata_df = pd.read_excel(io.BytesIO(response.content))
    except Exception as e:
        raise RuntimeError("Błąd przy wczytywaniu metadanych!") from e

    updated_metadata_df = updated_metadata_df.rename(columns={"Stary Kod stacji \n(o ile inny od aktualnego)": "Stary Kod stacji"})
    updated_metadata_df["Kod stacji"] = updated_metadata_df["Kod stacji"].astype(str).str.strip() # jedna stacja (patrzę na ciebie LuLubsStrzelMOB) ma spację na końcu nazwy

    return updated_metadata_df


def build_station_code_mapping(updated_metadata, verbose=True):
    """
    Buduje mapowanie kodów stacji (stary -> nowy) na podstawie metadanych.

    Zwraca słownik, w którym:
    - aktualne kody mapują się do siebie (tożsamość),
    - stare kody mapują się do odpowiadających im nowych kodów.
    """

    # Mapowanie tożsamości: nowy kod -> nowy kod
    mapping_current = pd.Series(
        updated_metadata["Kod stacji"].astype(str).str.strip().values,
        index=updated_metadata["Kod stacji"].astype(str).str.strip().values,
    )

    # Rozwinięcie list starych kodów (comma-separated list), bez pętli po wierszach
    old_codes = updated_metadata[["Kod stacji", "Stary Kod stacji"]].copy()
    old_codes["Kod stacji"] = updated_metadata["Kod stacji"].astype(str).str.strip()
    old_codes = old_codes[old_codes["Stary Kod stacji"].notna()]
    old_codes["old_code"] = (
        old_codes["Stary Kod stacji"]
        .astype(str)
        .str.split(",")
    )

    # Rozbij listy na wiersze i usuń białe znaki (częste po split po przecinku)
    old_codes = old_codes.explode("old_code")
    old_codes["old_code"] = old_codes["old_code"].astype(str).str.strip()
    old_codes = old_codes[old_codes["old_code"] != ""]

    # Mapowanie stary -> nowy
    mapping_old = pd.Series(
        old_codes["Kod stacji"].values,
        index=old_codes["old_code"].values,
    )

    # Połączenie mapowań (stare mogą nadpisać tożsamość, jeśli wystąpią duplikaty)
    mapping = mapping_current.to_dict()
    mapping.update(mapping_old.to_dict())

    if verbose:
        n_current = updated_metadata["Kod stacji"].nunique()
        distinct_old = len(mapping) - n_current
        print(
            f"[mapowanie] {n_current} aktualnych kodów, "
            f"{distinct_old} unikalnych starych kodów"
        )

    return mapping


def update_station_names_metadata(metadata_df, updated_metadata, code_mapping, label="metadata"):
    """
    Aktualizuje kody stacji w metadanych na podstawie mapowania (stary -> nowy).

    - Wypisuje diagnostykę (ile kodów było aktualnych, zaktualizowanych i nieznanych).
    - W przypadku nieznanych kodów przerywa działanie błędem.
    """

    df = metadata_df.copy()

    # Normalizacja kodów wejściowych
    df["Kod stacji"] = df["Kod stacji"].astype(str).str.strip()
    original_codes = df["Kod stacji"].copy()

    # Mapowanie kodów; jeśli brak w mapie -> zostaje wartość oryginalna
    mapped_codes = original_codes.map(code_mapping).fillna(original_codes)

    in_mapping = original_codes.isin(code_mapping.keys())
    updated_mask = in_mapping & (mapped_codes != original_codes)
    current_mask = in_mapping & (mapped_codes == original_codes)
    unmatched_mask = ~in_mapping

    n_updated = int(updated_mask.sum())
    n_current = int(current_mask.sum())
    n_unmatched = int(unmatched_mask.sum())
    n_total = len(df)

    # Spójność zliczeń (powinny sumować się do liczby wierszy)
    if n_updated + n_current + n_unmatched != n_total:
        print(
            f"[{label}] UWAGA: zliczenia się nie sumują! "
            f"zaktualizowane={n_updated}, aktualne={n_current}, brak_mapy={n_unmatched}, "
            f"razem={n_total}"
        )

    print(f"[{label}] liczba stacji (wierszy): {n_total}")
    print(f"[{label}]   aktualne (już nowe):     {n_current}")
    print(f"[{label}]   zaktualizowane:           {n_updated}")
    print(f"[{label}]   brak mapowania:           {n_unmatched}")

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
        print(f"[{label}] zaktualizowane kody (stary -> nowy):")
        print(changes)

    # Brak mapowania = błąd (chroni dalsze operacje)
    if n_unmatched > 0:
        unmatched_codes = (
            original_codes[unmatched_mask]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"[{label}] KODY BEZ MAPOWANIA (brak w updated_metadata):")
        print(unmatched_codes)
        raise ValueError(
            f"[{label}] Found {n_unmatched} unmatched station codes: "
            f"{', '.join(unmatched_codes.astype(str))}"
        )

    # Zapisz zmapowane kody i dołącz informację o starych kodach
    df["Kod stacji"] = mapped_codes
    df = df.merge(
        updated_metadata[["Kod stacji", "Stary Kod stacji"]],
        on="Kod stacji",
        how="left",
        suffixes=("", "_from_updated"),
    )

    # Sanity check: wszystkie kody muszą istnieć w updated_metadata (po normalizacji)
    updated_codes = set(updated_metadata["Kod stacji"].astype(str).str.strip())
    missing = set(df["Kod stacji"]) - updated_codes
    assert not missing, (
        f"[{label}] Some station codes in metadata_df are not in updated_metadata: "
        f"{sorted(missing)}"
    )

    return df


def update_station_names_data(data_df, code_mapping, label="measurements"):
    """
    Aktualizuje nazwy kolumn (kody stacji) w danych pomiarowych.

    - Stare kody są mapowane na nowe zgodnie z code_mapping.
    - Brak mapowania pozostawia kod bez zmian (po normalizacji).
    - Wypisuje diagnostykę o zmianach i potencjalnych duplikatach.
    """

    original_cols = pd.Index(data_df.columns)
    normalized_cols = original_cols.astype(str).str.strip()

    # Mapowanie kodów; jeśli brak w mapie -> zostaje wartość znormalizowana
    mapped_cols = pd.Index(normalized_cols.map(code_mapping))
    new_cols = mapped_cols.where(mapped_cols.notna(), normalized_cols)

    # Diagnostyka zmian (uwzględnia również korektę białych znaków)
    changed_mask = original_cols.astype(str) != new_cols
    n_changed = int(changed_mask.sum())
    print(f"[{label}] pomiary: zmienionych kolumn stacji: {n_changed}")

    if n_changed > 0:
        changes = (
            pd.DataFrame({"old_code": original_cols.astype(str), "new_code": new_cols})
            .loc[changed_mask]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(f"[{label}] zmienione kolumny (stary -> nowy):")
        print(changes)

    # Ostrzeżenie o duplikatach po mapowaniu (wiele starych -> jeden nowy)
    vc = pd.Series(new_cols).value_counts()
    duplicates = vc[vc > 1]
    if not duplicates.empty:
        print(
            f"[{label}] UWAGA: {len(duplicates)} kodów stacji występuje wielokrotnie "
            f"po zmianie (prawdopodobnie kilka starych -> jeden nowy):"
        )
        print(duplicates)

    # Zastosuj nowe nazwy kolumn
    rename_map = dict(zip(original_cols, new_cols))
    data_df = data_df.rename(columns=rename_map)

    return data_df


def extend_metadata_with_station_info(metadata_df, updated_metadata, extra_cols, label="metadata"):
    """
    Rozszerza metadane stacji o dodatkowe kolumny z updated_metadata.

    Parametry:
    - metadata_df: metadane stacji dla danego roku (z kolumną "Kod stacji")
    - updated_metadata: pełne metadane referencyjne (z kolumną "Kod stacji")
    - extra_cols: lista dodatkowych kolumn do dołączenia
    """

    df = metadata_df.copy()

    # Normalizacja kodów po obu stronach
    df["Kod stacji"] = df["Kod stacji"].astype(str).str.strip()
    upd = updated_metadata.copy()
    upd["Kod stacji"] = upd["Kod stacji"].astype(str).str.strip()

    # Sprawdzenie spójności: wszystkie kody muszą istnieć w updated_metadata
    missing = set(df["Kod stacji"]) - set(upd["Kod stacji"])
    assert not missing, (
        f"[{label}] Some station codes in metadata_df are not in updated_metadata: "
        f"{sorted(missing)}"
    )

    # Wybrane kolumny do dołączenia
    cols_to_use = ["Kod stacji"] + list(extra_cols)
    extra = upd[cols_to_use]

    # Dołącz dodatkowe informacje po kodzie stacji
    df = df.merge(extra, on="Kod stacji", how="left")

    return df


def combine_metadata_frames(meta_dfs, code_col="Kod stacji"):
    """
    Łączy metadane z wielu lat, zachowując tylko wspólne stacje.

    Wykorzystuje złączenia typu inner na kolumnie code_col, aby znaleźć
    kody obecne we wszystkich ramkach danych.
    """

    # Normalizacja kodów i diagnostyka liczby unikalnych stacji (bez kopiowania całych ramek)
    code_dfs = []
    for i, df in enumerate(meta_dfs):
        codes = df[code_col].astype(str).str.strip()
        print(f"[meta {i}] unikalne stacje: {codes.nunique()}")
        code_dfs.append(codes.drop_duplicates().to_frame(code_col))

    # Inner join na samym kodzie stacji => wspólne kody we wszystkich ramkach
    common = code_dfs[0]
    for df_codes in code_dfs[1:]:
        common = common.merge(df_codes, on=code_col, how="inner")

    common_codes = set(common[code_col])
    print(f"[meta] stacje obecne we WSZYSTKICH latach: {len(common_codes)}")

    # Zachowaj strukturę pierwszej ramki, ale tylko wspólne kody
    base = meta_dfs[0].copy()
    base[code_col] = base[code_col].astype(str).str.strip()
    combined = base.merge(common, on=code_col, how="inner").reset_index(drop=True)

    return combined, common_codes


def combine_data_frames(data_dfs):
    """
    Łączy dane pomiarowe z wielu lat, pozostawiając tylko wspólne kolumny stacji.

    Wykorzystuje concat z join="inner", aby zachować wyłącznie kolumny
    obecne we wszystkich ramkach danych.
    """

    for i, df in enumerate(data_dfs):
        print(f"[data {i}] liczba kolumn stacji: {df.columns.astype(str).nunique()}")

    # Inner join na kolumnach (wspólne stacje we wszystkich latach)
    combined = pd.concat(data_dfs, axis=0, join="inner", sort=False).sort_index()
    common_cols = list(combined.columns)
    print(f"[data] stacje obecne we WSZYSTKICH latach: {len(common_cols)}")

    return combined, common_cols
