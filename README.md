# Projekt 3 – analiza jakości powietrza

[![pytest](https://github.com/vardwyn/ZTP_projekt3/actions/workflows/pytest.yml/badge.svg)](https://github.com/vardwyn/ZTP_projekt3/actions/workflows/pytest.yml)

Refaktoryzacja notebooka do modułów Pythona i notebooka orkiestrującego.

## Uruchamianie

``` sh
git clone https://github.com/vardwyn/ZTP_projekt3
cd ZTP_projekt3
uv venv
uv sync
uv run jupyter lab analiza.ipynb
```

## Konfiguracja PubMed (Entrez)

Konfiguracja pipeline PubMed znajduje się w `config/task4.yaml`:
- `pubmed.email` – wymagany przez NCBI (identyfikacja użytkownika).
- `pubmed.api_key` – opcjonalny (zwiększa limity).
- `pubmed.tool` – identyfikator narzędzia.
- `pubmed.retmax` – limit rekordów na zapytanie.
- `pubmed.queries_file` – ścieżka do pliku z zapytaniami.

Same zapytania definiujemy w `config/pubmed_queries.yaml`:
``` yaml
queries:
  - id: pm25_poland
    term: '(PM2.5 OR "fine particulate matter")'
```
Możesz dodać kolejne zapytania, np.:
``` yaml
  - id: pm25_warsaw
    term: '(PM2.5 OR "fine particulate matter") AND Warsaw'
```

## Pipeline (Snakemake)

Pipeline uruchamiany jest przez `Snakefile_task4` i korzysta z listy lat w `config/task4.yaml`.

Przykład: uruchomienie dla lat `[2021, 2024]` (domyślnie w repo):
``` sh
snakemake -s Snakefile_task4
```

Zmiana lat na `[2019, 2024]`:
1) Edytuj `config/task4.yaml`:
``` yaml
years: [2019, 2024]
```
2) Uruchom:
``` sh
snakemake -s Snakefile_task4
```

### Dlaczego 2024 nie jest przeliczany ponownie?

Snakemake nie przelicza roku, jeśli **wszystkie pliki wynikowe dla tego roku już istnieją**
i **nie zmieniły się wejścia reguły**. Zmiana listy `years` tylko dodaje/usuwa targety.
W praktyce po zmianie z `[2021, 2024]` na `[2019, 2024]`:
- **2019** zostanie policzony (brak wyników),
- **2024** nie zostanie przeliczony (wyniki już istnieją).

### Jak to sprawdzić?

1) Dry‑run (bez wykonywania):
``` sh
snakemake -s Snakefile_task4 -n
```
Wypisze tylko zadania, które faktycznie będą uruchomione.

2) Logi wykonania:
Jeśli nie ma nic do zrobienia, Snakemake wypisze `Nothing to be done.`.
Jeśli dodano nowy rok (np. 2019), Snakemake wypisze tylko joby dla tego roku.

## Przed commitowaniem

1) Wyczyść notebook z outputów i liczników wykonania:

``` sh
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace analiza.ipynb
```

2) (Opcjonalnie) dodaj pre-commit hook uruchamiający testy:

``` sh
cat > .git/hooks/pre-commit <<EOF
#!/usr/bin/env bash
set -euo pipefail

uv run pytest
EOF

chmod +x .git/hooks/pre-commit
```
