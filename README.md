# Projekt 3 – analiza jakości powietrza

Refaktoryzacja notebooka do modułów Pythona i notebooka orkiestrującego.

## Uruchamianie

``` sh
git clone https://github.com/vardwyn/ZTP_projekt3
cd ZTP_projekt3
uv venv
uv sync
uv run jupyter lab analiza.ipynb
```

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
