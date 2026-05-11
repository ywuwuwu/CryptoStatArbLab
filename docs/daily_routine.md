# Daily Update Routine

Use this routine when you want to refresh the project without pushing raw market data, generated run outputs, or draft Medium copy to GitHub.

## Morning refresh

```bash
cd /home/yashuo/rc/CryptoStatArbLab
source .venv/bin/activate
git pull --ff-only
make test
```

## Refresh local data

Keep downloaded data local. The `data/`, `data_train/`, and `data_holdout/` folders are ignored by Git.

```bash
make data_momentum_kraken
```

Use the longer Binance command from `README.md` only when the endpoint is available for your location.

## Run research

```bash
make sweep_momentum
make diagnostics
```

Review the new local output under `runs/latest/`. Only promote small summary artifacts manually if they are useful for the public repo, such as a curated CSV or image copied into `reports/figures/`.

## Commit code and docs only

```bash
git status --ignored
git add README.md Makefile pyproject.toml requirements.txt configs src tests reports/report.md reports/figures docs .gitignore
git diff --cached --stat
git commit -m "Update research code and docs"
git push
```

Before pushing, confirm these paths are not staged:

```bash
git status --short | rg '^(A|M|R).*(data/|data_train/|data_holdout/|runs/|reports/medium_article.md)' || true
```

## Weekly cleanup

```bash
make clean
python -m pytest
```
