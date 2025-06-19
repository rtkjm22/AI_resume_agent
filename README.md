#　導入方法
## プロジェクト読み込み

```bash
poetry install
```

## データ読み込み(初回のみ)

```bash
poetry run python -c "from loader import build_vectorstore; build_vectorstore('data/rirekisho.txt')"
```

## プロジェクト立ち上げ

```bash
poetry run python src/main.py
```