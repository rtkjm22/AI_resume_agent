# 履歴書検索 RAG システム

## 概要

このプロジェクトは、**LangChain + ローカル LLM + FAISS によるベクトル検索**を活用した、履歴書ベースの質問応答 RAG システムです。

## 主な特徴・機能

主な機能は以下内容になります。

- **ELYZA Japanese Llama 2 7B** モデル（`gguf`形式）をローカル実行
- 履歴書（Markdown やテキスト）を **FAISS + HuggingFaceEmbeddings** でベクトル化し、意味ベースで検索
- **LangChain の RetrievalQA チェーン** により、ドキュメントをもとにした回答を自動生成
- 質問に対して履歴書に情報がない場合は、**DuckDuckGo 検索で外部回答**も自動実行

## 導入方法

### サンプルデータ追加

`data/rirekisho.txt`を作成し、自分自身の経歴を入力する。

### プロジェクト読み込み

```bash
poetry install
```

### データ読み込み(初回のみ)

```bash
poetry run python -c "from loader import build_vectorstore; build_vectorstore('data/rirekisho.txt')"
```

### プロジェクト立ち上げ

```bash
poetry run python src/main.py
```

### 画面にアクセス

```
http://127.0.0.1:7860/
```

## 処理フロー

1. 質問を受け取る（例:「この人の得意な技術は？」）
2. FAISS ベースの Retrieval で関連文書を抽出
3. ELYZA-Llama にプロンプトを渡して回答生成
4. 回答が「わかりません」または「関係なし」の場合 → DuckDuckGo 検索を実行
5. 最終的な回答を返却（履歴書内 or 外部検索結果）

## 技術スタック

| 技術                                   | 用途                                     |
| -------------------------------------- | ---------------------------------------- |
| **LangChain**                          | チェーン構築（RetrievalQA）              |
| **FAISS**                              | ベクトルストア                           |
| **HuggingFace Embeddings**             | 文書の意味ベクトル化                     |
| **ELYZA-japanese-Llama-2-7b-instruct** | 回答生成（ローカル LLM）                 |
| **DuckDuckGo Search**                  | フォールバック検索（履歴書外の質問対応） |

## プロンプト例（PromptTemplate）

```text
以下の文章を参考にして、質問に対して日本語で簡潔かつ正確に答えてください。

【参考文書】
{context}

【質問】
{question}

※ 履歴書に記載がない場合は「わかりません」と答えてください。
※ 履歴書に関係ない場合は「関係なし」と答えてください。
```
