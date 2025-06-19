from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from duckduckgo_search import DDGS
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf",
    filename="ELYZA-japanese-Llama-2-7b-instruct-q4_K_M.gguf",
)

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=35,
    max_tokens=1024,
    n_ctx=4096,
    temperature=0.7,
    top_p=0.95,
    verbose=True,
)

template = """
以下の文章を参考にして、質問に対して日本語で簡潔かつ正確に答えてください。

【参考文書】
{context}

【質問】
{question}

※ 履歴書に記載がない場合は「わかりません」と答えてください。
※ 履歴書に関係ない場合は「関係なし」と答えてください。
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

retriever = FAISS.load_local(
    "./faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True,
).as_retriever(search_kwargs={"k": 6})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)


def search_online(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join([r["body"] for r in results])


def answer_query(query: str) -> str:
    print(f"Query: {query}")

    answer = qa_chain.invoke(query)
    result = answer["result"]

    if is_related_answer(result):
        return result
    else:
        print("RAGでは答えられなかったのでDuckDuckGoで検索")
        online_result = search_online(query)
        return f"質問の内容は履歴書に記載がないため、外部検索結果を参考にお答えします：\n\n{online_result}"


# 履歴書に関係ない or 書いてないと判断される場合を判定
def is_related_answer(result: str) -> bool:
    return not any(keyword in result for keyword in ["わかりません", "関係なし"])
