from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from duckduckgo_search import DDGS
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf",
    filename="ELYZA-japanese-Llama-2-7b-instruct-q4_K_M.gguf"
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

retriever = FAISS.load_local(
    "./faiss_index",
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 10})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

def search_online(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join([r["body"] for r in results])

def answer_query(query: str) -> str:
    answer = qa_chain.invoke(query)

    print(answer)
    # if "わかりません" in answer or "知りません" in answer:
    #     extra = search_online(query)
    #     return f"{answer}\n\n---\n外部検索より補足:\n{extra}"

    return answer["result"]