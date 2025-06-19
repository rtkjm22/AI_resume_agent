import gradio as gr
from rag_engine import answer_query

with gr.Blocks() as demo:
    gr.Markdown("# 履歴書AIエージェント")
    with gr.Row():
        txt = gr.Textbox(label="ご質問はこちらへ")
        output = gr.Textbox(label="エージェントの回答", lines=10)

    txt.submit(fn=answer_query, inputs=txt, outputs=output)

if __name__ == "__main__":
    demo.launch()
