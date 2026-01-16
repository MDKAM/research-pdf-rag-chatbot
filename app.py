import gradio as gr


def hello(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "there"
    return f"Hello, {name}! 👋\n\nThis Space is alive. Next: PDF ingestion + FAISS + RAG."


with gr.Blocks(title="Research PDF RAG Chatbot (v0)") as demo:
    gr.Markdown(
        """
# Research PDF RAG Chatbot (v0)
Hello Gradio app ✅
"""
    )

    with gr.Row():
        name = gr.Textbox(label="Your name", placeholder="Mohammad", value="Mohammad")
    out = gr.Textbox(label="Output", lines=4)

    btn = gr.Button("Say hello")
    btn.click(fn=hello, inputs=[name], outputs=[out])

if __name__ == "__main__":
    demo.launch()
