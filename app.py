import gradio as gr
from src.rag_pipeline import RAGPipeline

# Init RAG (loads once)
rag = RAGPipeline()


def chat_response(message, history):
    """Handle chat input."""
    result = rag.rag_query(message)
    answer = result['answer']

    # Format sources
    sources_html = "<h3>Sources:</h3><ul>"
    for i, src in enumerate(result['sources'][:3], 1):  # Top 3
        meta = src['metadata']
        sources_html += f"<li><strong>Chunk {i} (ID: {meta['complaint_id']}, Product: {meta['product_category']})</strong><br>{src['text'][:300]}...</li>"
    sources_html += "</ul>"

    # Append to history
    history.append((message, answer + "\n\n" + sources_html))
    return history, ""


# Gradio Interface
with gr.Blocks(title="CrediTrust Complaint Insights Chatbot") as demo:
    gr.Markdown(
        "# CrediTrust RAG Chatbot\nAsk about customer complaints across products (e.g., 'Issues with Credit Cards?')")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your question...", label="Query")
    clear = gr.Button("Clear")

    msg.submit(chat_response, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0",
                server_port=7860)  # Share for demo
