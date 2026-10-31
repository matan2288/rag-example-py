import gradio as gr
from dotenv import load_dotenv

from knowledge import find_closest, load_knowledge
from openai_client import answer_from_context, require_api_key

load_dotenv(override=True)
require_api_key()
PIECES = load_knowledge()


def format_context(pieces) -> str:
    lines = ["## Retrieved context\n"]
    for piece in pieces:
        lines.append(f"**{piece.source}**\n\n{piece.text}\n")
    return "\n".join(lines)


def chat(history: list[dict]):
    question = history[-1]["content"]
    prior = history[:-1]
    closest = find_closest(question, PIECES)
    context = "\n\n".join(piece.text for piece in closest)
    answer = answer_from_context(question, prior, context)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(closest)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# Insurellm Expert Assistant\nAsk anything about Insurellm.")
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=600, type="messages")
                box = gr.Textbox(placeholder="Ask anything about Insurellm...", show_label=False)
            with gr.Column():
                context = gr.Markdown("*Retrieved context will appear here*", height=600)

        box.submit(put_message_in_chatbot, [box, chatbot], [box, chatbot]).then(
            chat, chatbot, [chatbot, context]
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
