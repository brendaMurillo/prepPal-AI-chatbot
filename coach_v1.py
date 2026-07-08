# coach_v1.py — Student Success Coach v1
# Purple/blue Gradio app with memory, streaming, temperature, live prompt,
# session-only saved chats, downloadable chat transcript, and JSONL logging.

import json
import datetime
import pathlib
import tempfile

import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


MODEL_NAME = "llama3.2"
LOG = pathlib.Path("interactions.jsonl")

COACH_PROMPT = """
You are the Student Success Coach, a warm, encouraging, trustworthy academic assistant.

Your purpose:
- Help students understand concepts, plan studying, manage academic stress, and practice skills.
- Do not simply complete graded homework or exams for the student.
- Instead, guide the student with explanations, hints, examples, and practice steps.
- Be clear, supportive, and structured.

Use this format when helpful:
1. Quick answer
2. Step-by-step explanation
3. Practice or next step

Rules:
- If the student asks for direct answers to graded work, guide them instead of doing it for them.
- If you are unsure, say so clearly.
- Do not invent citations, deadlines, university policies, or facts.
- Keep answers skimmable and student-friendly.
"""


def build_messages(system_prompt, history, message):
    """
    Build the message list sent to the model.

    Gradio's current Chatbot format is a list of dictionaries:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}
    """
    msgs = [SystemMessage(content=system_prompt)]

    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")

        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))

    msgs.append(HumanMessage(content=message))
    return msgs


def log_interaction(user_message, coach_response, temperature, system_prompt):
    record = {
        "ts": datetime.datetime.now().isoformat(),
        "model": MODEL_NAME,
        "temperature": temperature,
        "user": user_message,
        "coach": coach_response,
        "system_prompt": system_prompt,
    }

    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def respond(message, history, temperature, system_prompt):
    if not message.strip():
        yield "", history
        return

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=float(temperature),
    )

    msgs = build_messages(system_prompt, history, message)

    answer = ""
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    try:
        for chunk in llm.stream(msgs):
            if chunk.content:
                answer += chunk.content
                updated_history[-1] = {"role": "assistant", "content": answer}
                yield "", updated_history

        log_interaction(message, answer, temperature, system_prompt)

    except Exception as e:
        error_message = (
            "I could not connect to the local Ollama model.\n\n"
            "Try opening the Ollama app, then run this file again.\n\n"
            f"Error: {e}"
        )
        updated_history[-1] = {"role": "assistant", "content": error_message}
        yield "", updated_history


def make_chat_title(history):
    if not history:
        return "Empty chat"

    first_user_message = "Saved chat"

    for turn in history:
        if turn.get("role") == "user":
            first_user_message = turn.get("content", "Saved chat")
            break

    title = first_user_message.strip()[:40]

    if len(first_user_message) > 40:
        title += "..."

    timestamp = datetime.datetime.now().strftime("%I:%M %p")
    return f"{timestamp} — {title}"


def save_current_chat(history, saved_chats):
    if not history:
        return saved_chats, gr.update(choices=list(saved_chats.keys())), "No chat to save yet."

    title = make_chat_title(history)
    saved_chats[title] = history

    return (
        saved_chats,
        gr.update(choices=list(saved_chats.keys()), value=title),
        f"Saved chat: {title}",
    )


def load_saved_chat(selected_chat, saved_chats):
    if not selected_chat or selected_chat not in saved_chats:
        return [], "Choose a saved chat first."

    return saved_chats[selected_chat], f"Loaded chat: {selected_chat}"


def new_chat():
    return [], "Started a new chat."


def download_chat(history):
    if not history:
        return None, "No chat to download yet."

    lines = []
    lines.append("Student Success Coach — Chat Transcript")
    lines.append(f"Exported: {datetime.datetime.now().isoformat()}")
    lines.append("")

    for turn in history:
        role = turn.get("role", "").title()
        content = turn.get("content", "")

        if role == "User":
            role = "Student"
        elif role == "Assistant":
            role = "Coach"

        lines.append(f"{role}: {content}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    transcript = "\n".join(lines)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".txt",
        prefix="student_success_coach_chat_",
        encoding="utf-8",
    )

    temp_file.write(transcript)
    temp_file.close()

    return temp_file.name, "Chat transcript ready to download."


CUSTOM_CSS = """
body {
    background:
        radial-gradient(circle at 20% 20%, rgba(124, 58, 237, 0.35), transparent 30%),
        radial-gradient(circle at 80% 10%, rgba(37, 99, 235, 0.35), transparent 25%),
        linear-gradient(135deg, #020617 0%, #111827 45%, #1e1b4b 100%);
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(168, 85, 247, 0.22), transparent 30%),
        radial-gradient(circle at top right, rgba(59, 130, 246, 0.22), transparent 30%),
        linear-gradient(135deg, #020617, #0f172a, #1e1b4b) !important;
    color: #e0e7ff !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    margin-top: 0.5rem;
    margin-bottom: 0.25rem;
    background: linear-gradient(90deg, #c084fc, #60a5fa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#subtitle {
    text-align: center;
    color: #c7d2fe;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}

.gr-button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: 1px solid rgba(191, 219, 254, 0.25) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 18px rgba(96, 165, 250, 0.25);
}

textarea, input {
    border-radius: 14px !important;
}

label, .label-wrap {
    color: #dbeafe !important;
}

.chatbot {
    border: 1px solid rgba(147, 197, 253, 0.35) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 30px rgba(96, 165, 250, 0.18);
}
"""


with gr.Blocks() as demo:
    saved_chats_state = gr.State({})

    gr.HTML("<div id='title'>Student Success Coach</div>")
    gr.HTML("<div id='subtitle'>A private local AI study coach powered by Ollama, LangChain, and Gradio</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Chat Options")

            saved_dropdown = gr.Dropdown(
                choices=[],
                label="Saved Chats",
                interactive=True,
            )

            save_button = gr.Button("Save Current Chat")
            load_button = gr.Button("Load Saved Chat")
            new_button = gr.Button("New Chat")
            download_button = gr.Button("Download Chat")

            download_file = gr.File(label="Download File")
            status = gr.Textbox(label="Status", interactive=False)

            temp = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.4,
                step=0.1,
                label="Creativity / Temperature",
                info="0 = focused and consistent. 1 = more creative and varied.",
            )

            sys_box = gr.Textbox(
                value=COACH_PROMPT,
                lines=8,
                label="Live System Prompt",
                info="Edit the Coach's personality and rules while the app is running.",
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Student Success Coach",
                height=520,
            )

            msg = gr.Textbox(
                label="Message",
                placeholder="Ask the Coach for study help...",
            )

            send_button = gr.Button("Send")

    msg.submit(
        respond,
        inputs=[msg, chatbot, temp, sys_box],
        outputs=[msg, chatbot],
    )

    send_button.click(
        respond,
        inputs=[msg, chatbot, temp, sys_box],
        outputs=[msg, chatbot],
    )

    save_button.click(
        save_current_chat,
        inputs=[chatbot, saved_chats_state],
        outputs=[saved_chats_state, saved_dropdown, status],
    )

    load_button.click(
        load_saved_chat,
        inputs=[saved_dropdown, saved_chats_state],
        outputs=[chatbot, status],
    )

    new_button.click(
        new_chat,
        inputs=[],
        outputs=[chatbot, status],
    )

    download_button.click(
        download_chat,
        inputs=[chatbot],
        outputs=[download_file, status],
    )


demo.launch(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
)
