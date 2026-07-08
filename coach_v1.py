# coach_v1.py — Student Success Coach v1
# Purple/blue Gradio app with memory, streaming, temperature, live prompt,
# session-only saved chats, downloadable chat transcript, and JSONL logging.

import json
import datetime
import pathlib
import tempfile

import gradio as gr #works
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
- Do not invent citations, deadlines, university policies, holidays, or facts.
- Keep answers skimmable and student-friendly.
- You do not have live web access in Project 1.
- You may use the current local date provided by the app, but you should not claim to have searched the web.
"""


def system_prompt_with_date(system_prompt):
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return (
        f"Current local date: {today}.\n"
        "Use this date for study planning and scheduling help. "
        "Do not claim live web access or official holiday lookup.\n\n"
        f"{system_prompt}"
    )


def build_messages(system_prompt, history, message):
    """
    Build the message list sent to the model.

    Gradio's Chatbot history uses messages like:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}

    The model itself is stateless, so we create memory by re-sending:
    system prompt + previous chat history + the new user message.
    """
    msgs = [SystemMessage(content=system_prompt_with_date(system_prompt))]

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
/* ===== PrepPal Purple Tech Theme ===== */

body {
    background:
        radial-gradient(circle at 12% 15%, rgba(147, 51, 234, 0.32), transparent 28%),
        radial-gradient(circle at 88% 8%, rgba(59, 130, 246, 0.25), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(88, 28, 135, 0.32), transparent 35%),
        linear-gradient(135deg, #030014 0%, #09001f 42%, #12002f 100%) !important;
}

/* Main app font + background */
.gradio-container {
    background:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(135deg, rgba(10, 3, 30, 0.94), rgba(25, 8, 56, 0.94)) !important;
    background-size: 28px 28px, 28px 28px, auto !important;
    color: #e9d5ff !important;
    font-family: "Courier New", "SF Mono", Monaco, Consolas, monospace !important;
}

/* Title */
#title {
    text-align: center;
    font-size: 2.7rem;
    font-weight: 900;
    letter-spacing: 1px;
    margin-top: 0.6rem;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    background: linear-gradient(90deg, #f0abfc, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 28px rgba(168, 85, 247, 0.35);
}

#subtitle {
    text-align: center;
    color: #c4b5fd;
    font-size: 0.95rem;
    margin-bottom: 1.1rem;
    letter-spacing: 0.5px;
}

/* All major cards/panels */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap,
.gradio-container .contain {
    background: rgba(18, 8, 45, 0.86) !important;
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    border-radius: 6px !important;
    box-shadow:
        0 0 0 1px rgba(96, 165, 250, 0.08),
        0 0 28px rgba(124, 58, 237, 0.18) !important;
}

/* Chat window */
.chatbot {
    background:
        linear-gradient(180deg, rgba(15, 8, 40, 0.96), rgba(25, 8, 55, 0.96)) !important;
    border: 1px solid rgba(192, 132, 252, 0.42) !important;
    border-radius: 8px !important;
    box-shadow:
        inset 0 0 30px rgba(30, 64, 175, 0.12),
        0 0 30px rgba(168, 85, 247, 0.18) !important;
}

/* Chat bubbles */
.message,
.message-wrap,
.bubble-wrap {
    font-family: "Courier New", "SF Mono", Monaco, Consolas, monospace !important;
}

.user,
.assistant {
    border-radius: 6px !important;
}

/* Text input boxes */
textarea,
input,
.gr-textbox,
.gr-textbox textarea {
    background: rgba(6, 2, 20, 0.92) !important;
    color: #f5f3ff !important;
    border: 1px solid rgba(147, 197, 253, 0.38) !important;
    border-radius: 6px !important;
    font-family: "Courier New", "SF Mono", Monaco, Consolas, monospace !important;
}

textarea::placeholder,
input::placeholder {
    color: #a78bfa !important;
}

/* Labels */
label,
.label-wrap {
    color: #c084fc !important;
    font-weight: 900 !important;
    letter-spacing: 0.4px;
}

/* Chat Options heading */
h3 {
    color: #f0abfc !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-shadow: 0 0 14px rgba(217, 70, 239, 0.45);
}

/* Buttons */
.gr-button {
    background:
        linear-gradient(90deg, rgba(126, 34, 206, 0.95), rgba(37, 99, 235, 0.95)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(216, 180, 254, 0.45) !important;
    border-radius: 6px !important;
    font-weight: 900 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase;
    box-shadow:
        0 0 18px rgba(147, 51, 234, 0.35),
        inset 0 0 12px rgba(255, 255, 255, 0.06) !important;
}

.gr-button:hover {
    filter: brightness(1.15);
    box-shadow:
        0 0 26px rgba(168, 85, 247, 0.55),
        0 0 18px rgba(59, 130, 246, 0.35) !important;
}

/* Dropdown */
select,
.dropdown,
[role="listbox"] {
    background: rgba(6, 2, 20, 0.92) !important;
    color: #f5f3ff !important;
    border-radius: 6px !important;
}

/* Slider */
input[type="range"] {
    accent-color: #a855f7 !important;
}

/* Accordion / Live System Prompt */
.accordion,
details {
    background: rgba(11, 4, 32, 0.92) !important;
    border: 1px solid rgba(96, 165, 250, 0.35) !important;
    border-radius: 6px !important;
}

.accordion span,
details summary,
details summary span {
    color: #93c5fd !important;
    font-weight: 900 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

/* Status box */
#component-8 textarea,
#component-8 input {
    color: #e9d5ff !important;
}

/* Footer */
footer {
    color: #a78bfa !important;
}

/* Subtle scanline overlay */
.gradio-container::before {
    content: "";
    pointer-events: none;
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        to bottom,
        rgba(255,255,255,0.025),
        rgba(255,255,255,0.025) 1px,
        transparent 1px,
        transparent 6px
    );
    opacity: 0.25;
    z-index: 0;
}


/* Remove white blocks: make individual controls dark neon cards */
#sidebar-panel .block,
#sidebar-panel .form,
#sidebar-panel .wrap,
#sidebar-panel .contain,
#sidebar-panel .panel {
    background: rgba(8, 2, 28, 0.96) !important;
    border: 1px solid rgba(192, 132, 252, 0.7) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 10px rgba(168, 85, 247, 0.38),
        inset 0 0 14px rgba(88, 28, 135, 0.25) !important;
}

/* Make each sidebar button an individual neon tab */
#sidebar-panel .gr-button {
    background: rgba(8, 2, 28, 0.96) !important;
    color: #f0abfc !important;
    border: 1px solid rgba(216, 180, 254, 0.85) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 10px rgba(192, 132, 252, 0.55),
        0 0 22px rgba(168, 85, 247, 0.25),
        inset 0 0 12px rgba(88, 28, 135, 0.32) !important;
    text-shadow: 0 0 10px rgba(240, 171, 252, 0.55);
}

#sidebar-panel .gr-button:hover {
    background: rgba(30, 10, 70, 0.98) !important;
    color: #ffffff !important;
    border-color: rgba(147, 197, 253, 0.95) !important;
    box-shadow:
        0 0 14px rgba(216, 180, 254, 0.7),
        0 0 34px rgba(96, 165, 250, 0.35),
        inset 0 0 16px rgba(124, 58, 237, 0.38) !important;
}

/* Dropdown and status boxes as dark neon tabs too */
#sidebar-panel select,
#sidebar-panel input,
#sidebar-panel textarea,
#sidebar-panel .gr-textbox,
#sidebar-panel .gr-textbox textarea {
    background: rgba(5, 1, 18, 0.96) !important;
    color: #e9d5ff !important;
    border: 1px solid rgba(147, 197, 253, 0.55) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 10px rgba(96, 165, 250, 0.22),
        inset 0 0 10px rgba(59, 7, 100, 0.35) !important;
}

/* Make label chips dark instead of pale/white */
#sidebar-panel label,
#sidebar-panel .label-wrap,
#sidebar-panel .wrap > label,
#sidebar-panel span {
    background: transparent !important;
    color: #c084fc !important;
    text-shadow: 0 0 8px rgba(192, 132, 252, 0.45);
}

/* Fix the pale label badges */
#sidebar-panel .label-wrap span,
#sidebar-panel label span {
    background: rgba(8, 2, 28, 0.96) !important;
    border: 1px solid rgba(192, 132, 252, 0.45) !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
}

/* Creativity slider container */
#sidebar-panel input[type="range"] {
    accent-color: #c084fc !important;
}

/* Live System Prompt accordion as dark neon tab */
#sidebar-panel details,
#sidebar-panel .accordion {
    background: rgba(8, 2, 28, 0.96) !important;
    border: 1px solid rgba(192, 132, 252, 0.7) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 10px rgba(192, 132, 252, 0.45),
        inset 0 0 14px rgba(88, 28, 135, 0.3) !important;
}

/* Main message input and send button also dark/neon */
textarea,
input {
    background: rgba(5, 1, 18, 0.96) !important;
    color: #e9d5ff !important;
}

.gradio-container .gr-button {
    background: rgba(8, 2, 28, 0.96) !important;
    color: #f0abfc !important;
    border: 1px solid rgba(216, 180, 254, 0.75) !important;
}

/* Keep the chat panel dark */
.chatbot {
    background: rgba(8, 2, 28, 0.96) !important;
    border: 1px solid rgba(192, 132, 252, 0.55) !important;
}



/* ===== Clean neon tab polish ===== */

/* Remove pale label-chip backgrounds everywhere */
label,
.label-wrap,
.label-wrap span,
.gradio-container label span,
.gradio-container .label-wrap span {
    background: transparent !important;
    border: none !important;
    color: #d8b4fe !important;
    box-shadow: none !important;
    text-shadow: 0 0 10px rgba(216, 180, 254, 0.45);
}

/* Rename/clean labels visually */
.chatbot label,
.chatbot .label-wrap,
.chatbot .label-wrap span {
    background: transparent !important;
    color: #c084fc !important;
    border: none !important;
}

/* Make all buttons dark neon, no white fill */
.gradio-container .gr-button,
#sidebar-panel .gr-button {
    background: rgba(6, 2, 22, 0.96) !important;
    color: #f0abfc !important;
    border: 1.5px solid rgba(216, 180, 254, 0.85) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 10px rgba(216, 180, 254, 0.45),
        0 0 22px rgba(168, 85, 247, 0.28),
        inset 0 0 16px rgba(88, 28, 135, 0.35) !important;
    text-shadow: 0 0 10px rgba(240, 171, 252, 0.6);
    font-weight: 900 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
}

/* Button hover */
.gradio-container .gr-button:hover,
#sidebar-panel .gr-button:hover {
    background: rgba(24, 8, 58, 0.98) !important;
    color: #ffffff !important;
    border-color: rgba(147, 197, 253, 0.95) !important;
    box-shadow:
        0 0 14px rgba(216, 180, 254, 0.75),
        0 0 32px rgba(96, 165, 250, 0.38),
        inset 0 0 18px rgba(124, 58, 237, 0.42) !important;
}

/* Send button: dark neon full-width tab */
button {
    background: rgba(6, 2, 22, 0.96) !important;
    color: #f0abfc !important;
}

/* Message input panel */
textarea,
input,
.gr-textbox textarea,
.gr-textbox input {
    background: rgba(4, 1, 16, 0.98) !important;
    color: #f5f3ff !important;
    border: 1.5px solid rgba(147, 197, 253, 0.48) !important;
    border-radius: 10px !important;
    box-shadow:
        inset 0 0 14px rgba(59, 7, 100, 0.32),
        0 0 8px rgba(96, 165, 250, 0.18) !important;
}

/* Placeholder text */
textarea::placeholder,
input::placeholder {
    color: #a78bfa !important;
    opacity: 0.9 !important;
}

/* Saved chats dropdown: no white */
#sidebar-panel select,
#sidebar-panel [role="listbox"],
#sidebar-panel .dropdown,
#sidebar-panel input {
    background: rgba(4, 1, 16, 0.98) !important;
    color: #f5f3ff !important;
    border: 1.5px solid rgba(147, 197, 253, 0.5) !important;
}

/* Sidebar sections stay dark */
#sidebar-panel,
#sidebar-panel .block,
#sidebar-panel .form,
#sidebar-panel .wrap,
#sidebar-panel .contain,
#sidebar-panel .panel {
    background: rgba(6, 2, 22, 0.94) !important;
}

/* Creativity / temperature box cleanup */
#sidebar-panel .range,
#sidebar-panel input[type="range"] {
    accent-color: #c084fc !important;
}

/* Center the temperature help text */
#sidebar-panel p,
#sidebar-panel .info,
#sidebar-panel .svelte-1gfkn6j,
#sidebar-panel .svelte-1gfkn6j p {
    text-align: center !important;
    color: #c4b5fd !important;
}

/* Hide the little numeric reset/value area if Gradio exposes it visually */
#sidebar-panel input[type="number"] {
    background: rgba(4, 1, 16, 0.95) !important;
    color: #f5f3ff !important;
    text-align: center !important;
}

/* Make the slider area feel like one clean neon module */
#sidebar-panel .wrap:has(input[type="range"]),
#sidebar-panel .block:has(input[type="range"]) {
    background: rgba(4, 1, 16, 0.96) !important;
    border: 1.5px solid rgba(192, 132, 252, 0.7) !important;
    border-radius: 12px !important;
    box-shadow:
        0 0 12px rgba(192, 132, 252, 0.35),
        inset 0 0 14px rgba(88, 28, 135, 0.28) !important;
    text-align: center !important;
}

/* Make Live System Prompt look like a simple dark neon dropdown */
#sidebar-panel details,
#sidebar-panel .accordion {
    background: rgba(6, 2, 22, 0.96) !important;
    border: 1.5px solid rgba(192, 132, 252, 0.75) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 12px rgba(192, 132, 252, 0.38),
        inset 0 0 14px rgba(88, 28, 135, 0.28) !important;
}

/* Chatbot label clean-up */
.chatbot .label-wrap,
.chatbot label,
.chatbot span {
    background: transparent !important;
}

/* Main chat panel */
.chatbot {
    background: rgba(6, 2, 22, 0.96) !important;
    border: 1.5px solid rgba(192, 132, 252, 0.65) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 12px rgba(192, 132, 252, 0.32),
        inset 0 0 24px rgba(88, 28, 135, 0.25) !important;
}

/* Remove white/pale message label chip */
#component-0,
#component-1,
#component-2 {
    background: transparent !important;
}

/* Keep headings clean */
h3 {
    background: transparent !important;
    color: #f0abfc !important;
    border: none !important;
    text-shadow: 0 0 12px rgba(240, 171, 252, 0.55);
}

"""


with gr.Blocks() as demo:
    saved_chats_state = gr.State({})

    gr.HTML("<div id='title'>prepPal - an AI chatbot</div>")
    gr.HTML("<div id='subtitle'>Your private AI study companion powered by Ollama, LangChain, and Gradio</div>")

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar-panel"):
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

            status = gr.Textbox(label="Status", interactive=False, lines=1)

            temp = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.4,
                step=0.1,
                label="Creativity / Temperature",
                info="0 = focused and consistent. 1 = more creative and varied.",
            )

            with gr.Accordion("Live System Prompt", open=False):
                sys_box = gr.Textbox(
                    value=COACH_PROMPT,
                    lines=8,
                    label="Coach Instructions",
                    info="Edit the Coach's personality and rules while the app is running.",
                )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="prepPal Chat",
                height=430,
            )

            msg = gr.Textbox(
                label="Ask prepPal",
                placeholder="Type your study question...",
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
        outputs=[gr.File(label="Downloaded Chat", visible=False), status],
    )


demo.launch(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
)
