# coach_v1.py — Student Success Coach v1
# Purple/blue Gradio app with memory, streaming, temperature, live prompt,
# session-only saved chats, downloadable chat transcript, and JSONL logging.

import json
import base64
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
- Remember details the student gives you earlier in the same chat, such as their name, and use them naturally later.
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
        f"Current local date: {today}." + chr(10)
        + "Use this date for study planning and scheduling help. "
        + "Do not claim live web access or official holiday lookup."
        + chr(10) + chr(10)
        + system_prompt
    )


def _extract_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _extract_text(value.get("content", ""))
    if hasattr(value, "content"):
        return _extract_text(value.content)
    if isinstance(value, (list, tuple)):
        return " ".join(_extract_text(v) for v in value)
    return str(value)


def build_messages(system_prompt, history, message):
    msgs = [SystemMessage(content=system_prompt_with_date(system_prompt))]

    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = _extract_text(turn.get("content", "")).strip()

            if role == "user" and content:
                msgs.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                msgs.append(AIMessage(content=content))

        elif hasattr(turn, "role") and hasattr(turn, "content"):
            role = turn.role
            content = _extract_text(turn.content).strip()

            if role == "user" and content:
                msgs.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                msgs.append(AIMessage(content=content))

        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg = _extract_text(turn[0]).strip()
            bot_msg = _extract_text(turn[1]).strip()

            if user_msg:
                msgs.append(HumanMessage(content=user_msg))
            if bot_msg:
                msgs.append(AIMessage(content=bot_msg))

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
        f.write(json.dumps(record, ensure_ascii=False) + chr(10))


def respond(message, display_history, memory_history, temperature, system_prompt):
    if not message or not message.strip():
        yield "", display_history, memory_history
        return

    if memory_history is None:
        memory_history = []

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=float(temperature),
    )

    msgs = build_messages(system_prompt, memory_history, message)

    answer = ""

    updated_memory = memory_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    try:
        for chunk in llm.stream(msgs):
            if chunk.content:
                answer += chunk.content
                updated_memory[-1] = {"role": "assistant", "content": answer}
                yield "", updated_memory, updated_memory

        log_interaction(message, answer, temperature, system_prompt)

    except Exception as e:
        error_message = (
            "I could not connect to the local Ollama model."
            + chr(10) + chr(10)
            + "Try opening the Ollama app, then run this file again."
            + chr(10) + chr(10)
            + f"Error: {e}"
        )
        updated_memory[-1] = {"role": "assistant", "content": error_message}
        yield "", updated_memory, updated_memory


def make_chat_title(history):
    if not history:
        return "Empty chat"

    first_user_message = "Saved chat"

    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") == "user":
            content = _extract_text(turn.get("content", "")).strip()
            if content:
                first_user_message = content
                break

        elif hasattr(turn, "role") and hasattr(turn, "content"):
            if turn.role == "user":
                content = _extract_text(turn.content).strip()
                if content:
                    first_user_message = content
                    break

        elif isinstance(turn, (list, tuple)) and len(turn) >= 1:
            content = _extract_text(turn[0]).strip()
            if content:
                first_user_message = content
                break

    title = first_user_message[:45]

    if len(first_user_message) > 45:
        title += "..."

    return title


def save_current_chat(history, saved_chats):
    if saved_chats is None:
        saved_chats = {}

    if not history:
        return saved_chats, gr.update(choices=list(saved_chats.keys())), "No chat to save yet."

    base_title = make_chat_title(history)
    title = base_title
    count = 2

    saved_chats = dict(saved_chats)

    while title in saved_chats:
        title = f"{base_title} ({count})"
        count += 1

    saved_chats[title] = history

    return (
        saved_chats,
        gr.update(choices=list(saved_chats.keys()), value=title),
        f"Saved chat: {title}",
    )


def load_saved_chat(selected_chat, saved_chats):
    if saved_chats is None:
        saved_chats = {}

    if not selected_chat or selected_chat not in saved_chats:
        return [], [], "Choose a saved chat first."

    loaded_chat = saved_chats[selected_chat]
    return loaded_chat, loaded_chat, f"Loaded chat: {selected_chat}"


def new_chat():
    return [], [], "Started a new chat."


def download_chat(history):
    if not history:
        return gr.update(value=None, visible=False), "No chat to download yet."

    lines = []
    lines.append("prepPal Chat Transcript")
    lines.append(f"Exported: {datetime.datetime.now().isoformat()}")
    lines.append("")

    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role", "").title()
            content = _extract_text(turn.get("content", ""))

            if role == "User":
                role = "Student"
            elif role == "Assistant":
                role = "prepPal"

            lines.append(f"{role}: {content}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("")

        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg = _extract_text(turn[0])
            bot_msg = _extract_text(turn[1])

            if user_msg:
                lines.append(f"Student: {user_msg}")
                lines.append("")
                lines.append("-" * 60)
                lines.append("")

            if bot_msg:
                lines.append(f"prepPal: {bot_msg}")
                lines.append("")
                lines.append("-" * 60)
                lines.append("")

    transcript = chr(10).join(lines)

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt",
        prefix="preppal_chat_transcript_",
        mode="w",
        encoding="utf-8",
    )

    with temp_file as f:
        f.write(transcript)

    return gr.update(value=temp_file.name, visible=True), "Chat transcript is ready to download."


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

/* Thin individual outlines for Chat Option buttons */
#save-chat-button,
#load-chat-button,
#new-chat-button,
#download-chat-button {
    width: 92% !important;
    margin: 10px auto !important;
    padding: 12px 18px !important;
    background: rgba(6, 2, 22, 0.94) !important;
    color: #f0abfc !important;
    border: 1px solid rgba(192, 132, 252, 0.9) !important;
    border-radius: 999px !important;
    box-shadow:
        0 0 8px rgba(192, 132, 252, 0.34),
        inset 0 0 10px rgba(88, 28, 135, 0.22) !important;
}

#save-chat-button:hover,
#load-chat-button:hover,
#new-chat-button:hover,
#download-chat-button:hover {
    border-color: rgba(147, 197, 253, 0.95) !important;
    box-shadow:
        0 0 12px rgba(216, 180, 254, 0.48),
        0 0 18px rgba(96, 165, 250, 0.24) !important;
}



/* Make chatbot replies pink with no highlighted background */
.chatbot .message,
.chatbot .message-wrap,
.chatbot .bubble-wrap,
.chatbot .assistant,
.chatbot [data-testid="bot"],
.chatbot [class*="bot"] {
    background: transparent !important;
    box-shadow: none !important;
}

.chatbot .assistant,
.chatbot .assistant *,
.chatbot [data-testid="bot"],
.chatbot [data-testid="bot"] *,
.chatbot [class*="bot"],
.chatbot [class*="bot"] * {
    color: #f0abfc !important;
    background: transparent !important;
    text-shadow: 0 0 8px rgba(240, 171, 252, 0.35) !important;
}

/* Force assistant replies to be pink and remove the white/gray bubble */
#main-chatbot .message,
#main-chatbot .message *,
#main-chatbot .message-wrap,
#main-chatbot .message-wrap *,
#main-chatbot .bubble,
#main-chatbot .bubble *,
#main-chatbot .bubble-wrap,
#main-chatbot .bubble-wrap *,
#main-chatbot .prose,
#main-chatbot .prose *,
#main-chatbot [data-testid="bot"],
#main-chatbot [data-testid="bot"] *,
#main-chatbot [class*="bot"],
#main-chatbot [class*="bot"] *,
#main-chatbot [class*="assistant"],
#main-chatbot [class*="assistant"] * {
    background: transparent !important;
    background-color: transparent !important;
    color: #f0abfc !important;
    box-shadow: none !important;
    text-shadow: 0 0 8px rgba(240, 171, 252, 0.35) !important;
}

/* Remove white code/markdown-looking panels inside replies */
#main-chatbot pre,
#main-chatbot code,
#main-chatbot p,
#main-chatbot ol,
#main-chatbot ul,
#main-chatbot li {
    background: transparent !important;
    background-color: transparent !important;
    color: #f0abfc !important;
}

/* Make streaming/thinking assistant text match finished assistant text */
#main-chatbot,
#main-chatbot *,
#main-chatbot .message,
#main-chatbot .message *,
#main-chatbot .message.pending,
#main-chatbot .message.pending *,
#main-chatbot .message.generating,
#main-chatbot .message.generating *,
#main-chatbot [class*="pending"],
#main-chatbot [class*="pending"] *,
#main-chatbot [class*="generating"],
#main-chatbot [class*="generating"] *,
#main-chatbot [class*="streaming"],
#main-chatbot [class*="streaming"] *,
#main-chatbot [aria-live],
#main-chatbot [aria-live] * {
    color: #f0abfc !important;
    opacity: 1 !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    text-shadow: 0 0 8px rgba(240, 171, 252, 0.35) !important;
}

/* Keep the chat area dark, not white, while the bot is typing */
#main-chatbot .prose,
#main-chatbot .prose *,
#main-chatbot p,
#main-chatbot ol,
#main-chatbot ul,
#main-chatbot li,
#main-chatbot pre,
#main-chatbot code {
    color: #f0abfc !important;
    opacity: 1 !important;
    background: transparent !important;
    background-color: transparent !important;
}

/* Move prepPal Chat label above the chatbot so it does not overlap messages */
#main-chatbot {
    position: relative !important;
    padding-top: 34px !important;
}

#main-chatbot label,
#main-chatbot .label-wrap,
#main-chatbot .label-wrap span {
    position: absolute !important;
    top: 8px !important;
    left: 16px !important;
    z-index: 10 !important;
    background: rgba(6, 2, 22, 0.96) !important;
    color: #f0abfc !important;
    padding: 2px 10px !important;
    border-radius: 8px !important;
    text-shadow: 0 0 8px rgba(240, 171, 252, 0.55) !important;
}

/* Put prepPal Chat outside/above the chat box */
#chat-title-outside {
    margin: 0 0 6px 12px !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#chat-title-outside h3 {
    color: #f0abfc !important;
    font-size: 1rem !important;
    text-transform: none !important;
    letter-spacing: 0.4px !important;
    margin: 0 !important;
    text-shadow: 0 0 8px rgba(240, 171, 252, 0.55) !important;
}

/* Hide the built-in label inside the chatbot */
#main-chatbot label,
#main-chatbot .label-wrap,
#main-chatbot .label-wrap span {
    display: none !important;
}

/* Remove extra top padding from the previous label attempt */
#main-chatbot {
    padding-top: 0 !important;
}



/* ===== Load Saved Chat dropdown styling ===== */

#load-chat-dropdown {
    background: rgba(6, 2, 22, 0.96) !important;
    border: 1px solid rgba(216, 180, 254, 0.85) !important;
    border-radius: 10px !important;
    box-shadow:
        0 0 10px rgba(216, 180, 254, 0.42),
        0 0 22px rgba(168, 85, 247, 0.24),
        inset 0 0 14px rgba(88, 28, 135, 0.32) !important;
}

#load-chat-dropdown label,
#load-chat-dropdown .label-wrap,
#load-chat-dropdown .label-wrap span {
    color: #f0abfc !important;
    background: transparent !important;
    border: none !important;
    text-shadow: 0 0 10px rgba(240, 171, 252, 0.55) !important;
}

#load-chat-dropdown input,
#load-chat-dropdown select,
#load-chat-dropdown [role="combobox"],
#load-chat-dropdown [role="listbox"] {
    background: rgba(4, 1, 16, 0.98) !important;
    color: #f5f3ff !important;
    border: 1.5px solid rgba(147, 197, 253, 0.5) !important;
    border-radius: 10px !important;
}
"""


with gr.Blocks() as demo:
    saved_chats_state = gr.State({})
    chat_memory_state = gr.State([])

    gr.HTML("<div id='title'>prepPal - an AI chatbot</div>")
    gr.HTML("<div id='subtitle'>Your private AI study companion powered by Ollama, LangChain, and Gradio</div>")

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar-panel"):
            gr.Markdown("### Chat Options")

            saved_dropdown = gr.Dropdown(
                choices=[],
                label="Load Saved Chat",
                info="Click the dropdown, then choose the chat title you want to load.",
                interactive=True,
                elem_id="load-chat-dropdown",
            )

            save_button = gr.Button("Save Current Chat", elem_id="save-chat-button")
            new_button = gr.Button("New Chat", elem_id="new-chat-button")
            download_button = gr.Button("Download Chat", elem_id="download-chat-button")

            download_file = gr.File(
                label="Download Transcript",
                visible=False,
                interactive=False,
            )

            status = gr.Textbox(label="Status", interactive=False, lines=1)

            temp = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Creativity / Temperature",
                info="0 = focused and consistent. 0.5 = balanced. 1 = more creative and varied.",
            )

            with gr.Accordion("Live System Prompt", open=False):
                sys_box = gr.Textbox(
                    value=COACH_PROMPT,
                    lines=8,
                    label="Coach Instructions",
                    info="Edit the Coach's personality and rules while the app is running.",
                )

        with gr.Column(scale=3):
            gr.Markdown("### prepPal Chat", elem_id="chat-title-outside")

            chatbot = gr.Chatbot(
                elem_id="main-chatbot",
                label="",
                height=430,
            )

            msg = gr.Textbox(
                label="Ask prepPal",
                placeholder="Type your study question...",
            )

            send_button = gr.Button("Send")

    msg.submit(
        respond,
        inputs=[msg, chatbot, chat_memory_state, temp, sys_box],
        outputs=[msg, chatbot, chat_memory_state],
    )

    send_button.click(
        respond,
        inputs=[msg, chatbot, chat_memory_state, temp, sys_box],
        outputs=[msg, chatbot, chat_memory_state],
    )

    save_button.click(
        save_current_chat,
        inputs=[chat_memory_state, saved_chats_state],
        outputs=[saved_chats_state, saved_dropdown, status],
    )

    saved_dropdown.change(
        load_saved_chat,
        inputs=[saved_dropdown, saved_chats_state],
        outputs=[chatbot, chat_memory_state, status],
    )

    new_button.click(
        new_chat,
        inputs=[],
        outputs=[chatbot, chat_memory_state, status],
    )

    download_button.click(
        download_chat,
        inputs=[chat_memory_state],
        outputs=[download_file, status],
    )


demo.launch(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
)
