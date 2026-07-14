"""
coach_v2.py — The Agent Coach (Project 2)

Adds to the v1 tools (web_search, arxiv_search):
  - calculator      guarded arithmetic (only digits and + - * / ( ) . reach eval())
  - gpa_calculator  grade-point average from letter grades + credit hours
  - course_search   RAG over the student's own PDFs — run ingest.py first

Also adds:
  - conversation memory   LangGraph checkpointer, keyed by a thread_id, so the
                           agent actually remembers earlier turns
  - semantic memory        a separate Chroma collection of past Q&A, recalled
                           by embedding similarity (not just replayed in order)
  - guardrails              validated tool inputs, a recursion cap, and a
                           JSONL log of every tool call and result
  - a Gradio web UI        run `python coach_v2.py` and it opens in your
                           browser; tool traces still print in this terminal

NOTE: system_prompt below is a placeholder. Swap in your actual v1
COACH_PROMPT persona text (keep the tool-usage rules appended after it).
"""

import json
import re
from datetime import datetime, timezone

import arxiv
from ddgs import DDGS
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

# ── config ───────────────────────────────────────────────────────────────

EMBED_MODEL = "nomic-embed-text"

# Must match ingest.py exactly — this is where course_search reads from.
COURSE_CHROMA_DIR = "./chroma_course_db"
COURSE_COLLECTION = "course_materials"

MEMORY_CHROMA_DIR = "./chroma_memory_db"
MEMORY_COLLECTION = "semantic_memory"

TOOL_LOG_PATH = "tool_calls.jsonl"
SESSION_THREAD_ID = "coach-session-1"  # one running conversation for now

# Guardrail: LangGraph counts each model call AND each tool round as a
# "step," so this caps roughly 5-6 reasoning/tool cycles per question
# instead of the create_agent default of 25. Prevents the loop-forever
# failure from quiz Q6.
RECURSION_LIMIT = 12


# ── v1 tools (unchanged) ────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current facts or recent information."""
    if not query or not query.strip():
        return "Error: search query is empty."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No web results found."

        return "\n".join(
            f"- {r.get('title', 'No title')}: {r.get('body', 'No snippet')}"
            for r in results
        )

    except Exception as e:
        return f"Web search error: {e}"


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers."""
    if not query or not query.strip():
        return "Error: arXiv query is empty."

    try:
        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []

        for paper in client.results(search):
            papers.append(f"- {paper.title} ({paper.published.year})")

        return "\n".join(papers) or "No papers found."

    except Exception as e:
        return f"Arxiv error: {e}"


# ── new tool: calculator ────────────────────────────────────────────────

# Quiz Q3: only digits and + - * / ( ) . are allowed, so there's nothing an
# injected expression could do inside eval() besides arithmetic.
_CALC_PATTERN = re.compile(r"^[0-9+\-*/(). \t]+$")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "3.5 + 2 * (10 - 4)".
    Only supports numbers and + - * / ( ) . — use this for any math that
    does NOT involve letter grades (use gpa_calculator for grades/GPA).
    """
    if not expression or not expression.strip():
        return "Error: expression is empty."

    expression = expression.strip()

    if len(expression) > 200:
        return "Error: expression is too long."

    if not _CALC_PATTERN.match(expression):
        return (
            "Error: only digits and + - * / ( ) . are allowed. This guard "
            "is what stops unsafe input from reaching eval()."
        )

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Calculator error: {e}"

    return str(result)


# ── new tool: gpa_calculator ────────────────────────────────────────────

GRADE_POINTS = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
}


@tool
def gpa_calculator(courses: str) -> str:
    """Calculate a GPA from letter grades and credit hours.
    Input format: comma-separated "grade:credits" pairs, e.g. "A:3, B+:4, A-:3".
    Valid grades: A+ A A- B+ B B- C+ C C- D+ D D- F (4.0 scale).
    Use this whenever the student asks about their GPA or grade average.
    """
    if not courses or not courses.strip():
        return "Error: no courses given. Use the format 'A:3, B+:4'."

    total_points = 0.0
    total_credits = 0.0
    breakdown = []

    for entry in courses.split(","):
        entry = entry.strip()
        if not entry:
            continue

        if ":" not in entry:
            return f"Error: '{entry}' is not in 'grade:credits' format."

        grade_str, credit_str = entry.split(":", 1)
        grade_str = grade_str.strip().upper()
        credit_str = credit_str.strip()

        if grade_str not in GRADE_POINTS:
            return f"Error: '{grade_str}' is not a recognized grade. Use A+ through F."

        try:
            credits = float(credit_str)
        except ValueError:
            return f"Error: '{credit_str}' is not a valid number of credits."

        if credits <= 0:
            return f"Error: credits must be positive (got {credits} for {grade_str})."

        points = GRADE_POINTS[grade_str] * credits
        total_points += points
        total_credits += credits
        breakdown.append(f"{grade_str} ({credits:g} credits = {points:.2f} pts)")

    if total_credits == 0:
        return "Error: no valid courses parsed."

    gpa = total_points / total_credits
    return f"GPA: {gpa:.3f} across {total_credits:g} credits. Breakdown: " + "; ".join(breakdown)


# ── new tool: course_search (RAG over the student's own PDFs) ──────────

_embeddings = OllamaEmbeddings(model=EMBED_MODEL)

_course_db = Chroma(
    collection_name=COURSE_COLLECTION,
    embedding_function=_embeddings,
    persist_directory=COURSE_CHROMA_DIR,
)


@tool
def course_search(question: str) -> str:
    """Search the student's OWN course PDFs (syllabus, lecture notes) that
    were ingested with ingest.py. Try this FIRST for anything that could be
    in a syllabus: grades, project/exam weights or percentages, due dates,
    attendance or late policy, topics covered, instructor contact info —
    even if the student's question doesn't say "syllabus" or "class." Use
    web_search only for questions about the outside world that have
    nothing to do with the student's own course.
    """
    if not question or not question.strip():
        return "Error: question is empty."

    try:
        results = _course_db.similarity_search(question, k=3)
    except Exception as e:
        return f"course_search error: {e}"

    if not results:
        return "No matching passage found in the ingested course PDFs. Have you run ingest.py yet?"

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        page_note = f", page {page + 1}" if isinstance(page, int) else ""
        formatted.append(f"[{source}{page_note}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(formatted)


# ── semantic memory (automatic recall, not an agent-callable tool) ─────
# Conversation memory (below, via checkpointer) replays recent turns in
# order. Semantic memory instead retrieves only facts related to THIS
# question, by embedding similarity — the distinction quiz Q11 is testing.

_memory_db = Chroma(
    collection_name=MEMORY_COLLECTION,
    embedding_function=_embeddings,
    persist_directory=MEMORY_CHROMA_DIR,
)


def recall_relevant_facts(query, k=2):
    try:
        docs = _memory_db.similarity_search(query, k=k)
    except Exception:
        return []
    return [doc.page_content for doc in docs]


RECALL_QUESTION_HINTS = [
    "mentioned before", "said before", "told you before", "earlier",
    "did i mention", "did i say", "did i tell you", "what did i",
    "remember", "recall",
]


def looks_like_recall_question(text):
    lowered = text.lower()
    return any(hint in lowered for hint in RECALL_QUESTION_HINTS)


def remember_fact(user_prompt, final_answer):
    try:
        _memory_db.add_texts(
            texts=[f"Student asked: {user_prompt}\nCoach answered: {final_answer}"],
            metadatas=[{"timestamp": datetime.now(timezone.utc).isoformat()}],
        )
    except Exception as e:
        print(f"(semantic memory not saved: {e})")


# ── agent ────────────────────────────────────────────────────────────────

tools = [web_search, arxiv_search, calculator, gpa_calculator, course_search]

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Conversation memory: LangGraph persists state per thread_id, so we only
# ever need to pass the NEW message each turn — the checkpointer restores
# everything before it.
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    system_prompt=(
        # TODO: replace this paragraph with your actual v1 COACH_PROMPT
        # persona text. Keep the tool-usage rules below it.
        "You are a helpful academic assistant. "
        "You have five tools: web_search, arxiv_search, calculator, gpa_calculator, and course_search. "
        "Use arxiv_search only for academic papers. "
        "Use web_search only for current or recent facts. "
        "Use calculator for any arithmetic that does not involve letter grades. "
        "Use gpa_calculator whenever the student asks about their GPA or grade average. "
        "Use course_search for ANY question that could be answered by a syllabus: "
        "grades, project or exam weights/percentages, due dates, attendance or late "
        "policy, topics covered, or instructor/office-hour info — even if the student "
        "doesn't say the word 'syllabus' or 'class.' Try course_search before assuming "
        "something needs a web search. "
        "If the message includes a 'Relevant facts from earlier in our conversations' "
        "section, that is information the student already told you — answer directly "
        "from it instead of calling a tool, unless the student is also asking for "
        "something new that isn't in those facts. For example, if the student asks "
        "what they mentioned before and the facts already answer that, just answer — "
        "do not call course_search or web_search to look it up again. "
        "For greetings like hello, answer normally without tools. "
        "Never invent a tool name. "
        "If you use a tool, your final answer must be based only on the tool result. "
        "Do not ignore the tool result. "
        "Do not use outdated knowledge when the tool gives current information. "
        "Do not mention your knowledge cutoff. "
        "Your final answer should be plain English, not JSON."
    )
)


# ── observability: console trace + JSONL log ────────────────────────────

def get_current_turn_messages(result):
    """result['messages'] now holds the WHOLE thread (memory persists it),
    so isolate just what happened since the newest human message."""
    messages = result["messages"]
    last_human_index = 0
    for i, msg in enumerate(messages):
        if msg.__class__.__name__ == "HumanMessage":
            last_human_index = i
    return messages[last_human_index:]


def print_what_happened(result):
    print("\n--- WHAT HAPPENED ---")

    tool_was_used = False

    for msg in get_current_turn_messages(result):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tool_was_used = True
                print(f"\nAI selected tool: {call['name']}")
                print(f"Tool input: {call['args']}")

        elif msg.__class__.__name__ == "ToolMessage":
            print("\nTool result:")
            print(msg.content)

    if not tool_was_used:
        print("No tool was used.")


def get_tool_results(result):
    return [
        msg.content
        for msg in get_current_turn_messages(result)
        if msg.__class__.__name__ == "ToolMessage"
    ]


def log_interaction(user_prompt, result, final_answer):
    """Guardrails & observability: every tool call/result/answer, one JSON
    object per line, so the run can be audited later (quiz Q12)."""
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []

    for msg in get_current_turn_messages(result):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                records.append({
                    "timestamp": timestamp,
                    "user_prompt": user_prompt,
                    "event": "tool_call",
                    "tool": call["name"],
                    "input": call["args"],
                })
        elif msg.__class__.__name__ == "ToolMessage":
            records.append({
                "timestamp": timestamp,
                "user_prompt": user_prompt,
                "event": "tool_result",
                "tool": getattr(msg, "name", None),
                "output": msg.content,
            })

    records.append({
        "timestamp": timestamp,
        "user_prompt": user_prompt,
        "event": "final_answer",
        "answer": final_answer,
    })

    try:
        with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"(logging failed: {e})")


# ── grounding fallback ───────────────────────────────────────────────────
# Generalized version of the old hardcoded "Donald Trump" check: if a tool
# actually ran and the model's answer still hedges like it didn't trust the
# result, prefer the tool's own output instead of one hardcoded fact.

UNGROUNDED_PHRASES = [
    "knowledge cutoff",
    "training data",
    "i don't have real-time",
    "i do not have real-time",
    "i don't have access to real-time",
    "as an ai",
    "i can't provide current",
    "i cannot provide current",
    "up to my last update",
    "i'm not able to browse",
]


def get_final_answer(user_prompt, result):
    final_answer = result["messages"][-1].content
    tool_outputs = get_tool_results(result)

    if tool_outputs:
        lowered = final_answer.lower()
        if any(phrase in lowered for phrase in UNGROUNDED_PHRASES):
            grounded = "\n\n".join(tool_outputs)
            return (
                "Based on the tool result, here is the current information:\n\n"
                + grounded
            )

    return final_answer


# ── web interface (Gradio) ──────────────────────────────────────────────

import gradio as gr


def respond(message, history):
    """Gradio calls this for every message. `history` is Gradio's own chat
    history — it's ignored here since the LangGraph checkpointer (keyed by
    SESSION_THREAD_ID) already handles conversation memory; using both
    would just duplicate the same job.
    """
    cleaned = message.lower().strip()

    if cleaned in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]:
        print("\n--- WHAT HAPPENED ---")
        print("No tool was used.")
        print("\n--- FINAL ANSWER ---")
        print("Hello! How can I help you?")
        print("\n" + "=" * 60 + "\n")
        return "Hello! How can I help you?"

    # Semantic memory: pull related past facts before asking the agent.
    recalled_facts = recall_relevant_facts(message)
    if recalled_facts:
        print("\n--- MEMORY RECALLED ---")
        for fact in recalled_facts:
            print(f"- {fact}")

    # Obvious "what did I say before" questions get answered straight from
    # memory in code — llama3.2 doesn't reliably skip a tool call on its
    # own even when told to, so we don't leave that decision up to it here.
    if recalled_facts and looks_like_recall_question(message):
        final_answer = "Here's what's on record from earlier:\n\n" + "\n\n".join(recalled_facts)
        print("\n--- FINAL ANSWER (answered directly from memory, no agent call) ---")
        print(final_answer)
        print("\n" + "=" * 60 + "\n")
        return final_answer

    if recalled_facts:
        memory_note = "Relevant facts from earlier in our conversations:\n" + "\n".join(
            f"- {fact}" for fact in recalled_facts
        )
        augmented_prompt = f"{memory_note}\n\nStudent: {message}"
    else:
        augmented_prompt = message

    config = {
        "configurable": {"thread_id": SESSION_THREAD_ID},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": augmented_prompt}]},
            config=config,
        )

        print_what_happened(result)

        final_answer = get_final_answer(message, result)
        log_interaction(message, result, final_answer)

        print("\n--- FINAL ANSWER ---")
        print(final_answer)

        # Only remember this exchange if it wasn't already just a restatement
        # of something we recalled — otherwise recall-type questions keep
        # re-storing near-duplicates of themselves.
        if not recalled_facts:
            remember_fact(message, final_answer)

        return final_answer

    except Exception as e:
        print("\nERROR:")
        print(e)
        return f"Something went wrong on my end: {e}"


demo = gr.ChatInterface(
    fn=respond,
    title="The Agent Coach",
    description=(
        "Ask about your course (from your ingested syllabus), do math, check your "
        "GPA, or search the web/arXiv. Full tool traces still print in this terminal."
    ),
    examples=[
        "What's the late policy for projects in my syllabus?",
        "What is 47 times 23, minus 15?",
        "I got an A in a 3-credit class and a B+ in a 4-credit class. What's my GPA?",
    ],
)


if __name__ == "__main__":
    print("Academic Agent is ready. Opening in your browser...")
    demo.launch()