# hello_web.py — minimal Gradio web app to confirm the UI works
import gradio as gr

def greet(name):
    return f"Hello, {name}! Your web interface works."

gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Hello Web Test"
).launch()
