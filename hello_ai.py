# hello_ai.py — confirm Python can talk to your local LLM
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
print(llm.invoke("Say hello in one short sentence.").content)
