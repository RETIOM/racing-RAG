# Computes correct answer
from retrieve import retrieve_context, Node
from HyDE import generate_regulations
import pickle
import gradio as gr
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

def wrapper(query: str, do_generate: bool) -> str:
    f = open("rules_store.dat", 'rb')
    f.seek(0)
    rules = pickle.load(f)
    root=rules[0]
    # hyde = generate_regulations(query)
    embedder = OllamaTextEmbedder()
    hyde = embedder.run(text=query)["embedding"]
    context = retrieve_context(root, hyde, 3)

    text = ''
    for i in context:
        text += f"{i}\n\n"
    return text


demo = gr.Interface(fn=wrapper, inputs=["text", "checkbox"], outputs="textbox")

if __name__ == "__main__":
    demo.launch()