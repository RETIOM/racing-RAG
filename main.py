from retrieve import retrieve_context, Node
from ingest import clean_abbrev
from HyDE import generate_regulations
import pickle
import gradio as gr
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.readers import ExtractiveReader
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def wrapper(query: str, do_generate: bool) -> str:
    f = open("data/rules_native_raptor.dat", 'rb')
    f.seek(0)
    rules = pickle.load(f)
    f.close()
    root=rules[0]

    template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}"""
    clean_query = clean_abbrev(query)
    hyde = generate_regulations(clean_query)

    # embedder = OllamaTextEmbedder()

    builder = PromptBuilder(template=template, required_variables=["documents", "question"])
    context = retrieve_context(root, hyde, 3)
    generator = GoogleAIGeminiGenerator()

    pipe = Pipeline()
    pipe.add_component("builder", builder)
    pipe.add_component("llm", generator)
    pipe.connect("builder", "llm")

    result = pipe.run({"builder" : {"documents" : context, "question" : query}})["llm"]["replies"][0]

    if do_generate:
        text = result
    else:
        text = ''
        for i in context:
            text += f"{i.content}\n\n"

    return text


demo = gr.Interface(fn=wrapper, inputs=["text", "checkbox"], outputs="textbox")

if __name__ == "__main__":
    demo.launch(share=True)