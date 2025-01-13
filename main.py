from retrieve import retrieve_context, Node
from ingest import clean_abbrev
from HyDE import generate_regulations
import pickle
import gradio as gr
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.readers import ExtractiveReader


# ADD QUERY TRANSLATION FROM ABBREVIATIONS!!
def wrapper(query: str, do_generate: bool) -> str:
    f = open("rules_store.dat", 'rb')
    f.seek(0)
    rules = pickle.load(f)
    f.close()
    root=rules[0]

    # hyde = generate_regulations(query)
    embedder = OllamaTextEmbedder()
    reader = ExtractiveReader(model="deepset/tinyroberta-squad2")

    # clean_query = clean_abbrev(query) # Uncomment to add query translation

    hyde = embedder.run(text=query)["embedding"]
    context = retrieve_context(root, hyde, 3)

    reader.warm_up()
    result = reader.run(query=query, documents=context)["answers"][0]

    if do_generate:
        text = f"Answer: {result.data} ({round(result.score*100,2)}%)\n\nDocument: {result.document.content}"
    else:
        text = ''
        for i in context:
            text += f"{i.content}\n\n"

    return text


demo = gr.Interface(fn=wrapper, inputs=["text", "checkbox"], outputs="textbox")

if __name__ == "__main__":
    demo.launch()