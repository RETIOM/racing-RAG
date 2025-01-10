# Encoding, maybe document preprocessing
import PyPDF2
import re

from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline


class Node:
    def __init__(self, content=None, vec=None, children=[]):
        self.content = content
        self.vec = vec
        self.children = children
    def add_child(self, child):
        self.children.append(child)


def prep_pdf(path: str) -> str:
    pdf = open(path, 'rb')

    pdf_reader = PyPDF2.PdfReader(pdf)

    pre_text = ''

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pre_text += page.extract_text()

    # Processing
    text = re.sub("Formula Student Rules 2025($| Version: 1.0 [0-9]+ of [0-9]+)","", pre_text, flags=re.M)
    text = text.replace("•","-")

    return text

def create_tree(text: str) -> Node:
    top = re.split("^[A-Z]+ [A-Z ]+$", text, flags=re.M)[1:]
    first_layer = [] # SECTION NODES
    for section in top:
        second_layer = [] # SUBSECTION NODES
        sec_summary, sec_embedding = embed_summarize(section, True)
        umiddle = re.split("^[A-Z]+ [1-9] ", section, flags=re.M)[1:]
        for subsection in umiddle:
            third_layer = [] # SUBSECTION NODES
            sub_summary, sub_embedding = embed_summarize(subsection, True)
            lmiddle = re.split("^[A-Z]+[1-9]+.[1-9]+ ", subsection, flags=re.M)[1:]
            for subsubsection in lmiddle:
                subsub_summary, subsub_embedding = embed_summarize(subsubsection, True)
                bottom = re.split("^[A-Z]+[1-9]+.[1-9]+.[1-9]+ ", subsubsection, flags=re.M)[1:]
                leaves = [Node(leaf, embed_summarize(leaf, False)) for leaf in bottom]
                third_layer.append(Node(subsub_summary, subsub_embedding, leaves))
            second_layer.append(Node(sub_summary, sub_embedding, third_layer))
        first_layer.append(Node(sec_summary, sec_embedding, second_layer))

    return Node(children=first_layer)

def embed_summarize(text: str, summarize: bool):
    template = '''Summarize the following text: {{text}}\n Output should be just the summary in a single paragraph with no pretext.'''
    print("HEY")
    builder = PromptBuilder(template=template)
    generator = OllamaGenerator(model="llama3.1",
                                url="http://localhost:11434",
                                generation_kwargs={
                                    "num_predict": 100,
                                    "temperature": 0.9,
                                })
    embedder = OllamaTextEmbedder()

    pipe = Pipeline() # Pomyslec o czyszczeniu zanim damy do streszczenia
    pipe.add_component("builder", builder)
    pipe.add_component("llm", generator)
    pipe.connect("builder", "llm")

    if summarize:
        summary = pipe.run({"builder":{"text":text}})["llm"]["replies"][0]
        embedding = embedder.run(text=summary)["embedding"][0]
        return summary, embedding
    else:
        embedding = embedder.run(text=text)["embedding"][0]
        return embedding


if __name__ == '__main__':
    # prep_pdf("Rules.pdf")
    text = '''EV E LECTRIC VEHICLES
EV 1 D EFINITIONS
EV1.1 Tractive System
EV1.1.1 Tractive System (TS) – every part that is electrically connected to the motors and TS
accumulators. The LVS may be supplied by the TS if a galvanic isolation between both
systems is ensured.
EV1.1.2 TS enclosures – every housing or enclosure that contains parts of the TS.
EV1.2 Electrical
EV1.2.1 Galvanic Isolation – two electric circuits are defined as galvanically isolated if all of the
following conditions are true:
-The resistance between both circuits is ≥500Ω/V, related to the maximum TS voltage
of the vehicle, at a test voltage of maximum TS voltage or 250 V , whichever is higher.
-The isolation test voltage RMS, AC for 1 min , between both circuits is higher than
three times the maximum TS voltage or 750 V, whichever is higher.
-The working voltage of the isolation barrier, if specified in the datasheet, is higher than
the maximum TS voltage.
Capacitors that bridge galvanic isolation must be class-Y capacitors.
EV1.2.2 High Current Path – any path of a circuitry that, during normal operation, carries more than
1 A.
EV 2 E LECTRIC POWERTRAIN
EV2.1 Motors
EV2.1.1 Only electric motors are allowed.
EV2.1.2 Motor attachments must follow T10.
EV2.1.3 Motor casings must follow T7.3.
EV2.1.4 The motor(s) must be connected to the TS accumulator through a motor controller.
EV2.2 Power Limitation
EV2.2.1 The TS power at the outlet of the TSAC must not exceed 80 kW.'''
    a = create_tree(text)

    while len(a.children) > 0:
        print(a.children)
        a = a.children[0].content
    # embed = embed_summarize("Motor attachments must follow T10.", False)
    pass
