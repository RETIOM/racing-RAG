# Encoding, maybe document preprocessing
import PyPDF2
import re

import pickle

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

# substitute all abbrev
def prep_pdf(path: str) -> str:
    abbreviations = [
        ("AIP", "Anti Intrusion Plate"),
        ("AIR", "Accumulator Isolation Relay"),
        ("AMI", "Autonomous Mission Indicator"),
        ("AMS", "Accumulator Management System"),
        ("APPS", "Accelerator Pedal Position Sensor"),
        ("AS", "Autonomous System"),
        ("ASB", "Autonomous System Brake"),
        ("ASES", "Accumulator Structural Equivalency Spreadsheet"),
        ("ASF", "Autonomous System Form"),
        ("ASMS", "Autonomous System Master Switch"),
        ("ASR", "Autonomous System Responsible"),
        ("ASRQ", "ASR Qualification"),
        ("ASSI", "Autonomous System Status Indicator"),
        ("BOM", "Bill of Material"),
        ("BOTS", "Brake Over-Travel Switch"),
        ("BPP", "Business Plan Presentation Event"),
        ("BPPV", "Business Plan Pitch Video"),
        ("BSPD", "Brake System Plausibility Device"),
        ("CCBOM", "Costed Carbonized Bill of Material"),
        ("CGS", "Compressed Gas System"),
        ("CO2e", "Carbon Dioxide Equivalents"),
        ("CRD", "Cost Report Documents"),
        ("CV", "Internal Combustion Engine Vehicle"),
        ("DC", "Driverless Cup"),
        ("DI", "Direct Injection"),
        ("DNF", "Did Not Finish"),
        ("DOO", "Down or Out"),
        ("DQ", "Disqualified"),
        ("DSS", "Design Spec Sheet"),
        ("DV", "Driverless"),
        ("EBS", "Emergency Brake System"),
        ("EDR", "Engineering Design Report"),
        ("EI", "Flexural Rigidity"),
        ("ESF", "Electrical System Form"),
        ("ESO", "Electrical System Officer"),
        ("ESOQ", "Electrical System Officer Qualification"),
        ("ETC", "Electronic Throttle Control"),
        ("EV", "Electric Vehicle"),
        ("GWP", "Global Warming Potential"),
        ("HPI", "High Pressure Injection"),
        ("HSC", "Hybrid Storage Container"),
        ("HSF", "Hybrid System Form"),
        ("HV", "High Voltage"),
        ("HVD", "High Voltage Disconnect"),
        ("HY", "Combustion Hybrid Vehicle"),
        ("IA", "Impact Attenuator"),
        ("IAD", "Impact Attenuator Data"),
        ("IMD", "Insulation Monitoring Device"),
        ("LCA", "Life Cycle Assessment"),
        ("LPI", "Low Pressure Injection"),
        ("LV", "Low Voltage"),
        ("LVMS", "Low Voltage Master Switch"),
        ("LVS", "Low Voltage System"),
        ("OC", "Off-Course"),
        ("R2D", "Ready-to-drive"),
        ("RES", "Remote Emergency System"),
        ("SCS", "System Critical Signal"),
        ("SDC", "Shutdown Circuit"),
        ("SE3D", "Structural Equivalency 3D Model"),
        ("SES", "Structural Equivalency Spreadsheet"),
        ("SESA", "SES Approval"),
        ("TPS", "Throttle Position Sensor"),
        ("TS", "Tractive System"),
        ("TSAC", "Tractive System Accumulator Container"),
        ("TSAL", "Tractive System Active Light"),
        ("TSMP", "Tractive System Measuring Point"),
        ("TSMS", "Tractive System Master Switch"),
        ("USS", "Unsafe Stop"),
        ("VSV", "Vehicle Status Video")
    ]
    pdf = open(path, 'rb')

    pdf_reader = PyPDF2.PdfReader(pdf)

    pre_text = ''

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pre_text += page.extract_text()

    # Processing
    text = re.sub("Formula Student Rules 2025($| Version: 1.0 [0-9]+ of [0-9]+)","", pre_text, flags=re.M)
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    text = text.replace("•","-")
    text = text.replace("’","'")

    for element in abbreviations:
        text = re.sub(f" {element[0]}[ ,.s']", f" {element[1]} ", text, flags=re.M)

    return text

def create_tree(text: str) -> Node:
    top = re.split("^[A-Z]+ [A-Za-z ]+$", text, flags=re.M)[1:] # BIG section
    first_layer = [] # SECTION NODES
    for section in top:
        second_layer = [] # SUBSECTION NODES
        sec_summary, sec_embedding = embed_summarize(section, True)
        umiddle = re.split("^[A-Z]+ [0-9]", section, flags=re.M)[1:] # separate by <num>
        for subsection in umiddle:
            third_layer = [] # SUBSECTION NODES
            sub_summary, sub_embedding = embed_summarize(subsection, True)
            lmiddle = re.split("^[A-Z]+[0-9]+.[0-9]+ ", subsection, flags=re.M)[1:] # separate by <num>.<num>
            for subsubsection in lmiddle:
                subsub_summary, subsub_embedding = embed_summarize(subsubsection, True)
                bottom = re.split("^[A-Z]+[0-9]+.[0-9]+.[0-9]+ ", subsubsection, flags=re.M)[1:] # separate by <num>.<num>.<num>
                leaves = [Node(leaf.replace("\n", "", 1), embed_summarize(leaf, False)) for leaf in bottom] # ADD BACK
                third_layer.append(Node(subsub_summary, subsub_embedding, leaves))
            second_layer.append(Node(sub_summary, sub_embedding, third_layer))
        first_layer.append(Node(sec_summary, sec_embedding, second_layer))

    return Node(children=first_layer)

def embed_summarize(text: str, summarize: bool):
    template = '''Summarize the following text: {{text}}\n Output should be just the summary in a single paragraph with no pretext.'''
    # print("HEY")
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
        print("summarizing&embedding")
        summary = pipe.run({"builder":{"text":text}})["llm"]["replies"][0]
        embedding = embedder.run(text=summary)["embedding"]
        return summary, embedding
    else:
        print("embedding")
        embedding = embedder.run(text=text)["embedding"]
        return embedding


def collapse_tree(current_node: Node, tree: list) -> None:
    # Base case: if the node is None, return (no node to process)
    if len(current_node.children)==0:
        tree.append(current_node)
        return

    # Process the current node (you can customize this action)
    tree.append(current_node)

    # Recur on all the children of the current node
    for child in current_node.children:
        collapse_tree(child,tree)

def save_tree(root: Node) -> None:
    tree = []
    collapse_tree(root, tree)

    f = open('rules_store.dat', 'wb')
    pickle.dump(tree, f)
    f.close()


if __name__ == '__main__':
    rules =  prep_pdf("data/Rules-63-90.pdf")
