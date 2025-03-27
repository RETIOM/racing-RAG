# Encoding, maybe document preprocessing
import PyPDF2
import re

import pickle

from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline
import os
import time


os.environ["GOOGLE_API_KEY"] = "AIzaSyDod0-UiNyMzNPQhpmHanN86GT0jrH8aGY"

class Node:
    def __init__(self, content=None, vec=None, children=[]):
        self.content = content
        self.vec = vec
        self.children = children
    def add_child(self, child):
        self.children.append(child)

# substitute all abbrev
def clean_abbrev(text: str) -> str:
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
    for element in sorted(abbreviations, key=lambda abbreviations : -len(abbreviations[0])):
        text = text.replace(element[0], element[1])
    return text

def prep_pdf(path: str) -> str:
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

    return clean_abbrev(text)

def create_tree(text: str) -> Node:
    top = re.split("^[A-Z]+ [A-Z ]+$", text, flags=re.M)[1:] # BIG section
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


# SPLIT TO LEAVES, GROUP SUMMARIZE REPEAT
def native_raptor(text: str) -> Node:
    # SPLITTING INTO LEAVES - to be improved in the future
    top = re.split("^[A-Z]+ [A-Z ]+$", text, flags=re.M)[1:]


def embed_summarize(text: str, summarize: bool):
    template = '''Here is a collection of rules from the Formula Student Germany competition.
    
The rules dictate the regulations students must obey while designing and creating their vehicle.

Respond with a single paragraph, without any pretext.
    
Give a detailed summary of the provided text:
{{text}}'''
    
    '''Create a detailed summary of the following text: {{text}}\n Output should be just the summary in a single paragraph with no pretext.'''
    # print("HEY")
    builder = PromptBuilder(template=template)
    # generator = OllamaGenerator(model="llama3.1",
    #                             url="http://localhost:11434",
    #                             generation_kwargs={
    #                                 "num_predict": 100,
    #                                 "temperature": 0,
    #                             })
    generator = GoogleAIGeminiGenerator(model="gemini-1.5-flash")
    embedder = OllamaTextEmbedder()

    pipe = Pipeline() # Pomyslec o czyszczeniu zanim damy do streszczenia
    pipe.add_component("builder", builder)
    pipe.add_component("llm", generator)
    pipe.connect("builder", "llm")

    if summarize:
        # print("summarizing&embedding")
        summary = pipe.run({"builder":{"text":text}})["llm"]["replies"][0]
        embedding = embedder.run(text=summary)["embedding"]
        # prevents throtttle
        # time.sleep(5)
        return summary, embedding
    else:
        print("embedding")
        embedding = embedder.run(text=text)["embedding"]
        return embedding

# Basic DFS to save tree into list
def collapse_tree(current_node: Node, tree: list) -> None:
    if len(current_node.children)==0:
        tree.append(current_node)
        return
    tree.append(current_node)
    for child in current_node.children:
        collapse_tree(child,tree)

# Calls collapse and pickles the output
def save_tree(root: Node, path: str) -> None:
    tree = []
    collapse_tree(root, tree)

    f = open(f'{path}.dat', 'wb')
    pickle.dump(tree, f)
    f.close()

# Wrapper for embedding (preps pdf, creates tree, saves)
def encode_pdf(input_path: str, output_path: str) -> None:
    rules = prep_pdf(input_path)
    root = create_tree(rules)
    save_tree(root, output_path)


if __name__ == '__main__':
    # rules =  prep_pdf("data/Rules-63-90.pdf")
    f = open("data/output_txt", 'rb')
    text = """CV4.1.2 The SDC is defined as a series connection of at least the LVMS, see T11.3, the BSPD, see
T11.6, three shutdown buttons, see T11.4, the BOTS, see T6.2 and the inertia switch, see
T11.5.
CV4.1.3 All circuits that are part of the SDC must be designed in a way, that in the de-energized/disconnected
state they open the SDC.
CV4.1.4 [HY ONLY ]The HSC AIR as per CV5 .2.2 must be part of the SDC in such a way that one
side of the relay coil is directly incorporated into the SDC and the other side is controlled by
the hybrid control system.
CV 5 H YBRID SYSTEM
CV5.1 Hybrid System General
CV5.1.1 Hybrid System – the hybrid storage container, motors and every part that is electrically
connected to them.
CV5.1.2 The hybrid system must be a LVS, T11.1 and T11.7 are applied for all hybrid system
components.
CV5.1.3 All electrical parts of the hybrid system except for ground terminals must be covered at least
according to IPxxB when energized.
CV5.1.4 Hybrid Storage Container (HSC) – the electric energy storage system, including the AIR and
overcurrent protection, that is used in the hybrid system.
CV5.1.5 Moving energy into the Hybrid Storage Container (HSC) from a different electric storage
system is prohibited during any dynamic event.
CV5.1.6 A firewall, see T4.8, must be present between the HSC and the fuel tank.
CV5 Hybrid System
CV5.1.7 The HSC must be positioned according to T11.7.2, all other hybrid system components must
be positioned within the surface envelope, see T1.1.18.
CV5.1.8 The high current path, see EV1.2.2, of the hybrid system must meet EV4.5.15.
CV5.1.9 Motors must meet EV2.1.
CV5.1.10 The hybrid system may only be activated when the combustion engine is running or during
engine start.
CV5.2 Hybrid Storage Container
CV5.2.1 The HSC must be attached to the primary structure, see T1.1.12, according to T9.3.1.
CV5.2.2 A disconnection mechanism, designed as an AIR must be integrated inside of the HSC,
disconnecting the positive pole of the HSC. The AIR must be compliant with EV5.6.3.
CV5.2.3 The maximum total weight of all elements in the hybrid system that store the electric energy,
e.g. battery cells or supercapacitors, including all casings and tabs that are integral to them,
is 3 kg.
CV5.2.4 Holes, both internal and external, in the HSC, are only allowed for the wiring harness,
ventilation, cooling, or fasteners. The total cutout area must be below 25 % of the area of the
respective single wall.
CV5.2.5 The HSC must be removable to be inspected at the mechanical inspection and it must be
possible to easily check the weight limit.
CV5.3 Hybrid System Form
CV5.3.1 A Hybrid System Form (HSF) has to be submitted using the HSF template.
CV5.3.2 The HSF template will be available on the competition website.
CV5.3.3 If no HSF is submitted, the team must not use the hybrid system at the competition. A5.4.2
will not be applied for the HSF.

EV E LECTRIC VEHICLES
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
1 A."""
    encode_pdf("data/Rules.pdf", "data/Rules")
    f.close()
    # native_raptor(text)
