# Divides the query into subquestions, proceeds to routing
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline, Document

def decompose_query(query: str) -> list[Document]:
    template = """Generate sub-questions related to the input question. \n
    The goal is to break down the input into a set of sub-problems/sub-questions strictly related to the original question that can be answered in isolation without any context. \n
    Reply with just the questions, unnumbered, separated with newlines\n
    Generate 3 search queries related to: {{question}}"""

    builder = PromptBuilder(template=template)

    generator = OllamaGenerator(model="llama3.1",
                                url="http://localhost:11434",
                                generation_kwargs={
                                    "num_predict": 100,
                                    "temperature": 0.9,
                                })

    pipe = Pipeline()
    pipe.add_component("builder", builder)
    pipe.add_component("llm", generator)
    pipe.connect("builder", "llm")

    sub_questions = pipe.run({"builder": {"question" : query}})
    # print(sub_questions["llm"]["replies"][0])

    return [Document(content=i) for i in sub_questions["llm"]["replies"][0].split("\n")]


if __name__ == '__main__':
    a = decompose_query("Who won more Grand Slams, Iga Swiatek or Serena Williams?")
    # print("\n")

    for i in a:
        print(i)
