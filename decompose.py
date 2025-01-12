# Divides the query into subquestions, proceeds to routing
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline

'''!Work out if can be used as component!'''
'''!check if each query can be passed into pipeline separately!'''
"""Generate sub-questions related to the input question. \n
    The goal is to break down the input into a set of sub-problems/sub-questions strictly related to the original question that can be answered in isolation without any context. \n
    Reply with just the questions, unnumbered, separated with newlines\n
    Generate 3 search queries related to: {{question}}"""

'''You are a helpful AI Assistant who generates sub-questions related to the input question. \n
    The original question is related to the Formula Student competition, so should be the sub-questions. \n
    The goal is to break down the input into a set of 3 sub-problems/sub-questions strictly related to the original question that can be answered in isolation without any context. \n
    Reply with just the questions, unnumbered, separated with newlines\n
    Generate 3 questions related to: {{question}}'''

def decompose_query(query: str) -> list[str]:
    template = """You are a helpful AI Assistant who generates sub-questions based on the input question.\n
    The original question is related to the Formula Student competition, so should be the sub-questions.\n
    Break the following question down into at most 3 lowest-level subquestions, if they exist: {{question}} \n
    Reply with just the questions, unnumbered separated with newlines\n
    Do not break up already low level questions."""

    builder = PromptBuilder(template=template)

    generator = OllamaGenerator(model="llama3.1",
                                url="http://localhost:11434",
                                generation_kwargs={
                                    "num_predict": 100,
                                    "temperature": 0,
                                })

    pipe = Pipeline()
    pipe.add_component("builder", builder)
    pipe.add_component("llm", generator)
    pipe.connect("builder", "llm")

    sub_questions = pipe.run({"builder": {"question" : query}})

    return [i for i in sub_questions["llm"]["replies"][0].split("\n")]

# make it split into lowest  level
if __name__ == '__main__':
    a = decompose_query("what is the max engine capacity and what fuels are allowed")
    for i in a:
        print(i)
