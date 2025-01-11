from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline

from numpy import array, mean

def generate_regulations(query: str, n_iter: int) -> list[str]:
    template = """You are a Formula Student Germany rulemaker. \n
    Create a rule related to the following question: {{question}} \n
    Answer with only the regulatory part, do not label the rule"""

    embedder = OllamaTextEmbedder()

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

    regs = []
    for i in range(n_iter):
        regs.append(pipe.run({"builder": {"question" : query}})["llm"]["replies"][0])
        print(regs[-1])

    regs_enc = array([embedder.run(text=i)["embedding"] for i in regs])
    avg_enc = mean(regs_enc, axis=0)
    vector = avg_enc.reshape((1, len(avg_enc)))[0].tolist()

    return vector


if __name__ == '__main__':
    a = generate_regulations("what is the max engine capacity", 5)
    print(a)

