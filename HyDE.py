from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import Pipeline

from numpy import array, mean

def generate_regulations(query: str, n_iter=5) -> list[float]:
    template = """You are a Formula Student Germany rulemaker. \n
    Create a rule related to the following question: {{question}} \n
    Answer with only the regulatory part, do not label the rule or add any additional comment"""

    embedder = OllamaTextEmbedder()

    builder = PromptBuilder(template=template)

    generator = OllamaGenerator(model="llama3.1",
                                url="http://localhost:11434",
                                generation_kwargs={
                                    "num_predict": 100,
                                    "temperature": 0.2,
                                })

    pipe = Pipeline()
    pipe.add_component("builder", builder)
    pipe.add_component("llm", generator)
    pipe.connect("builder", "llm")

    regs = [pipe.run({"builder": {"question" : query}})["llm"]["replies"][0] for i in range(n_iter)]
    # print(regs)
    regs_enc = array([embedder.run(text=i)["embedding"] for i in regs])
    avg_enc = mean(regs_enc, axis=0)
    vector = avg_enc.reshape((1, len(avg_enc)))[0].tolist()

    return vector


if __name__ == '__main__':
    a = generate_regulations("What is the maximum engine capacity allowed in Formula Student competitions?", 5)
    print(a)

