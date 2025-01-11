import numpy as np
import pickle
from ingest import Node

from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def traverse_tree(root: Node, query: list[float], k: int) -> list[str]:
    best_nodes = []
    s_current = root.children
    for layer in range(4):   # 4 is num_layers, possibly replace with while children
        top_k = []
        for node in s_current:
            score = cosine_similarity(query, node.vec)
            top_k.append((node,score))
        selected = sorted(top_k, key=lambda x: x[1], reverse=True)[:k] # could change to just nodes if not sorting in return
        best_nodes += selected
        s_current = []
        for pair in selected:
            for children in pair[0].children:
                s_current.append(children)
        # s_current = [pair[0].children for pair in selected] ; Consider fixing
    return [x[0].content for x in sorted(best_nodes, key=lambda x:x[1])][-5:]

def rebuild_tree():
    pass

if __name__ == '__main__':
    f = open('rules_store.dat', 'rb')
    f.seek(0)
    tree = pickle.load(f)
    root = tree[0]
    q_text = "what does it mean when the tsal is red"
    embedder = OllamaTextEmbedder()
    query = embedder.run(text=q_text)["embedding"]

    docs = traverse_tree(root, query, 3)

    for i in docs[::-1]:
        print(i)