from haystack import Document
import numpy as np
# from ingest import Node
# Routes split questions across subteam databases

# Database encoded into tree structure; pipe: decompose -> HyDE -> Route -> Rerank

# create similarity assessment function(cosine similarity)

# in: embedded query, tree database, k-number of returned thingy
# out: list of retrieved documents

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# TO BE MODIFIED ACCORDING TO DATABASE TREE STRUCTURE, also adapt from paper !!!ADD TYPE TO ROOT
def traverse_tree(root, query: list[float], k: int) -> list[str]:
    best_nodes = []
    s_current = root.children
    for layer in range(4):   # 4 is num_layers, possibly replace with while children
        top_k = []
        for node in s_current:
            print(node)
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