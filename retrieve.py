from haystack import Document
import numpy as np
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

# TO BE MODIFIED ACCORDING TO DATABASE TREE STRUCTURE, also adapt from paper
def traverse_tree(tree, query: list[float], k: int) -> list[Document]:
    documents = []
    s_current = tree[0] # first node(?) of tree/temporary
    for layer in range(4):   # 4 is temporary, arbitrary; replace with number of layers
        top_k = []
        for node in s_current:
            score = cosine_similarity(query, node)
            top_k.append((node,score))
        documents.append(sorted(top_k, key=lambda x: x[1], reverse=True)[:k]) # zmienic na slownik z powrotem, z podzialem na warstwy
        s_current = layer
    return [doc[0] for doc in sorted(documents, key=lambda x: x[1])]

def rebuild_tree():
    pass