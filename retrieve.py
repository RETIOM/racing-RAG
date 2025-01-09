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

def traverse_tree(tree, query: list[float], k: int) -> list[Document]:
    s_0 = tree[0] # first layer of tree/temporary mark
