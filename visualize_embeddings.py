import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import plotly
import plotly.graph_objs as go

EMBEDDINGS_PATH="data/embeddings/custom_market_dataset_unique/query/embeddings.npy"
PATHS_PATH="data/embeddings/custom_market_dataset_unique/query/paths.npy"

def load_embeddings():
    return np.load(EMBEDDINGS_PATH), np.load(PATHS_PATH)

def extract_ids_from_paths(paths):
    ids = np.ones(paths.shape) * -1
    for pi, pth in enumerate(paths):
        ids[pi] = os.path.basename(pth).split("_")[0]
    return ids

def id_to_RGB(n):
    n = ((n ^ n >> 15) * 2246822519) & 0xffffffff
    n = ((n ^ n >> 13) * 3266489917) & 0xffffffff
    n = (n ^ n >> 16) >> 8
    return list(map( lambda x: x/255 , n.to_bytes(3, 'big') ))

def display_tsne_scatterplot(
    embeddings,
    ids=None,
    method="dummy",
    show_points=20,
    num_dims=2,
    perplexity = 5,
    learning_rate = 500,
    iteration = 1000,
):
    assert num_dims == 2 or num_dims == 3

    if show_points is None:
        show_points = int(embeddings.shape[0])

    print("Displaying the scatterplot in {:d}D for {:d} points".format(
        num_dims,
        show_points
    ))

    row_idxs = np.random.choice(embeddings.shape[0], show_points, replace=False)

    if ids is not None:
        ids = ids[row_idxs]

    if method.upper() == "TSNE":
        compact_embeddings = TSNE(
            n_components = num_dims,
            random_state = 0,
            perplexity = perplexity,
            learning_rate = learning_rate,
            n_iter = iteration
        ).fit_transform(embeddings[row_idxs, :])
    else:
        compact_embeddings = embeddings[row_idxs, :2]

    print(compact_embeddings.shape)

    if num_dims == 2:
        for pi, pt in enumerate(compact_embeddings):
            if ids is None:
                color = "blue"
            else:
                color = [id_to_RGB(int(ids[pi]))] 

            plt.scatter(
                pt[0],
                pt[1],
                c = color,
                alpha=0.5
            )

    plt.savefig("tmp_img.jpg")


if __name__=="__main__":

    print("Loading embeddings")
    embeddings, paths = load_embeddings()
    print("Embeddings loaded")

    ids = extract_ids_from_paths(paths)

    # print(ids)

    display_tsne_scatterplot(
        embeddings,
        ids=ids,
    )
