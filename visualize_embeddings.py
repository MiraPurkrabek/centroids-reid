import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import plotly
import plotly.graph_objs as go

# EMBEDDINGS_PATH="data/embeddings/custom_market_dataset_unique/query/embeddings.npy"
# PATHS_PATH="data/embeddings/custom_market_dataset_unique/query/paths.npy"
EMBEDDINGS_PATH="data/embeddings/U19_SKV_MIL_08_01_2022_1st_period_synced_1min_all/embeddings.npy"
PATHS_PATH="data/embeddings/U19_SKV_MIL_08_01_2022_1st_period_synced_1min_all/paths.npy"

def load_embeddings():
    return np.load(EMBEDDINGS_PATH), np.load(PATHS_PATH)

def extract_ids_from_paths(paths):
    ids = np.ones(paths.shape) * -1
    for pi, pth in enumerate(paths):
        ids[pi] = os.path.basename(pth).split("_")[0]
    return ids

def id_to_RGB(n, normalized=False):
    n = ((n ^ n >> 15) * 2246822519) & 0xffffffff
    n = ((n ^ n >> 13) * 3266489917) & 0xffffffff
    n = (n ^ n >> 16) >> 8
    
    if normalized:
        return list(map( lambda x: x/255 , n.to_bytes(3, 'big') ))
    else:
        return list(n.to_bytes(3, 'big') )

def display_tsne_scatterplot(
    embeddings,
    ids=None,
    method="TSNE",
    show_points=1000,
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

    # Filter out values with ID -1
    filtered_embeddings = embeddings[ids != -1, :]
    filtered_ids = ids[ids != -1]

    row_idxs = np.random.choice(filtered_embeddings.shape[0], show_points, replace=False)

    if filtered_ids is not None:
        filtered_ids = filtered_ids[row_idxs]

    if method.upper() == "TSNE":
        compact_embeddings = TSNE(
            n_components = num_dims,
            random_state = 0,
            perplexity = perplexity,
            learning_rate = learning_rate,
            n_iter = iteration,
            init = 'pca',
        ).fit_transform(filtered_embeddings[row_idxs, :])
    else:
        compact_embeddings = filtered_embeddings[row_idxs, :2]

    print(compact_embeddings.shape)

    data = []

    if num_dims == 2:
        # for pi, pt in enumerate(compact_embeddings):
        # if ids is None:
        #     color = "blue"
        # else:
        #     color = id_to_RGB(int(ids[pi]))

        trace = go.Scatter(
                x = compact_embeddings[:, 0], 
                y = compact_embeddings[:, 1], 
                text = list(map(lambda x: str(int(x)), filtered_ids)),
                # name = ids,
                textposition = "top center",
                textfont_size = 20,
                mode = 'markers+text',
                marker = {
                    'size': 10,
                    'opacity': 0.8,
                    'color': filtered_ids
                }
        )
        data.append(trace)

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )

    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_html(
        "scatter_img.html",
        full_html=False,
        include_plotlyjs='cnd',
    )

def display_tsne_mean_id_scatterplot(
    embeddings,
    ids,
    method="TSNE",
    num_dims=2,
    perplexity = 5,
    learning_rate = 500,
    iteration = 1000,
):
    assert num_dims == 2 or num_dims == 3

    show_points = np.unique(ids[ids != -1]).shape[0]

    print("Displaying the scatterplot in {:d}D for {:d} points".format(
        num_dims,
        show_points
    ))

    means = np.zeros((show_points, embeddings.shape[1]))

    for mc_id in range(show_points):
        means[mc_id] = np.mean(embeddings[ids == mc_id, :], axis=0)

    if method.upper() == "TSNE":
        compact_embeddings = TSNE(
            n_components = num_dims,
            random_state = 0,
            perplexity = perplexity,
            learning_rate = learning_rate,
            n_iter = iteration,
            init = 'pca',
        ).fit_transform(means)
    else:
        compact_embeddings = means[:, :2]

    print(compact_embeddings.shape)

    data = []

    if num_dims == 2:
        # for pi, pt in enumerate(compact_embeddings):
        # if ids is None:
        #     color = "blue"
        # else:
        #     color = id_to_RGB(int(ids[pi]))

        trace = go.Scatter(
                x = compact_embeddings[:, 0], 
                y = compact_embeddings[:, 1], 
                text = list(map(lambda x: str(int(x)), range(show_points))),
                # name = ids,
                textposition = "top center",
                textfont_size = 20,
                mode = 'markers+text',
                marker = {
                    'size': 10,
                    'opacity': 0.8,
                    'color': list(range(show_points))
                }
        )
        data.append(trace)

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )

    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_html(
        "means_img.html",
        full_html=False,
        include_plotlyjs='cnd',
    )


if __name__=="__main__":

    print("Loading embeddings")
    embeddings, paths = load_embeddings()
    print("Embeddings loaded - {}".format(embeddings.shape))

    ids = extract_ids_from_paths(paths)

    # print(ids)

    # display_tsne_scatterplot(
    #     embeddings,
    #     ids=ids,
    # )

    display_tsne_mean_id_scatterplot(
        embeddings,
        ids=ids,
    )
