import numpy as np
import torch


def get_top_k_frequent_tokens(tokenizer, dataset, k: int) -> tuple[list[int], list[str]]:
    """
    Compute the k most frequent tokens in the dataset.

    :param tokenizer: The tokenizer used to encode the text
    :param dataset: The dataset (HuggingFace Dataset or list of texts)
    :param k: Number of most frequent tokens to return
    :return: (token_ids, token_texts) for the top-k most frequent tokens
    """
    from collections import Counter

    # Assume dataset is a HuggingFace Dataset with 'input_ids' or a list of lists
    if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
        all_token_ids = [tid for seq in dataset["input_ids"] for tid in seq]
    else:
        # fallback: try to use 'text' column or treat as list of lists
        if hasattr(dataset, "column_names") and "text" in dataset.column_names:
            all_token_ids = [tid for text in dataset["text"] for tid in tokenizer.encode(text)]
        else:
            all_token_ids = [tid for seq in dataset for tid in seq]

    freq = Counter(all_token_ids)
    most_common = freq.most_common(k)
    token_ids = [tid for tid, _ in most_common]
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]
    return token_ids, token_texts


def get_token_embeddings(model, token_ids: list[int]) -> torch.Tensor:
    """
    Get the embeddings for the given token ids from the model's embedding layer.

    :param model: The trained model
    :param token_ids: List of token ids
    :return: Tensor of shape (len(token_ids), embedding_dim)
    """
    # Assumes model.criterion.model.module.word_embedding.weight
    embedding_matrix = model.criterion.model.module.word_embedding.weight
    token_ids_tensor = torch.tensor(token_ids, device=embedding_matrix.device)
    embeddings = embedding_matrix[token_ids_tensor]
    return embeddings.detach().cpu()


def plot_token_embeddings_2d(
    embeddings: torch.Tensor,
    token_texts: list[str],
    method: str = "tsne",
    save_path: str | None = None,
    random_state: int = 42,
    figsize: tuple[int, int] = (14, 12),
    title: str | None = None,
):
    """
    Project embeddings to 2D using t-SNE or UMAP and plot with token labels.

    :param embeddings: Tensor of shape (n_tokens, embedding_dim)
    :param token_texts: List of token strings
    :param method: 'tsne' or 'umap'
    :param save_path: Optional path to save the plot
    :param random_state: Random seed for reproducibility
    :param figsize: Figure size
    :param title: Optional plot title
    """
    import matplotlib.pyplot as plt

    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(token_texts) - 1))
        emb_2d = reducer.fit_transform(embeddings.numpy())
    elif method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        emb_2d = reducer.fit_transform(embeddings.numpy())
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    plt.figure(figsize=figsize)
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=40, alpha=0.7)
    for i, token in enumerate(token_texts):
        plt.text(emb_2d[i, 0], emb_2d[i, 1], token, fontsize=9, ha="center", va="center")
    plt.title(title or f"{method.upper()} of Top-{len(token_texts)} Token Embeddings")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def calculate_embedding_statistics(embeddings: torch.Tensor) -> dict:
    """
    Calculate various statistics for analyzing embedding space properties.

    :param embeddings: Tensor of shape (n_tokens, embedding_dim)
    :return: Dictionary containing the following statistics:
        - isotropy: Measure of uniformity of embedding directions (higher is better)
        - avg_norm: Average L2 norm of embeddings
        - norm_std: Standard deviation of L2 norms
        - mean_nearest_dist: Mean distance to nearest neighbor
        - nearest_dist_std: Standard deviation of nearest neighbor distances
        - mean_pairwise_dist: Mean pairwise distance between embeddings
        - pairwise_dist_std: Standard deviation of pairwise distances
    """
    # Convert to numpy for calculations
    X = embeddings.numpy()

    # Calculate norms
    norms = np.linalg.norm(X, axis=1)
    avg_norm = float(np.mean(norms))
    norm_std = float(np.std(norms))

    # Calculate pairwise distances
    dists = torch.cdist(embeddings, embeddings, p=2).numpy()
    np.fill_diagonal(dists, np.inf)  # Exclude self-distances

    # Nearest neighbor distances
    min_dists = np.min(dists, axis=1)
    mean_nearest_dist = float(np.mean(min_dists))
    nearest_dist_std = float(np.std(min_dists))

    # Mean pairwise distance
    mean_pairwise_dist = float(np.mean(dists[dists != np.inf]))
    pairwise_dist_std = float(np.std(dists[dists != np.inf]))

    # Calculate isotropy (based on singular values)
    U, S, Vh = np.linalg.svd(X)
    isotropy = float(np.min(S) / np.max(S))

    return {
        "isotropy": isotropy,
        "avg_norm": avg_norm,
        "norm_std": norm_std,
        "mean_nearest_dist": mean_nearest_dist,
        "nearest_dist_std": nearest_dist_std,
        "mean_pairwise_dist": mean_pairwise_dist,
        "pairwise_dist_std": pairwise_dist_std,
    }
