from sklearn.decomposition import PCA
import numpy as np
import torch


def explain_variance(data, amount=.95, writer=lambda x: None):
    N, T, D = data.shape

    for t in range(T):
        pca = PCA()
        pca.fit(data[:,t,:])
        cumulative = np.cumsum(pca.explained_variance_ratio_)

        dim = np.argmax(cumulative >= amount) + 1

        for fn in (print, writer):
            fn(f"At time step {t}: Dimensionality of embeddings is D={D} but {int(amount*100)}% of variance is in only {dim} dimensions")
            fn(f"At time step {t}: Explained Variance by 20 most significant dims:")
            fn(pca.explained_variance_ratio_[:20])

    return dim


def pca_batched(X: np.ndarray, ratio=0.95):
    # batched across time dimension

    N, T, D = X.shape

    X -= np.mean(X, axis=0)

    _, S, V = np.linalg.svd(X.transpose((1,0,2)), full_matrices=False)

    # Determine the number of dimensions to keep based on the ratio
    eigenvalues = S ** 2 / (N - 1) # squared singular values
    total_eigenvalues = np.sum(eigenvalues, axis=1, keepdims=True)
    eigenvalues /= total_eigenvalues

    cumulative_eigenvalues = np.cumsum(eigenvalues, axis=1)

    d = np.argmax(cumulative_eigenvalues >= ratio, axis=1)

    print(f"Fitted {T} SVDs. How many dimensions were needed to explain {ratio} of the variance?")
    print(d)

    return V, d

def pca_sklearn(X: np.ndarray, ratio=0.95):

    reshaped_x = X.reshape(X.shape[0]*X.shape[1], -1)

    print("shapes:", X.shape, reshaped_x.shape)

    pca = PCA(n_components=ratio)

    pca.fit(reshaped_x)

    transform = pca.components_ # C x D

    return transform

def noise_significant_dims(emb, V, mu):
    B, T, D = emb.shape
    V = torch.tensor(V, dtype=emb.dtype).to(emb.device)
    C = V.shape[0] # number of components
    V = V.unsqueeze(0).unsqueeze(0) # 1, 1, C, D

    noise = torch.randn(B, T, C, 1, device=emb.device)
    normed = torch.linalg.norm(noise, axis=2, keepdims=True)
    noise *= mu / normed
    noised_EVs = (noise * V).sum(dim=2) # B, T, D

    emb_plus = emb + noised_EVs
    emb_minus = emb - noised_EVs

    return emb_plus, emb_minus, noised_EVs


def noise_pca_dims_batched(emb, V, d, mu):
    # emb is torch Tensor, V, d are np.ndarray, mu is scalar

    B, T, D = emb.shape

    V = torch.tensor(V, dtype=emb.dtype).to(emb.device)
    d = torch.tensor(d, dtype=torch.int32).to(emb.device)

    transformed_emb = torch.einsum('btd,tdp->btp', emb, V)

    emb_plus = emb.clone()
    emb_minus = emb.clone()

    noise = torch.randn(B, T, D, device=emb.device)
    normed = torch.linalg.norm(noise, axis=2, keepdims=True)
    noise /= normed / mu

    # at time t, mask is 1 when the idx of the dimension is < the number of dims
    # found by pca for that timestep, else 0
    mask = torch.arange(D).to(emb.device)[None,:] < d[:, None]

    # TODO: think hard whether I can just einsum the noise once and add it
    noise_d = mask[None,:,:] * noise

    transformed_emb += noise_d
    emb_plus = torch.einsum("btp,tdp->btd", transformed_emb, V)

    transformed_emb -= 2 * noise_d
    emb_minus = torch.einsum("btp,tdp->btd", transformed_emb, V)

    # remove noise for return
    transformed_emb -= noise_d

    return emb_plus, emb_minus, noise_d, transformed_emb


if __name__ == "__main__":
    # 1. Test pca_batched
    N, T, D = 2000, 24, 512

    X = np.random.normal(loc=0, scale=200, size=(N, T, D))

    V, d = pca_batched(X, ratio=0.8)

    assert V.shape[0] == T, V.shape
    assert V.shape[1] == D, V.shape
    assert V.shape[2] == D, V.shape

    assert d.shape[0] == T, d.shape

    # 2. Test noise_pca_dims_batched
    B = 32
    emb = torch.randn(B, T, D)

    emb_plus, emb_minus, noise_d, transformed_emb = noise_pca_dims_batched(emb, V, d, 1.0)

    assert transformed_emb.shape[0] == B, transformed_emb.shape
    assert transformed_emb.shape[1] == T, transformed_emb.shape
    assert transformed_emb.shape[2] == D, transformed_emb.shape

    # Sklearn implementation:

    V = pca_sklearn(X, ratio=0.95)

    emb_plus, emb_minus, noised_evs = noise_significant_dims(emb, V, 1.0)

