import PyPDF2
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
import pandas as pd
import umap
import tiktoken

nltk.download("punkt")


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# Extract content from the PDF
pdf_path = "privacy.pdf"
text_content = extract_text_from_pdf(pdf_path)

# Dummy document structure for demonstration
docs = [{"metadata": {"source": "page1"}, "page_content": text_content}]

# Sort and reverse documents by source
d_sorted = sorted(docs, key=lambda x: x["metadata"]["source"])
d_reversed = list(reversed(d_sorted))

# Concatenate content
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc["page_content"] for doc in d_reversed]
)


# Function to count tokens using tiktoken
def num_tokens_from_string(text, encoding="cl100k_base"):
    enc = tiktoken.get_encoding(encoding)
    tokens = enc.encode(text)
    return len(tokens)


# Check token count in concatenated content
total_tokens = num_tokens_from_string(concatenated_content, "cl100k_base")
print("Num tokens in all context: %s" % total_tokens)


# Function to chunk text into approximately 1000 tokens
def chunk_text(text, max_tokens=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if num_tokens_from_string(chunk + " " + sentence, "cl100k_base") <= max_tokens:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks


texts_split = chunk_text(concatenated_content)

# Print number of text splits generated
print(f"Number of text splits generated: {len(texts_split)}")

# Load SBERT model
model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings for each chunk
global_embeddings = model.encode(texts_split)

# Print dimension of the first embedding
print(len(global_embeddings[0]))


# Function to reduce cluster embeddings
def reduce_cluster_embeddings(embeddings, dim, n_neighbors=None, metric="cosine"):
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


# Reduce embeddings to 2 dimensions
dim = 2
global_embeddings_reduced = reduce_cluster_embeddings(global_embeddings, dim)

# Print the first reduced embedding
print(global_embeddings_reduced[0])

# Visualize the reduced embeddings
plt.figure(figsize=(10, 8))
plt.scatter(global_embeddings_reduced[:, 0], global_embeddings_reduced[:, 1], alpha=0.5)
plt.title("Global Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234
):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [
        GaussianMixture(n_components=n, random_state=random_state)
        .fit(embeddings)
        .bic(embeddings)
        for n in range(1, max_clusters)
    ]
    return np.argmin(bics) + 1


def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(
        embeddings
    )
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


labels, _ = gmm_clustering(global_embeddings_reduced, threshold=0.5)

plot_labels = np.array([label[0] if len(label) > 0 else -1 for label in labels])
plt.figure(figsize=(10, 8))

unique_labels = np.unique(plot_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = plot_labels == label
    plt.scatter(
        global_embeddings_reduced[mask, 0],
        global_embeddings_reduced[mask, 1],
        color=color,
        label=f"Cluster {label}",
        alpha=0.5,
    )

plt.title("Cluster Visualization of Global Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()


# Create a dataframe to check the texts associated with each cluster
simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

df = pd.DataFrame(
    {
        "Text": texts_split,
        "Embedding": list(global_embeddings_reduced),
        "Cluster": simple_labels,
    }
)
print(df.head(3))


# Function to format clustered texts
# def format_cluster_texts(df):
#     clustered_texts = {}
#     for cluster in df["Cluster"].unique():
#         cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
#         clustered_texts[cluster] = " --- ".join(cluster_texts)
#         print(f"Cluster {cluster}:\n{clustered_texts[cluster]}\n")
#     return clustered_texts


# make above function more debuggable along with how much left
def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df["Cluster"].unique():
        cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts)
        print(f"Cluster {cluster}:\n{clustered_texts[cluster]}\n")
    print(f"Total clusters: {len(clustered_texts)}")
    return clustered_texts


clustered_texts = format_cluster_texts(df)

# Print clustered texts
for cluster, text in clustered_texts.items():
    print(f"Cluster {cluster}:\n{text}\n")
