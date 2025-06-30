from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict
import umap.umap_ as umap
import plotly.express as px
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
from itertools import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

class TopicWeaver:
    def __init__(self, model_name, top_n, cluster_min):
        # Parameters
        self.top_n = top_n

        # Helper modules
        self.model = SentenceTransformer(model_name)
        self.keybert = KeyBERT(model=self.model)
        self.clusterer = HDBSCAN(min_cluster_size=cluster_min, metric='euclidean')
        self.reducer_2d = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric='euclidean', random_state=42)
        self.reducer_3d = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.1, metric='euclidean', random_state=42)
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name='gpt-4', chunk_size=1024, chunk_overlap=128)

    def get_keywords(self, doc: str) -> list[str]:
        """
        Splits and embeds doc and returns the top 10
        most semantically relevant keywords
        """
        chunks = self.splitter.split_text(doc)
        result = self.keybert.extract_keywords(docs=chunks, top_n=self.top_n, nr_candidates=self.top_n*2)
        
        keywords = list(set([item for item in chain.from_iterable(result) if isinstance(item, str)]))
        pooled = np.mean(self.model.encode(chunks), axis=0)

        kw_similarities = {}

        for kw, kw_emb in zip(keywords, self.model.encode(keywords)):
            distance = cos_sim(pooled, kw_emb)
            kw_similarities[kw] = float(distance)

        kw_similarities = dict(sorted(
            kw_similarities.items(), 
            key=lambda item: item[-1], 
            reverse=True
        ))

        top_keywords = list(kw_similarities.keys())[:10]
        
        return top_keywords

    def cluster_keywords(self, keywords: list[str]) -> dict:
        """
        Clusters keywords into semantic clusters
        and returns a map of their members
        """
        embeddings = self.model.encode(keywords, normalize_embeddings=True)
        clusters = self.clusterer.fit_predict(embeddings)

        cluster_map = defaultdict(list)
        keyword_embeddings = {}

        for keyword, cluster_id in zip(keywords, clusters):
            if cluster_id != -1: # Skip noise if using HDBSCAN
                cluster_map[int(cluster_id)].append(keyword)
                keyword_embeddings[keyword] = embeddings[keywords.index(keyword)]

        return cluster_map, keyword_embeddings

    def name_cluster(self, keywords: list[str], keyword_embeddings: dict) -> str:
        """
        Names a cluster of keywords by finding the keyword
        closest to the centroid of the cluster.
        """
        embeddings = [keyword_embeddings[kw] for kw in keywords]
        embeddings = torch.tensor(embeddings)

        # Calculate centroid and find the keyword closest to it
        centroid = util.normalize_embeddings(embeddings.mean(dim=0, keepdim=True))
        name_index = int(util.cos_sim(centroid, embeddings)[0].argmax())

        return keywords[name_index]

    def display_2d(self, cluster_map: dict, cluster_names: dict, keyword_embeddings: dict):
        """
        Displays clusters and their member keywords
        """

        flat_keywords, flat_ids = self.flatten_cluster_map(cluster_map)
        flat_names = [cluster_names[cluster_id] for cluster_id in flat_ids]

        embeddings = [keyword_embeddings[kw] for kw in flat_keywords]
        embeddings = self.reducer_2d.fit_transform(embeddings)

        coords = pd.DataFrame({
            'keyword': flat_keywords,
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'cluster': flat_names
        })

        fig = px.scatter(
            coords, 
            x='x', 
            y='y',
            title=f'2D plot of {len(cluster_map)} topics',
            color=flat_names,
            hover_data=['keyword'],
            height=800
        )

        return fig

    def display_3d(self, cluster_map: dict, cluster_names: dict, keyword_embeddings: dict):
        """
        Displays clusters and their member keywords
        """

        flat_keywords, flat_ids = self.flatten_cluster_map(cluster_map)
        flat_names = [cluster_names[cluster_id] for cluster_id in flat_ids]

        embeddings = [keyword_embeddings[kw] for kw in flat_keywords]
        embeddings = self.reducer_3d.fit_transform(embeddings)

        coords = pd.DataFrame({
            'keyword': flat_keywords,
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'z': embeddings[:, 2],
            'cluster': flat_names
        })

        fig = px.scatter_3d(
            coords,
            x='x',
            y='y',
            z='z',
            title=f'2D plot of {len(cluster_map)} topics',
            color='cluster',
            hover_data=['keyword'],
            height=800
        )

        return fig

    def flatten_cluster_map(self, cluster_map: dict) -> tuple:
        """
        Flattens a cluster map and returns two equally long lists
        """
        flat_keywords = [
            kw 
            for group in cluster_map.values() 
            for kw in group
        ]
        flat_ids = [
            mapped_id 
            for cluster_id in cluster_map.keys() 
            for mapped_id in [cluster_id] * len(cluster_map[cluster_id]) 
        ]
        return flat_keywords, flat_ids



"""
def create_graph(cluster_map, cluster_names, titles, extracted_keywords):
    # Map each keyword to its cluster
    keyword_to_cluster = {
        keyword: cluster_id
        for cluster_id, keywords in cluster_map.items()
        for keyword in keywords
    }

    G = nx.Graph()

    # Add cluster nodes
    for cluster_id, cluster_name in cluster_names.items():
        G.add_node(f'cluster_{cluster_id}',  label=cluster_name + ' (topic)', type='cluster')
    
    # Add article nodes
    for i, pair in enumerate(zip(titles, extracted_keywords)):
        title, keywords = pair[0], pair[1]
        G.add_node(f'article_{i}', label=title + ' (article)', type='article')

        added_clusters = set()
        
        for kw in keywords:
            if kw in keyword_to_cluster:
                cluster_id = keyword_to_cluster[kw]
                # Add edge between cluster and article
                if cluster_id and cluster_id not in added_clusters:
                    G.add_edge(f'article_{i}', f'cluster_{cluster_id}')

    return G

def graph_html(G):
    nodes = [{"id": n, "label": G.nodes[n]['label'], "type": G.nodes[n]['type']} for n in G.nodes()]
    edges = [{"from": u, "to": v} for u, v in G.edges()]
    graph_data = {'nodes': nodes, 'edges': edges}
    
    env = Environment(loader=FileSystemLoader('./'))
    html_string = str(env.get_template('/template.jinja').render(data=graph_data))
    script_string = str(env.get_template('/script.jinja').render(data=graph_data))

    return html_string, script_string

if __name__ == '__main__':
    # Run the script
    None
"""