import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import os
from peewee import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

def cluster_similar_sentences(sentences, similarity_threshold=0.8, min_cluster_size=0):
   
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    distance_matrix = 1 - similarity_matrix
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-similarity_threshold,
        metric='precomputed',
        linkage='complete'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    clusters = defaultdict(list)
    for sentence, label in zip(sentences, cluster_labels):
        clusters[label].append(sentence)
    
    significant_clusters = {
        label: sentences 
        for label, sentences in clusters.items() 
        if len(sentences) >= min_cluster_size
    }
    
    return significant_clusters

def get_cluster_representatives(clusters):
  
    model = SentenceTransformer('all-MiniLM-L6-v2')
    representatives = {}
    
    for cluster_id, sentences in clusters.items():
        if len(sentences) == 1:
            representatives[cluster_id] = sentences[0]
            continue
            
        embeddings = model.encode(sentences)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        representative_idx = np.argmax(avg_similarities)
        representatives[cluster_id] = sentences[representative_idx]
    
    return representatives

if __name__ == "__main__":
     db = SqliteDatabase('clustersDB.db')

class BaseModel(Model):
        class Meta:
            database = db

class Cluster(BaseModel):
        cluster_id = AutoField(primary_key=True)
        name = TextField(null=False)                 

class Item(BaseModel):
        item_id = AutoField(primary_key=True)     
        description = TextField()                    
        cluster_id = ForeignKeyField(Cluster, backref='items', on_delete='CASCADE')

db.connect()


file_path = os.path.join(os.path.dirname(__file__), "Data")

sample_sentences = []
for filename in os.listdir(file_path):
    if filename.endswith('.txt'): 
        file_path = os.path.join(file_path, filename)  
        sample_sentences.clear()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()
                if sentence:
                 sample_sentences.append(sentence)

    clusters = cluster_similar_sentences(
        sentences=sample_sentences,
        similarity_threshold=0.8,
        min_cluster_size=0
    )
    
    representatives = get_cluster_representatives(clusters)
                
    db.create_tables([Cluster, Item], safe = True)
    for cluster_id, sentences in clusters.items():
        cluster = Cluster.create(name=representatives[cluster_id])
        for sentence in sentences:
            Item.create(description=sentence, cluster_id=cluster)
    db.close()
    