import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import os
from peewee import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

def cluster_similar_sentences(sentences, similarity_threshold=0.9, min_cluster_size=0):
   
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
#  db = SqliteDatabase('clustersDB.db')
   db = SqliteDatabase(r'D:\benito\Requirement2\cluster-visualization\\clustersDB.db')

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


folder_path = r"D:\benito\Requirement2\TransformedSentence"
sample_sentences = []
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'): 
        file_path = os.path.join(folder_path, filename)  
        sample_sentences.clear()
        with open(file_path, 'r') as file:
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
    output_file_path = fr"D:\benito\Requirement2\FinalResult\{filename}"

    with open(output_file_path, 'w') as file:
        file.write("Clustered Sentences:\n")
    
        for cluster_id, sentences in clusters.items():
            file.write(f"\nCluster {cluster_id}:\n")
            file.write(f"Representative: {representatives[cluster_id]}\n")
            file.write("All sentences in cluster:\n")
        
            for sentence in sentences:
                file.write(f"- {sentence}\n")
                
    db.create_tables([Cluster, Item], safe = True)
    for cluster_id, sentences in clusters.items():
        cluster = Cluster.create(name=representatives[cluster_id])
        for sentence in sentences:
            Item.create(description=sentence, cluster_id=cluster)
    db.close()
    