Text Clustering and Analysis System  

1. Project Overview  
   - Purpose and functionalities of the system  

2. System Requirements
   - 2.1 Dependencies  
   - 2.2 Installation  

3. System Architecture  
     - 3.1 Core Components  
     - Text Processing Module  
     - Database Layer  
     - Web Interface  

4. Implementation Details 
   - 4.1 Text Processing  
     - Embedding Generation  
     - Clustering Process  

   - 4.2 Database Schema  
     - Base Model  
     - Cluster Model  
     - Item Model  

   - 4.3 API Endpoints  
     - Fetch Clusters  
     - Fetch Cluster Description  

5. Usage Guide
   - 5.1 System Setup  
   - 5.2 Running the System  
   - 5.3 Accessing Results  

6. Conclusion

Appendix  
- Installation Commands  
- Key Code Snippets for Reference  
- Error Handling Recommendations  
- Future Enhancements Ideas  
- Github Link - https://github.com/BenitoJD/Text-Clustering-and-Analysis-System

1. Project Overview  
This project is designed to help analysed and group textual data efficiently by leveraging modern NLP (Natural Language Processing) techniques. It uses machine learning models to understand the underlying semantics of text, groups similar sentences together, and provides a web interface to visualize and interact with the results. The primary functionalities include:  
- Generating embeddings for text to convert it into machine-readable vectors.  
- Using clustering algorithms to group similar sentences together.  
- Storing the clustered results in a database for easy retrieval and analysis.  

The system is built to be modular, scalable, and user-friendly, making it suitable for a wide range of text analysis tasks, such as incident reporting, sentiment analysis, or content categorization.  

 2. System Requirements  

2.1 Dependencies  
The system relies on several Python libraries for its functionality:  
numpy: For numerical computations and handling matrices like embeddings.  
scikit-learn: For implementing clustering algorithms and managing data preprocessing.  
sentence-transformers: To generate semantic embeddings for sentences.  
-transformers: A robust NLP library used alongside `sentence-transformers`.  
peewee: A lightweight ORM (Object-Relational Mapping) to interact with the SQLite database.  
flask: For building the web server and API endpoints.  
flask-cors: To enable secure cross-origin requests between frontend and backend components.  
sqlite3: A database engine for storing and querying data.  

2.2 Installation  
Ensure all dependencies are installed by running the following command:  
bash  
pip install numpy sklearn sentence-transformers transformers peewee flask flask-cors  
3. System Architecture  

3.1 Core Components  

1. Text Processing Module  
   - This module handles all NLP tasks, such as generating embeddings, calculating similarity, and clustering similar content.  
   - It ensures that input text is preprocessed and transformed into a format suitable for analysis.  

2. Database Layer  
   - Manages data persistence, ensuring that clustered results are saved for future use.  
   - Uses SQLite, a lightweight and reliable database engine, and Peewee ORM for simplified query handling.  

3. Web Interface
   - Built using Flask to provide a REST API for interaction.  
   - Enables users to fetch, view, and analyze clusters in a user-friendly way.  
4. Implementation Details  

 4.1 Text Processing  

1. Embedding Generation  
   - The system converts sentences into high-dimensional vectors (embeddings) using a pre-trained `sentence-transformers` model.  
   - Embeddings capture the semantic meaning of sentences, enabling effective similarity comparison.  

```python  
def cluster_similar_sentences(sentences, similarity_threshold=0.8):  
    # Load a pre-trained sentence embedding model  
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    
    # Generate embeddings for input sentences  
    embeddings = model.encode(sentences, show_progress_bar=True)  
    
    # Normalize embeddings to improve similarity computations  
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]  
    
    # Compute a similarity matrix  
    similarity_matrix = np.dot(embeddings, embeddings.T)  
    distance_matrix = 1 - similarity_matrix  # Convert similarity to distance  
    
    return distance_matrix  
```  

2. Clustering Process  
   - Clustering is performed using hierarchical clustering from `scikit-learn`.  
   - Similar sentences are grouped based on their distance in the embedding space.  

```python  
def perform_clustering(distance_matrix, similarity_threshold):  
    clustering = AgglomerativeClustering(  
        n_clusters=None,  # Automatically determine the number of clusters  
        distance_threshold=1 - similarity_threshold,  # Use threshold for grouping  
        metric='precomputed',  # Distance matrix is precomputed  
        linkage='complete'  # Use complete linkage for clustering  
    )  
    
    cluster_labels = clustering.fit_predict(distance_matrix)  
    return cluster_labels  
```  

---

4.2 Database Schema  

1. **Base Model**  
   - Represents the foundation of all database tables.  

```python  
class BaseModel(Model):  
    class Meta:  
        database = db  
```  

2. Cluster Model 
   - Stores information about each cluster.  

```python  
class Cluster(BaseModel):  
    cluster_id = AutoField(primary_key=True)  
    name = TextField(null=False)  
```  

3. Item Model
   - Links individual sentences or items to their respective clusters.  

```python  
class Item(BaseModel):  
    item_id = AutoField(primary_key=True)  
    description = TextField()  
    cluster_id = ForeignKeyField(Cluster, backref='items', on_delete='CASCADE')  
```  

---

4.3 API Endpoints  

1. Fetch Clusters  
   - Returns a list of all clusters from the database.  

```python  
@app.route('/api/clusters')  
def get_clusters():  
    data = get_cluster_data()  
    if data is None:  
        return jsonify({'error': 'Database error'}), 500  
    return jsonify(data)  
```  

2. Fetch Cluster Description
   - Fetches all items associated with a specific cluster ID.  

```python  
@app.route('/description/<int:cluster_id>')  
def get_desc_data(cluster_id):  
    data = Get_Description(cluster_id)  
    if data is None:  
        return jsonify({'error': 'Database error'}), 500  
    return jsonify(data)  
```  

---

 5. Usage Guide  

5.1 System Setup  
1. Clone the repository.  
2. Install all dependencies using the provided `pip` command.  
3. Set up the database:  

```python  
def setup_database():  
    db = SqliteDatabase('clustersDB.db')  
    db.create_tables([Cluster, Item], safe=True)  
```  

5.2 Running the System  
1. Start the Flask server:  
```python  
if __name__ == '__main__':  
    port = get_available_port()  
    app.run(host='127.0.0.1', port=port, debug=True)  
```  
2. Process text files by placing them in the `Data` directory and running the main clustering script.  

5.3 Accessing Results  
- Access the clusters at `http://localhost:<port>/api/clusters`.  
- View details of a specific cluster using `http://localhost:<port>/description/<cluster_id>`.  
 

6. Conclusion  
This intelligent text clustering system provides a comprehensive solution for processing, grouping, and analyzing textual data. It combines state-of-the-art NLP models with robust clustering techniques to make text analysis accessible and effective. Its web interface and API further enhance usability, enabling users to integrate it into their workflows seamlessly. Future enhancements can further scale the system for real-time and large-scale text processing applications.
