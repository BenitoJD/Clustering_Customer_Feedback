# app.py
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import sqlite3
import socket

app = Flask(__name__, static_folder='Js')
CORS(app)

def get_available_port(start=5000, max_port=5100):
    for port in range(start, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError('No available ports')
def Get_Description(clusterID):
    try:
        conn = sqlite3.connect('clustersDB.db')
        cursor = conn.cursor()
        
        query = """
        select description from item where cluster_id = ?
        """
        
        cursor.execute(query, (clusterID,))
        rows = cursor.fetchall()
        data = [{'Description': row[0]} for row in rows]
        cursor.close()
        conn.close()
        return data
    except Exception as e:
        print(f"Database Error: {str(e)}")
        return None
def get_cluster_data():
    try:
        conn = sqlite3.connect('clustersDB.db')
        cursor = conn.cursor()
        
        query = """
   WITH ClusterCounts AS (
    SELECT 
        i.cluster_id, 
        c.name AS cluster_name, 
        COUNT(*) AS count_of_similar_rows 
    FROM 
        item i
    JOIN 
        cluster c ON i.cluster_id = c.cluster_id
    GROUP BY 
        i.cluster_id, c.name 
), RankedClusters AS (
    SELECT 
        cluster_id,
        cluster_name,
        count_of_similar_rows,
        PERCENT_RANK() OVER (ORDER BY count_of_similar_rows DESC) AS percentile
    FROM 
        ClusterCounts
)
SELECT 
    cluster_id,
    cluster_name,
    count_of_similar_rows
FROM 
    RankedClusters
WHERE 
    percentile <= 0.9  -- Top 10%
ORDER BY 
    count_of_similar_rows DESC;

        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        data = [{'cluster_id': row[0], 'count_of_similar_rows': row[2], 'Description': row[1]} for row in rows]
        
        cursor.close()
        conn.close()
        return data
    except Exception as e:
        print(f"Database Error: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/description')
def description():
    return render_template('description.html')

@app.route('/api/clusters')
def get_clusters():
    data = get_cluster_data()
    if data is None:
        return jsonify({'error': 'Database error'}), 500
    return jsonify(data)
@app.route('/description/<int:cluster_id>')
def get_desc_data(cluster_id):
    data = Get_Description(cluster_id)
    if data is None:
        return jsonify({'error': 'Database error'}), 500
    elif not data:
        return jsonify({'error': 'No data found for the specified cluster ID'}), 404
    return jsonify(data)


if __name__ == '__main__':
    try:
        # Try to find an available port
        port = get_available_port()
        print(f"Server starting on port {port}")
        app.run(host='127.0.0.1', port=port, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        
        # If all else fails, try running on a very different port
        try:
            alt_port = 8080
            print(f"Attempting to start on alternative port {alt_port}")
            app.run(host='127.0.0.1', port=alt_port, debug=True)
        except Exception as e:
            print(f"Failed to start server on alternative port: {e}")