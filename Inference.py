from flask import Flask, request, jsonify
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

app = Flask(__name__)

# Load the FAISS index
index_path = 'beer_embeddings.index'
if not os.path.exists(index_path):
    raise FileNotFoundError(f"FAISS index file not found at {index_path}")
index = faiss.read_index(index_path)

# Configuration for beer attributes
config = ['ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour',
          'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']

# Load beer metadata
datadir = "/home/karthiktiwari/Downloads/Beer Data"
metadata_path = os.path.join(datadir, "beer_profile_and_ratings.csv")
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Beer metadata file not found at {metadata_path}")
beer_metadata = pd.read_csv(metadata_path, index_col=0)
beer_metadata.reset_index(drop=True, inplace=True)



# Initialize the neural network model
input_dim = len(config)  # Number of beer attributes
embedding_dim = 32  # Desired embedding size
model = BeerAttributeNN(input_dim, embedding_dim)
model.eval()  # Set to evaluation mode

# Load model weights (adjust the path to your model's location)
model_weights_path = 'beer_attribute_nn_weights.pth'
if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path))
else:
    raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Retrieve beer attribute data from the request
        data = request.get_json()
        beer_attributes = np.array([data[attr] for attr in config], dtype='float32').reshape(1, -1)

        # Pass the beer attributes through the neural network to get the embedding
        beer_tensor = torch.tensor(beer_attributes)
        embedding = model(beer_tensor).detach().numpy().reshape(1, -1)

        # Validate embedding dimension
        if embedding.shape[1] != embedding_dim:
            return jsonify({"error": "Generated embedding dimension does not match expected size"}), 400

        # Query FAISS for the 5 most similar embeddings
        k = 5
        distances, indices = index.search(embedding.astype('float32'), k)

        # Retrieve beer IDs and relevant metadata for recommended items
        recommendations = [
            {
                'beer_id': int(beer_metadata.index[int(idx)]),  # ID from metadata
                'distance': float(dist),  # Similarity distance
                'beer_name': beer_metadata.iloc[int(idx)]['Beer Name (Full)'],  # Beer name
                'style': beer_metadata.iloc[int(idx)]['Style'],  # Beer style
                'ABV': beer_metadata.iloc[int(idx)]['ABV'],  # Alcohol by volume
                'rating': beer_metadata.iloc[int(idx)]['Average Rating']  # Beer rating
            }
            for dist, idx in zip(distances[0], indices[0])
        ]

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
