from PIL import Image
import numpy as np
import torch
import cv2
import io
import os
import datetime
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    session,
    jsonify,
    flash,
)
import pandas as pd
import config
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tfidf_Vectorizer import TfidfEmbeddingVectorizer
from preprocessing import ingredient_parser,get_and_sort_corpus
from recommender import get_recommendations

app = Flask(__name__)

def load_det_model():
    return torch.hub.load('./yolo', 'custom','./yolo/best.pt', source='local',trust_repo=True)

det_model = load_det_model()

@app.route("/")
def Home():
    return render_template('Home.html')

@app.route("/recipes", methods=['GET', 'POST'])
def Recipe():
    model = Word2Vec.load(config.MODEL)
    model.init_sims(replace=True)

    data = pd.read_csv(config.RECIPES_DETAILS)
    data["parsed"] = data.ingredients.apply(ingredient_parser)
    corpus = get_and_sort_corpus(data)
    tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
    tfidf_vec_tr.fit(corpus)
    doc_vec = tfidf_vec_tr.transform(corpus)
    doc_vec = [doc.reshape(1, -1) for doc in doc_vec]

    #clean input
    input = request.form['ingredients']
    input = input.split(",")
    iIngred = [s.strip() for s in input]
    
    input_embedding = tfidf_vec_tr.transform([iIngred])[0].reshape(1, -1)
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)

    scores = list(cos_sim)
    recommendations = get_recommendations(5, scores)
    return render_template('Recipes.html', reco = recommendations)

@app.route("/recipeDetails/<recipeName>")
def RecipeDetails(recipeName):
    print(recipeName)
    return render_template('RecipeDetails.html')

@app.route("/inference2")
def Inference2():
    return render_template('Inference2.html')

def get_det_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    results = det_model(img, size=640)
    return results

@app.route('/detect', methods=['GET', 'POST'])
def det_predict():
    if request.method == 'POST':
        if request.files['file'].filename == '':
            return render_template('Home.html')
        file = request.files.get('file')
        if not file:
            return render_template('Home.html')
        img_bytes = file.read()
        results = get_det_prediction(img_bytes)
        filename = results.save(save_dir=f'static/prediction')
        predictions = results.pred[0]
        detected_classes = [int(x) for x in list(predictions[:, 5])]
        all_classes = results.names
        detected_classes = [all_classes[i] for i in detected_classes]
        data = {
            "filename" : filename,
            "classes": detected_classes
        }
    return jsonify(data)

def inferenceResult(filename, classes, coordinates):
    bounding_box_data = []
    for box in coordinates:
        bounding_box_data.append({
            "image": filename,
            "xyxyn": box,
            "labels": classes[box[-1]]
        })
    json_data = json.dumps(bounding_box_data)
    return json_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)