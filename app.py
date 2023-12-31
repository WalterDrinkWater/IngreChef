from PIL import Image
from pathlib import Path
import numpy as np
import torch
import cv2
import io
import os
import datetime
import json
import base64

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
    Response
)
import pandas as pd
import config
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from recommendation_system.tfidf_Vectorizer import TfidfEmbeddingVectorizer
from recommendation_system.preprocessing import ingredient_parser, get_and_sort_corpus
from recommendation_system.recommender import get_recommendations
from pymysql import connections, cursors

db_conn = connections.Connection(
    host=config.customhost,
    port=3306,
    user=config.customuser,
    password=config.custompass,
    connect_timeout=600,
    db=config.customdb,
)
app = Flask(__name__)
camera = cv2.VideoCapture(0)


def load_det_model():
    return torch.hub.load(
        "./yolo", "custom", "./yolo/best.pt", source="local", trust_repo=True
    )

model = Word2Vec.load(config.MODEL)
model.init_sims(replace=True)
data = pd.read_csv(config.RECIPES_DETAILS)
data["parsed"] = data.ingredients.apply(ingredient_parser)
corpus = get_and_sort_corpus(data)
tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
tfidf_vec_tr.fit(corpus)
doc_vec = tfidf_vec_tr.transform(corpus)
doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
det_model = load_det_model()

@app.route("/")
def Home():
    return render_template("Home.html")

@app.route("/recipes", methods=["GET", "POST"])
def Recipe():
    # clean input
    input = request.form["ingredients"].lower()
    input = input.split(",")
    iIngred = [s.strip() for s in input]

    input_embedding = tfidf_vec_tr.transform([iIngred])[0].reshape(1, -1)
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)

    scores = list(cos_sim)
    recommendations = get_recommendations(5, scores)

    return render_template("Recipes.html", reco=recommendations, ingred=iIngred)

@app.route("/recipeDetails/<recipeName>")
def RecipeDetails(recipeName):
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM recipes WHERE recipe_name=%s LIMIT 1", (recipeName))
    recipeDetails = cursor.fetchone()
    ingredients = strToCleanList(recipeDetails[3])
    steps = strToCleanList(recipeDetails[4])
    return render_template(
        "RecipeDetails.html",
        recipeDetails=recipeDetails,
        ingredients=ingredients,
        steps=steps,
    )

def strToCleanList(str):
    str = str[1 : len(str) - 1]
    splited_list = str.split("', '")
    splited_list[0] = splited_list[0][1 : len(splited_list[0])]
    splited_list[len(splited_list) - 1] = splited_list[len(splited_list) - 1][
        0 : len(splited_list[len(splited_list) - 1]) - 1
    ]
    return splited_list

def get_det_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    results = det_model(img, size=640)
    return results

def path_get_det_prediction(file_path):
    img = Image.open(file_path)
    results = det_model(img, size=640)
    return results

@app.route("/detect", methods=["GET", "POST"])
def det_predict():
    if request.method == "POST":
        if request.files["file"].filename == "":
            return render_template("Home.html")
        file = request.files.get("file")
        if not file:
            return render_template("Home.html")
        img_bytes = file.read()
        results = get_det_prediction(img_bytes)
        filename = results.save(save_dir=f"static/prediction")
        predictions = results.pred[0]
        detected_classes = [int(x) for x in list(predictions[:, 5])]
        all_classes = results.names
        detected_classes = [all_classes[i] for i in detected_classes]
        data = {"filename": filename, "classes": detected_classes}
    return jsonify(data)

def generate_frames():
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video")
def video():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get the data URL from the request
        data_url = request.json.get('photo_data')

        # Extract the base64-encoded image data
        _, encoded_data = data_url.split(",", 1)
        # Decode the base64 data
        binary_data = base64.b64decode(encoded_data)

        save_folder = 'static/photo/'
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        
        image_file = os.path.join("static", "photo", f"photo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")

        # Save the image to a file
        with open(image_file, "wb") as f:
            f.write(binary_data)
        results = path_get_det_prediction(image_file)
        filename = results.save(save_dir=f"static/prediction")
        predictions = results.pred[0]
        detected_classes = [int(x) for x in list(predictions[:, 5])]
        all_classes = results.names
        detected_classes = [all_classes[i] for i in detected_classes]
        data = {"success": True, "photo_path": image_file, "filename": filename, "classes": detected_classes}
        return jsonify(data)
    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
