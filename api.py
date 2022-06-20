import logging

from os import getenv
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

from utils.constants import HIRAGANA, BASE_URL, HOST, PORT
from classes.PredictionHandler import PredictionHandler


def create_app():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        handlers=[logging.FileHandler(
            "remakeIT_api.log"), logging.StreamHandler()]
    )

    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def redirect_to_base_url():
        return redirect(BASE_URL)

    @app.route(f'{BASE_URL}/', methods=['GET', 'POST'])
    def welcome():
        return jsonify(
            message="Welcome to the HiraganAI API!",
            version="0.1.0",
        )

    @app.route(f'{BASE_URL}/predict', methods=['POST'])
    def hiragana():
        if 'file' not in request.files:
            return jsonify(
                message={
                    'fr': "Aucun fichier n'a été envoyé.",
                    'en': "No file has been sent.",
                },
                error=True,
            ), 400
            return redirect(request.url)

        file = request.files['file']
        image = PredictionHandler(file)
        prediction = image.make_prediction()

        return jsonify(
            message={
                'fr': f"La prédiction est : {prediction}",
            },
            prediction=prediction,
            error=False,
        ), 200

    @app.route(f'{BASE_URL}/hiragana', methods=['POST'])
    def predict():
        pass

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=HOST, port=PORT, debug=True)
