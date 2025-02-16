from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizers
model = tf.keras.models.load_model("model_files/translation_model.h5")

with open("model_files/tokenizer_eng.pkl", "rb") as f:
    tokenizer_eng = pickle.load(f)

with open("model_files/tokenizer_fr.pkl", "rb") as f:
    tokenizer_fr = pickle.load(f)

max_seq_length = 21

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    input_text = data.get("text", "")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    input_sequence = tokenizer_eng.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')

    predictions = model.predict(input_padded, verbose=0)
    predicted_indices = np.argmax(predictions, axis=-1)[0]

    translated_words = [tokenizer_fr.index_word.get(idx, "") for idx in predicted_indices if idx != 0]
    translated_sentence = " ".join(translated_words)

    return jsonify({"input": input_text, "translated": translated_sentence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
