
from flask import Flask, request, jsonify
import torch
import clip
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically select GPU if available
model, preprocess = clip.load("ViT-B/32", device=device)

# Define NSFW categories
categories = ["safe", "nudity", "violence", "explicit"]

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Save the uploaded image
    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(image_path)

    try:
        # Preprocess the image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Encode text categories
        text = clip.tokenize(categories).to(device)

        # Compute logits and probabilities
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()

        # Clean up the saved image
        os.remove(image_path)

        # Prepare the response
        response = {category: prob for category, prob in zip(categories, probs[0])}

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
                                                        
