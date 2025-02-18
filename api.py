from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for API accessibility

# ////// Basic API Endpoints //////////////

@app.route('/')
def index():
    """Root endpoint to check if the API is running."""
    return 'Hello, World....!'

@app.route('/user', methods=['GET'])
def get_user():
    """Returns a sample user data as JSON."""
    user = {'name': 'John Doe', 'email': 'john@example.com'}
    return jsonify(user)

# List to store user data
users = []

@app.route('/user', methods=['POST'])
def create_user():
    """Creates a new user and adds to the list."""
    new_user = request.get_json()
    users.append(new_user)
    return jsonify(new_user), 201

@app.route('/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API functionality."""
    return jsonify({'message': 'Flask API is working!'})

# ////// End of Basic API //////////////

# ////// LLM API (Transformers) //////////////

# Load a pretrained model for text classification
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/classify', methods=['POST'])
def classify_text():
    """
    Text classification endpoint.
    Accepts a JSON request containing a text field and returns the predicted label.
    """
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    # Set padding token if necessary
    tokenizer.pad_token = tokenizer.eos_token

    # Encode the input text
    inputs = tokenizer.encode_plus(
        text,
        padding='max_length',  # Ensures consistent input size
        truncation=True,       # Truncate if text exceeds max_length
        max_length=128,        # Maximum token length
        return_tensors='pt'    # Convert input to PyTorch tensor
    )

    # Get model output
    outputs = model(**inputs)

    # Extract the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1)
    return jsonify({'label': predicted_class.item()})

# Load a pretrained conversational AI model (DialoGPT)
model_name = "microsoft/DialoGPT-medium"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chatbot endpoint.
    Accepts a JSON request with a user message and returns a generated chatbot response.
    """
    data = request.get_json()
    user_input = data.get('text', '')

    if not user_input:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    # Encode the user input
    inputs = tokenizer.encode(user_input, return_tensors="pt")

    # Generate chatbot response
    reply_ids = model.generate(
        inputs,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True, 
        temperature=0.7,  # Controls randomness in output (higher = more creative)
        top_p=0.9         # Nucleus sampling to generate diverse responses
    )

    # Decode the generated response
    bot_reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": bot_reply})

# ////// End of LLM API (Transformers) //////////////

if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask application in debug mode
