
# from flask import Flask, request, render_template
# import google.auth
# from google.cloud import aiplatform

# app = Flask(__name__)

# # Initialize the Vertex AI client
# credentials, project = google.auth.default()
# client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
# client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

# # Endpoint path for your Vertex AI model
# endpoint_name = client.endpoint_path(
#     project="restaurant-recommender-408302", location="us-central1", endpoint="7552406832728244224"
# )
from flask import Flask, request, render_template
import os
import google.auth
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()  # This loads the .env file at the start of your app

app = Flask(__name__)

# Initialize the Vertex AI client
credentials, project = google.auth.default()
client_options = {"api_endpoint": os.getenv("API_ENDPOINT")}
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

# Endpoint path for your Vertex AI model
endpoint_name = client.endpoint_path(
    project=os.getenv("PROJECT_ID"), location="us-central1", endpoint=os.getenv("ENDPOINT_ID")
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collecting input data from the form
    customer_id = request.form['customer_id']
    location_number = request.form['location_number']
    # Add additional fields if needed

    # Prepare the data in the format expected by your model
    instances = [{"customer_id": customer_id, "location_number": location_number}]  # Update based on your model's input format

    # Make a prediction request
    response = client.predict(endpoint=endpoint_name, instances=instances)
    
    # Extract prediction result
    prediction = response.predictions[0]

    # Render the prediction result in the predict.html template
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


