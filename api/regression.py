from http.server import BaseHTTPRequestHandler
from urllib import parse
from pathlib import Path
import os
import numpy as np
import tflite_runtime.interpreter as tflite


# Load the TFLite model and allocate tensors
root_directory = Path(__file__).resolve().parent.parent
model_path = os.path.join(root_directory, 'model', 'regression_model.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to make a prediction
def predict(number):
    # Preprocess the input
    number = float(number)
    input_data = np.array([[number]], dtype=np.float32)

    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Extract the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Take the url query string and create a dictionary of parameters
        url = self.path
        url_components = parse.urlsplit(url)
        query_string_list = parse.parse_qsl(url_components.query)
        dictionary = dict(query_string_list)  # /?number=something

        # Do stuff
        if dictionary.get("number"):
            prediction = predict(dictionary["number"])

        # Forming the response
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(f"The model predicted: {prediction}".encode())


# if __name__ == "__main__":
#     # Ask user for a number and convert to float
#     user_input = float(input("Enter a number: "))
#
#     # Make a prediction
#     prediction = predict(user_input)
#
#     # Print the prediction
#     print("Prediction:", prediction)
