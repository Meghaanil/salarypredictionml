import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Dummy training data for demonstration
# Replace this with your actual training data loading process
X_train = np.array([[2, 9, 6]])
y_train = np.array([53713.86677124])

# Create a pipeline with Linear Regression
regressor = Pipeline([
    ('linear_regression', LinearRegression())
])

# Fit the model with the training data
regressor.fit(X_train, y_train)

# Save the fitted model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(regressor.named_steps['linear_regression'], model_file)

# Load the fitted model
with open('model.pkl', 'rb') as model_file:
    fitted_model = pickle.load(model_file)

# Recreate the pipeline
model = Pipeline([
    ('linear_regression', fitted_model)
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
