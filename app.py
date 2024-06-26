from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the info page
def infoPage():
    return render_template("info.html")

@app.route('/input', methods=['GET'])  # route to display the input page
def inputPage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            age = int(request.form['age'])
            sex = request.form['sex']
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker']
            region = request.form['region']

            # Encoding categorical variables
            sex = 0 if sex == 'male' else 1
            smoker = 0 if smoker == 'yes' else 1
            region_dict = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
            region = region_dict[region]

            data = [age, sex, bmi, children, smoker, region]
            data = np.array(data).reshape(1, -1)  # Reshaping the data

            # Creating a DataFrame for the input data
            columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            input_data = pd.DataFrame(data, columns=columns)

            obj = PredictionPipeline()
            predict = obj.predict(input_data)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
