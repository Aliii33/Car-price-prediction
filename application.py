from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('Lasso.pkl', 'rb'))
car = pd.read_csv('Cleaned dataset.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    brand = sorted(car['brand'].unique())
    model = sorted(car['model'].unique())
    model_year = sorted(car['model_year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    brand.insert(0, 'Select Company')


    return render_template('index.html', brand=brand, model=model, model_year=model_year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    brand = request.form.get('brand')

    model = request.form.get('model')
    model_year = request.form.get('model_year')
    fuel_type = request.form.get('fuel_type')
    milage = request.form.get('milage')

    prediction = model.predict(pd.DataFrame(columns=['model', 'brand', 'model_year', 'milage', 'fuel_type'],
                                            data=np.array([model, brand, model_year, milage, fuel_type]).reshape(1, 5)))
    print(str(np.round(prediction[0], 2)))

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()
