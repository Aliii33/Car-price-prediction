from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('Lasso.pkl','rb'))
car=pd.read_csv('Cleaned dataset.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['brand'].unique())
    car_models=sorted(car['model'].unique())
    year=sorted(car['model_year'].unique(),reverse=True)
    fuel_types=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, year=year,fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    brand = request.form.get('brand')
    model_name = request.form.get('model')
    model_year = int(request.form.get('model_year'))
    fuel_type = request.form.get('fuel_type')
    mileage = int(request.form.get('milage'))

    # Create a DataFrame with user input
    user_input = pd.DataFrame([[brand, model_name, model_year, mileage, fuel_type]],
                              columns=['brand', 'model', 'model_year', 'milage', 'fuel_type'])

    # Make predictions using the loaded model
    prediction = model.predict(user_input)

    print(prediction)

    return str(np.round(prediction[0],2))

if __name__=='__main__':
    app.run()
