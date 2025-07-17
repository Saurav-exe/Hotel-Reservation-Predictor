import joblib
import numpy as np
from config.paths_configs import MODEL_OUTPUT_PATH
from flask import Flask, render_template, request

app = Flask(__name__)
#MODEL_OUTPUT_PATH='D:/MLOPS/Project1-Hotel Reservation Predictor/MLOPS p1/artifacts/model/lgbm_model.pkl/lgbm_model.pkl'
loaded_model= joblib.load(MODEL_OUTPUT_PATH)



#loaded_model = joblib.load(MODEL_OUTPUT_PATH)
print("Model loaded successfully")
@app.route('/', methods=['GET', 'POST'])

def index():
  if request.method=='POST':

    no_of_adults= int(request.form['no_of_adults'])
    no_of_children= int(request.form['no_of_children'])
    no_of_weekend_nights= int(request.form['no_of_weekend_nights'])
    no_of_week_nights= int(request.form['no_of_week_nights'])
    type_of_meal_plan= int(request.form['type_of_meal_plan'])
    required_car_parking_space= int(request.form['required_car_parking_space'])
    room_type_reserved= int(request.form['room_type_reserved'])
    lead_time= int(request.form['lead_time'])
    arrival_month= int(request.form['arrival_month'])
    arrival_year= int(request.form['arrival_year'])
    arrival_date= int(request.form['arrival_date'])
    market_segment_type= int(request.form['market_segment_type'])
    repeated_guest= int(request.form['repeated_guest'])
    no_of_previous_cancellations= int(request.form['no_of_previous_cancellations'])
    no_of_previous_bookings_not_canceled= int(request.form['no_of_previous_bookings_not_canceled'])
    avg_price_per_room= float(request.form['avg_price_per_room'])
    no_of_special_requests= int(request.form['no_of_special_requests'])

    features= np.array([[no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan,required_car_parking_space, room_type_reserved, lead_time, arrival_month, arrival_year, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests]])

    prediction= loaded_model.predict(features)

    return render_template('index.html', prediction=prediction[0])
  
  return render_template('index.html', prediction=None)

if __name__ == '__main__':
  app.run(host='0.0.0.0' ,port=5000)