from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


def init():
    # load the saved model.
    global  predictionmodal
    predictionmodal = joblib.load("EPP_DTC_MLMODEL.pkl")


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    total_score = 0
    try:
        if request.method == 'POST':
            age = int(request.form['Age'])
            gender = int(request.form['Gender'])
            stream = int(request.form['Stream'])
            internships = int(request.form['Internships'])
            CGPA = (request.form['CGPA'])
            hostel	= int(request.form['Hostel'])
            historyofbacklogs = int(request.form['HistoryOfBacklogs'])
            
            # Predict Apparent temperature
             # Same order as the x_train dataframe
            features = [np.array([age, gender, stream, internships, CGPA, hostel, historyofbacklogs ])]
            prediction = predictionmodal.predict(features)
            finalresult = ''
            if prediction == 0:
               finalresult ='Congratulations!! You Are Placed'
            else:
                finalresult ='Opps!! Sorry You Are Not Placed'
            return render_template('index.html', result = finalresult)
    except Exception as e:
        print(e)
        return 'Calculation Error' + str(e), 500

if __name__ == '__main__':
    init()
    app.run(debug=True)
