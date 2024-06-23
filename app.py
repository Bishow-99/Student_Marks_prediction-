from flask import Flask,request,render_template
import numpy as np
import pandas as pd
app = Flask(__name__)
import joblib
model=joblib.load('Marks_model.pkl')

df=pd.DataFrame()

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global df
    input_features=[int(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    if input_features[0]<0 or input_features[0]> 24:
        return render_template("index.html",  prediction_message='Hours must be between 1 to 24')
    output=model.predict( features_value)[[0][0]].round(2) 
    
    #cant exceed more than 100%
    if output<=100:   
        df=pd.concat([df, pd.DataFrame({'study hours':input_features, 'predicted_ouput':[output]})], ignore_index=False)
        df.to_csv('new_marksheeet.csv')
        return render_template('index.html', prediction_message='Your score will be [{}%], if you study like [{}] hours per day'.format(output,int(features_value[0])))
    else:
        return render_template("index.html",prediction_message=f"If you study like {int(features_value[0])} hrs a day, then you can score almost 100% in your subject  ")

#To run our flask application
if __name__=="__main__":
    app.run(debug=True)


  
