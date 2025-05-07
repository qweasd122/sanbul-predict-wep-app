import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qweasd122'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ 12-Dec)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        x_test = pd.DataFrame([{
            'longitude': float(form.longitude.data),
            'latitude': float(form.latitude.data),
            'month': form.month.data,
            'day': form.day.data,
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data)}])
        print(x_test.shape)
        print(x_test)

        data = pd.read_csv('sanbul2district-divby100.csv', sep=',')
        x, y = train_test_split(data, test_size=0.2, random_state=42)
        data = x.drop('burned_area', axis=1)
        data_num = data.drop(['month', 'day'], axis=1)

        num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
        cat_attribs = ['month', 'day']

        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler())
        ])
        x_test_tr = num_pipeline.fit_transform(data_num)

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs)
        ])

        full_pipeline.fit(data)
        x_test = full_pipeline.transform(x_test)

        model = keras.models.load_model('fires_model.keras')

        prediction = model.predict(x_test)
        res = prediction[0][0]
        res = np.round(res, 2)

        return render_template('result.html', res=res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()