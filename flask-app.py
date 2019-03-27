from flask import Flask, render_template, redirect, request
import gain

app = Flask(__name__)
model = gain.GainModel()

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict_page():
    if request.is_json():
        output = model.run_game(str(request.get_json()))
        return render_template('predict.html', output)

    return redirect('/')

