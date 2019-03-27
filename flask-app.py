from flask import Flask, render_template, redirect, request
import gain

app = Flask(__name__)
model = gain.GainModel()

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict_page():
    instring = request.form['input']
    if instring.startswith('{'):
        output = model.run_game(instring)
        return render_template('predict.html', output=output)
    return render_template('predict.html', output="Incorrect Input")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
