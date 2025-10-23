from flask import Flask, request,render_template
from test import *

app = Flask(__name__)
@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']   
        result = predict(file)
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)