from commons import get_tensor
from detection import get_class_name
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        image = file.read()
        class_name, phase = get_class_name(input_image=image)
        return render_template('result.html', class_label=class_name, phase = phase)


if __name__ == '__main__':
    app.run(debug=True)
