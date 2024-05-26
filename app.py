from flask import Flask, request, jsonify
from Main import Main

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        main = Main()
        pred_class = main.main(request.json.get("img_path"), False)
        return jsonify({'classe': pred_class})
    
if __name__ == '__main__':
    app.run(port=5000)