'''from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_input():
    data = request.json
    input_data = data.get('input')
    
    # 處理輸入資料並生成輸出
    output_data = process(input_data)
    
    return jsonify({'output': output_data})

    # 這裡是處理輸入資料的邏輯
def process(input_data):
    # 假設我們只是將輸入資料轉為大寫
    return input_data.upper()

if __name__ == '__main__':
    app.run(debug=True)'''