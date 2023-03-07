import Lenet5Detection
import DetectTableIMG2
import Detect2
from flask import Flask, request, jsonify
from flask_cors import CORS
# r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# app = Flask(__name__)
app = Flask(__name__, static_folder='uploads')
app.config['JSON_AS_ASCII'] = False
CORS(app, resources=r'/*')
'''
@app.route("/")
def hello():
    print("1111111111111111")
    results=CnnMain.evaluate_one_image()
    imageFile = request.files.get('file')
    if imageFile is None:
        print( "未上传文件")
    else:
        imageFile.save("file.jpg")
        print("成功接收文件")

    #imgfileid=request.args.get('imgfileid')
    return "<h1 style='color:blue'>Hello There! from :腾讯云服务器+nginx+uwgsi+flask+深度学习模型<br></h1><h1 style='color:green'>深度学习模型运行结果："+results+"</h1>"
'''

@app.route("/",methods=["GET"])
def hello():
    # results = Lenet5Detection.evaluate_one_image('./Lenet5TestJpg/testnumber_9.jpg')
    # return "<h1 style='color:blue'>Hello There! from HTTP Get:我的服务器（向日葵）"+"<br>"+results+"</h1>"
    return "<h1 style='color:blue'>Hello There! from HTTP Get:我的服务器（花生壳）</h1>"
@app.route("/", methods=["POST"])
def save_file():
    print("接受文件")
    data = request.files
    # print(type(data))
    # print(data)
    file = data['img']
    # print(file.filename)# oiuWEhfPwlYe74d542d03a20e8c22fc1aa60de5a6b22.jpg
    # print(type(file.filename))  # <class 'str'>
    # print(type(request.headers))
    # print(request.headers)
    # print(request.headers.get('File-Name'))
    # 文件写入磁盘
    # file.save(file.filename)
    file.save(os.path.join('uploads', secure_filename(file.filename)))
    score = DetectTableIMG2.cutimage('./uploads/'+file.filename)
    # score = Detect2.cutimage('./uploads/'+file.filename)
    # results = Lenet5Detection.evaluate_one_image(file.filename)
    # results = Lenet5Detection.evaluate_one_image('./'+file.filename)
    # results = Lenet5Detection.evaluate_one_image('./uploads/'+file.filename)
    # results = Lenet5Detection.evaluate_one_image('./Lenet5TestJpg/testnumber_9.jpg')
    resp = {'filename': "results.png", 'results': score}
    return jsonify(resp)
    # return results
    # return "已接受保存"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)