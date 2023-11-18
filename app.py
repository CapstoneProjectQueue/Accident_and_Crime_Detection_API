from flask import Flask, render_template, jsonify, request
from flask_restx import Api, Resource #Api 개발을 위해 flask_restx 사용
# from tf import tensorflow
from PIL import Image # 이미지 처리 라이브러리
import cv2
import os
import glob

app = Flask(__name__)
api = Api(app) 
# model = tf.keras.models.load_model('C:/github/Accident_and_Crime_Detection_API/model.h5')


def prepare_img(file):
    video = cv2.VideoCapture(file)
    img_array=[]
    while True:
        ret, frame = video.read()
        frame = int(video.get(1))
        if(frame%60==0):
            img_array.append(cv2.imwrite(frame))
        if not ret:
            break
    video.release()
    return img_array

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def model():
    if request.method=='POST':
        file = request.files['video'] # 서버에서 이미지 받아오기
        if 'video' not in request.files: # 서버에서 받아온 이미지가 없을 경우
            return jsonify({'error':'영상 없음'})
        img_array=prepare_img(file)
        prediction=[]
        for i in img_array:
            prediction.append(model.predict(img_array[i]))
        for i in prediction:
            if prediction[i]==0:
                return jsonify({'result':'abnormal'})
        return jsonify({'result':'normal'}) # 결과 전송
    

if __name__=='__main__':
    app.run('0.0.0.0', port=5000, debug=True)