import numpy as np
import pandas as pd
import tensorflow as tf
import os 

import cv2
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

csv_file_path = 'C:\\Users\\USER\\Desktop\\capstonequeue\\code\\data.csv'  # CSV 파일 경로를 지정해야 함.
data_info = pd.read_csv(csv_file_path)

## Loading Dataset
data = []
labels = []
classes = 2
cur_path = os.getcwd()

# "비정상" "정상" 폴더에서 이미지 로드. 폴더명 임시로 넣음
abnormal_path = 'C:\\Users\\USER\\Desktop\\capstonequeue\\img\\abnormal'
normal_path = 'C:\\Users\\USER\\Desktop\\capstonequeue\\img\\normal'

# for i, path in enumerate([abnormal_path, normal_path]):
#     images = os.listdir(path)

#     for a in images:
#         try:
#             image = Image.open(os.path.join(path, a))
#             image = image.resize((30,30))
#             image = np.array(image)
#             data.append(image)
#             labels.append(i) #비정상에 0, 정상에 1
#         except:
#             print("Error loading image")

for index, row in data_info.iterrows():
    try:
        image = Image.open(row['Path'])  # 이미지 파일 경로는 CSV 파일에 있는 'Path' 컬럼을 사용합니다.
        image = image.resize((30, 30))
        image = np.array(image)
        data.append(image)
        labels.append(row['ClassId'])  # 라벨 정보는 CSV 파일에 있는 'ClassId' 컬럼을 사용합니다.
    except:
        print("Error loading image")

data = np.array(data)
labels = np.array(labels)

## Data Splitting and conversion
#Checking data shape
print(data.shape, labels.shape)

# #Splitting training and testing dataset
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# y_train = np.array(y_train)
# y_test = np.array(y_test)

# 이미지 경로와 라벨 추출
image_paths = data_info['Path'].values
labels = data_info['ClassId'].values


# 이미지 데이터와 라벨을 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)


# 분할된 데이터를 DataFrame으로 생성
train_data = pd.DataFrame({'Path': X_train, 'ClassId': y_train})
test_data = pd.DataFrame({'Path': X_test, 'ClassId': y_test})

# 분할된 데이터를 CSV 파일로 저장
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)


# 전처리
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((30, 30))
    image = np.array(image) / 255.0  # 이미지 데이터 정규화
    return image

# 이미지 경로를 통해 데이터 로드
X_train = np.array([load_and_preprocess_image(path) for path in train_data['Path']])
X_test = np.array([load_and_preprocess_image(path) for path in test_data['Path']])

y_train = np.array(train_data['ClassId'], dtype=np.float32)
y_test = np.array(test_data['ClassId'], dtype=np.float32)
####################################################################


## Creating and Compiling the Model
# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilation of the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#M odel display
model.summary()

## Training the Model
epochs = 30 #최적 수: 10~50
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

## Visualizing the performance of the Model during Training Phase
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

print("Success")

## Loading Test Dataset and Evaluating the Model
# testing accuracy on test dataset
from sklearn.metrics import accuracy_score

# Importing the test dataset
y_test = pd.read_csv('C:\\Users\\USER\\Desktop\\capstonequeue\\code\\test_data.csv') # 실제 csv파일 데이터셋 경로로 바꿔줘야함
# 그리고 우리가 현재 이 csv파일이 없음. 직접 만들어야함. 이 csv는 테스트 데이터에 대한
# 클래스, 레이블 및 경로 정보 포함하고 있어야함. 

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

# Retreiving the images
for img in imgs:
    image = Image.open(img)  # 실제 데이터셋 경로로 수정
    image = image.resize([30, 30])
    data.append(np.array(image))



X_test = np.array(data)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.8).astype("int32")

# 예측 결과를 "정상" 또는 "비정상"으로 매핑
class_labels = {0: "비정상", 1: "정상"}
predicted_labels = [class_labels[i] for i in y_pred.flatten()]

# 예측 결과 출력
for i, label in enumerate(predicted_labels):
    print(f"이미지 {i+1}: {label}")

#Accuracy with the test data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(labels, y_pred)
prec = precision_score(labels, y_pred)
rec = recall_score(labels, y_pred)
f1 = f1_score(labels, y_pred)

## print results
print(f"정확도(Accuracy): {acc:.4f}")
print(f"정밀도(Precision): {prec:.4f}")
print(f"재현율(Recall): {rec:.4f}")
print(f"F1 점수(F1 Score): {f1:.4f}")

## Save Model
model.save('abnormal_normal_classifier.h5')