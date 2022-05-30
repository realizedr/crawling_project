import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    #scikit-learn 설치
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df = pd.read_csv('./crawling_data(2)/wadiz_데이터3_20220530.csv')

X = preprocessed_data
Y = df['winner']


encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
# print(labeled_Y[:3])
label = encoder.classes_
# print(label)
with open('./models/encoder2.pickle', 'wb') as f:
    pickle.dump(encoder, f)


token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1

with open('./models/news_token2.pickle', 'wb') as f:
    pickle.dump(token, f)

max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)


X_train, X_test, Y_train, Y_test = np.load(
    './crawling_data(2)/hit.npy', allow_pickle = True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same',
#           input_shape = (64, 64, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same',
#            activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(32, kernel_size=(3, 3), padding = 'same',
#            activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['binary_accuracy'])
# early_stopping = EarlyStopping(monitor='val_binary_accuracy',
#                                patience=7)  # 매 에폭마다 밸리데이션을 하는데 어큐러시는 오르는데 벨리데이션 어큐러시는
# # 7 에폭까지 돌려서 더이상 좋아지지 않으면 멈춘다.
#
# fit_hist = model.fit(X_train, Y_train, batch_size=64,
#                      epochs=100, validation_split=0.15,
#                      callbacks=[early_stopping])
# score = model.evaluate(X_test, Y_test)
# print('Evaluation loss :', score[0])
# print('Evaluation accuracy :', score[1])
# model.save('./cat_and_dog_{}.h5'.format(str(np.around(score[1], 3))))
#
# plt.plot(fit_hist.history['binary_accuracy'], label='binary_accuracy')
# plt.plot(fit_hist.history['val_binary_accuracy'], label='val_binary_accuracy')
# plt.legend()
# plt.show()
#
# plt.plot(fit_hist.history['loss'], label='loss')
# plt.plot(fit_hist.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()