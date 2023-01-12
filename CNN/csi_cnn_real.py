# -*- coding: utf-8 -*-

from __future__ import print_function
import sklearn as sk
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import KFold, cross_val_score
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fix_cross_vali_input_data import csv_import, DataSet

keras = tf.keras

learning_rate = 0.0001  # 학습의 속도에 영향. 너무 크면 학습이 overshooting해서 학습에 실패하고, 너무 작으면 더디게 진행하다가 학습이 끝나 버린다.
training_iters = 2000
batch_size = 16
display_step = 100

# Network Parameters
window_size = 500
threshold = 60

n_input = 90  # WiFi activity data input (img shape: 90*window_size)
n_steps = window_size  # timesteps
n_hidden = 200  # hidden layer num of features original 200
n_classes = 6  # WiFi activity total classes

# Output folder
OUTPUT_FOLDER_PATTERN = "LR{0}_BATCHSIZE{1}_NHIDDEN{2}/"
output_folder = OUTPUT_FOLDER_PATTERN.format(learning_rate, batch_size, n_hidden)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

leaky_relu = tf.nn.leaky_relu

def CNN(
        input_shape=None,
        classes = 6):
    model = tf.keras.Sequential([
        # 특징 추출 부분
        # Conv 1
        tf.keras.layers.Conv2D(filters=4,
                               kernel_size=(5, 5),
                               strides=3,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=input_shape),
        # Max Pool 1
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding="valid"),
        tf.keras.layers.BatchNormalization(),

        # Conv 1
        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same",
                               activation=tf.keras.activations.relu,
                               input_shape=input_shape),
        # Max Pool 1
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=(2, 3),
                                  padding="valid"),
        tf.keras.layers.BatchNormalization(),

        # Conv 2
        tf.keras.layers.Conv2D(filters=8,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same",  # 크기 유지, 즉 여기서는 2임
                               activation=tf.keras.activations.relu),
        # Max Pool 2
        tf.keras.layers.MaxPool2D(pool_size=(3, 5),
                                  strides=(2, 3),
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        # 분류 층 부분
        # Fully connected layer 1
        tf.keras.layers.Dense(units=64,
                              activation=tf.keras.activations.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        # Fully connected layer 2
        tf.keras.layers.Dense(units=32,
                              activation=tf.keras.activations.relu),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Dense(units=32,
                              activation=tf.keras.activations.relu, name="my_intermediate_layer"),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),

        # Fully connected layer 3
        tf.keras.layers.Dense(units=classes,
                              activation=tf.keras.activations.softmax)
    ])

    return model

my_model = CNN((500, 90,1), n_classes)

print(my_model.summary())

my_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# data import
x_go_sleep_, x_pass_out_, x_just_lay_, y_go_sleep_, y_pass_out_,y_just_lay_  = csv_import()
# d = readCsiAmp_PCA()
print(" go_sleep_ =", len(x_go_sleep_), " just_lay_=", len(x_just_lay_),  " pass_out_=", len(x_pass_out_))

# data shuffle
x_go_sleep_, y_go_sleep_ = shuffle(x_go_sleep_, y_go_sleep_, random_state=0)
x_just_lay_, y_just_lay_ = shuffle(x_just_lay_, y_just_lay_, random_state=0)
# x_lay, y_lay = shuffle(x_lay, y_lay, random_state=0)
x_pass_out_, y_pass_out_ = shuffle(x_pass_out_, y_pass_out_, random_state=0)
# x_sit, y_sit = shuffle(x_sit, y_sit, random_state=0)
# x_stand, y_stand = shuffle(x_stand, y_stand, random_state=0)
# x_stn, y_stn = shuffle(x_stn, y_stn, random_state=0)


testset_size = 0.25

x_go_sleep_, x_go_sleep__t, y_go_sleep_, y_go_sleep__t = train_test_split(
    x_go_sleep_, y_go_sleep_, test_size=testset_size, shuffle=False)

x_just_lay_, x_just_lay__t, y_just_lay_, y_just_lay__t = train_test_split(
    x_just_lay_, y_just_lay_, test_size=testset_size, shuffle=False)

# x_lay, x_lay_t, y_lay, y_lay_t = train_test_split(
#     x_lay, y_lay, test_size=testset_size, shuffle=False)

x_pass_out_, x_pass_out__t, y_pass_out_, y_pass_out__t = train_test_split(
    x_pass_out_, y_pass_out_, test_size=testset_size, shuffle=False)

# x_sit, x_sit_t, y_sit, y_sit_t = train_test_split(
#     x_sit, y_sit, test_size=testset_size, shuffle=False)

# x_stand, x_stand_t, y_stand, y_stand_t = train_test_split(
#     x_stand, y_stand, test_size=testset_size, shuffle=False)

# x_stn, x_stn_t, y_stn, y_stn_t = train_test_split(
#     x_stn, y_stn, test_size=testset_size, shuffle=False)

test_x = np.r_[x_go_sleep__t, x_pass_out__t,x_just_lay__t]
test_y = np.r_[y_go_sleep__t, y_pass_out__t,y_just_lay__t]
# test_y = test_y[:, 1:]

##### main #####


# k_fold
kk = 6  # =========================================================================================
cnt = 0
# Launch the graph
for ii in range(kk * 10):
    cnt+= 1
    if cnt == 5:
      break
    i = ii % kk
    # Roll the data
    x_go_sleep_ = np.roll(x_go_sleep_, int(len(x_go_sleep_) / kk), axis=0)
    y_go_sleep_ = np.roll(y_go_sleep_, int(len(y_go_sleep_) / kk), axis=0)
    x_just_lay_ = np.roll(x_just_lay_, int(len(x_just_lay_) / kk), axis=0)
    y_just_lay_ = np.roll(y_just_lay_, int(len(y_just_lay_) / kk), axis=0)
    # x_lay = np.roll(x_lay, int(len(x_lay) / kk), axis=0)
    # y_lay = np.roll(y_lay, int(len(y_lay) / kk), axis=0)
    x_pass_out_ = np.roll(x_pass_out_, int(len(x_pass_out_) / kk), axis=0)
    y_pass_out_ = np.roll(y_pass_out_, int(len(y_pass_out_) / kk), axis=0)
    # x_sit = np.roll(x_sit, int(len(x_sit) / kk), axis=0)
    # y_sit = np.roll(y_sit, int(len(y_sit) / kk), axis=0)
    # x_stand = np.roll(x_stand, int(len(x_stand) / kk), axis=0)
    # y_stand = np.roll(y_stand, int(len(y_stand) / kk), axis=0)
    # x_stn = np.roll(x_stn, int(len(x_stn) / kk), axis=0)
    # y_stn = np.roll(y_stn, int(len(y_stn) / kk), axis=0)

  
  # data separation // np.r_은 concatenate와 동일하다고 생각됨.
    wifi_x_train = np.r_[
        x_go_sleep_[int(len(x_go_sleep_) / kk):], x_pass_out_[int(len(x_pass_out_) / kk):], x_just_lay_[int(len(x_just_lay_) / kk):]]
    print(wifi_x_train.shape)

    wifi_y_train = np.r_[
          y_go_sleep_[int(len(y_go_sleep_) / kk):], y_pass_out_[int(len(y_pass_out_) / kk):], y_just_lay_[int(len(y_just_lay_) / kk):]]

    # wifi_y_train = wifi_y_train[:, 1:] #분류 클래스를 한개 줄임?
    print(wifi_y_train.shape)
    
    wifi_x_train = tf.expand_dims(wifi_x_train, axis=-1)

    wifi_x_validation = np.r_[
        x_go_sleep_[int(len(x_go_sleep_) / kk):], x_pass_out_[int(len(x_pass_out_) / kk):], x_just_lay_[int(len(x_just_lay_) / kk):]]

    wifi_y_validation = np.r_[
          y_go_sleep_[int(len(y_go_sleep_) / kk):], y_pass_out_[int(len(y_pass_out_) / kk):], y_just_lay_[int(len(y_just_lay_) / kk):]]

    wifi_x_validation = tf.expand_dims(wifi_x_validation, axis=-1)

    # wifi_y_validation = wifi_y_validation[:, 1:]
    print("shape:", wifi_x_train.shape)

    # history = my_model.fit(wifi_x_train, wifi_y_train,
    #                        epochs=6, validation_data=(wifi_x_validation, wifi_y_validation),
    #                        batch_size=batch_size)

    history = my_model.fit(wifi_x_train, wifi_y_train,
                            epochs=6,
                            validation_data=(wifi_x_validation, wifi_y_validation),
                            batch_size=batch_size)

my_model.save('iris.h5')

# wifi_x_validation = np.r_[
#     x_go_sleep_[int(len(x_go_sleep_) / kk):], x_pass_out_[int(len(x_pass_out_) / kk):], x_just_lay_[int(len(x_just_lay_) / kk):]]

# wifi_y_validation = np.r_[
#       y_go_sleep_[int(len(y_go_sleep_) / kk):], y_pass_out_[int(len(y_pass_out_) / kk):], y_just_lay_[int(len(y_just_lay_) / kk):]]

# # wifi_y_validation = wifi_y_validation[:, 1:]
# print("shape:", wifi_x_train.shape)

# # history = my_model.fit(wifi_x_train, wifi_y_train,
# #                        epochs=6, validation_data=(wifi_x_validation, wifi_y_validation),
# #                        batch_size=batch_size)

# history = my_model.fit(wifi_x_train, wifi_y_train,
#                         epochs=18,
#                         validation_data=(wifi_x_validation, wifi_y_validation),
#                         batch_size=batch_size)

#정확성 사진 테스트
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# # plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(min(plt.ylim()), 0.7), 1])
# plt.title('Training and Validation Accuracy ' + str(ii))

# '''
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0, 1.0])
# plt.title('Training and Validation Loss' + str(ii))
# plt.xlabel('epoch')
# '''
# plt.savefig("figs/fig_" + str(ii) + ".png", dpi=150)

# print("saved!: " + str(ii))

# 실제 테스트 진행

# print("test size: ", test_x.shape)
# my_model.summary()
# loss_and_metrics = my_model.evaluate(test_x, test_y)
# print("테스트 성능 : {}%".format(round(loss_and_metrics[1] * 100, 4)))
# print(loss_and_metrics[1])

# asdf = my_model.predict(test_x)
# lst = []
# lst2 = []
# for a in asdf:
#     lst.append(np.argmax(a))
# for a in test_y:
#     lst2.append(np.argmax(a))

# print("lst", len(lst), "lst2", len(lst2))

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(lst, lst2)
# print(cm)




# # 실제 테스트 진행

# feature_extractor = keras.Model(
#     inputs=my_model.inputs,
#     outputs=my_model.get_layer(name="my_intermediate_layer").output,
# )
# features = feature_extractor(test_x)
# features = features.numpy()

# f_max = max(map(max, features))
# f_min = min(map(min, features))
# # for k in range(len(test_y)):
# #   print(np.argmax(test_y[i]))
# names = [ "none","go_sleep_","stand","pass_out", "just_lay_","stn"]

# for i, a in enumerate(features):

#     fig = plt.figure(1)
#     image = test_x[i]
#     # plt.subplot(121)
#     plt.pcolormesh(image)
#     plt.axis('off')
#     # plt.title(names[np.argmax(test_y[i])])

# #     plt.plot(122)
# #     # plt.bar(range(32), a)
# #     # plt.ylim(f_min, f_max)

#     plt.savefig("Kalman_FT_Feature/cv_"+names[np.argmax(test_y[i])] + str(i) + ".png", dpi=150)
#     plt.close(fig)
#     print('done:', i)

