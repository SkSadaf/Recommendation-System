# ------------ Importing Libraries -------------- #

import json
import gzip
import gensim
import random
import keras
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.layers.core import Dropout
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import confusion_matrix
# from mealpy.swarm_based.BES import BaseBES as BES
from mealpy.swarm_based.BA import ModifiedBA as BA
from mealpy.swarm_based.CSO import BaseCSO as CSO
from mealpy.swarm_based.CSO import ModifiedCSO as hybrid
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Add
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Lambda, Reshape


"""## Preprocessing"""
#
# # Hyperparameter
# dataName = "Baby"  # name of your product
# vector_size = 100  # vector size of review text and item description
# epochs = 5


def get_max_users_max_items(filename):
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        # print(line)
        while line is not None and line != "":
            arr = line.split("  ")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
            # print(line)
    return num_users, num_items

# Construct  sparse matrix with size of maxmuim of users and items


def load_rating_file_as_matrix(filename):
    num_users, num_items = get_max_users_max_items(filename)
    # print(num_users , num_items)
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("  ")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if rating > 0:
                mat[user, item] = rating
            line = f.readline()
    return mat
# get the  text vector for  user_review and item_description


def load_review_feature(filename):
    dict = {}
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            fea = line.strip('\n').split(',')
            index = int(fea[0])
            if index not in dict:
                dict[index] = fea[1:]
            line = f.readline()
    return dict


# # get read all files we have saved in previous stage
# path = 'Data/baby_recom/' + dataName
# trainMatrix = load_rating_file_as_matrix(path + ".train.rating")
# user_review_fea = load_review_feature(path + ".user")
# item_review_fea = load_review_feature(path + ".item")
# testRatings = load_rating_file_as_matrix(path + ".test.rating")
# num_users, num_items = trainMatrix.shape

# print(num_users, num_items)


# ----------------------- Optimization -------------------- #
def fitness_function1(solution, model, test_data, test_label):
    # try:
    opt_val = model.get_weights()
    opt_val_1 = opt_val[0]
    solution.reshape(opt_val_1.shape[0], opt_val_1.shape[1])
    opt_val[0] = solution.reshape(opt_val_1.shape[0], opt_val_1.shape[1])
    model.set_weights(opt_val)
    y_pred = model.predict(test_data)
    # Acc =sklearn.metrics.accuracy_score(np.argmax(test_label, axis = 1), np.argmax(y_pred, axis = 1))
    # Acc = perf_evalution_CM(test_label, y_pred)
    Acc = mean_squared_error(test_label, y_pred)
    return Acc


def update_weights_model_BA(model, test_data, test_lab, opt):
    # named_model = dict(model.named_modules())
    to_opt = model.get_weights()
    to_opt_1 = np.array(to_opt[0])
    to_opt_2 = to_opt_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    wei_to_train_1 = main_weight_updation_optimization(to_opt_2, model, test_data, test_lab, opt)
    to_opt[0] = wei_to_train_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    model.set_weights(to_opt)
    # model.predict(test_data)
    # print("MSE OF OPT : ", mean_squared_error(model.predict(test_data), test_lab))
    # model = model
    return model


def update_weights_model_CSO(model, test_data, test_lab, opt):
    # named_model = dict(model.named_modules())
    to_opt = model.get_weights()
    to_opt_1 = np.array(to_opt[0])
    to_opt_2 = to_opt_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    wei_to_train_1 = main_weight_updation_optimization(to_opt_2, model, test_data, test_lab, opt)
    to_opt[0] = wei_to_train_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    model.set_weights(to_opt)
    # model.predict(test_data)
    # print("MSE OF OPT : ", mean_squared_error(model.predict(test_data), test_lab))
    # model = model
    return model


def update_weights_model_hybrid(model, test_data, test_lab, opt):
    # named_model = dict(model.named_modules())
    to_opt = model.get_weights()
    to_opt_1 = np.array(to_opt[0])
    to_opt_2 = to_opt_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    wei_to_train_1 = main_weight_updation_optimization(to_opt_2, model, test_data, test_lab, opt)
    to_opt[0] = wei_to_train_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    model.set_weights(to_opt)
    # model.predict(test_data)
    # print("MSE OF OPT : ", mean_squared_error(model.predict(test_data), test_lab))
    # model = model
    return model


def main_weight_updation_optimization(curr_wei, model, test_data, test_lab, opt):

    problem_dict1 = {
        "fit_func": fitness_function1,
        "lb": [curr_wei.min(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "ub": [curr_wei.max(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "minmax": "min",
        "log_to": None,
        "save_population": True,
        "Curr_Weight": curr_wei,
        "Model_trained_Partial": model,
        "test_label": test_lab,
        "test_loader": test_data,
                    }
    if opt == 1:
        print(" BAT Optimization")
        model = BA(problem_dict1, epoch=1, pop_size=10)
    elif opt == 2:
        print(" Cat Optimization")
        model = CSO(problem_dict1, epoch=1, pop_size=10)
    elif opt == 3:
        print(" Hybrid Optimization")
        model = hybrid(problem_dict1, epoch=1, pop_size=10)
    # best_position2, best_fitness2 = curr_wei, model.solve()
    best_position2, best_fitness2 = model.solve()
    print(best_position2)
    return best_position2
# build model
# Intiliaze the model input


def user_recom_stat(vector_size, num_users, user_input1, user_fea1, labels1, user_input_test, user_fea_test, epochs):

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_sent = Input(shape=(vector_size,), dtype='float32', name='user_sentiment')

    # Embedding layer for user
    Embedding_User = Embedding(input_dim=num_users, input_length=1, output_dim=vector_size, name='user_embedding')

    # User Sentiment Dense Network

    def user_Sentiment(user_latent, user_sent):
        latent_size = user_latent.shape[1]
        inputs = user_sent
        layer = Dense(latent_size, activation='relu', name='user_attention_layer')(inputs)
        sent = Lambda(lambda x: K.softmax(x), name='user_Sentiment_softmax')(layer)
        output = Multiply()([user_latent, sent])
        return output

    # Crucial to flatten an embedding vector
    user_latent = Reshape((vector_size,))(Flatten()(Embedding_User(user_input)))
    user_latent_atten = user_Sentiment(user_latent, user_sent)

    user_latent = Dense(vector_size, activation='relu')(user_latent_atten)

    # review-based attention calculation
    vec = Multiply()([user_latent, user_sent])
    user_item_concat = Concatenate()([user_sent, user_latent])
    att = Dense(vector_size, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)

    # Element-wise product of user and item embeddings
    predict_vec = Multiply()([vec, att])
    # Final prediction layer
    prediction = Dense(vector_size, activation='relu')(predict_vec)
    # For overfitting
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(1, name='prediction')(prediction)

    model = Model(inputs=[user_input, user_sent], outputs=prediction)

    model.summary()

    # compile our model
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['mean_squared_error'])

    # using callbacks to avoid overfitting and save best model during training
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    checkpoint = ModelCheckpoint('Data/model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # fit our model
    hist = model.fit([user_input1, user_fea1], labels1, batch_size=64, epochs=epochs,
                     validation_split=0.2, verbose=1, callbacks=[earlystopping, checkpoint])
    # pred = model.predict([user_input_test, user_fea_test])
    # pred = model.predict([user_input_test, user_fea_test])
    return model


def item_recom_stat(vector_size, num_items, item_input1, item_fea1, labels1, item_input_test, item_fea_test, epochs):

    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_cont = Input(shape=(vector_size,), dtype='float32', name='item_content')

    # Embedding layer for item
    Embedding_Item = Embedding(input_dim=num_items, input_length=1, output_dim=vector_size, name='item_embedding')

    def item_Content(item_latent, item_cont):
        latent_size = item_latent.shape[1]
        inputs = item_cont
        layer = Dense(latent_size, activation='relu', name='item_attention_layer')(inputs)
        cont = Lambda(lambda x: K.softmax(x), name='item_Content_softmax')(layer)
        output = Multiply()([item_latent, cont])
        return output

    # Crucial to flatten an embedding vector
    item_latent = Reshape((vector_size,))(Flatten()(Embedding_Item(item_input)))
    item_latent_atten = item_Content(item_latent, item_cont)

    item_latent = Dense(vector_size, activation='relu')(item_latent_atten)

    # review-based attention calculation
    vec = Multiply()([item_cont, item_latent])
    user_item_concat = Concatenate()([ item_cont, item_latent])
    att = Dense(vector_size, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)

    # Element-wise product of user and item embeddings
    predict_vec = Multiply()([vec, att])
    # Final prediction layer
    prediction = Dense(vector_size, activation='relu')(predict_vec)
    # For overfitting
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(1, name='prediction')(prediction)

    model = Model(inputs=[item_input, item_cont], outputs=prediction)

    model.summary()

    # compile our model
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['mean_squared_error'])

    # using callbacks to avoid overfitting and save best model during training
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    checkpoint = ModelCheckpoint('Data/model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # fit our model
    hist = model.fit([item_input1, item_fea1], labels1, batch_size=64, epochs=epochs,
                     validation_split=0.2, verbose=1, callbacks=[earlystopping, checkpoint])
    # pred = model.predict([item_input_test, item_fea_test])
    # pred = model.predict([item_input_test, item_fea_test])

    return model


def hybrid_cnn_model(vector_size, num_users,num_items,  user_input1, user_fea1, item_input1, item_fea1, labels1,user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt):

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_sent = Input(shape=(vector_size,), dtype='float32', name='user_sentiment')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_cont = Input(shape=(vector_size,), dtype='float32', name='item_content')

    # Embedding layer for user
    Embedding_User = Embedding(input_dim=num_users, input_length=1, output_dim=vector_size, name='user_embedding')

    # Embedding layer for item
    Embedding_Item = Embedding(input_dim=num_items, input_length=1, output_dim=vector_size, name='item_embedding')

    # User Sentiment Dense Network

    def user_Sentiment(user_latent, user_sent):
        latent_size = user_latent.shape[1]
        inputs = user_sent
        layer = Dense(latent_size, activation='relu', name='user_attention_layer')(inputs)
        sent = Lambda(lambda x: K.softmax(x), name='user_Sentiment_softmax')(layer)
        output = Multiply()([user_latent, sent])
        return output
    # Item Content Dense Network

    def item_Content(item_latent, item_cont):
        latent_size = item_latent.shape[1]
        inputs = item_cont
        layer = Dense(latent_size, activation='relu', name='item_attention_layer')(inputs)
        cont = Lambda(lambda x: K.softmax(x), name='item_Content_softmax')(layer)
        output = Multiply()([item_latent, cont])
        return output

    # Crucial to flatten an embedding vector
    user_latent = Reshape((vector_size,))(Flatten()(Embedding_User(user_input)))
    item_latent = Reshape((vector_size,))(Flatten()(Embedding_Item(item_input)))
    user_latent_atten = user_Sentiment(user_latent, user_sent)
    item_latent_atten = item_Content(item_latent, item_cont)

    user_latent = Dense(vector_size, activation='relu')(user_latent_atten)
    item_latent = Dense(vector_size, activation='relu')(item_latent_atten)

    # review-based attention calculation
    vec = Multiply()([user_latent, item_latent])
    user_item_concat = Concatenate()([user_sent, item_cont, user_latent, item_latent])
    att = Dense(vector_size, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)

    # Element-wise product of user and item embeddings
    predict_vec = Multiply()([vec, att])
    # Final prediction layer
    prediction = Dense(vector_size, activation='relu')(predict_vec)
    # For overfitting
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(1, name='prediction')(prediction)

    model = Model(inputs=[user_input, user_sent, item_input, item_cont], outputs=prediction)

    model.summary()

    # compile our model
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['mean_squared_error'])

    # using callbacks to avoid overfitting and save best model during training
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    checkpoint = ModelCheckpoint('Data/model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # fit our model
    hist = model.fit([user_input1, user_fea1, item_input1, item_fea1], labels1, batch_size=64, epochs=epochs,
                     validation_split=0.2, verbose=1, callbacks=[earlystopping, checkpoint])
    # pred = model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])

    if opt == 0:
        model = model
    elif opt == 1:
        model = model
        model = update_weights_model_BA(model, [user_input1, user_fea1, item_input1, item_fea1], labels1, opt)
    elif opt == 2:
        model = model
        model = update_weights_model_CSO(model, [user_input1, user_fea1, item_input1, item_fea1], labels1, opt)
    elif opt == 3:
        model = model
        model = update_weights_model_hybrid(model, [user_input1, user_fea1, item_input1, item_fea1], labels1, opt)
    # preds = model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
    return model


def A3NCF_model(num_users, num_items, user_input1, user_fea1, item_input1, item_fea1, labels1,user_input_test, user_fea_test, item_input_test, item_fea_test, k, latent_dim, epochs, regs=[0, 0]):

    # get_custom_objects().update({'vallina_relu': Activation(vallina_relu)})
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_fea = Input(shape=(k,), dtype='float32', name='user_fea')

    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_fea = Input(shape=(k,), dtype='float32', name='item_fea')

    MF_Embedding_User = Embedding( name='user_embedding', output_dim=latent_dim,
                                  embeddings_regularizer=l2(regs[0]), input_dim=num_users, input_length=1)
    MF_Embedding_Item = Embedding( name='item_embedding', output_dim=latent_dim,
                                  embeddings_regularizer=l2(regs[0]), input_dim=num_items, input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    user_latent = Add()([user_fea, user_latent])
    item_latent = Add()([item_fea, item_latent])
    user_latent = Dense(latent_dim, kernel_initializer='glorot_normal', activation='relu')(user_latent)
    item_latent = Dense(latent_dim, kernel_initializer='glorot_normal', activation='relu')(item_latent)
    user_item_concat = Concatenate()([user_fea, item_fea, user_latent, item_latent])
    att = Dense(latent_dim, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)
    vec = keras.layers.Multiply()([user_latent, item_latent])

    # Element-wise product of user and item embeddings
    predict_vec = keras.layers.Multiply()([vec, att])

    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(latent_dim, kernel_initializer='glorot_normal', activation='relu')(predict_vec)
    prediction = keras.layers.core.Dropout(0.5)(prediction)
    prediction = Dense(1, kernel_initializer='glorot_normal', name='prediction')(prediction)

    model = Model(inputs=[user_input, user_fea, item_input, item_fea], outputs=prediction)
    model.summary()

    # compile our model
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['mean_squared_error'])

    # using callbacks to avoid overfitting and save best model during training
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    checkpoint = ModelCheckpoint('Data/model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # fit our model
    hist = model.fit([user_input1, user_fea1, item_input1, item_fea1], labels1, batch_size=64, epochs=epochs,
                     validation_split=0.2, verbose=1, callbacks=[earlystopping, checkpoint])
    # pred = model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
    return model


# # # get instance of training dataset for train the moodel
# user_input = np.load("npy_dat/baby_npy/user_input.npy")
# user_fea = np.load("npy_dat/baby_npy/user_fea.npy")
# item_input = np.load("npy_dat/baby_npy/item_input.npy")
# item_fea = np.load("npy_dat/baby_npy/item_fea.npy")
# labels = np.load("npy_dat/baby_npy/labels.npy")


# hist0, model1 = user_recom_stat(vector_size, num_users, user_input, user_fea, labels, epochs)
# hist11, model2 = item_recom_stat(vector_size, num_items, item_input, item_fea, labels, epochs)
# hist, model = uter_model(vector_size, num_users, user_input, user_fea, item_input, item_fea, labels, epochs, opt=0)
# hist1, opt_model = uter_model(vector_size, num_users, user_input, user_fea, item_input, item_fea, labels, epochs, opt=1)
# #
# user_input_test = np.load("npy_dat/baby_npy/user_input_test.npy")
# user_fea_test = np.load("npy_dat/baby_npy/user_fea_test.npy")
# item_input_test = np.load("npy_dat/baby_npy/item_input_test.npy")
# item_fea_test = np.load("npy_dat/baby_npy/item_fea_test.npy")
# test_label = np.load("npy_dat/baby_npy/test_label.npy")

# # --------------- evulate the test dataset ----------------- #
# print("MAE of UTER model : ", mean_absolute_error(model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# print("MSE of UTER model : ", mean_squared_error(model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# pred = model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
#
# # --------- evaluate the test dataset ------------ #
# print("MAE of Optimization model : ", mean_absolute_error(opt_model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# print("MSE of Optimization model : ", mean_squared_error(opt_model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# # pred_opt = opt_model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
# # print("Prediction of UTER : \n", pred)
# # print("Prediction of Opt : \n", pred_opt)
#
# # ------------- Draw the progress during the training -------------- #
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# epochs = range(1, len(loss) + 1)
#
#
# # -------------- loss plot ----------------- #
#
# plt.plot(epochs, loss, color='red', label='Training Loss')
# plt.plot(epochs, val_loss, color='blue', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # ------------- Draw the progress during the training -------------- #
# loss = hist1.history['loss']
# val_loss = hist1.history['val_loss']
# epochs = range(1, len(loss) + 1)
#
#
# # -------------- loss plot ----------------- #
# plt.plot(epochs, loss, color='red', label='Training Loss')
# plt.plot(epochs, val_loss, color='blue', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# model_2, pred_4 = hybrid_cnn_model(vector_size, num_users, user_input, user_fea, item_input, item_fea, labels,
#                                     user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=0)
#
# # model1 = A3NCF_model(num_users, num_items, user_input, user_fea, item_input, item_fea, labels, vector_size, latent_dim=100, regs=[0, 0])
# print("MAE of UTER model : ", mean_absolute_error(pred_4.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
# a = np.load("npy_dat/baby_npy/test_label.npy")
# a1 = []
# a2 = []
# a3 = []
# a4 = []
# a5 = []
# for i in range(len(a)):
#     if a[i] == 1:
#         a1.append(a[i])
#     elif a[i] == 2:
#         a2.append(a[i])
#     elif a[i] == 3:
#         a3.append(a[i])
#     elif a[i] == 4:
#         a4.append(a[i])
#     elif a[i] == 5:
#         a5.append(a[i])

#
# def orginal_Rating(test_label, db):
#
#     perf_A = []
#     perf_B = []
#     perf_C = []
#     perf_D = []
#     perf_E = []
#     perf_F = []
#     perf_G = []
#
#     epochs = 50
#     tr_per = 0.8
#     a = test_label
#     a1 = []
#     a2 = []
#     a3 = []
#     a4 = []
#     a5 = []
#     for i in range(len(a)):
#         if a[i] == 1:
#             a1.append(a[i])
#         elif a[i] == 2:
#             a2.append(a[i])
#         elif a[i] == 3:
#             a3.append(a[i])
#         elif a[i] == 4:
#             a4.append(a[i])
#         elif a[i] == 5:
#             a5.append(a[i])
#
#     for rating in range(1, 6):
#         # ----------------- Varying the Rating  ---------------- #
#
#         [MAE1, MSE1, RMSE1] = error_metrics(a1, test_label)
#         [MAE2, MSE2, RMSE2] = error_metrics(a2, test_label)
#         [MAE3, MSE3, RMSE3] = error_metrics(a3, test_label)
#         [MAE4, MSE4, RMSE4] = error_metrics(a4, test_label)
#         [MAE5, MSE5, RMSE5] = error_metrics(a5, test_label)
#         # [MAE6, MSE6, RMSE6] = error_metrics(a6, test_label)
#         # [MAE7, MSE7, RMSE7] = error_metrics(a7, test_label)
#
#         per_1 = [MAE1, MSE1, RMSE1]
#         per_2 = [MAE2, MSE2, RMSE2]
#         per_3 = [MAE3, MSE3, RMSE3]
#         per_4 = [MAE4, MSE4, RMSE4]
#         per_5 = [MAE5, MSE5, RMSE5]
#         # per_6 = [MAE6, MSE6, RMSE6]
#         # per_7 = [MAE7, MSE7, RMSE7]
#
#         perf_A.append(per_1)
#         perf_B.append(per_2)
#         perf_C.append(per_3)
#         perf_D.append(per_4)
#         perf_E.append(per_5)
#         # perf_F.append(per_6)
#         # perf_G.append(per_7)
#
#     if db == 0:
#         np.save('{0}\\npy\\db_1\\recom_O_comp_A'.format(os.getcwd()), perf_A)
#         np.save('{0}\\npy\\db_1\\recom_O_comp_B'.format(os.getcwd()), perf_B)
#         np.save('{0}\\npy\\db_1\\recom_O_comp_C'.format(os.getcwd()), perf_C)
#         np.save('{0}\\npy\\db_1\\recom_O_comp_D'.format(os.getcwd()), perf_D)
#         np.save('{0}\\npy\\db_1\\recom_O_comp_E'.format(os.getcwd()), perf_E)
#         # np.save('{0}\\npy\\db_1\\recom_O_comp_F'.format(os.getcwd()), perf_F)
#         # np.save('{0}\\npy\\db_1\\recom_O_comp_G'.format(os.getcwd()), perf_G)
#     elif db == 1:
#         np.save('{0}\\npy\\db_2\\recom_O_comp_A'.format(os.getcwd()), perf_A)
#         np.save('{0}\\npy\\db_2\\recom_O_comp_B'.format(os.getcwd()), perf_B)
#         np.save('{0}\\npy\\db_2\\recom_O_comp_C'.format(os.getcwd()), perf_C)
#         np.save('{0}\\npy\\db_2\\recom_O_comp_D'.format(os.getcwd()), perf_D)
#         np.save('{0}\\npy\\db_2\\recom_O_comp_E'.format(os.getcwd()), perf_E)
#         # np.save('{0}\\npy\\db_2\\recom_O_comp_F'.format(os.getcwd()), perf_F)
#         # np.save('{0}\\npy\\db_2\\recom_O_comp_G'.format(os.getcwd()), perf_G)
#     elif db == 2:
#         np.save('{0}\\npy\\db_3\\recom_O_comp_A'.format(os.getcwd()), perf_A)
#         np.save('{0}\\npy\\db_3\\recom_O_comp_B'.format(os.getcwd()), perf_B)
#         np.save('{0}\\npy\\db_3\\recom_O_comp_C'.format(os.getcwd()), perf_C)
#         np.save('{0}\\npy\\db_3\\recom_O_comp_D'.format(os.getcwd()), perf_D)
#         np.save('{0}\\npy\\db_3\\recom_O_comp_E'.format(os.getcwd()), perf_E)
#         # np.save('{0}\\npy\\db_3\\recom_O_comp_F'.format(os.getcwd()), perf_F)
#         # np.save('{0}\\npy\\db_3\\recom_O_comp_G'.format(os.getcwd()), perf_G)
