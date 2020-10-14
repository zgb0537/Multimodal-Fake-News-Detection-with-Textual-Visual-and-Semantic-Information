from numpy.random import seed
seed(100)
import keras.callbacks as callbacks
import os, cv2
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout
import pickle
from skimage import feature
from scipy.spatial.distance import cosine
from sklearn.utils import shuffle
import performance
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def train_test_data_generation(data_file,label_file):
    # features loading
    im_LBP_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_image_LBP_feature.npy').item()
    im_tag_inception_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_image_tag_inception_feature.npy').item()
    im_tag_xception_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_image_tag_xception_feature.npy').item()
    im_tag_vgg16_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_image_tag_vgg16_feature.npy').item()
    im_tag_vgg19_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_image_tag_vgg19_feature.npy').item()
    im_tag_resnet_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_image_tag_resnet_feature.npy').item()
    text_sentiment_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_text_sentiment_dict.npy').item()
    text_embedding_feature_dict = np.load('feature_extraction/'+data_file.split('_')[0]+'_text_embedding_dict.npy').item()


    #image data path
    im_path = 'data/'+data_file.split('_')[0]+'/'+data_file.split('_')[0]+'_images/'

    data = []
    label = []

    tweets_file = open('data/'+data_file.split('_')[0]+'/'+data_file.split('_')[0]+'_twitter_data.txt', 'r', encoding='utf8')
    for line in tweets_file.readlines():
        twitter_id = line.split('\t')[0]
        twitter_text = line.split('\t')[1]
        twitter_label = line.split('\t')[-1]
        if twitter_label.strip().replace('\n', '') == 'fake':
            twitter_label = 1
        elif twitter_label.strip().replace('\n', '') == 'real':
            twitter_label = 0
        else:
            continue
        image_name = line.split('\t')[-2] + '.jpg'

        if image_name in os.listdir(im_path):
            try:
                # image features
                tag_vgg16_feature = im_tag_vgg16_feature_dict[image_name]
                tag_vgg19_feature = im_tag_vgg19_feature_dict[image_name]
                tag_xception_feature = im_tag_xception_feature_dict[image_name]
                tag_inception_feature = im_tag_inception_feature_dict[image_name]
                tag_resnet_feature = im_tag_resnet_feature_dict[image_name]

                tags_embedding_features = np.array(tag_vgg16_feature) + np.array(tag_vgg19_feature) + np.array(
                    tag_xception_feature) + np.array(tag_inception_feature) + np.array(tag_resnet_feature)

                tags_embedding_features = tags_embedding_features / 5
                tags_embedding_features = tags_embedding_features.tolist()

                im_lbp_feature = im_LBP_feature_dict[image_name]

                #text features
                text_embedding_feature = text_embedding_feature_dict[twitter_id]
                text_sentiment_feature = text_sentiment_feature_dict[twitter_id]

                # image and text semantic similarity feature
                image_text_similarity_feature = []
                text_im_similarity_vgg16 = cosine(text_embedding_feature, tag_vgg16_feature)
                text_im_similarity_vgg19 = cosine(text_embedding_feature, tag_vgg19_feature)
                text_im_similarity_resnet = cosine(text_embedding_feature, tag_resnet_feature)
                text_im_similarity_inception = cosine(text_embedding_feature, tag_inception_feature)
                text_im_similarity_xception = cosine(text_embedding_feature, tag_xception_feature)

                image_text_similarity_feature.append(text_im_similarity_vgg16)
                image_text_similarity_feature.append(text_im_similarity_vgg19)
                image_text_similarity_feature.append(text_im_similarity_resnet)
                image_text_similarity_feature.append(text_im_similarity_inception)
                image_text_similarity_feature.append(text_im_similarity_xception)

                #final feature
                tweet_feature = text_embedding_feature + text_sentiment_feature + im_lbp_feature + tags_embedding_features + image_text_similarity_feature

                data.append(tweet_feature)
                label.append(twitter_label)
            except Exception:
                print(image_name)
                continue
    np.save(data_file, data)
    np.save(label_file, label)
    return data, label

def neural_network_classifer(train_X, train_Y, test_X, test_Y, val_X, val_Y):
    callback = callbacks.EarlyStopping(monitor='val_acc', patience=100, verbose=1, mode='max')
    tensorboard = TensorBoard(log_dir='log/')
    checkpoint = ModelCheckpoint(filepath='model/best_weights.h5', monitor='val_accuracy', mode='auto', save_best_only='True')
    callback_lists = [callback, tensorboard, checkpoint]

    # input layer
    input_layer = Input(shape=(input_dim,))
    input_layer = BatchNormalization()(input_layer)

    # fc
    dense = Dense(1000)(input_layer)
    dense = Dropout(0.6)(dense)

    dense = Dense(500)(dense)
    dense = Dropout(0.6)(dense)

    dense = Dense(300)(dense)
    dense = Dropout(0.6)(dense)

    dense = Dense(100)(dense)
    dense = Dropout(0.6)(dense)

    # output layer
    output_layer = Dense(1, activation='sigmoid')(dense)
    model = Model([input_layer], outputs=[output_layer])
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_X, train_Y, batch_size=64, epochs=50, verbose=1, validation_data=(val_X, val_Y),callbacks=callback_lists, shuffle=True)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    loss, acc = model.evaluate(test_X, test_Y, batch_size=64, verbose=0)
    print(acc)
    result_predict = model.predict(test_X)
    result_predict = [round(list(i)[0]) for i in list(result_predict)]
    return result_predict

def main(data_name):

    data_file = data_name+'_data.npy'
    label_file = data_name+'_label.npy'

    if os.path.exists(data_file):
        data = np.load(data_file).tolist()
        label = np.load(label_file).tolist()
    else:
        data, label = train_test_data_generation(data_file,label_file)

    data = np.nan_to_num(data)

    print('all data number:',len(data))
    print('real news number:', labels.count(0))
    print('fake news number:', labels.count(1))

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    nn_X_train, nn_X_val, nn_Y_train, nn_Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

    #clf = RandomForestClassifier(n_estimators=100)
    #clf = svm.SVC()
    #clf.fit(X_train, Y_train)
    #results = clf.predict(X_test)

    results = neural_network_classifer(nn_X_train, nn_Y_train, X_test, Y_test, nn_X_val, nn_Y_val)
    acc,P,R,F1,report = performance.performance_measure(Y_test.tolist(),results)
    print('F1:',F1)
    print(report)

if __name__ == '__main__':
    data_names = ['politifact','gossipcop']
    for data_name in data_names:
        main(data_name)