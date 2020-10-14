import cv2,os
import numpy as np
from keras.applications.vgg16 import decode_predictions
from keras.applications import ResNet50, Xception, InceptionV3, VGG16, VGG19
from keras.preprocessing import image as Image
from keras.applications.vgg16 import preprocess_input
from tqdm import tqdm
from skimage import feature

#LBP feature
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="nri_uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints*(self.numPoints-1) + 3),
            range=(0, self.numPoints*(self.numPoints-1) + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

def embedding_load(embedding_path):
    embedding_vector = {}
    f = open(embedding_path,'r',encoding='utf8')
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        embedding_vector[word] = coef
    f.close()
    return embedding_vector

#tag embedding feature
def im_tag_embedding_feature_extraction(img_path, model,embedding_vector,im_size):
    img = Image.load_img(img_path, target_size=(im_size, im_size))
    x = Image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    yhat = model.predict(x)

    labels = decode_predictions(yhat,top=10)
    #print(labels[0])
    words = []
    for label in labels[0]:
        word = label[1]
        #print(word)
        words.append(word)
    tags_matrix = []
    zero_array = np.zeros(300)
    for tag in words:
        if tag in embedding_vector.keys():
            tag_embedding = embedding_vector[tag]
            tags_matrix.append(np.array(tag_embedding))
            zero_array = zero_array+np.array(tag_embedding)
    tag_feature = zero_array / len(tags_matrix)
    return list(tag_feature)

if __name__ == '__main__':

    embedding_path = '../data/GoogleNews_vectors_negative_300d.txt'
    embedding_vector = embedding_load(embedding_path)
    im_path = '../data/politifact_images/'   #../data/gossipcop_images/

    model_vgg16 = VGG16(weights='imagenet', include_top=True)
    model_vgg19 = VGG19(weights='imagenet', include_top=True)
    model_resnet = ResNet50(weights='imagenet', include_top=True)
    model_inception = InceptionV3(weights='imagenet', include_top=True)
    model_xception = Xception(weights='imagenet', include_top=True)

    lbp_feature_dict = {}
    vgg16_tags_embedding_feature_dict = {}
    vgg19_tags_embedding_feature_dict = {}
    resnet_tags_embedding_feature_dict = {}
    inception_tags_embedding_feature_dict = {}
    xception_tags_embedding_feature_dict = {}

    i=0

    for im in os.listdir(im_path):
        try:
            print(im)
            i += 1
            if i%100 == 0:
                print(i)
            #read image data
            image = cv2.imread(im_path+im)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # features extraction
            desc = LocalBinaryPatterns(8, 1.0)
            hist_LBP = desc.describe(gray)
            vgg16_tags_embedding_feature = im_tag_embedding_feature_extraction(im_path+im, model_vgg16, embedding_vector, 224)
            vgg19_tags_embedding_feature = im_tag_embedding_feature_extraction(im_path+im, model_vgg19, embedding_vector, 224)
            resnet_tags_embedding_feature = im_tag_embedding_feature_extraction(im_path+im, model_resnet, embedding_vector, 224)
            inception_tags_embedding_feature = im_tag_embedding_feature_extraction(im_path+im, model_inception, embedding_vector, 299)
            xception_tags_embedding_feature = im_tag_embedding_feature_extraction(im_path+im, model_xception, embedding_vector, 299)

            
            lbp_feature_dict[im] = list(hist_LBP)
            vgg16_tags_embedding_feature_dict[im] = vgg16_tags_embedding_feature
            vgg19_tags_embedding_feature_dict[im] = vgg19_tags_embedding_feature
            resnet_tags_embedding_feature_dict[im] = resnet_tags_embedding_feature
            inception_tags_embedding_feature_dict[im] = inception_tags_embedding_feature
            xception_tags_embedding_feature_dict[im] = xception_tags_embedding_feature
        except Exception:
            print('error image',im)
            continue
    # save features
    np.save('politifact_image_LBP_feature.npy', lbp_feature_dict)
    np.save('politifact_image_tag_vgg16_feature.npy', vgg16_tags_embedding_feature_dict)
    np.save('politifact_image_tag_vgg19_feature.npy', vgg19_tags_embedding_feature_dict)
    np.save('politifact_image_tag_resnet_feature.npy', resnet_tags_embedding_feature_dict)
    np.save('politifact_image_tag_inception_feature.npy', inception_tags_embedding_feature_dict)
    np.save('politifact_image_tag_xception_feature.npy', xception_tags_embedding_feature_dict)