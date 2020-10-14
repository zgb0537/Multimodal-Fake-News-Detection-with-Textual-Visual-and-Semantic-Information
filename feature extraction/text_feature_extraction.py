from keras.preprocessing.text import Tokenizer
import numpy as np
import aidrtokenize as aidrtokenize
import emoji
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def text_sentiment(text,analyzer):
    vs = analyzer.polarity_scores(text)
    polarity_num=[]
    polarity_num.append(float(vs['pos']))
    polarity_num.append(float(vs['compound']))
    polarity_num.append(float(vs['neu']))
    polarity_num.append(float(vs['neg']))
    print(polarity_num)
    return polarity_num

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

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def text_proprecess(text):
    txt = text.strip().lower()
    txt = give_emoji_free_text(txt)
    txt = aidrtokenize.tokenize(txt)
    return txt

def text_embedding(text,embedding_vector):
    token = Tokenizer()
    token.fit_on_texts(text)
    tags_matrix = []
    zero_array = np.zeros(300)
    for word in text.strip().split(' '):
        if word in embedding_vector.keys():
            tag_embedding = embedding_vector[word]
            tags_matrix.append(np.array(tag_embedding))
            zero_array = zero_array + np.array(tag_embedding)
        else:
            continue
    tag_feature = zero_array / len(tags_matrix)
    return list(tag_feature)


def text_feature_extraction(text,embedding_vector,analyzer):
    sentiment_feature = text_sentiment(text,analyzer) #sentiment feature size is 4
    text = text_proprecess(text)
    print(text)
    embedding_feature = text_embedding(text, embedding_vector) # embedding feature size is 300
    return sentiment_feature, list(embedding_feature)

if __name__ == '__main__':

    analyzer = SentimentIntensityAnalyzer()
    embedding_vector = embedding_load('../data/GoogleNews_vectors_negative_300d.txt')

    sentiment_feature_dict = {}
    text_embedding_feature_dict = {}

    tweets_dev_file = open('./politifact/politifact_twitter_data.txt', 'r', encoding='utf8')
    for line in tweets_dev_file.readlines():
        twitter_id = line.split('\t')[0]
        twitter_text = line.split('\t')[1]
        label = line.split('\t')[-1]
        if label.strip().replace('\n', '') == 'fake':
            label = 0
        elif label.strip().replace('\n', '') == 'real':
            label = 1
        else:
            continue

        text_sentiment_feature, text_embedding_feature = text_feature_extraction(twitter_text, embedding_vector, analyzer)
        sentiment_feature_dict[twitter_id] = text_sentiment_feature
        text_embedding_feature_dict[twitter_id] = text_embedding_feature

    np.save('politifact_text_sentiment_dict.npy', sentiment_feature_dict)
    np.save('politifact_text_embedding_dict.npy', text_embedding_feature_dict)
