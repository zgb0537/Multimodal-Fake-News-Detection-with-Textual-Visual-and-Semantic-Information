import requests
import tweepy
import csv
import sys

def image_download(image_url, image_out_path, image_name):
    try:
        pic = requests.get(image_url, headers=headers)
        f = open(image_out_path + image_name + '.jpg', 'wb')
        f.write(pic.content)
        f.close()
    except Exception as ex:
        print(image_url)

def twitter_crawler(api, files):
    for file in files:
        label = file.split('_')[1]
        with open(file + '.txt', 'w', encoding='utf8') as outfile:
            with open('FakeNewsNet/' + file + '.csv', 'r', encoding="utf8", errors='ignore') as csvfile:
                reader = csv.DictReader(csvfile)
                for news in reader:
                    print(news)
                    news_id = news['id']
                    news_title = news['title']
                    news_url = news['news_url']
                    tweet_ids = news['tweet_ids']
                    print(news_id)
                    if len(tweet_ids) > 10:
                        for twitter_id in tweet_ids.split('\t'):
                            try:
                                tweet = api.get_status(twitter_id)
                                tweets = tweet.text
                                tweets = tweets.replace('\n', ' ')
                                print(tweet.text)
                                media = tweet.entities.get('media', [])
                                print(media)
                                # print(media['type'])
                                # tweet = twitter.show_status(id=twitter_id)
                                image_name = news_id + '_' + label + '_' + twitter_id + '.jpg'
                                if len(media) > 0:
                                    media_url = media[0]['media_url']
                                    print(media_url)
                                    image_download(media_url, './' + file.split('_')[0] + '_images/', image_name)
                                    image_name_true = image_name
                                    outfile.write(
                                        twitter_id + '\t' + tweets + '\t' + news_title + '\t' + news_url + '\t' + image_name_true + '\n')
                            except tweepy.TweepError:
                                print("Failed to run the command on that user, Skipping...")


if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)

    CONSUMER_KEY = ""
    CONSUMER_SECRET = ""
    OAUTH_TOKEN = ""
    OAUTH_TOKEN_SECRET = ""

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    files = ['politifact_real','politifact_fake','gossipcop_real','gossipcop_fake']
    twitter_crawler(api, files)


