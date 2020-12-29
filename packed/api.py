from utils import preprocess

def sentiment_classify(comment, vectorizer, model):
    comment = preprocess(comment)
    tfidf = vectorizer.transform([comment])
    pred = model.predict(tfidf)[0]

    return pred