__author__ = 'xead'
from sklearn.externals import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("SGD.pkl")
        self.vectorizer = joblib.load("Tfidf.pkl")
        self.classes_dict = {0: "Отрицательный", 1: "Положительный", -1: "Ошибка предсказания"}

    def predict_text(self, text):
        try:
            print("Nik, i am here!")
            vectorized = self.vectorizer.transform([text])
            print("vec ok!")
            return self.model.predict(vectorized)[0]
        except:
            print("Ошибка предсказания")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print('Ошибка предсказания')
            return None


    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        print('hello Nik!',prediction)
        return self.classes_dict[prediction]