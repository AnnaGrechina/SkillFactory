from flask import Flask, request, jsonify
from application import app 
from spam_classifier import Classifier
import pandas as pd

@app.route('/') 
def hello():
    email1 = 'Hi, My name is Warren E. Buffett an American business magnate, investor and philanthropist. am the most successful investor in the world. I believe strongly in‘giving while living’ I had one idea that never changed in my mind? that you should use your wealth to help people and i have decided to give {$1,500,000.00} One Million Five Hundred Thousand United Dollars, to randomly selected individuals worldwide. On receipt of this email, you should count yourself as the lucky individual. Your email address was chosen online while searching at random. Kindly get back to me at your earliest convenience before i travel to japan for my treatment , so I know your email address is valid. Thank you for accepting our offer, we are indeed grateful You Can Google my name for more information: God bless you. Best Regard Mr.Warren E. Buffett Billionaire investor !' 
    email2 = "Hi guys I want to build a website like REDACTED and I wanted to get your perspective of whether that site is good from the users' perspective before I go ahead and build something similar. I think that the design of the site is very modern and nice but I am not sure how people would react to a similar site? I look forward to your feedback. Many thanks!'"
    email3 = 'As a result of your application for the position of Data Engineer, I would like to invite you to attend an interview on May 30, at 9 a.m. at our office in Washington, DC. You will have an interview with the department manager, Moris Peterson. The interview will last about 45 minutes. If the date or time of the interview is inconvenient, please contact me by phone or email to arrange another appointment. We look forward to seeing you.'
    test_set = [email1, email2, email3]
    result_dict = dict()
    for email in test_set:
        result_dict[email] = cls_Kaggle.classify(email)
    result = ';   ----- NEXT TEXT ----  '.join(str(key) + '   :   ' + str(value) for key, value in result_dict.items())
    return (result)

@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text')
    if text is None:
        params = ', '.join(data.keys())
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400
    else:
        result = cls_Kaggle.classify(text)
        return jsonify({'result': result})

# обучение на данных из п. 2.9
dfspam = pd.read_csv('spam_or_not_spam.csv')
dfspam.dropna(inplace=True)
dfspam['label'] = dfspam['label'].apply(lambda x: 'SPAM' if x == 1 else 'NOT_SPAM')
data_train_Kaggle = dfspam.values.tolist()
cls_Kaggle = Classifier()
cls_Kaggle.train(data_train_Kaggle)


