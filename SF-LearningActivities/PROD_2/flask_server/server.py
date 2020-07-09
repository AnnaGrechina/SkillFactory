from flask import Flask
from flask import request #в этом объекте в поле args находятся параметры
from flask import jsonify
import numpy as np
import pickle

import datetime

#создаем объект приложения
app = Flask(__name__)  #передаем имя, обычно передают __name__, чтобы приложение имело то же имя, что и питоновый процесс


# пишем функцию для обработки запросов и прикрепляем ее к определенному пути
@app.route('/hello')
def hello_func():
    name = request.args.get('name')
    return f'hello, {name}!'

@app.route('/time')
def current_time():
    return {'time': datetime.datetime.now()}

@app.route('/add', methods = ['POST'])
def add_func():
    num = request.json['num']
    if num > 10:
        return 'too much', 400  # передаем кортеж, второе значение - код ошибки
    return jsonify({'result' : num + 1})

#Задание 17.2
# написать Flask-приложение, которое по эндпойнту /predict будет слушать POST-запросы на предсказания.
# В теле POST-запроса — список из четырёх чисел в формате json.
# Ответом на запрос должен быть json-формат {"prediction": *число - предсказание модели*}.

@app.route('/predict', methods=['POST'])
def pred_func():
    x_pred = request.get_json()
    x_pred = np.array(x_pred).reshape(1, -1)
    with open('hw1.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
    y_pred = model.predict(x_pred)

    print(x_pred)
    print(type(x_pred))

    return {'prediction': y_pred[0]}




# для запуска приложения есть метод run, который нужно выполнить в питоновском main
if __name__ == '__main__':
    app.run('localhost', 5000)


