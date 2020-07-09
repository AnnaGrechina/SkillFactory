# клиент для сервиса predict, который отправляет POST-запрос со
# список из четырёх чисел в формате json
import requests

if __name__ == '__main__':
    r = requests.post('http://localhost:5000/predict', json=[0.5, 1.5, 20.23, 40.54])

    if r.status_code == 200:
        print (r.json()['prediction'])

    else:
        print(r.text)