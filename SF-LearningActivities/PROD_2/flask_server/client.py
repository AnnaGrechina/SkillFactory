# браузеры не умеют делать пост-запросы. Для проверки сервера напишем клиент
import requests

if __name__ == '__main__':
    r = requests.post('http://localhost:5000/add', json={'num': -10})

    print(r.status_code)

    if r.status_code == 200:
        print (r.json()['result'])
    else:
        print(r.text)