from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка сохранённой модели
try:
    print("Загрузка модели...")
    model = joblib.load('/Users/aryurkov/Documents/1ml/model.pkl')
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

# Функция для предсказания стоимости
def model_prediction(data):
    try:
        # Преобразуем входные данные в numpy массив
        data = np.array(data).reshape(1, -1)  # Преобразуем в форму (1, 9) для 9 признаков
        prediction = model.predict(data)
        return prediction[0]
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    print("Запрос на главную страницу")  # Логирование для отладки
    if request.method == 'POST':
        try:
            # Логируем полученные данные из формы
            print(f"Получены данные формы: {request.form}")
            
            totsp = float(request.form['totsp'])
            livesp = float(request.form['livesp'])
            kitsp = float(request.form['kitsp'])
            dist = float(request.form['dist'])
            metrdist = float(request.form['metrdist'])
            walk = float(request.form['walk'])
            brick = float(request.form['brick'])
            floor = float(request.form['floor'])
            code = float(request.form['code'])

            # Формируем входные данные
            input_data = [totsp, livesp, kitsp, dist, metrdist, walk, brick, floor, code]

            # Получаем предсказание
            prediction = model_prediction(input_data)
            print(f"Предсказание: {prediction}")  # Логирование предсказания

            return render_template('index.html', prediction=prediction)

        except Exception as e:
            print(f"Ошибка обработки данных формы: {e}")
            return render_template('index.html', prediction=None)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    # Тестируем модель перед запуском
    if model:
        try:
            print("Проверяем модель...")
            test_data = np.array([50, 30, 10, 5, 10, 1, 1, 1, 3]).reshape(1, -1)  # 9 признаков
            print(f"Пример предсказания: {model.predict(test_data)}")
        except Exception as e:
            print(f"Ошибка тестирования модели: {e}")

    app.run(debug=True)