import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Загрузка данных
data = pd.read_csv('/Users/aryurkov/Documents/1ml/flats_moscow.csv')

# Удаляем первый столбец, который является индексом, если он есть
data = data.drop(columns=['Unnamed: 0'], errors='ignore')  # 'errors="ignore"' игнорирует ошибку, если столбец отсутствует

# Обработка данных и нормализация
scaler = MinMaxScaler()
data[['totsp', 'livesp', 'kitsp', 'dist', 'metrdist', 'walk', 'brick', 'floor', 'code']] = scaler.fit_transform(
    data[['totsp', 'livesp', 'kitsp', 'dist', 'metrdist', 'walk', 'brick', 'floor', 'code']]
)

# Разделение на X (признаки) и y (целевая переменная)
X = data.drop(columns=['price'])
y = data['price']

# Проверка на количество признаков
print(f"Количество признаков в данных: {X.shape[1]}")  # Это должно быть 9

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели в файл
model_path = '/Users/aryurkov/Documents/1ml/model.pkl'
joblib.dump(model, model_path)

# Выводим сообщение об успешном сохранении модели
print(f"Модель успешно сохранена в {model_path}")