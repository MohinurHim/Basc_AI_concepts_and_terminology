# Шаг 1 Установка библиотек
# Шаг 2  Загрузка и предобработка данных
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Загрузка данных MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Алгоритм машинного обучения (k-NN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Создание и обучение модели k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f'Accuracy of k-NN: {accuracy_knn:.4f}')

# Шаг 4: Глубокое обучение (MLP)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Предобработка меток для MLP
y_train_mlp = to_categorical(y_train, 10)
y_test_mlp = to_categorical(y_test, 10)

# Создание модели MLP
model_mlp = Sequential([
    Flatten(input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция и обучение модели MLP
model_mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_mlp.fit(X_train, y_train_mlp, epochs=10, batch_size=32, validation_split=0.2)

# Оценка модели на тестовой выборке
loss_mlp, accuracy_mlp = model_mlp.evaluate(X_test, y_test_mlp)

print(f'Accuracy of MLP: {accuracy_mlp:.4f}')

# Шаг 5: Нейронные сети (CNN)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

# Предобработка данных для CNN
X_train_cnn = X_train.values.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.values.reshape(-1, 28, 28, 1)
y_train_cnn = to_categorical(y_train, 10)
y_test_cnn = to_categorical(y_test, 10)

# Создание модели CNN
model_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Компиляция и обучение модели CNN
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_split=0.2)

# Оценка модели на тестовой выборке
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_cnn, y_test_cnn)

print(f'Accuracy of CNN: {accuracy_cnn:.4f}')


# Теоритические вопросы:

# Какие преимущества и недостатки использованных методов вы увидели?
# Преимущества: k-NN быстро проводит тестирование, благодаря своей простоте и легкости, без предварительного обучения модели. Недостатки: Для больших наборов данных вычисление расстояний может быть очень затратным. Наличие шумных данных может существенно ухудшить производительность модели.Неэффективен при высокой размерности данных.

# Преимущества: MLP способность моделировать сложные нелинейные зависимости, быстрое выполнение прогнозов. Недостатки: необходимо значительный объём данных для хороших результатов, чтобы не было переобучение.

# Преимущества: CNN один из лучших алгоритмов по распознаванию и классификации изображений. Автоматически определяет важные признаки. Недостатки: сложность процесса обучения, уходит больше времени. Может быть подвержена переобучению при недостаточном количестве данных.

# В чем, на ваш взгляд, заключается принципиальная разница между многослойным перцептроном и сверточной нейронной сетью?
# Многослойные персептроны – наиболее простой тип нейросети, который состоит из одного или нескольких слоев нейронов. Каждый нейрон обрабатывает входящие данные и передает выходные данные следующему слою нейронов. MLP может использоваться для решения широкого спектра задач – от прогнозирования временных рядов до распознавания образов на изображениях. Сверточные нейронные сети – это тип нейросетей, основаной на принципе свертки, который позволяет выявлять визуальные признаки изображения. Каждый нейрон в CNN обрабатывает только небольшую область изображения, что позволяет учитывать локальные свойства каждого фрагмента.

# Какие методы предобработки данных были использованы в этом задании?
# Нормализация: Данные были нормалтзованы путем деления на 255, чтобы привести значение пикселей в диапазон от 0 до 1.

# Разделение выборки train_test_split: на обучающую и тестовую.

# Преобразование меток to_categorical: что позволяет модели корректно интепретировать многоклассовую задачу.

# Изменение формы данных reshape: для получения трехмерного массива, соответствующий формату изображений

