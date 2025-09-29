'''
Задача 1. Обнаружение фальшивых новостей
Фальшивые новости — это ложная информация, распространяемая через
социальные сети и другие сетевые СМИ для достижения политических
или идеологических целей.

Твоя  задача -  используя библиотеку sklearn построить модель
классического машинного обучения, которая может с высокой
точностью более 90% определять, является ли новость реальной (REAL）
или фальшивой（FAKE).

Ты должен самостоятельно изучить и применить к задаче
TfidfVectorizer для извлечения признаков из текстовых
данных и PassiveAggressiveClassifier.

Построй матрицу ошибок (confusion matrix).
Представь, что ваш заказчик очень любит графики и диаграммы.
Визуализируй для него результаты там, где это возможно.
'''

import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import pivot_table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Установка стилей графиков
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

def LoadAnalizeData(file_path):
    # Загрузка данных
    try:
        df = pd.read_csv('E:/pyton/practic_1/data_fake_news.csv')
        print('датасет загружен')
    except Exception as e:
        print(f'Ошибка при загрузке датасета: {e}')

    print(f"Размер датасета: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")

    # Проверяем нули
    print(df.isnull().sum())

    # Удаляем строки с пропущенными значениями в тексте
    df = df.dropna(subset=['text'])
    print(f"Размер датасета после удаления: {df.shape}")

    return df

def preprocessorTextData(df):
    def preprocess_text(text):
        """Функция для очистки и предобработки текста"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Проверяем результат предобработки
    print("Примеры текстов после очистки:")
    for i in range(2):
        print(f"Оригинал {i + 1}: {df['text'].iloc[i][:100]}")
        print(f"Очищенный {i + 1}: {df['cleaned_text'].iloc[i][:100]}")
        print()

    return df

def figure(df, plot_type, title, x_field=None, y_field=None, rotation=45, figsize=(10, 6)):
    plt.figure(figsize=figsize)

    if plot_type == 'bar':
        # Столбчатая диаграмма
        if y_field is None:
            data = df[x_field].value_counts()
            x = data.index
            y = data.values
        else:
            data = df.groupby(x_field)[y_field].sum()
            x = data.index
            y = data.values
        bars = plt.bar(x, y)
        plt.title(title)
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.xticks(rotation=rotation)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height, f'{int(height)}', ha="center", va="bottom")

    elif plot_type == 'pie':
        # Круговая диаграмма
        if y_field is None:
            data = df[x_field].value_counts()
        else:
            data = df.groupby(x_field)[y_field].sum()

        plt.pie(data.values, labels=data.index, startangle=90)
        plt.title(title)
        plt.xlabel(x_field)
        plt.ylabel(y_field)

    elif plot_type == 'heatmap':
        # Тепловая карта
        if x_field and y_field:
            pivot_table = df.pivot_table(index=x_field, columns=y_field, aggfunc='size', fill_value=0)
            sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
            plt.title(title)
            plt.xlabel(y_field)
            plt.ylabel(x_field)
        else:
            # Тепловая карта корреляции для числовых данных
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title(f'{title}\n(Матрица корреляции)')
            else:
                print("Нет числовых данных для тепловой карты корреляции")
                return

    elif plot_type == 'confusion_matrix':
        # Матрица ошибок
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            cm = confusion_matrix(df['y_true'], df['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(title)
            plt.xlabel(x_field)
            plt.ylabel(y_field)
        else:
            print("Для матрицы ошибок нужны колонки 'y_true' и 'y_pred'")
            return

    else:
        print(f'неизвестный тип графика {plot_type}')
        return

    plt.tight_layout()
    plt.show()

def main():
    # Основная функция
    print(f'=' * 60)
    print('Найдем фальшивые новости')
    print(f'=' * 60)

    # загрузка данных
    df = LoadAnalizeData('E:/pyton/PythonProject/data_fake_news.csv')
    if df is None:
        return
    # Предобработка текста
    df = preprocessorTextData(df)

    # Анализ распределения классов
    label_counts = df['label'].value_counts()
    print(label_counts)

    # Визуализация
    figure(df, 'bar', 'Распределение новостей по классам',
                x_field='label', rotation=0)

    figure(df, 'pie', 'Процентное распределение классов',
                x_field='label')

    # Разделение на обучающие и тестовые данные
    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Обучающая выборка: {X_train.shape[0]} записей")
    print(f"Тестовая выборка: {X_test.shape[0]} записей")

    # Обучение модели TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    # Обучение TF-IDF на тренировочных данных
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"📐 Размерность TF-IDF матрицы (train): {X_train_tfidf.shape}")
    print(f"📐 Размерность TF-IDF матрицы (test): {X_test_tfidf.shape}")


    # Создание и обучение PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42,
        C=0.5,
        early_stopping=True
    )

    pac.fit(X_train_tfidf, y_train)

    # Предсказания
    y_pred = pac.predict(X_test_tfidf)

    # Оценка модели
    accuracy = accuracy_score(y_test, y_pred)
    print(f"РЕЗУЛЬТАТЫ МОДЕЛИ")
    print(f"Точность модели: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Проверка достижения целевой точности
    if accuracy >= 0.90:
        print("Целевая точность 90% ДОСТИГНУТА!")
    else:
        print("Целевая точность 90% НЕ ДОСТИГНУТА")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print(f"Матрица ошибок:")
    print(cm)

    # Создаем DataFrame для матрицы ошибок и используем нашу функцию
    cm_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })

    # Визуализация матрицы ошибок с помощью нашей функции
    figure(cm_df, 'confusion_matrix', 'Матрица ошибок классификации')

    # Дополнительная визуализация: важные слова
    print("Анализ важных признаков...")
    feature_names = tfidf_vectorizer.get_feature_names_out()

    if hasattr(pac, 'coef_'):
        coefficients = pac.coef_[0]
        # Создаем DataFrame с важностью слов
        feature_importance = pd.DataFrame({
            'word': feature_names,
            'importance': coefficients
        })

        # Топ-10 слов для FAKE и REAL
        top_fake = feature_importance.nlargest(10, 'importance')
        top_real = feature_importance.nsmallest(10, 'importance')

        print("📝 Топ-10 слов для FAKE новостей:")
        print(top_fake[['word', 'importance']])

        print("\n📝 Топ-10 слов для REAL новостей:")
        print(top_real[['word', 'importance']])

        # Отчет о классификации
    print("\n📋 Детальный отчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # Тестирование на примерах
    print("\n🧪 ТЕСТИРОВАНИЕ НА ПРИМЕРАХ:")
    test_samples = [
        "Scientists discovered a new cancer treatment in clinical trials",
        "The government is hiding the truth about alien invasion",
        "Economic growth showed positive dynamics this quarter",
        "Secret method to lose 10 kg in 3 days without diet"
    ]

    for i, text in enumerate(test_samples, 1):
        # Предобработка текста
        cleaned_text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Преобразование в TF-IDF и предсказание
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        prediction = pac.predict(text_tfidf)[0]

        print(f"{i}. '{text[:50]}...' → {prediction}")

    print("\n🎉 ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
# Запуск программы
if __name__ == "__main__":
    main()
