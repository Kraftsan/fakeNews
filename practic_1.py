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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import re

# Установка стилей графиков
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Загрузка данных
try:
    df = pd.read_csv('E:/pyton/practic_1/data_fake_news.csv')
    print('датасет загружен')
except Exception as e:
    print(f'Ошибка при загрузке датасета: {e}')

# print(df.head())
# print(len(df))
# print(df.columns.tolist())
# print(df.isnull().sum())

# Распределение классов
label_counts = df['label'].value_counts()
print(label_counts)

# Визуализация распределения классов
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Распределение классов')

plt.subplot(1, 2, 2)
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Количество записей по классам')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.tight_layout()
plt.show()

def preprocessorText(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['cleaned_text'] = df['text'].apply(preprocessorText)


# Разделение на обучающие и тестовые данные
X = df['cleaned_text']
y = df['label']

print(f"Оригинал: {df['text'].iloc[0][:100]}...")
print(f"Очищенный: {df['cleaned_text'].iloc[0][:100]}...")

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

print(f"Размерность матрицы (train): {X_train_tfidf.shape}")
print(f"Размерность матрицы (test): {X_test_tfidf.shape}")

# Создание и обучение PassiveAggressiveClassifier