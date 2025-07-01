import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar el dataset
df = pd.read_csv("Tabla_Politicas_Publicas.csv", sep=";", encoding="latin-1")

# 2. Crear la etiqueta binaria: 1 = Éxito, 0 = No Éxito
df["label"] = df["Evaluación"].str.contains("Éxito", case=False, na=False).astype(int)

# 3. Combinar columnas textuales en un solo campo
text_cols = ["Política pública", "Objetivo principal", "Grupo", "Resultado final"]
df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

# 4. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# 5. Crear pipeline: TF-IDF + Regresión Logística
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# 6. Validación cruzada (5-fold)
cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=5,
    scoring=["accuracy", "f1"]
)

print("=== Validación cruzada (5-fold) ===")
print(f"Accuracy promedio: {cv_results['test_accuracy'].mean():.3f}")
print(f"F1 promedio:       {cv_results['test_f1'].mean():.3f}")

# 7. Entrenar modelo final
pipeline.fit(X_train, y_train)

# 8. Evaluación sobre el conjunto de prueba
y_pred = pipeline.predict(X_test)

print("\n=== Evaluación en test ===")
print(classification_report(y_test, y_pred, target_names=["No Éxito", "Éxito"]))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
