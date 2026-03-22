import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 📥 Carregar dataset
data = pd.read_csv("dataset.csv", header=None)

# 🎯 Separar entrada (X) e saída (y)
X = data.iloc[:, :-1]  # todas colunas menos a última
y = data.iloc[:, -1]   # última coluna (letra)

# 🔀 Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Criar modelo
model = RandomForestClassifier()

# 🏋️ Treinar modelo
model.fit(X_train, y_train)

# 📊 Avaliar modelo
accuracy = model.score(X_test, y_test)
print(f"Acurácia do modelo: {accuracy:.2f}")

# 💾 Salvar modelo
joblib.dump(model, "modelo.pkl")

print("Modelo treinado e salvo como modelo.pkl")