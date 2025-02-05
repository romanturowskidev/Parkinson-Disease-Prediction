# import all necessary libraries
import pandas
from pandas.plotting import scatter_matrix  # Zmiana na aktualny moduł Pandas
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # Zmiana cross_validation na model_selection
from sklearn.metrics import matthews_corrcoef, accuracy_score  # Aktualne importy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler  # Skalowanie danych

# Wczytanie zbioru danych z lokalizacji
url = "data.csv"
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
            "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
            "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "status"]
dataset = pandas.read_csv(url, names=features)

# Przygotowanie danych
array = dataset.values
X = array[:, 0:16]  # Cechy
Y = array[:, 16]    # Etykiety (0 = zdrowy, 1 = chory)

# Skalowanie danych (zalecane dla algorytmów takich jak LogisticRegression i MLPClassifier)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i walidacyjny
validation_size = 0.3
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Przygotowanie modeli
models = []
models.append(('LR', LogisticRegression(max_iter=500)))  # Zwiększono max_iter, aby uniknąć problemów ze zbieżnością
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NN', MLPClassifier(solver='lbfgs', max_iter=2000)))  # Zwiększono max_iter
models.append(('NB', GaussianNB()))
models.append(('GB', GradientBoostingClassifier(n_estimators=10000)))

# Ewaluacja modeli
results = []
names = []
print("Scores for each algorithm:")
for name, model in models:
    # Użycie aktualnego KFold
    kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    
    # Walidacja krzyżowa
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    
    # Trenowanie modelu
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    
    # Wyświetlenie wyników
    print(f"{name}: Accuracy: {accuracy_score(Y_validation, predictions) * 100:.2f}%")
    print(f"{name}: MCC: {matthews_corrcoef(Y_validation, predictions):.4f}")
    print()

