#INTERCONNECT_PROJECT
#Importando librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from boruta import BorutaPy

from IPython.display import display, HTML


#cargamos los datasets
df_contract = pd.read_csv('/home/nick/datasets/contract.csv')
df_personal = pd.read_csv('/home/nick/datasets/personal.csv')
df_internet = pd.read_csv('/home/nick/datasets/internet.csv')
df_phone = pd.read_csv('/home/nick/datasets/phone.csv')

pd.set_option('display.max_columns', None)#Para observar todas las columnas en DataFrames


#1.- EXPLORANDO LOS DATOS
#Función que muestra la información inicial de los datos

def data_info(df, nombre_df):

    print(f'Datos {nombre_df}:')

    display(df.head())
    #inserto espacio en blanco doble
    display(HTML('<br><br>'))

    print(f'Información de {nombre_df}:')
    display(df.info())
    
    #Inserta línea horizontal que seprara secciones
    display(HTML('<hr>'))

data_info(df_contract, 'df_contract')
print(df_contract['BeginDate'].max())
data_info(df_personal, 'df_personal')
data_info(df_internet, 'df_internet')
data_info(df_phone, 'df_phone')


#funcion que imprime valores unicos y cuenta frecuencia de los mismos
def unique_values(df, df_name):
    for column in df.columns:
        print(f"Valores únicos en '{column}' de {df_name}: {df[column].unique()}")

def values_freq(df, df_name):
    for column in df.columns:
        print(f"Conteo de valores en '{column}' de {df_name}:")
        print(df[column].value_counts(dropna=False))

print()

#Función que reemplaza valores en blanco
def replace_spaces(DataFrames):
    for df in DataFrames:
        df.replace(' ', np.nan, inplace=True)

dataframes = [df_contract, df_personal, df_internet, df_phone]
replace_spaces(dataframes)
#Mostrando la operación de la función
for df in dataframes:
    print(df.isna().sum())
print()
print(df_contract.isna().sum())#observamos los valores en DF

#Mostramos las filas que poseen tales valores para saber como tratarlos
print(df_contract[df_contract['TotalCharges'].isna()])

#rellenamos los valores ausentes en la columna TotalCharges con 0
df_contract['TotalCharges'] = df_contract['TotalCharges'].fillna(0)
print(df_contract.isna().sum())

print()

#función para duplicados
def duplicates(dfs):
    duplicated = {}
    for i, df in enumerate(dfs):
        count = df.duplicated().sum()
        duplicated[f'DataFrame_{i+1}'] = count

    return duplicated

dfs = [df_contract, df_personal, df_internet, df_phone]
duplicate = duplicates(dfs)
#print(duplicate)


#2 .- TRATANDO TIPOS DE DATOS

#df_contract. Como la columna BeginDate siempre empieza desde el primero de cualquier mes
#se tratará de tal manera. Lo mismo pasará con columna EndDate
# Convertir 'BeginDate' y 'EndDate' a formato datetime

print(df_contract['BeginDate'].max())
df_contract['BeginDate'] = pd.to_datetime(df_contract['BeginDate'], errors='coerce')
df_contract['EndDate'] = pd.to_datetime(df_contract['EndDate'], errors='coerce')# Fecha de referencia para contratos sin EndDate
reference_date = pd.to_datetime('2020-02-01')


#función que calcula duración en meses
def calculate_duration_months(begin_date, end_date):
    if pd.isna(begin_date):
        return np.nan
    if pd.isna(end_date):
        end_date = reference_date
    duration = (end_date.year - begin_date.year) * 12 + end_date.month - begin_date.month
    return duration


#Duración en meses
df_contract['BeginDate'] = pd.to_datetime(df_contract['BeginDate'], errors='coerce')
df_contract['duration_months'] = df_contract.apply(
    lambda row: calculate_duration_months(row['BeginDate'], row['EndDate']),
    axis=1
)


# Crear la columna objetivo
df_contract['target'] = (df_contract['EndDate'] < reference_date).astype(int)

# Mostrar resultados
print(df_contract[['BeginDate', 'EndDate', 'duration_months', 'target']].head())
print()
print(df_contract['EndDate'].value_counts())
print()
print(df_contract['BeginDate'].value_counts())
print()
print(df_contract['target'].value_counts())

#funcion que convierte a categrico
def categorical_value(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
    return df

categorical_contract = ['Type', 'PaymentMethod', 'PaperlessBilling']
df_contract = categorical_value(df_contract, categorical_contract)

#convirtiendo TotalCharges a númerico
df_contract['TotalCharges'] = pd.to_numeric(df_contract['TotalCharges'], errors='coerce')

df_contract['EndDate'] = df_contract['EndDate'].fillna(df_contract['EndDate'].max())
print(df_contract.head())

print()
print()
print()
#df_personal

categorical_personal = ['gender', 'Partner', 'Dependents']
df_personal = categorical_value(df_personal, categorical_personal)


#df_internet
categorical_internet = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_internet = categorical_value(df_internet, categorical_internet)


#df_phone
df_phone = categorical_value(df_phone, ['MultipleLines'])


# 3 .- COMBINANDO DATAFRAMES
data = df_contract.merge(df_personal, how='left', on='customerID')
data = data.merge(df_internet, how='left', on='customerID')
data = data.merge(df_phone, how='left', on='customerID')

#corregimos el formato de las columnas
for i in data.columns:
    data.columns = data.columns.str.lower()

data = data.rename(columns={'customerid': 'customer_id', 'begindate': 'begin_date', 
                                                'enddate': 'end_date', 'paperlessbilling': 'paperless_billing',
                                                'paymentmethod': 'payment_method', 'monthlycharges': 'monthly_charges',
                                                'totalcharges': 'total_charges', 'internetservice': 'internet_service', 
                                                'onlinesecurity': 'online_security', 'onlinebackup': 'online_backup',
                                                'deviceprotection': 'device_protection', 'techsupport': 'tech_support',
                                                'streamingtv': 'streaming_tv', 'streamingmovies': 'streaming_movies',
                                                'multiplelines': 'multiple_lines', 'seniorcitizen': 'senior_citizen',
                                                'type': 'contract_type'})


#4 .- OBSERVANDO LOS DATOS
#Función que desplega distintos gráficos referentes a la tasa de abandono
def plot_groupby_target(data, groupby_col, colors=None):
    group_data = data.groupby([groupby_col, 'target'], observed=True)['customer_id'].count().unstack(fill_value=0)

    ind = np.arange(len(group_data))
    width = 0.25

    if colors is None:
        colors = ['g', 'violet', 'blue', 'orange', 'red', 'purple', 'cyan', 'brown']

    plt.figure(figsize=(12, 7))

    for i, target_val in enumerate(group_data.columns):
        label = 'No abandonó' if target_val == 0 else 'Abandonó'
        plt.bar(ind + i * width, group_data[target_val], width, color=colors[i % len(colors)], label=label)

    plt.xlabel('Categoría')
    plt.ylabel('Cantidad de clientes')
    plt.title(f'{groupby_col.capitalize()} vs Abandono')
    plt.xticks(ind + width, group_data.index.tolist())
    plt.legend()
    plt.show()

#Contratos de servicios de clientes Interconnect
data['internet_multilines'] = np.where(~data['internet_service'].isna() & ~data['multiple_lines'].isna(), 'Ambos', 
                                       np.where(data['internet_service'].isna() & ~data['multiple_lines'].isna(), 'Solo telefono', 
                                                np.where(~data['internet_service'].isna() & data['multiple_lines'].isna(), 'Solo internet',
                                                         'no_info')))


data['internet_multilines'] = data['internet_multilines'].astype('category')


col_stats = ['monthly_charges', 'total_charges', 'target', 'senior_citizen']
print(data[col_stats].corr())


print()


#5 .- INGENIERÍA DE CARACTERÍSTICAS

data['automatic_pay'] = np.where(data['payment_method'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']), 
                                 'automatic', 
                                 'manual')


for col in data.columns:
    print(f"Valores únicos de {col}: {data[col].unique()}\n")

#días transcurridos antes de haber abandonado
data['duration_days'] = (data['end_date'] - data['begin_date']).dt.days

data['extra_payment'] = data['total_charges'] - data['monthly_charges'] * data['duration_months']

#rellenamos valores ausentes en las columnas donde aún existen los mismos
#Usamos 'No' asumiendo que los mismos no contrataron o aún no contratan tal servicio
col_nan_values = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
for col in col_nan_values:
    if data[col].dtype.name == 'category':
        if 'No' not in data[col].cat.categories:
            data[col] = data[col].cat.add_categories('No')
        data[col] = data[col].fillna('No')
print(data.isna().sum())
print()

#Verificar si columna 'multiple_lines' tipo categórico
if data['multiple_lines'].dtype.name == 'category':
    #Renombrar categorías para evitar la advertencia
    data['multiple_lines'] = data['multiple_lines'].cat.rename_categories({'No': '0', 'Yes': '1'})
else:
    #replace si no es categórico
    data['multiple_lines'] = data['multiple_lines'].replace({'No': '0', 'Yes': '1'})


for row in range(len(data)):
    
    if pd.isna(data.loc[row, 'multiple_lines']):
        data.loc[row, 'atleast_one_line'] = 0
    
    else:
        data.loc[row, 'atleast_one_line'] = 1
        
data['atleast_one_line'] = data['atleast_one_line'].astype('int')
print(data.nunique())
print()

print(data.isna().sum())
for col in data.select_dtypes(['category']).columns:
    if -1 not in data[col].cat.categories:
        data[col] = data[col].cat.add_categories([-1])

#rellenar valores nulos con -1
data = data.fillna(-1)
#convertimos a categorico la columna faltante
categorical_value(data, ['internet_multilines', 'automatic_pay', 'senior_citizen'])
data['target'] = data.pop('target')
print(data.head())


print()


#6 .- ENCONDIFICADO  Y ESCALADO DE VARIABLES.

#divide conjunto de datos
valid_set = data[data['begin_date'] >= reference_date]
train_test_set = data[data['begin_date'] < reference_date]

train_set, test_set = train_test_split(train_test_set, test_size=0.25, random_state=12345, stratify=train_test_set['target'])

#eliminar columnas irrelevantes
columns_to_drop = ['customer_id', 'begin_date', 'end_date']
train_features = train_set.drop(columns=columns_to_drop + ['target'])
train_target = train_set['target']

valid_features = valid_set.drop(columns=columns_to_drop + ['target'])
valid_target = valid_set['target']

test_features = test_set.drop(columns=columns_to_drop + ['target'])
test_target = test_set['target']


#categóricas y numéricas
cat_cols = train_features.select_dtypes(include=['category']).columns.tolist()
num_cols = train_features.select_dtypes(include=['float64', 'int64']).columns.tolist()

#categóricas a string
train_features[cat_cols] = train_features[cat_cols].astype(str)
valid_features[cat_cols] = valid_features[cat_cols].astype(str)
test_features[cat_cols] = test_features[cat_cols].astype(str)

#preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

#Aplicando a los datos
feature_train_transformed = preprocessor.fit_transform(train_features)
feature_valid_transformed = preprocessor.transform(valid_features)
feature_test_transformed = preprocessor.transform(test_features)

ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
new_feature_names = num_cols + list(ohe_feature_names)

#Convetir a DataFrame
features_train_OHE = pd.DataFrame(feature_train_transformed, columns=new_feature_names)
features_valid_OHE = pd.DataFrame(feature_valid_transformed, columns=new_feature_names)
features_test_OHE =pd.DataFrame(feature_test_transformed, columns=new_feature_names)
print(features_train_OHE.shape, features_valid_OHE.shape)

#SMOTE
smote = SMOTE(random_state=12345)
feature_train_balanced, target_train_balanced = smote.fit_resample(features_train_OHE, train_target)
print(f'Clases después de SMOTE: \n {target_train_balanced.value_counts()}')

print()


#TOTRY: RECORDAR O NO DROP='FIRST'. PROBAR.


#BORUTA
rf_boruta = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=12345)
boruta_selector = BorutaPy(rf_boruta, n_estimators='auto', perc=100, random_state=12345)

#Boruta al conjunto de datos balanceado
boruta_selector.fit(feature_train_balanced, target_train_balanced)

#Seleccionando las características relevantes
selected_features = features_train_OHE.columns[boruta_selector.support_].tolist()
print(f"Características seleccionadas por Boruta: {selected_features}")


#Reduciendo conjunto a las caracteristícas seleccionadas por boruta
feature_train_selected = feature_train_balanced[selected_features]
feature_valid_selected = features_valid_OHE[selected_features]
feature_test_selected = features_test_OHE[selected_features]

print()

#Validación cruzada para verificar estabilidad de características seleccionadas
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)

#Validando. cross_val_score para ver si las características seleccionadas son estables

cv_scores = cross_val_score(rf_boruta, feature_train_selected, target_train_balanced, cv=skf, scoring='accuracy')
print(f"Puntuaciones de validación cruzada: {cv_scores}")
print(f"Promedio de las puntuaciones: {cv_scores.mean()}")


# Definir los pipelines para cada modelo
pipelines = {
    'RandomForest': Pipeline(steps=[
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'LogisticRegression': Pipeline(steps=[
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'MLPClassifier': Pipeline(steps=[
        ('classifier', MLPClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline(steps=[
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'AdaBoost': Pipeline(steps=[
        ('classifier', AdaBoostClassifier(random_state=42))
    ]),
    'DecisionTree': Pipeline(steps=[
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'LGBMClassifier': Pipeline(steps=[
        ('classifier', LGBMClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline(steps=[
        ('classifier', XGBClassifier(random_state=42))
    ]),
    'CatBoost': Pipeline(steps=[
        ('classifier', CatBoostClassifier(random_state=42, verbose=0))
    ])
}

for name, pipeline in pipelines.items():
    print(f"\nModelo: {name}")

    cv_scores = cross_val_score(pipeline, feature_train_selected, target_train_balanced, cv=skf, scoring='accuracy')
    print(f"Puntuaciones de validación cruzada: {cv_scores}")
    print(f"Promedio de las puntuaciones: {np.mean(cv_scores)}")
        
    pipeline.fit(feature_train_selected, target_train_balanced)
    valid_predictions = pipeline.predict(feature_valid_selected)
    valid_report = classification_report(valid_target, valid_predictions)
    print("Reporte de clasificación en conjunto de validación:")
    print(valid_report)
        
    test_predictions = pipeline.predict(feature_test_selected)
    test_report = classification_report(test_target, test_predictions)
    print("Reporte de clasificación en conjunto de prueba:")
    print(test_report)
        
    #Bootstraping
    n_iterations = 1000
    bootstrap_scores = []
    for i in range(n_iterations):
        indices = np.random.choice(range(len(feature_test_selected)), size=len(feature_test_selected), replace=True)
        feature_test_bootstrap = feature_test_selected.iloc[indices]
        target_test_bootstrap = test_target.iloc[indices]
        bootstrap_predictions = pipeline.predict(feature_test_bootstrap)
        bootstrap_scores.append(np.mean(bootstrap_predictions == target_test_bootstrap))
       
    print("Media de las puntuaciones Bootstrap:", np.mean(bootstrap_scores))
    print("Desviación estándar de las puntuaciones Bootstrap:", np.std(bootstrap_scores))


print()


#9 .- ENTRENAMIENTO DE MODELOS
#Entrenando el modelo con la métrica AUC-ROC >= 0.88

#GB_model
gb_model = GradientBoostingClassifier()

#hallo hiperparametros
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
gb_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo Gradient Boosting:")
print(gb_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(gb_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(gb_grid_search.best_score_)

best_gb = gb_grid_search.best_estimator_
best_gb.fit(feature_train_selected, target_train_balanced)

y_pred = best_gb.predict(feature_test_selected)
y_pred_proba = best_gb.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



#MLP_model
mlp_model = MLPClassifier(max_iter=1000)

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

mlp_grid_search = GridSearchCV(estimator=mlp_model, param_grid=mlp_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

mlp_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo MLP:")
print(mlp_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(mlp_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(mlp_grid_search.best_score_)

best_mlp = mlp_grid_search.best_estimator_
best_mlp.fit(feature_train_selected, target_train_balanced)

y_pred = best_mlp.predict(feature_test_selected)
y_pred_proba = best_mlp.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadero Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



#rf_model
rf_model = RandomForestClassifier()

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

rf_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo Random Forest:")
print(rf_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(rf_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(rf_grid_search.best_score_)

best_rf = rf_grid_search.best_estimator_
best_rf.fit(feature_train_selected, target_train_balanced)

y_pred = best_rf.predict(feature_test_selected)
y_pred_proba = best_rf.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadero Positivo')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


#MODELOS EXCLUDOS

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

xgb_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo XGBoost:")
print(xgb_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(xgb_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(xgb_grid_search.best_score_)

best_xgb = xgb_grid_search.best_estimator_
best_xgb.fit(feature_train_selected, target_train_balanced)

y_pred = best_xgb.predict(feature_test_selected)
y_pred_proba = best_xgb.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

#cb_model
cb_model = CatBoostClassifier(silent=True)

catboost_param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

catboost_grid_search = GridSearchCV(estimator=cb_model, param_grid=catboost_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

catboost_grid_search.fit(feature_train_selected, target_train_balanced)

print("Mejor modelo CatBoost:")
print(catboost_grid_search.best_estimator_)

print("Mejores hiperparámetros:")
print(catboost_grid_search.best_params_)

print("Mejor puntuación AUC-ROC:")
print(catboost_grid_search.best_score_)

best_catboost = catboost_grid_search.best_estimator_
best_catboost.fit(feature_train_selected, target_train_balanced)

y_pred = best_catboost.predict(feature_test_selected)
y_pred_proba = best_catboost.predict_proba(feature_test_selected)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


#RED NEURONAL

neural_model = tf.keras.models.Sequential()
neural_model.add(tf.keras.layers.Dense(units=15, input_dim=feature_train_selected.shape[1], activation='tanh'))
neural_model.add(tf.keras.layers.Dense(units=5, activation='tanh'))
neural_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compilo
neural_model.compile(loss='binary_crossentropy', #función de pérdida. Problemas de clasificación binaria
             optimizer=tf.keras.optimizers.Adam(),#optimización estocástica del gradiente
             metrics=[tf.keras.metrics.AUC(name='auc')])



early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = neural_model.fit(feature_train_selected, 
                   target_train_balanced, 
                   epochs=100, 
                   batch_size=32, 
                   validation_data=(feature_valid_selected, valid_target),
                   callbacks=[early_stopping])#detiene el entrenamiento ayudando a evitar un número innecesario de epocas


loss, auc_score = neural_model.evaluate(feature_test_selected, test_target, verbose=0)
print(f"Puntuación AUC-ROC en conjunto de prueba: {auc_score}")

# Realizar predicciones
y_pred_proba = neural_model.predict(feature_test_selected).ravel()  #probabilidad AUC-ROC
y_pred = (y_pred_proba > 0.5).astype(int)

print("Reporte de Clasificación:")
print(classification_report(test_target, y_pred))

print("Puntuación AUC-ROC:")
print(roc_auc_score(test_target, y_pred_proba))

fpr, tpr, _ = roc_curve(test_target, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()