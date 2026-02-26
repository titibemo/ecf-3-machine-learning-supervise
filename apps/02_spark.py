# %%
# docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /apps/02_spark.py

import numpy as np
import time
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from itertools import chain
from pyspark.ml.feature import Imputer, StringIndexer, StandardScaler, OneHotEncoder, VectorAssembler, MinMaxScaler, Bucketizer, SQLTransformer
import pyspark.sql.functions as F
from pyspark.sql.functions import sum as spark_sum
import numpy as np
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
import time
import builtins
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# %% [markdown]
# # PYSPARK

# %% [markdown]
# ### 1. EXPLORATION ET PRÉPARATION - 5a. Setup Spark

# %% [markdown]
# - Charger les données et afficher info/describe

# %%
builder: SparkSession.Builder = SparkSession.builder

spark = builder.master('local').appName("demo_rdd").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df_spark = spark.read \
    .option('header', 'true') \
    .option('inferSchema', 'true') \
    .option('sep', ',') \
    .csv('/data/03_DONNEES.csv')

# %%
# # --- Statistiques descriptives ---
print("=== STATISTIQUES DESCRIPTIVES ===")
df_spark.describe().show()

# %%
# --- Dimensions et types de colonnes ---
print("=== DIMENSIONS ET TYPES DE COLONNES ===\n")
df_spark.printSchema()
num_rows = df_spark.count()
num_cols = len(df_spark.columns)
print(f"Nombre de lignes : {num_rows}")
print(f"Nombre de colonnes : {num_cols}\n")

# %%
# - Analyser la distribution de Churn
total_count = df_spark.count()

df_spark.groupBy("Churn").count().withColumn(
    "percentage", F.col("count") / total_count * 100
).show()

# %%
# --- Valeurs manquantes ---
missing_df = df_spark.select([
    (F.count(F.when(F.col(c).isNull(), c)) \
     .alias(c)) for c in df_spark.columns
])
missing_counts = missing_df.collect()[0].asDict()
missing_stats = [(col, count, round(100*count/num_rows,2)) 
                 for col, count in missing_counts.items() if count > 0]

print(missing_counts)
print("=== VALEURS MANQUANTES ===")
for col, count, percent in missing_stats:
    print(f"{col}: {count} valeurs manquantes ({percent}%)")
print()

# %%
# créer une colonne 0/1 à partir de "Churn"
df_spark = df_spark.withColumn(
    "Churn_num",
    when(df_spark["Churn"] == "Yes", 1).otherwise(0)
)

# vérifier
df_spark.select("Churn", "Churn_num").show(5)

# %%
# - Encoder les variables catégorielles (One-hot encoding)

string_indexer = StringIndexer(
    inputCols=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract'],
    outputCols=['gender_idx', 'Partner_idx', 'Dependents_idx', 'PhoneService_idx', 'MultipleLines_idx',
                'InternetService_idx', 'OnlineSecurity_idx', 'OnlineBackup_idx', 'DeviceProtection_idx',
                'TechSupport_idx', 'StreamingTV_idx', 'StreamingMovies_idx', 'Contract_idx'],
    handleInvalid="keep"
)

ohe = OneHotEncoder(
    inputCols=['gender_idx', 'Partner_idx', 'Dependents_idx', 'PhoneService_idx', 'MultipleLines_idx',
                'InternetService_idx', 'OnlineSecurity_idx', 'OnlineBackup_idx', 'DeviceProtection_idx',
                'TechSupport_idx', 'StreamingTV_idx', 'StreamingMovies_idx', 'Contract_idx'],
    outputCols=["gender_ohe", "Partner_ohe", "Dependents_ohe", "PhoneService_ohe", "MultipleLines_ohe", "InternetService_ohe", "OnlineSecurity_ohe", "OnlineBackup_ohe", "DeviceProtection_ohe", "TechSupport_ohe", "StreamingTV_ohe", "StreamingMovies_ohe", "Contract_ohe"],
    dropLast=True
)

# %% [markdown]
# ### **5b - Préparation Spark**

# %%
assembler = VectorAssembler(
    inputCols=[
       "gender_ohe", "Partner_ohe", "Dependents_ohe", "PhoneService_ohe", "MultipleLines_ohe", "InternetService_ohe", "OnlineSecurity_ohe", "OnlineBackup_ohe", "DeviceProtection_ohe", "TechSupport_ohe", "StreamingTV_ohe", "StreamingMovies_ohe", "Contract_ohe", "SeniorCitizen", "tenure", "InternetCharges", "MonthlyCharges", "TotalCharges"
    ],
    outputCol="features",
    handleInvalid="skip"
)

# %%
# Normalisation des données
scaler = StandardScaler(
    inputCol="features",
    outputCol="features_std"
)

# %%
# pipeline prête
preprocessing = Pipeline(
    stages=[
        string_indexer,
        ohe,
        assembler,
        scaler
    ]
)

# %% [markdown]
# ### **5c. Modélisation Spark**

# %%
# - Split train/test
train_df, test_df = df_spark.randomSplit([0.7, 0.3], seed=42)

# %%
# entrainement des modèles Logistic Regression - Random Forest - Gradient Boosting
model_lr = LogisticRegression(
    featuresCol="features_std",
    labelCol="Churn_num",
    maxIter=100,
)

model_rf = RandomForestClassifier(
    featuresCol="features_std",
    labelCol="Churn_num",
    numTrees=100,
    seed=42
)

model_gb = GBTClassifier(
    featuresCol="features_std",
    labelCol="Churn_num",
    maxIter=100,
    seed=42
)

pipeline_lr = Pipeline(
    stages=[string_indexer, ohe, assembler, scaler, model_lr]
)
pipeline_rf = Pipeline(
    stages=[string_indexer, ohe, assembler, scaler, model_rf]
)
pipeline_gb = Pipeline(
    stages=[string_indexer, ohe, assembler, scaler, model_gb]
)

# %%
# - Calculer accuracy, precision, recall, f1-score pour chaque
models = {
    "Logistic Regression": pipeline_lr,
    "Random Forest": pipeline_rf,
    "gradient boosting": pipeline_gb
}

results = {}

for name, pipeline in models.items():
    start_time = time.time()
    model_fit = pipeline.fit(train_df)
    elapsed_time = time.time() - start_time
    preds = model_fit.transform(test_df)
    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="Churn_num", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="Churn_num", predictionCol="prediction", metricName="f1"
    )
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="Churn_num", predictionCol="prediction", metricName="weightedPrecision"
    )
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="Churn_num", predictionCol="prediction", metricName="weightedRecall"
    )
    
    # calcul métriques
    acc = evaluator_acc.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)
    prec = evaluator_precision.evaluate(preds)
    rec = evaluator_recall.evaluate(preds)
    
    # stocker résultats
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "time": elapsed_time
    }
    
    # affichage
    print(name)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1),
    print("time", elapsed_time)
    print()

# %%
# Sauvegarder les résultats dans un fichier JSON
try:
    with open('/data/model_results_pyspark_comparaison.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to 'model_results.json'")
except Exception as e:
    print("Error saving results:", e)

# %%
# Ajouter le nom du modèle dans ses propres métriques
for name, metrics in results.items():
    metrics["model_name"] = name

# - Sélectionner le meilleur modèle
best_model_name = None
best_f1 = -1

for name, metrics in results.items():
    if metrics["recall"] > best_f1:
        best_f1 = metrics["recall"]
        best_model_name = name

best_model_metrics = results[best_model_name]

print(f"=== BEST MODEL ===")
print(f"Model: {best_model_name}")
print(f"Metrics: {best_model_metrics}")
print(best_model_metrics["model_name"])  # ici on récupère le nom directement depuis les métriques

# %%
# Construire le JSON du meilleur modèle
best_model_json = {
    best_model_name: best_model_metrics
}

# Supprimer la clé "model_name" si elle existe
best_model_json[best_model_name].pop("model_name", None)

# Sauvegarder
import json
with open("/data/model_results_pyspark_best_model.json", "w") as f:
    json.dump(best_model_json, f, indent=4)

print("Saved best model metrics as JSON.")


