# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col,log
from hyperopt import fmin, tpe, hp,Trials
import mlflow
import mlflow.spark
import mlflow.pyfunc
from datetime import datetime

def load_data(data_path):
    return spark.read.parquet(data_path, header=True)

def get_feature_columns(df, data_type):
    return [col_name for col_name, dt in df.dtypes if dt == data_type]

def preprocess_data(df, target_column):
    try:
        # Identify numerical and categorical columns
        numerical_columns = get_feature_columns(df, 'double') + get_feature_columns(df, 'int')
        categorical_columns = list(set(get_feature_columns(df, 'string')))
        
        # Remove target column from numerical columns if present
        if target_column in numerical_columns:
            numerical_columns.remove(target_column)
        elif target_column in categorical_columns:
            categorical_columns.remove(target_column)
        else:
            raise ValueError("Target column not found in either numerical or categorical columns.")

        # Create Imputer for numerical columns
        numerical_imputers = [Imputer(inputCols=[col_name], outputCols=[f"{col_name}_imputed"], strategy="median") for col_name in numerical_columns]

        # Create StringIndexer for categorical columns
        categorical_stages = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="skip") for col_name in categorical_columns]

        # Create OneHotEncoder for categorical columns
        categorical_stages += [OneHotEncoder(inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_onehot") for col_name in categorical_columns]

        # Create VectorAssembler to combine features
        assembler = VectorAssembler(inputCols=[f"{col_name}_imputed" for col_name in numerical_columns] +
                                              [f"{col_name}_onehot" for col_name in categorical_columns], outputCol="features")

        # Create StandardScaler to scale features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

        # Create the preprocessing pipeline
        preprocessing_pipeline = Pipeline(stages=numerical_imputers + categorical_stages + [assembler, scaler])

        # Fit the preprocessing pipeline to the data
        fitted_preprocessing_pipeline = preprocessing_pipeline.fit(df)

        # Transform the data
        transformed_data = fitted_preprocessing_pipeline.transform(df)

        return fitted_preprocessing_pipeline, transformed_data.select('scaled_features', target_column)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def build_model(train_data, model, target_feature, features_col="scaled_features"):
    model.setFeaturesCol(features_col)
    model.setLabelCol(target_feature)
    model_pipeline = Pipeline(stages=[model])
    trained_model = model_pipeline.fit(train_data)
    return trained_model

def evaluate_regression_model(model, test_data, label_col="Price"):
    predictions = model.transform(test_data)
    evaluator_mse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mse")
    mse = evaluator_mse.evaluate(predictions)
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
    mae = evaluator_mae.evaluate(predictions)
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def create_experiment(experiment_name, performance_metrics, transformation_pipeline, model_pipeline, run_params=None):
    run_name = experiment_name + str(datetime.now().strftime("%d-%m-%y"))
    with mlflow.start_run(run_name=run_name, nested=True):
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric, value in performance_metrics.items():
            mlflow.log_metric(metric, value)

        # for stage in model_pipeline.stages:
        #     mlflow.log_params(stage.extractParamMap())

        mlflow.spark.log_model(model_pipeline, "model")
        mlflow.spark.log_model(transformation_pipeline, "transformation")

def main(data_path, target_feature, experiment_name, model, param_space,hyeropt_metric):
    df = load_data(data_path)
    transformation_pipeline, transformed_data = preprocess_data(df, target_feature)
    transformed_data = transformed_data.withColumn("target_log", log(col(target_feature)))
    train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)

    def objective(params):
        model_instance = model()
        model_instance.setParams(**params)
        model_pipeline = build_model(train_data, model_instance, "target_log")
        performance = evaluate_regression_model(model_pipeline, test_data)
        return performance[hyeropt_metric]
    
    trials = Trials()
    mlflow.pyspark.ml.autolog(log_models=False)
    max_evals = 2
    # Check if there's an active run, and end it
    if mlflow.active_run():
        mlflow.end_run()

    best_params = fmin(fn=objective,
                       space=param_space,
                       algo=tpe.suggest,
                       max_evals=max_evals,
                       trials=trials)
    print(best_params)

    # Retrain model on the entire dataset with best hyperparameters
    best_model_instance = model()
    best_model_instance.setParams(**best_params)
    trained_best_model_pipeline = build_model(transformed_data, best_model_instance, target_feature)

    # Evaluate the best model on the test set
    performance_best_model = evaluate_regression_model(trained_best_model_pipeline, test_data)

    # Save the best model and relevant information
    create_experiment(experiment_name, performance_best_model, transformation_pipeline, trained_best_model_pipeline, best_params)


# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
if __name__ == "__main__":
    data_path = "/mnt/silver_layer/transformed_data.parquet"
    target_feature = "Price"
    experiment_name = "car_price_prediction_rf" 
    hyeropt_metric = "RMSE"
    param_space_rf = {
        'maxDepth': hp.choice('maxDepth', range(5, 16)),
        'numTrees': hp.choice('numTrees', range(10, 101))
    }

    main(data_path, target_feature, experiment_name, RandomForestRegressor, param_space_rf,hyeropt_metric)

# COMMAND ----------


if __name__ == "__main__":
    from pyspark.ml.regression import GBTRegressor 
    data_path = "/mnt/silver_layer/transformed_data.parquet"
    target_feature = "Price"
    experiment_name = "car_price_prediction_gbt_"
    param_space = {
        'maxDepth': hp.choice('maxDepth', range(5, 16)),
        'maxBins': hp.choice('maxBins', [32, 64, 128]),
        #'minInstancesPerNode': hp.choice('minInstancesPerNode', [5, 10]),
        'subsamplingRate': hp.uniform('subsamplingRate', 0.5, 1.0)
    }
    hyeropt_metric = "RMSE"

    main(data_path, target_feature, experiment_name, GBTRegressor, param_space, hyeropt_metric)


# COMMAND ----------


