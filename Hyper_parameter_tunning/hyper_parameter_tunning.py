# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, log
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from datetime import datetime
import mlflow
import mlflow.spark

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


from pyspark.ml.tuning import ParamGridBuilder, CrossValidatorModel

def get_best_params_from_model(model):
    best_params = {}
    for param_name in model.getEstimatorParamMaps()[model.avgMetrics.index(min(model.avgMetrics))]:
        best_params[param_name.name] = model.bestModel.getOrDefault(param_name)
    return best_params

def build_and_tune_model(train_data, model, target_feature, param_grid, features_col="scaled_features"):
    model.setFeaturesCol(features_col)
    model.setLabelCol(target_feature)

    evaluator = RegressionEvaluator(labelCol=target_feature, predictionCol="prediction", metricName="rmse")

    cross_validator = CrossValidator(estimator=model,
                                   estimatorParamMaps=param_grid,
                                   evaluator=evaluator,
                                   numFolds=3)  # You can adjust the number of folds as needed

    trained_model = cross_validator.fit(train_data)
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
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "Mean Absolute Error (MAE)": mae,
        "R-squared": r2
    }

def create_experiment(experiment_name, performance_metrics, transformation_pipeline, model_pipeline, run_params=None):
    run_name = experiment_name + str(datetime.now().strftime("%d-%m-%y"))

    with mlflow.start_run(run_name=run_name):
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        for metric in performance_metrics:
            mlflow.log_metric(metric, performance_metrics[metric])

        mlflow.spark.log_model(model_pipeline.bestModel, "model")
        mlflow.spark.log_model(transformation_pipeline, "transformation")

def main(data_path, target_feature, experiment_name, model, param_grid):
    df = load_data(data_path)
    transformation_pipeline, transformed_data = preprocess_data(df, target_feature)
    transformed_data = transformed_data.withColumn("target_log", log(col(target_feature)))
        # Split the data into training and test sets

    train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)
    

    tuned_model_pipeline = build_and_tune_model(train_data, model, "target_log", param_grid)
    
    # Extract best hyperparameters from the tuned model
    best_params = get_best_params_from_model(tuned_model_pipeline)
    
    # Evaluate the model with the best parameters
    performance_metrics = evaluate_regression_model(tuned_model_pipeline, test_data)
    
    # Create the experiment and pass the best parameters
    create_experiment(experiment_name, performance_metrics, transformation_pipeline, tuned_model_pipeline, run_params=best_params)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("CarPricePrediction").getOrCreate()

    data_path = "/FileStore/tables/ToyotaCorolla_sample"
    model = LinearRegression()  # Use Linear Regression
    experiment_name = "car_price_prediction_"
    target_feature = "Price"

    param_grid = ParamGridBuilder() \
        .addGrid(model.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0]) \
        .addGrid(model.maxIter, [10, 20, 30]) \
        .build()

    main(data_path, target_feature, experiment_name, model, param_grid)

