# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, log


def load_data(data_path, format_type, version=None):
    if format_type == 'parquet':
        return spark.read.parquet(data_path, header=True)
    elif format_type == 'delta':
        if version is not None:
            delta_table = spark.read.format("delta").option("versionAsOf", version).load(data_path)
            delta_dataframe = delta_table.toDF()
            return delta_dataframe
        else:
            return spark.read.format("delta").load(data_path).toDF()
    else:
        raise ValueError("Invalid data format. Supported formats are 'parquet' and 'delta'.")


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
    # Assuming 'model' is an instance of a Spark MLlib model (e.g., LogisticRegression)
    model.setFeaturesCol(features_col)
    model.setLabelCol(target_feature)
    model_pipeline = Pipeline(stages=[model])
    trained_model = model_pipeline.fit(train_data)
    return trained_model

def evaluate_classification_model(model, test_data, label_col="cid"):
    predictions = model.transform(test_data)
    evaluator_multi = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
    accuracy = evaluator_multi.evaluate(predictions)
    evaluator_binary = BinaryClassificationEvaluator(labelCol=label_col)
    auc_roc = evaluator_binary.evaluate(predictions)
    confusion_matrix = predictions.groupBy(label_col, "prediction").count()
    tp_row = confusion_matrix.filter((col(label_col) == 1) & (col("prediction") == 1)).first()
    true_positive = tp_row["count"] if tp_row else 0
    fp_row = confusion_matrix.filter((col(label_col) == 0) & (col("prediction") == 1)).first()
    false_positive = fp_row["count"] if fp_row else 0
    fn_row = confusion_matrix.filter((col(label_col) == 1) & (col("prediction") == 0)).first()
    false_negative = fn_row["count"] if fn_row else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "Accuracy": accuracy,
        "AUC-ROC": auc_roc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }


from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_regression_model(model, test_data, label_col="Price"):
    predictions = model.transform(test_data)
    # RegressionEvaluator for MSE
    evaluator_mse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mse")
    mse = evaluator_mse.evaluate(predictions)
    # RegressionEvaluator for RMSE
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    # RegressionEvaluator for MAE
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")
    mae = evaluator_mae.evaluate(predictions)
    # R-squared
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    r2 = evaluator_r2.evaluate(predictions)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R-squared": r2
    }

def create_experiment(experiment_name, performance_metrics, transformation_pipeline,model_pipeline,run_params=None):
    import mlflow
    import mlflow.spark
    from datetime import datetime
    run_name=experiment_name+str(datetime.now().strftime("%d-%m-%y"))
    
    with mlflow.start_run(run_name=run_name):
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in performance_metrics:
            mlflow.log_metric(metric, performance_metrics[metric])
        #mlflow.sklearn.log_model(pipeline, "ml_pipline_2")
        mlflow.spark.log_model(model_pipeline, "model")
        mlflow.spark.log_model(transformation_pipeline, "transformation")
        #mlflow.set_tag("model", model_name)
            

def main(data_path,format_type,target_feature,experiment_name,model):
    df = load_data(data_path,format_type)
    # Preprocess the data
    transformation_pipeline,transformed_data = preprocess_data(df, target_feature)
    # Split the data into training and test sets
    transformed_data = transformed_data.withColumn("target_log", log(col(target_feature)))
        # Split the data into training and test sets
    train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)
    display(train_data)
    # Build and train the model
    model_pipeline = build_model(train_data, model, "target_log")
    # Evaluate the model
    performance_metrics = evaluate_regression_model(model_pipeline, test_data)
    print(performance_metrics)
    create_experiment(experiment_name, performance_metrics, transformation_pipeline, model_pipeline)

if __name__ == "__main__":
    # Specify the data path and target feature
    from pyspark.ml.regression import LinearRegression
    data_path = "/mnt/silver_layer/transformed_data.parquet"
    model = LinearRegression()
    experiment_name = "car_price_prediction_"
    target_feature="Price"
    format_type = "parquet"
    # Call the main function
    main(data_path,format_type,target_feature,experiment_name,model)   

# COMMAND ----------



# COMMAND ----------


