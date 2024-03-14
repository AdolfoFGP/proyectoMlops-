import os
import io
import json
import logging
import boto3
import time
import botocore
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from botocore.exceptions import ClientError
from sklearn.metrics import f1_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client("s3")
sm_client = boto3.client("sagemaker-runtime")
##

def evaluate_model(bucket, key, endpoint_name):
    """
    Description:
    ------------
    Executes model predictions on the testing dataset.
    
    :bucket: (str) Pipeline S3 Bucket.
    :key: (str) Path to "testing" dataset.
    :endpoint_name: (str) Name of the 'Dev' endpoint to test.

    :returns: Lists of ground truth, prediction labels and response times.
    
    """
    #column_names = ["Amount", "length", "diameter", "height", "whole weight", "shucked weight",
    #                "viscera weight", "shell weight", "sex_F", "sex_I", "sex_M"]
    
    column_names = ["Class", "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
       "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
       "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
       
    response_times = []
    predictions = []
    y_test = []
    obj = s3.get_object(Bucket=bucket, Key=key)
    test_df = pd.read_csv(io.BytesIO(obj['Body'].read()), names=column_names)
    y = test_df['Class']#.to_numpy()
    X = test_df.drop(['Class'], axis=1)#.to_numpy()
    #X = preprocessing.normalize(X)
    
    # Cycle through each row of the data to get a prediction
    for row in range(len(X)):
        payload = ",".join(map(str, X.iloc[row]))  # Usar X.iloc[row] para acceder a la fila por su posición
        #payload = ",".join(map(str, X[row]))
        elapsed_time = time.time()
        try:
            response = sm_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType = "text/csv",
                Body=payload
            )
        except ClientError as e:
            error_message = e.response["Error"]["Message"]
            logger.error(error_message)
            raise Exception(error_message)
        response_times.append(time.time() - elapsed_time)
        result = np.asarray(response['Body'].read().decode('utf-8').rstrip('\n'))
        predictions.append(float(result))
        #y_test.append(float(y[row]))
        y_test.append(float(y.iloc[row]))
    
    return y_test, predictions, response_times


def handler(event, context):
    logger.debug("## Environment Variables ##")
    logger.debug(os.environ)
    logger.debug("## Event ##")
    logger.debug(event)
    
    # Ensure variables passed from Model Evaluation Step
    if ("Bucket" in event):
        bucket = event["Bucket"]
    else:
        raise KeyError("S3 'Bucket' not found in Lambda event!")
    if ("Key" in event):
        key = event["Key"]
    else:
        raise KeyError("S3 'Key' not found in Lambda event!")
    if ("Output_Key" in event):
        output_key = event["Output_Key"]
    else:
        raise KeyError("S3 'Output_Uri' not found in Lambda event!")
    if ("Endpoint_Name" in event):
        endpoint_name = event["Endpoint_Name"]
    else:
        raise KeyError("'SageMaker Endpoint Name' not found in Lambda event!")
    
    # Get the evaluation results from SageMaker hosted model
    logger.info("Evaluating SageMaker Hosted Model ...")
    y, y_pred, times = evaluate_model(bucket, key, endpoint_name)
    
    # Convertir y a enteros (0 o 1)
    #y_int = y.astype(int)
    
    # Convertir y a enteros (0 o 1)
    y_int = [int(value) for value in y]
    
    # Adjust decision threshold to obtain binary predictions
    threshold = 0.5
    y_pred_binary = [1 if pred > threshold else 0 for pred in y_pred]
    
    # Calculate the metrics
    #mse = mean_squared_error(y, y_pred)
    #rmse = mean_squared_error(y, y_pred, squared=False)
    #std = np.std(np.array(y) - np.array(y_pred))
        # Calculate the F1-score
    #f1 = f1_score(y, y_pred)
    f1 = f1_score(y_int, y_pred_binary)

    # Save Metrics to S3 for Model Package
    logger.info("f1-score: {}".format(f1))
    logger.info("Average Endpoint Response Time: {:.2f}s".format(np.mean(times)))
    report_dict = {
            "classification_metrics": {
                "f1_score": f1
            },
        }
    try:
        s3.put_object(
            Bucket=bucket,
            Key="{}/{}".format(output_key, "evaluation.json"),
            Body=json.dumps(report_dict, indent=4)
        )
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

    # Return results
    logger.info("Done!")
    return {
        "statusCode": 200,
        "Result": f1,
        "AvgResponseTime": "{:.2f} seconds".format(np.mean(times))
    }