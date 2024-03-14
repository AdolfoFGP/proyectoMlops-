import os
import sys
import json
import re
import traceback
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#tf.get_logger().setLevel('ERROR')
##
# Declare communication channel between Sagemaker and container
prefix = '/opt/ml'
# Sagemaker stores the dataset copied from S3
input_path = os.path.join(prefix, 'input/data')
# If something bad happens, write a failure file with the error messages and store here
output_path = os.path.join(prefix, 'output')
# Everything stored here will be packed into a .tar.gz by Sagemaker and copied into S3
model_path = os.path.join(prefix, 'model')
# These are the hyperparameters sent to the training algorithms through the Estimator
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')


# Define function called for training
def train():
    print("Training mode ...")
    
    try:
        # This algorithm has a single channel of input data called 'training'. Since we run in
        # File mode, the input files are copied to the directory specified here.
        channel_name = 'training'
        training_path = os.path.join(input_path, channel_name)

        params = {}
        # Read in any hyperparameters that the are passed with the training job
        with open(param_path, 'r') as tc:
            params = json.load(tc)

        # Confirm that training files exists and the channel was correctly configured
        input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        
        
        # Specify the Column names in order to manipulate the specific columns for pre-processing

        column_names = ["Class", "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
       "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
       "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]

        # Load the training dataset
        train_data = pd.read_csv(os.path.join(input_path, 'training', 'train.csv'), sep=',', names=column_names)

        X_train = train_data.drop('Class', axis=1)
        y_train = train_data['Class']


        algorithm = 'XGBoostClassifier'
        print("Training Algorithm: %s" % algorithm)

        for key in params:
            if isinstance(params[key], str):
                try:
            # Convertir solo si es un valor que debe ser flotante
                    if key in ['learning_rate']:  # Ejemplo: 'learning_rate' y 'n_estimators' son flotantes
                        params[key] = float(params[key])
                except ValueError:
                    pass  # Dejarlo como cadena si la conversión falla


        # Initialize XGBoost classifier
        model = xgb.XGBClassifier(
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.1),
            #n_estimators=params.get('n_estimators', 100),
            objective=params.get('objective', 'binary:logistic'),
            eval_metric=params.get('eval_metric', 'logloss')
        )
        
        #early_stopping_rounds = int(params.get('early_stopping_rounds', 10))
        
        # Train the model
        model.fit(X_train, y_train)

        # Save the model as a single 'h5' file without the optimizer
        print("Saving Model ...")
        model.save_model(os.path.join(model_path, 'model.model'))


    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # `DescribeTrainingJob` result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\\n' + trc)
            
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\\n' + trc, file=sys.stderr)
        
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
    
# Define function called for local testing
def predict(payload, algorithm):
    print("Local Testing Mode ...")
    if algorithm is None:
        raise ValueError("Please provide the algorithm specification")
    payload = np.asarray(payload) # Convert the payload to numpy array
    # Convierte el payload a un objeto DMatrix
    dtest = xgb.DMatrix(payload)
    return algorithm.predict(dtest).tolist()