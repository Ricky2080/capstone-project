import os
import json
import pickle
import joblib
import uuid
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField, BooleanField, CharField, TextField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging

# LOGGING
class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)
    

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # this should be just "logger.setLevel(logging.INFO)" but markdown is interpreting it wrong here...
    handler = logging.StreamHandler()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = CustomRailwayLogFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()

########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    id = CharField(primary_key=True, max_length=50)
    observation = TextField()
    pred_class = BooleanField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

class API_call_log(Model):
    id = CharField(primary_key=True, max_length=50)
    log = TextField()
    
    class Meta:
        database = DB

DB.create_tables([API_call_log], safe=True)

# End database stuff
########################################


########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

########################################
# Input validation functions


def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
    'id',
    'name',
    'sex',
    'dob',
    'race', 
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'priors_count',
    'c_case_number',
    'c_charge_degree',
    'c_charge_desc',
    'c_offense_date',
    'c_arrest_date',
    'c_jail_in'
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) < 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        
        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_category_map = {
        "sex": ['Male', 'Female'],
        "race": ['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Others'],
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""


########################################
# Begin webserver stuff


app = Flask(__name__)

def process_observation(observation):
    logger.info("Processing observation, %s", observation)
    # A lot of processing
    return observation

def log_api_call():
    # Generate a unique ID for each call
    call_id = str(uuid.uuid4())
    # Prepare the log content
    log_content = {
        'method': request.method,
        'url': request.url,
        'headers': dict(request.headers),
        'body': request.get_data(as_text=True)
    }
    # Store the log in the database
    API_call_log.create(id=call_id, log=json.dumps(log_content))

#@app.before_request
#def before_request():
#    log_api_call()

@app.route('/will_recidivate', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    logger.info('Observation: %s', obs_dict)
    _id = obs_dict['id']
    observation = obs_dict
    
    # a single observation into a dataframe that will work with a pipeline.
    #obs = pd.DataFrame([observation])
    
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    if not _id:
        logger.warning('Returning error: no id provided')
        return jsonify({'error': 'id is required'}), 400
    
    if Prediction.select().where(Prediction.id == _id).exists():
        prediction = Prediction.get(Prediction.id == _id)

        # Update the prediction
        prediction.observation = str(observation)
        prediction.save()

        logger.warning('Returning error: already exists id %s', _id)
        return jsonify({'error': 'id already exists'}), 400

    try:
        obs = pd.DataFrame([observation], columns=columns)
    except ValueError as e:
        logger.error('Returning error: %s', str(e), exc_info=True)
        default_response = {'id': _id, 'outcome': False}
        return jsonify(default_response), 200
    
    predicted_outcome = bool(pipeline.predict(obs))
    response = {'id': _id, 'outcome': predicted_outcome}
    p = Prediction(
        id=_id,
        observation=request.data,
        pred_class=predicted_outcome,
    )
    p.save()
    logger.info('Saved: %s', model_to_dict(p))
    logger.info('Prediction: %s', response)

    return jsonify(response)


@app.route('/recidivism_result', methods=['POST'])
def update():
    obs = request.get_json()
    logger.info('Observation:', obs)
    _id = obs['id']
    outcome = obs['outcome']

    if not _id:
        logger.warning('Returning error: no id provided')
        return jsonify({'error': 'id is required'}), 400
    if not Prediction.select().where(Prediction.id == _id).exists():
        logger.warning(f'Returning error: id {_id} does not exist in the database')
        return jsonify({'error': 'id does not exist'}), 400
    
    p = Prediction.get(Prediction.id == _id)
    p.true_class = outcome
    p.save()
    logger.info('Updated: %s', model_to_dict(p))
    
    predicted_outcome = p.pred_class
    response = {'id': _id, 'outcome': outcome, 'predicted_outcome': predicted_outcome}
    return jsonify(response)



@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000) # always check configured port