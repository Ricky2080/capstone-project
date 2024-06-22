import os
import json
import joblib
import uuid
import pandas as pd
from flask import Flask, jsonify, request
from peewee import Model, CharField, BooleanField, TextField, SqliteDatabase
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
    logger.setLevel(logging.INFO)
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

DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db'
logger.info(f"Connecting to database at {DATABASE_URL}")
DB = connect(DATABASE_URL)

class Prediction(Model):
    id = CharField(primary_key=True, max_length=50)
    observation = TextField()
    pred_class = BooleanField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB
        table_name = 'prediction'

class APICallLog(Model):
    id = CharField(primary_key=True, max_length=50)
    log = TextField()
    
    class Meta:
        database = DB
        table_name = 'api_call_log'

def initialize_database():
    try:
        DB.connect(reuse_if_open=True)
        DB.create_tables([Prediction, APICallLog], safe=True)
        logger.info("Database initialized and tables created")
    except Exception as e:
        logger.error("Error initializing database: %s", str(e), exc_info=True)
        raise

initialize_database()

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

########################################
# Input validation functions

def validate_observation(observation):
    required_columns = {
        'id', 'name', 'sex', 'dob', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_case_number', 'c_charge_degree',
        'c_charge_desc', 'c_offense_date', 'c_arrest_date', 'c_jail_in'
    }
    valid_categories = {
        "sex": {'Male', 'Female'},
        "race": {'African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Others'}
    }

    missing_columns = required_columns - observation.keys()
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"

    invalid_columns = observation.keys() - required_columns
    if invalid_columns:
        return False, f"Invalid columns: {invalid_columns}"

    for col, valid_values in valid_categories.items():
        if observation[col] not in valid_values:
            return False, f"Invalid value for {col}: {observation[col]}"

    return True, ""

########################################
# Flask app and endpoints

app = Flask(__name__)

@app.route('/will_recidivate/', methods=['POST'])
def predict():
    try:
        obs_dict = request.get_json()
        logger.info('Observation: %s', obs_dict)
        _id = obs_dict.get('id')

        is_valid, error_message = validate_observation(obs_dict)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        if not _id:
            return jsonify({'error': 'id is required'}), 400

        if Prediction.select().where(Prediction.id == _id).exists():
            return jsonify({'error': 'id already exists'}), 400

        obs = pd.DataFrame([obs_dict], columns=columns)
        predicted_outcome = bool(pipeline.predict(obs)[0])
        response = {'id': _id, 'outcome': predicted_outcome}

        Prediction.create(
            id=_id,
            observation=json.dumps(obs_dict),
            pred_class=predicted_outcome,
        )
        logger.info('Saved: %s', response)
        return jsonify(response)
    except Exception as e:
        logger.error("Error in predict endpoint: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/recidivism_result/', methods['POST'])
def update():
    try:
        obs = request.get_json()
        logger.info('Observation: %s', obs)
        _id = obs.get('id')
        outcome = obs.get('outcome')

        if not _id:
            return jsonify({'error': 'id is required'}), 400
        if not Prediction.select().where(Prediction.id == _id).exists():
            return jsonify({'error': 'id does not exist'}), 400

        prediction = Prediction.get(Prediction.id == _id)
        prediction.true_class = outcome
        prediction.save()
        response = {'id': _id, 'outcome': outcome, 'predicted_outcome': prediction.pred_class}
        logger.info('Updated: %s', response)
        return jsonify(response)
    except Exception as e:
        logger.error("Error in update endpoint: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/list-db-contents')
def list_db_contents():
    try:
        return jsonify([model_to_dict(obs) for obs in Prediction.select()])
    except Exception as e:
        logger.error("Error in list-db-contents endpoint: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
