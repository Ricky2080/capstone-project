import os
import json
import joblib
import uuid
import pandas as pd
from flask import Flask, jsonify, request
from peewee import Model, CharField, BooleanField, TextField
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

class API_call_log(Model):
    id = CharField(primary_key=True, max_length=50)
    log = TextField()
    
    class Meta:
        database = DB

# Ensure database connection is established and tables are created
def initialize_database():
    try:
        DB.connect(reuse_if_open=True)
        DB.create_tables([Prediction, API_call_log], safe=True)
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

def check_valid_column(observation):
    valid_columns = {
        'id', 'name', 'sex', 'dob', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_case_number', 'c_charge_degree',
        'c_charge_desc', 'c_offense_date', 'c_arrest_date', 'c_jail_in'
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

def check_categorical_values(observation):
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
            error = "Categorical field {} missing".format(key)
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
    try:
        call_id = str(uuid.uuid4())
        log_content = {
            'method': request.method,
            'url': request.url,
            'headers': dict(request.headers),
            'body': request.get_data(as_text=True)
        }
        API_call_log.create(id=call_id, log=json.dumps(log_content))
        DB.commit()  # Ensure commit of log entry
    except Exception as e:
        logger.error("Error logging API call: %s", str(e), exc_info=True)

@app.route('/will_recidivate/', methods=['POST'])
def predict():
    try:
        obs_dict = request.get_json()
        logger.info('Observation: %s', obs_dict)
        _id = obs_dict['id']
        observation = obs_dict
        
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
            prediction.observation = json.dumps(observation)
            prediction.save()
            DB.commit()  # Ensure commit of updated entry
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
            observation=json.dumps(observation),
            pred_class=predicted_outcome,
        )
        p.save()
        DB.commit()  # Ensure commit of new entry
        logger.info('Saved: %s', model_to_dict(p))
        logger.info('Prediction: %s', response)

        return jsonify(response)
    except Exception as e:
        logger.error("Error in predict endpoint: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/recidivism_result/', methods=['POST'])
def update():
    try:
        obs = request.get_json()
        logger.info('Observation: %s', obs)
        _id = obs['id']
        outcome = obs['outcome']

        if not _id:
            logger.warning('Returning error: no id provided')
            return jsonify({'error': 'id is required'}), 400
        if not Prediction.select().where(Prediction.id == _id).exists():
            logger.warning('Returning error: id %s does not exist in the database', _id)
            return jsonify({'error': 'id does not exist'}), 400
        
        p = Prediction.get(Prediction.id == _id)
        p.true_class = outcome
        p.save()
        DB.commit()  # Ensure commit of updated entry
        logger.info('Updated: %s', model_to_dict(p))
        
        predicted_outcome = p.pred_class
        response = {'id': _id, 'outcome': outcome, 'predicted_outcome': predicted_outcome}
        return jsonify(response)
    except Exception as e:
        logger.error("Error in update endpoint: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/list-db-contents')
def list_db_contents():
    try:
        return jsonify([
            model_to_dict(obs) for obs in Prediction.select()
        ])
    except Exception as e:
        logger.error("Error in list-db-contents endpoint: %s", str(e), exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
