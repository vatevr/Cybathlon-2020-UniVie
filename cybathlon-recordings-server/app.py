import datetime
import json
import logging
import os
import uuid
import warnings

from flask import Flask, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, joinedload

from src.model.eeg_recording import EEGRecording
from src.model.eeg_recording_metadata import EEGRecordingMetadata
from src.model.my_model import MyModel

app = Flask(__name__)

dbuser = os.environ.get('DBUSER', 'postgres')
dbpassword = os.environ.get('DBPASSWORD', 'docker')
dbhost = os.environ.get('DBHOST', '0.0.0.0')
dbport = os.environ.get('DBPORT', '5432')

print(f'postgres environment {dbhost}:{dbport}')

if not os.environ.get('TEST'):
    warnings.warn('TEST environment variable not found.')

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{dbuser}:{dbpassword}@{dbhost}:{dbport}/postgres'

engine = create_engine(f'postgresql://{dbuser}:{dbpassword}@{dbhost}:{dbport}/postgres')
Session = sessionmaker(bind=engine)
session = Session()

db = SQLAlchemy(app)

model = MyModel()

def find_recording(id):
    rec = session.query(EEGRecording.id) \
        .with_entities(EEGRecording.id) \
        .filter(EEGRecording.id == uuid.UUID(id)) \
        .scalar()

    return rec is not None


def api_error(message):
    return Response(json.dumps({'message': message}), status=500)


@app.route('/api/record', methods=['POST'])
def upload_recording():
    """
    accepts file uploads to server. maximum limit of the file imposed by postgresql is 1GB
    operation is transactional
    :return: success response after the succssful upload, internal server error after failure
    """
    if 'file' not in request.files:
        return Response(json.dumps({'message': 'internal server error!'}), status=500)

    file = request.files['file']
    recording = EEGRecording(recording_file=file.read(), filename=file.filename)

    try:
        session.add(recording)
        session.commit()
        session.refresh(recording)
        print(str(recording.id))
    except:
        session.rollback()
        logging.exception('')
        api_error("couldn't save the file")
        raise
    finally:
        session.close()

    return Response(json.dumps({'id': str(recording.id), 'filename': file.filename, 'message': 'upload successfull'}))


@app.route('/api/record/<recording_id>', methods=['GET'])
def download_recording(recording_id):
    # if not find_recording(recording_id):
    #     return api_error(f'file with id {recording_id} was not found')

    recording: EEGRecording = session \
        .query(EEGRecording) \
        .filter(EEGRecording.id == uuid.UUID(recording_id)) \
        .first()

    return api_error(f'file {recording_id} could not be found') if recording is None \
        else Response(recording.recording_file, headers={"Content-disposition": "attachment; filename=" + recording.filename})


@app.route('/api/label/<recording_id>', methods=['POST'])
def mark_metadata(recording_id):
    if not find_recording(recording_id):
        return api_error(f'file with id {recording_id} was not found')

    content = request.json

    recording_uuid = uuid.UUID(recording_id)

    metadata = EEGRecordingMetadata(int(content['subject_id']),
                                    int(content['paradigm_id']),
                                    datetime.datetime.now(),
                                    str(content['comment']),
                                    str(content['recorded_by']),
                                    bool(content['with_feedback']),
                                    recording_uuid)

    session.query(EEGRecordingMetadata) \
        .filter(EEGRecordingMetadata.recording == recording_uuid) \
        .delete()

    session.add(metadata)
    session.commit()

    session.refresh(metadata)
    print(str(content))

    return jsonify({'uuid': recording_id})


@app.route('/api/recordings', methods=['GET'])
def find_recordings():
    query = build_query()

    try:
        result = query.all()
        return Response(json.dumps([row.as_dict() for row in result]), mimetype='application/json')
    except:
        logging.exception('')

        return api_error('failed to retrieve recordings')


def build_query():
    fields = [EEGRecording.id, EEGRecording.filename, EEGRecording.eeg_metadata]

    query = session \
        .query(EEGRecording)\
        .options(joinedload(EEGRecording.eeg_metadata))

    if request.args.get('subject_id'):
        query = query.filter(
            EEGRecordingMetadata.subject_id == int(request.args.get('subject_id'))
        )
    if request.args.get('paradigm_id'):
        query = query.filter(
            EEGRecordingMetadata.paradigm_id == int(request.args.get('paradigm_id')),
        )
    if request.args.get('recorded_by'):
        query = query.filter(
            EEGRecordingMetadata.recorded_by.ilike(str(request.args.get('recorded_by'))),
        )

    # TODO query by feedback too
    return query


@app.route('/api/verify', methods=['GET'])
def verify_connection():
    connection_str = 'postgresql://' + dbuser + ':' + dbpassword + '@' + dbhost + ':' + dbport + '/postgres'
    verify_engine = create_engine(connection_str)
    engine_connection = verify_engine.connect()

    # Create a metadata instance
    metadata = MetaData(verify_engine)
    # Declare a table
    table = db.Table('Example', metadata,
                     db.Column('id', db.Integer, primary_key=True),
                     db.Column('name', db.String))

    # Create all tables
    metadata.create_all()
    tables = []
    for _t in metadata.tables:
        tables.append(_t)

    engine_connection.close()
    return Response(json.dumps(tables), mimetype='application/json')


def main(args=None):
    bind_to = {'hostname': "0.0.0.0", 'port': 9888}
    app.run(port=9888, host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main(['-h'])
