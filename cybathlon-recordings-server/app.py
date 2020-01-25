import datetime
import json
import logging
import uuid

from flask import Flask, request, jsonify, Response
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import joinedload

from src.model.eeg_recording import EEGRecording
from src.model.eeg_recording_metadata import EEGRecordingMetadata
from src.service.transfer_manager import TransferManager

app = Flask(__name__)

transfer_manager = TransferManager()


def find_recording(id):
    rec = transfer_manager.session.query(EEGRecording.id) \
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
        transfer_manager.session.add(recording)
        transfer_manager.session.commit()
        transfer_manager.session.refresh(recording)
        print(str(recording.id))
    except:
        transfer_manager.session.rollback()
        logging.exception('')
        api_error("couldn't save the file")
        raise
    finally:
        transfer_manager.session.close()

    return Response(json.dumps({'id': str(recording.id), 'filename': file.filename, 'message': 'upload successfull'}))


@app.route('/api/record/<recording_id>', methods=['GET'])
def download_recording(recording_id):
    # if not find_recording(recording_id):
    #     return api_error(f'file with id {recording_id} was not found')

    recording: EEGRecording = transfer_manager.session \
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

    transfer_manager.session.query(EEGRecordingMetadata) \
        .filter(EEGRecordingMetadata.recording == recording_uuid) \
        .delete()

    transfer_manager.session.add(metadata)
    transfer_manager.session.commit()

    transfer_manager.session.refresh(metadata)
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
    query = transfer_manager.session \
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
    connection_str = transfer_manager.connection_str
    verify_engine = create_engine(connection_str)
    engine_connection = verify_engine.connect()

    # Create a metadata instance
    metadata = MetaData(verify_engine)
    # Declare a table
    table = transfer_manager.db.Table('verify_table', metadata,
                     transfer_manager.db.Column('id', transfer_manager.db.Integer, primary_key=True),
                     transfer_manager.db.Column('name', transfer_manager.db.String))

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
