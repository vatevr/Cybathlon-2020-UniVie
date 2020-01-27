import datetime
import json
import logging
import uuid

from flask import Flask, request, Response
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import joinedload

from src.model.eeg_recording import EEGRecording
from src.model.eeg_recording_label import EEGRecordingLabel
from src.service.transfer_manager import TransferManager
from src.service.base import connection_str

app = Flask(__name__)

transfer_manager = TransferManager()


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
        return api_error('file required')

    file = request.files['file']

    if not file:
        return api_error('file is empty')

    recording = EEGRecording(recording_file=file.read(), filename=file.filename)

    try:
        transfer_manager.save(recording)
        print(str(recording.id))
    except:
        return api_error("couldn't save the file")

    return Response(json.dumps(recording.as_dict()))


@app.route('/api/record/<recording_id>', methods=['GET'])
def download_recording(recording_id):
    if not transfer_manager.find_recording(recording_id):
        return api_error(f'file with id {recording_id} was not found')

    recording: EEGRecording = transfer_manager.session \
        .query(EEGRecording) \
        .filter(EEGRecording.id == uuid.UUID(recording_id)) \
        .first()

    return api_error(f'file {recording_id} could not be found') if recording is None \
        else Response(recording.recording_file, headers={"Content-disposition": "attachment; filename=" + recording.filename})


@app.route('/api/label/<recording_id>', methods=['POST'])
def mark_metadata(recording_id):
    if not transfer_manager.find_recording(recording_id):
        return api_error(f'file with id {recording_id} was not found')

    content = request.json

    recording_uuid = uuid.UUID(recording_id)

    metadata = EEGRecordingLabel(int(content['subject_id']),
                                 int(content['paradigm_id']),
                                 datetime.datetime.now(),
                                 str(content['comment']),
                                 str(content['recorded_by']),
                                 bool(content['with_feedback']),
                                 recording_uuid)

    transfer_manager.session.query(EEGRecordingLabel) \
        .filter(EEGRecordingLabel.recording == recording_uuid) \
        .delete()

    transfer_manager.session.add(metadata)
    transfer_manager.session.commit()

    transfer_manager.session.refresh(metadata)
    print(str(content))

    return json.dumps({'uuid': recording_id})


@app.route('/api/recordings', methods=['GET'])
def find_recordings():
    query = transfer_manager.build_query(subject_id=request.args.get('subject_id'),
                                         paradigm_id=request.args.get('paradigm_id'),
                                         recorded_by=request.args.get('recorded_by'))

    try:
        result = query.all()
        return Response(json.dumps([row.as_dict() for row in result]), mimetype='application/json')
    except:
        logging.exception('')

        return api_error('failed to retrieve recordings')


@app.route('/api/verify', methods=['GET'])
def verify_connection():
    verify_engine = create_engine(connection_str)
    engine_connection = verify_engine.connect()


    # List all tables
    tables = verify_engine.table_names()
    engine_connection.close()

    return Response(json.dumps({'tables': tables}), mimetype='application/json')


def main(args=None):
    bind_to = {'hostname': "0.0.0.0", 'port': 9888}
    app.run(port=9888, host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main(['-h'])
