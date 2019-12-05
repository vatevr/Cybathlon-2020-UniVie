import datetime
import json
import uuid

from flask import Flask, request, make_response, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, LargeBinary, text as sa_text
from sqlalchemy.dialects.postgresql.base import UUID, BYTEA
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:docker@localhost:5433/postgres'

engine = create_engine('postgresql://postgres:docker@localhost:5433/postgres')
Session = sessionmaker(bind=engine)
session = Session()

db = SQLAlchemy(app)


class EEGRecording(db.Model):
    __tablename__ = 'eeg_recordings'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
                   server_default=sa_text('uuid_generate_v4()'))
    recording_file = db.Column(BYTEA(), nullable=False)

    def __init__(self, recording_file):
        self.recording_file = recording_file


class EEGRecordingMetadata(db.Model):
    __tablename__ = 'eeg_recording_metadata'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
                   server_default=sa_text('uuid_generate_v4()'))
    subject_id = db.Column(db.Integer, nullable=False)
    paradigm_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, nullable=False)
    comment = db.Column(db.Text, nullable=True)
    recorded_by = db.Column(db.Unicode(60), nullable=False)
    with_feedback = db.Column(db.Boolean, nullable=False)
    recording = db.Column(UUID(as_uuid=True), nullable=False)

    def __init__(self,
                 subject_id,
                 paradigm_id,
                 created_at,
                 comment,
                 recorded_by,
                 with_feedback,
                 recording):
        self.subject_id = subject_id
        self.paradigm_id = paradigm_id
        self.created_at = created_at
        self.comment = comment
        self.recorded_by = recorded_by
        self.with_feedback = with_feedback
        self.recording = recording

    def as_dict(self):
        return {
            'id': str(self.id),
            'subject_id': self.subject_id,
            'paradigm_id': self.paradigm_id,
            'created_at': str(self.created_at),
            'comment': self.comment,
            'recorded_by': self.recorded_by,
            'with_feedback': self.with_feedback,
            'recording': str(self.recording),
        }


# TODO db level validation
def find_recording(id):
    return True


@app.route('/api/record', methods=['POST'])
def upload_recording():
    if 'file' not in request.files:
        return Response(json.dumps({'message': 'internal server error!'}), status=500)

    file = request.files['file']
    recording = EEGRecording(recording_file=file.read())
    session.add(recording)
    session.commit()

    session.refresh(recording)

    print(str(recording.id))

    return Response(json.dumps({'id': str(recording.id), 'message': 'upload successfull'}))


@app.route('/api/label/<recording_id>', methods=['POST'])
def mark_metadata(recording_id):
    if not find_recording(recording_id):
        return 'not found any recordings with id: ' + recording_id

    content = request.json

    metadata = EEGRecordingMetadata(int(content['subject_id']),
                                    int(content['paradigm_id']),
                                    datetime.datetime.now(),
                                    str(content['comment']),
                                    str(content['recorded_by']),
                                    bool(content['with_feedback']),
                                    uuid.UUID(recording_id))

    session.add(metadata)
    session.commit()

    session.refresh(metadata)
    print(str(content))

    return jsonify({'uuid': recording_id})


@app.route('/api/recordings', methods=['GET'])
def find_recordings():
    query_params = {key: int(request.args.get(key))
                    for key in ['subject_id', 'paradigm_id']
                    if request.args.get(key) is not None}

    if 'recorded_by' in request.args:
        query_params['recorded_by'] = request.args.get('recorded_by')

    query = session.query(EEGRecordingMetadata)

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

    result = query.all()

    return Response(json.dumps([row.as_dict() for row in result]), mimetype='application/json')


def main(args=None):
    app.run(port=5003, debug=True)


if __name__ == '__main__':
    main(['-h'])
