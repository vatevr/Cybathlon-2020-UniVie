import uuid

from flask import Flask, request, make_response, jsonify
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

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=sa_text('uuid_generate_v4()'))
    recording_file = db.Column(BYTEA(), nullable=False)

    def __init__(self, recording_file):
        self.recording_file = recording_file


def find_recording(id):
    return True

@app.route('/api/record', methods=['POST'])
def upload_recording():
    if 'file' not in request.files:
        return 'No file found!'

    file = request.files['file']
    recording = EEGRecording(recording_file=file.read())
    session.add(recording)
    session.commit()

    session.refresh(recording)

    print(str(recording.id))

    return str(recording.id) + ' uploaded to server'

@app.route('/api/label/<recording_id>', methods=['POST'])
def mark_metadata(recording_id):
    if not find_recording(recording_id):
        return 'not found any recordings with id: ' + recording_id

    content = request.json

    print(str(content))

    return jsonify({'uuid': recording_id})


def main(args=None):
    app.run(port=5003, debug=True)


if __name__ == '__main__':
    main(['-h'])