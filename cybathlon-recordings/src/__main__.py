import uuid

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, LargeBinary, text as sa_text
from sqlalchemy.dialects.postgresql.base import UUID, BYTEA
from sqlalchemy.orm import sessionmaker


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:docker@localhost:5433/postgres'

db = SQLAlchemy(app)


class EEGRecording(db.Model):
    __tablename__ = 'eeg_recordings'

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=sa_text('uuid_generate_v4()'))
    recording_file = db.Column(BYTEA(), nullable=False)

    def __init__(self, recording_file):
        self.recording_file = recording_file


@app.route('/record', methods=['POST'])
def upload_recording():
    return 'Rest API works!'


def main(args=None):
    # app.run(port=5003)

    engine = create_engine('postgresql://postgres:docker@localhost:5433/postgres')
    Session = sessionmaker(bind=engine)
    session = Session()

    with open('../files/dummy.log', mode='rb') as file:
        recording1 = EEGRecording(recording_file=file.read())
        session.add(recording1)
        session.commit()

    result = session.query(EEGRecording).one()

    print(result.id)


if __name__ == '__main__':
    main(['-h'])