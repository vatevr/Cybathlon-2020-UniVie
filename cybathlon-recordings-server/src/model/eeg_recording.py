import uuid

from sqlalchemy import Column, VARCHAR
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql.base import UUID, BYTEA
from sqlalchemy.orm import relationship

from src.service.base import Base


class EEGRecording(Base):
    __tablename__ = 'eeg_recordings'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
                server_default=sa_text('uuid_generate_v4()'))
    recording_file = Column(BYTEA(), nullable=False)
    filename = Column(VARCHAR(60), nullable=False)
    eeg_metadata = relationship("EEGRecordingMetadata", lazy='joined')

    def __init__(self, recording_file: bytes, filename: str):
        assert filename, 'filename required'
        assert recording_file, 'file should be present'

        self.recording_file = recording_file
        self.filename = filename

    def as_dict(self):
        base = {
            'id': str(self.id),
            'filename': self.filename,
        }

        if self.eeg_metadata:
            base.update(self.eeg_metadata.as_dict())

        return base