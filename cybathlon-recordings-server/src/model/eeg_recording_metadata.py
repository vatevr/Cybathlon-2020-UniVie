import uuid

from sqlalchemy import Column
from sqlalchemy import text as sa_text, ForeignKey, Integer, Text, Unicode, Boolean
from sqlalchemy.dialects.postgresql.base import UUID, TIMESTAMP

from src.service.base import Base


class EEGRecordingMetadata(Base):
    __tablename__ = 'eeg_recording_metadata'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
                server_default=sa_text('uuid_generate_v4()'))
    subject_id = Column(Integer, nullable=False)
    paradigm_id = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    comment = Column(Text, nullable=True)
    recorded_by = Column(Unicode(60), nullable=False)
    with_feedback = Column(Boolean, nullable=False)
    recording = Column(UUID, ForeignKey("eeg_recordings.id"))

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
