import uuid

from sqlalchemy import Column
from sqlalchemy import text as sa_text, ForeignKey, Integer, Text, Unicode, Boolean
from sqlalchemy.dialects.postgresql.base import UUID, TIMESTAMP

from src.model.paradigm import Paradigm
from src.model.subject import Subject
from src.service.base import Base


class EEGRecordingLabel(Base):
    __tablename__ = 'eeg_recording_label'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
                server_default=sa_text('uuid_generate_v4()'))
    paradigm = Column(Integer, ForeignKey(Paradigm.id), nullable=False)
    subject = Column(Integer, ForeignKey(Subject.id), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    comment = Column(Text, nullable=True)
    recorded_by = Column(Unicode(60), nullable=False)
    with_feedback = Column(Boolean, nullable=False)
    recording = Column(UUID(as_uuid=True), ForeignKey("eeg_recordings.id"))

    def __init__(self,
                 subject_id,
                 paradigm_id,
                 created_at,
                 comment,
                 recorded_by,
                 with_feedback,
                 recording):
        self.subject = subject_id
        self.paradigm = paradigm_id
        self.created_at = created_at
        self.comment = comment
        self.recorded_by = recorded_by
        self.with_feedback = with_feedback
        self.recording = recording

    def as_dict(self):
        return {
            'id': str(self.id),
            'subject_id': self.subject,
            'paradigm_id': self.paradigm,
            'created_at': str(self.created_at),
            'comment': self.comment,
            'recorded_by': self.recorded_by,
            'with_feedback': self.with_feedback,
            'recording': str(self.recording),
        }
