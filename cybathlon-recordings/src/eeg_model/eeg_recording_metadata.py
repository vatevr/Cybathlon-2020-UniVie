from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Unicode, Integer, Text, TIMESTAMP, Boolean
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


class EEGRecordingMetadata(Base):
    __tablename__ = 'eeg_recording_metadata'

    session_id = Column(Unicode(50), primary_key=True)
    subject_id = Column(Integer, nullable=False)
    paradigm_id = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    comment = Column(Text, nullable=True)
    recorded_by = Column(Unicode(60), nullable=False)
    with_feedback = Column(Boolean, nullable=False)
    recording = Column(UUID, nullable=False)

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
