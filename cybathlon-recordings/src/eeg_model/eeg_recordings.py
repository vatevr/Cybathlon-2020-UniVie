from sqlalchemy import Column, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EEGRecording(Base):
    __tablename__ = 'eeg_recordings'

    id = Column(UUID, primary_key=True)
    file = Column(LargeBinary, nullable=False)

    def __init__(self, file):
        self.file = file
