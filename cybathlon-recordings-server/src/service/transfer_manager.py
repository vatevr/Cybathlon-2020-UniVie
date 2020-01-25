import uuid

from src.model.eeg_recording import EEGRecording
from src.service.base import Session, Base, engine


class TransferManager:
    def __init__(self):
        print('wof!')

        Base.metadata.create_all(engine)
        self.session = Session()


    def find_recording(self, id):
        rec = self.session.query(EEGRecording.id) \
            .with_entities(EEGRecording.id) \
            .filter(EEGRecording.id == uuid.UUID(id)) \
            .scalar()

        return rec is not None
