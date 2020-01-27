import logging
import uuid

from sqlalchemy.orm import joinedload, defer

from src.model.eeg_recording import EEGRecording
from src.model.eeg_recording_label import EEGRecordingLabel
from src.service.base import Session, Base, engine


class TransferManager:
    def __init__(self):
        print('wof!')

        Base.metadata.create_all(engine)
        self.session = Session()

    def find_recording(self, id: str):
        rec = self.session.query(EEGRecording.id) \
            .with_entities(EEGRecording.id) \
            .filter(EEGRecording.id == uuid.UUID(id)) \
            .scalar()

        return rec is not None

    def save(self, recording: EEGRecording):
        try:
            self.session.add(recording)
            self.session.commit()
            self.session.refresh(recording)
            print(str(recording.id))
        except:
            self.session.rollback()
            logging.exception('')
            raise
        finally:
            self.session.close()

    def build_query(self, subject_id=None, paradigm_id=None, recorded_by=None):
        query = self.session \
            .query(EEGRecording) \
            .join(EEGRecordingLabel, isouter=True) \
            .options(joinedload(EEGRecording.eeg_metadata), defer(EEGRecording.recording_file))

        if subject_id:
            query = query.filter(
                EEGRecordingLabel.subject == int(subject_id)
            )
        if paradigm_id:
            query = query.filter(
                EEGRecordingLabel.paradigm == int(paradigm_id),
            )
        if recorded_by:
            query = query.filter(
                EEGRecordingLabel.recorded_by.ilike(str(recorded_by)),
            )

        # TODO query by feedback too
        return query

