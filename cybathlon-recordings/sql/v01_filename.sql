ALTER TABLE eeg_recordings
    ADD COLUMN filename VARCHAR(60);

UPDATE eeg_recordings SET filename = 'unknown';

ALTER TABLE eeg_recordings
ALTER COLUMN filename SET NOT NULL;
