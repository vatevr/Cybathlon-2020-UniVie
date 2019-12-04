CREATE TABLE eeg_recordings (
    id UUID PRIMARY KEY,
    recording_file BYTEA NOT NULL
);

CREATE TABLE eeg_recording_metadata
(
    session_id    VARCHAR(50) PRIMARY KEY,
    subject_id    INT         NOT NULL,
    paradigm_id   INT         NOT NULL,
    created_at    TIMESTAMP   NOT NULL,
    comment       TEXT,
    recorded_by   VARCHAR(60) NOT NULL,
    with_feedback BOOLEAN     NOT NULL,
    recording     uuid       NOT NULL,
    FOREIGN KEY (recording) REFERENCES eeg_recordings(id)
);