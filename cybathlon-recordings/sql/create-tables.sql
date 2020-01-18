CREATE TABLE eeg_recordings (
    id UUID PRIMARY KEY,
    recording_file BYTEA NOT NULL,
    filename VARCHAR(60) NOT NULL
);

CREATE TABLE eeg_recording_metadata
(
    id            UUID PRIMARY KEY,
    subject_id    INT         NOT NULL,
    paradigm_id   INT         NOT NULL,
    created_at    TIMESTAMP   NOT NULL,
    comment       TEXT,
    recorded_by   VARCHAR(60) NOT NULL,
    with_feedback BOOLEAN     NOT NULL,
    recording     UUID        NOT NULL UNIQUE,
    FOREIGN KEY (recording) REFERENCES eeg_recordings (id)
);