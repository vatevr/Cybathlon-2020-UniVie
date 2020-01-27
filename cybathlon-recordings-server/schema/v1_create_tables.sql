BEGIN;

CREATE TABLE public.migrations
(
    number  SERIAL PRIMARY KEY,
    comment VARCHAR(60)
);

CREATE TABLE public.current_version
(
    current_number SERIAL PRIMARY KEY,
    FOREIGN KEY (current_number) REFERENCES migrations (number)
);

CREATE TABLE public.eeg_recordings
(
    id             UUID PRIMARY KEY,
    recording_file BYTEA       NOT NULL,
    filename       VARCHAR(60) NOT NULL
);

CREATE TABLE public.paradigm
(
    id   SERIAL PRIMARY KEY,
    name VARCHAR(60) NOT NULL
);

CREATE TABLE public.subject
(
    id   SERIAL PRIMARY KEY,
    name VARCHAR(60) NOT NULL
);

CREATE TABLE public.eeg_recording_label
(
    id            UUID PRIMARY KEY,
    subject       INT         NOT NULL,
    paradigm      INT         NOT NULL,
    created_at    TIMESTAMP   NOT NULL,
    comment       TEXT,
    recorded_by   VARCHAR(60) NOT NULL,
    with_feedback BOOLEAN     NOT NULL,
    recording     UUID        NOT NULL UNIQUE,
    FOREIGN KEY (recording) REFERENCES eeg_recordings (id),
    FOREIGN KEY (subject) REFERENCES subject (id),
    FOREIGN KEY (paradigm) REFERENCES paradigm (id)
);

INSERT INTO public.migrations(comment) VALUES ('initial schema') RETURNING public.migrations.number, public.migrations.comment;

INSERT INTO public.paradigm(id, name)
VALUES (1, 'move right hand'),
       (2, 'move left hand'),
       (3, 'make right fist'),
       (4, 'make left fist'),
       (5, 'hug'),
       (6, 'move feet'),
       (7, 'stand up'),
       (8, 'music'),
       (9, 'taste'),
       (10, 'touch'),
       (11, 'calculate')
RETURNING public.paradigm.name;

INSERT INTO public.subject(name)
VALUES ('John Doe'),
       ('Jane Doe')
RETURNING public.subject.name;

COMMIT;