## Cybathlon recordings CLI

### Requirements

__cybathlon-cli__ connects by default to recordings server. Make sure to have this running

### Installation

```
pip install .
```

### Usage


### Cybathlon server api functions

```shell

cybathlon-cli

Usage:       cybathlon-cli <command>
             
```


### Playbook


Healthcheck

```shell script
python3 cybathlon-cli healthcheck
```


Upload a sample eeg
```shell script
python3 cybathlon-cli upload ~/Downloads/sample_eeg/A00054239004.raw
```

Put labels on it, either by correct order of subject, paradigm, feedback, comment filename
or by options --subject --paradigm --with-feedback

```shell script
python3 cybathlon-cli label 1 1 Hamlet true 'I am testing this functionality' 'd038f0bc-6e81-4768-91fb-e4bd6ec6e837'
```

All recordings with its labels
```shell script
python3 cybathlon-cli recordings
```

All recordings filtered by paradigm
```shell script
python3 cybathlon-cli recordings --paradigm 2
```

