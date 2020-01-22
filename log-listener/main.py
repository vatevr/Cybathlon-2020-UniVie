import time
from log_listener import LogListener
from config_reader import ConfigReader
from watchdog.observers import Observer

if __name__ == "__main__":
    config_reader = ConfigReader()
    event_handler = LogListener(config_reader)
    observer = Observer()
    observer.schedule(event_handler, path=(config_reader.get_path() + '/log'), recursive=False)
    observer.start()
    print('\nNow listening for log entries. Please start the race!', end='\n\n')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()