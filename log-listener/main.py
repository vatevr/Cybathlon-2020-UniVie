import time
from log_listener import LogListener
from dependencies import Dependencies
from watchdog.observers import Observer

if __name__ == "__main__":
    dependencies = Dependencies()
    event_handler = LogListener(dependencies)
    observer = Observer()
    observer.schedule(event_handler, path=(dependencies.get_config_reader().get_path() + '/log'), recursive=False)
    observer.start()
    print('\nNow listening for log entries. Please start the race!', end='\n\n')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()