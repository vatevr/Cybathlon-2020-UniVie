from src.service.transfer_manager import TransferManager

if __name__ == '__main__':
    transfer_manager = TransferManager()

    isExisting = transfer_manager.find_recording('5cc97694-fefc-4303-88cd-e101769b3404')

    assert isExisting, "file does not exist"

    print('file found, yay!')