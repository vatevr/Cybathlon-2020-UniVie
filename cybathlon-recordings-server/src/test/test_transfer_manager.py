from src.service.transfer_manager import TransferManager

if __name__ == '__main__':
    transfer_manager = TransferManager()

    isExisting = transfer_manager.find_recording('49a5970f-449c-4763-95fe-e7d83285232a')

    assert isExisting, "file does not exist"

    print('file found, yay!')