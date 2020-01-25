import requests as requests
from future.moves import sys

class Client(requests.Session):
    base: str

    def __init__(self, base: str):
        self.base = base
        super(Client, self).__init__()
        self.get = self.wrap(self.get)
        self.put = self.wrap(self.put)
        self.post = self.wrap(self.post)

    def wrap(self, method):
        def wrapper(url, *args, **kwargs):
            url = self.base + url

            print(url)

            response = method(url, verify=False, *args, **kwargs)
            try:
                res = response.json()
                if response.status_code != 200:
                    print("Error:", res["detail"], file=sys.stderr)
                    sys.exit(1)
            except ValueError:
                print(response.text)
                sys.exit(1)
            return res

        return wrapper
