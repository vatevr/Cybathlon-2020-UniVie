import json
import re
import sys

import fire
import requests
from fire.core import FireError
from tabulate import tabulate


class Client(requests.Session):
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


class HttpUtils:
    @staticmethod
    def get_filename_from_cd(headers: dict):
        """
        retrieve filename from content-disposition http header
        """

        cd = headers.get('content-disposition')

        if not cd:
            return None
        filename = re.findall('filename=(.+)', cd)
        if len(filename) == 0:
            return None
        return filename[0]


class RecordingsApi:
    def __init__(self):
        self._base = 'http://0.0.0.0:9888/api'
        self._last_id = None
        self._client = Client(self._base)

    def __output(self, data):
        if type(data) == list:
            print("Count:", len(data))
            for t in data:
                r = [[k, v] for k, v in t.items()]
                print(tabulate(r, tablefmt="fancygrid"))
        else:
            r = [[k, v] for k, v in data.items()]
            print(tabulate(r, tablefmt="fancygrid"))

    def healthcheck(self):
        print('starting healthcheck')

        response = self._client.get('/verify?port=5434&host=localhost')
        print(f'healthcheck successful {response}')

    def upload(self, filepath):
        if not filepath:
            print('error - please select a recording file to upload!')

        print('starting upload..')
        print(f'uploading file {filepath}')

        file = open(filepath, 'rb')

        files = {'file': file}

        response = requests.post(self._base + '/record', files=files)

        print(f'uploading a recording status code {response.status_code}')

        if response.status_code > 200:
            print('to verify connectivity, use cybathlon healthcheck command..')

        if response.status_code == 200:
            self._last_id = response.json()['id']
            self.__output(response.json())
            print(f'uploaded file with id {self._last_id}')

    def download(self, file_id):
        response = requests.get(f'{self._base}/record/{file_id}')

        if response.status_code > 200:
            print('file could not be retrieved, check if the id is correct')
        else:
            filename = HttpUtils.get_filename_from_cd(response.headers)
            open(f'../downloads/{filename}', 'wb').write(response.content)
            print(f'file {filename} was successfully downloaded to downloads folder')

    def label(self, subject, paradigm, recorded_by, with_feedback, comment=None, file_id=None):
        file_id = file_id if file_id is not None else self._last_id

        if file_id is None:
            raise FireError('no file was selected, please upload a recording or provide a valid recording id')

        arguments = {
            'subject_id': int(subject),
            'paradigm_id': int(paradigm),
            'recorded_by': str(recorded_by),
            'with_feedback': with_feedback.lower() == 'true',
            'comment': comment
        }

        response = requests.post(f'{self._base}/label/{file_id}', json=arguments)

        if response.status_code == 200:
            self.__output(response.json())
            print(f'file successfully labeled {file_id}')

    def recordings(self, subject=None, paradigm=None, recorded_by=None):
        params_raw = {
            'subject_id': int(subject) if subject is not None else None,
            'paradigm_id': int(paradigm) if paradigm is not None else None,
            'recorded_by': str(recorded_by) if recorded_by is not None else None
        }
        params = {k: v for k, v in params_raw.items() if v is not None}

        print(f'fetching recordings with params {json.dumps(params)}')

        response = requests.get(f'{self._base}/recordings', params=params)
        if response.status_code == 200:
            print(f'files successfully retrieved')
            self.__output(response.json())


def main():
    cybathlon = RecordingsApi()
    fire.Fire(cybathlon, name="cybathlon")


if __name__ == "__main__":
    main()
