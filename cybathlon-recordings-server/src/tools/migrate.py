import logging

import fire
from sqlalchemy import text
from sqlalchemy.engine import ResultProxy
from tabulate import tabulate

from src.service.base import Session


class Migrations():
    def __output(self, data):
        if type(data) == list:
            print("Count:", len(data))
            for t in data:
                r = [[k, v] for k, v in t.items()]
                print(tabulate(r, tablefmt="fancygrid"))
        else:
            r = [[k, v] for k, v in data.items()]
            print(tabulate(r, tablefmt="fancygrid"))

    def migrate(self):
        session: Session = Session()

        fd = open('../../schema/v1_create_tables.sql', 'r')
        raw_sql = fd.read()
        fd.close()
        command = text(raw_sql)
        # Execute commands
        try:
            print('executing command')
            print(command)
            result: ResultProxy = session.execute(command)
            print(result.fetchall())
        except:
            logging.exception('')
            session.rollback()
            session.close()

        session.close()


if __name__ == '__main__':
    fire.Fire(Migrations(), name="migrations")
