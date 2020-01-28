import warnings
from os import environ

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.model.my_model import MyModel

if not environ.get('TEST'):
    warnings.warn('TEST environment variable not found.')

dbuser = environ.get('DBUSER', 'postgres')
dbpassword = environ.get('DBPASSWORD', 'docker')
dbhost = environ.get('DBHOST', '0.0.0.0')
dbport = environ.get('DBPORT', '5432')

connection_str = f'postgresql://{dbuser}:{dbpassword}@{dbhost}:{dbport}/postgres'

print(f'postgres environment {dbhost}:{dbport}')

engine = create_engine(connection_str)
Session = sessionmaker(bind=engine)

# TEST code
model = MyModel()

Base = declarative_base()