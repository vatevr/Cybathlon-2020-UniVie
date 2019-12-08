from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.orm import sessionmaker

if __name__ == '__main__':
    dbuser = 'postgres'
    dbpassword = 'docker'
    dbhost = '0.0.0.0'
    dbport = '5434'

    engine = create_engine(f'postgresql://{dbuser}:{dbpassword}@{dbhost}:{dbport}/postgres', pool_size=6)
    connection = engine.connect()

    print(engine)
    print(connection)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a metadata instance
    metadata = MetaData(engine)
    # Declare a table
    table = Table('Example', metadata,
                  Column('id', Integer, primary_key=True),
                  Column('name', String))
    # Create all tables
    metadata.create_all()
    for _t in metadata.tables:
        print("Table: ", _t)

