from sqlalchemy import Column, VARCHAR, Integer

from src.service.base import Base


class Subject(Base):
    __tablename__ = 'subject'

    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(60), nullable=False)

    def __init__(self, name: str):
        assert name, 'subject name required'

        self.name = name