FROM python:3.7.1
COPY . /cybathlon-recordings
WORKDIR /cybathlon-recordings
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["./src/app.py"]
