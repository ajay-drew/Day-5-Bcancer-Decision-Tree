FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Model_dt.py .
COPY data.csv . 

CMD ["python", "Model_dt.py"]