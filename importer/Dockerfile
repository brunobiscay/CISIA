FROM python:3.11

WORKDIR /app

COPY importer/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY sources ./sources
COPY importer/ .

CMD ["python", "import.py"]