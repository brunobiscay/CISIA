FROM python:3.11

WORKDIR /app

COPY orchestration/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY ml ./ml
COPY orchestration ./orchestration

CMD ["python", "orchestration/deployments.py"]
