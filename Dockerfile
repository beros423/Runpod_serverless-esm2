FROM python:3.10-slim

COPY reqirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY esm_embeddings.py .
COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
