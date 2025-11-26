FROM python:3.10-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY esm_embedding.py .
COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
