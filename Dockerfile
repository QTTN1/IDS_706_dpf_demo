FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sales_analysis.py conftest.py ./
COPY tests ./tests

CMD ["python", "sales_analysis.py"]