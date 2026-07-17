FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from torchgeo.datasets import OSCD100; OSCD100(root='data/oscd100', split='train', bands=OSCD100.rgb_bands, download=True)"

EXPOSE 8501

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
