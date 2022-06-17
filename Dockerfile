

FROM python:3.9.1

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r  requirements.txt 


ENV PORT=


COPY app.py . 
COPY pages/ .

CMD streamlit run app.py --server.port=${PORT}  --browser.serverAddress="0.0.0.0"

