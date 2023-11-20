FROM python:3.10

WORKDIR /app

ENV Gradio_APP=app.py

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "app.py" ]
