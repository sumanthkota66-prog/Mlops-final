FROM python:3.12
WORKDIR /car-co2-volume
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python3.12 train.py
EXPOSE 5000
CMD ["python3.12", "app.py"]
