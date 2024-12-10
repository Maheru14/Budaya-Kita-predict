FROM tensorflow/tensorflow:2.18.0
WORKDIR /app
COPY . /app
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/budayakita-690e5dcb36ab.json"
RUN pip install --upgrade pip
RUN rm -rf /usr/lib/python3/dist-packages/blinker* && pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["python", "predicted.py"]