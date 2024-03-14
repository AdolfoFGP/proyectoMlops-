ARG REGION

# Use a Python base image
FROM python:3.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -U \
    flask \
    gevent \
    gunicorn \
    numpy \
    xgboost \
    pandas \
    scikit-learn

RUN mkdir -p /opt/program
RUN mkdir -p /opt/ml

COPY app.py /opt/program
COPY model.py /opt/program
COPY nginx.conf /opt/program
COPY wsgi.py /opt/program
WORKDIR /opt/program

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]