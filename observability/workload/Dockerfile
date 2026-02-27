FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu@sha256:d3dbf1a00bf1c5b61bbd1d7456cca3ec090fd3200536a4b61a8fbd3de8ef23f3

RUN pip install prometheus_client

COPY metric.py /app/metric.py

WORKDIR /app

EXPOSE 9000

CMD ["python", "metric.py"]
