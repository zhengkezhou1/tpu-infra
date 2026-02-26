FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu@sha256:33fd74d1ac4a45de18855cfec6c41644bf59d5b4e76a343d32f52b6553f0e804

RUN apt-get update && \
    apt-get install -y python3.12 python3.12-dev curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    python3.12 -m pip install --no-cache-dir jax[tpu] prometheus_client && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY metric.py /app/metric.py

WORKDIR /app

EXPOSE 8000

CMD ["python3.12", "metric.py"]
