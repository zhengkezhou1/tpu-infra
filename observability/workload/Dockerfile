FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu@sha256:33fd74d1ac4a45de18855cfec6c41644bf59d5b4e76a343d32f52b6553f0e804

RUN conda create -n pretrain -c conda-forge python=3.12 -y && \
    conda run -n pretrain pip install --no-cache-dir jax[tpu] prometheus_client

COPY metric.py /app/metric.py

WORKDIR /app

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "pretrain", "python", "metric.py"]
