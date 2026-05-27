FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.5-cuda12.1:latest

WORKDIR /workspace

# Install the project deps. We don't install the project itself in the image —
# the AML job mounts the source and runs `pip install -e .` at start.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# fasttext-langdetect ships the model file as a separate download on first use;
# we pre-warm it to avoid slow first-launch.
RUN python -c "import ftlangdetect; ftlangdetect.detect(text='hello world')" || true

# COMET model — large, so we pre-download in the image so workers don't all
# race for it at startup. Comment out if image size is a problem.
RUN python -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')" || true

CMD ["bash"]
