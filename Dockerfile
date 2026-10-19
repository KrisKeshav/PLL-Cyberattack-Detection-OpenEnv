FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Hugging Face Spaces run containers as uid 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/env

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=app:app pll_cyber ./pll_cyber
COPY --chown=app:app server ./server
COPY --chown=app:app inference.py openenv.yaml pyproject.toml README.md ./

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
