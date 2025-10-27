FROM agrigorev/zoomcamp-model:2025

WORKDIR /app_pk

RUN pip install --no-cache-dir uv

COPY ["pyproject.toml","uv.lock", "./"]
COPY . .

RUN uv sync --frozen --no-install-project --no-cache

EXPOSE 9696

# Use uv to run uvicorn via the env created above
CMD ["uv", "run", "uvicorn", "predict_fast:app", "--host", "0.0.0.0", "--port", "9696"]

