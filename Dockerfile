
FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Pin uv to use the system Python (3.11 from base image), not download its own
ENV UV_PYTHON=/usr/local/bin/python3.11
ENV UV_PYTHON_DOWNLOADS=never

# Make uv use the base Python and put venv on PATH
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . .

# Install the project itself
RUN uv sync --frozen

# Default command (overridden by Docker Compose)
CMD ["python", "--version"]
