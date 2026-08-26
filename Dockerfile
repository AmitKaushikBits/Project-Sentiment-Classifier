# Dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN python -c "from pathlib import Path; lines=Path('requirements.txt').read_text(encoding='utf-16').splitlines(); Path('requirements-linux.txt').write_text('\\n'.join(line for line in lines if not line.lower().startswith('pywin32==')) + '\\n', encoding='utf-8')" \
	&& pip install --no-cache-dir -r requirements-linux.txt
COPY src/ ./src
COPY models/ ./models
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]