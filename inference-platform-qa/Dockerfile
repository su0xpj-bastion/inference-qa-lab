FROM python:3.12-slim
RUN useradd --uid 10001 --no-create-home app
WORKDIR /app
COPY mockserve/ mockserve/
USER 10001
EXPOSE 8000
# exec form: python is PID 1 and receives SIGTERM directly (no shell swallowing it)
ENTRYPOINT ["python", "-m", "mockserve", "--port", "8000"]
