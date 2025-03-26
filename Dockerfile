# Use a base image with Python 3.13.2 (if available) or build it from source
FROM python:3.13-slim

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
RUN poetry install .

# Copy the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["poetry","run","uvicorn", "app:app", "--reload", "--env-file", ".env"]
