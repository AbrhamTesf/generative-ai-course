# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Set the environment variable for the model path
ENV MODEL_PATH=/app/models

# Copy the models directory explicitly
COPY models/ ./models

# Copy the application code explicitly
COPY app/ ./app

# Copy the requirements file explicitly
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "--app-dir", "app", "main:app", "--host", "0.0.0.0", "--port", "8000"]