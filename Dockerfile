# Base Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Flask default port
EXPOSE 5000

# Run Flask app
CMD ["python", "app/api/flask_app.py"]