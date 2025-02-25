# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the midterm project files
COPY midterm/requirements.txt .
COPY midterm/src ./src
COPY midterm/data ./data

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV PYTHONPATH=/app/src

# Run streamlit app
CMD ["streamlit", "run", "src/ui/app.py"]
