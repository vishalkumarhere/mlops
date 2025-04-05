# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Default command to run the app
CMD ["streamlit", "run", "app.py"]
