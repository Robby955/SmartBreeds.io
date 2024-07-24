FROM python:3.12

ENV PORT 8080
# Install system dependencies required for mysqlclient
RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the Python dependencies file and install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


ENV GOOGLE_APPLICATION_CREDENTIALS=/app/smartdogbreed-416219-c363a1e8919c.json

COPY breed_names.txt /app/breed_names.txt
COPY most_common_misclassifications.csv /app/most_common_misclassifications.csv
COPY result_summary_20240403-204919.txt /app/result_summary_20240403-204919.txt

# Copy the rest of the application
COPY . /app

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 8080
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]