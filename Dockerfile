FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Port
ENV PORT=8080

# Start
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
