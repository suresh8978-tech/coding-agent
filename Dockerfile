FROM python:3.12-slim

# Install Terraform and Git
RUN apt-get update && apt-get install -y wget unzip git && \
    wget -q https://releases.hashicorp.com/terraform/1.7.5/terraform_1.7.5_linux_amd64.zip && \
    unzip terraform_1.7.5_linux_amd64.zip -d /usr/local/bin/ && \
    rm terraform_1.7.5_linux_amd64.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent.py .
COPY tools/ tools/
COPY user_config.yaml .

EXPOSE 8001
CMD ["python", "agent.py"]
