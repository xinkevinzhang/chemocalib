FROM python:3.11-slim

LABEL maintainer="ChemoCalib Authors"
LABEL description="ChemoCalib: Chemometrics-Calibrated Constraint-Based Metabolic Modeling"
LABEL version="0.2.0"

# Install GLPK for COBRApy FBA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        glpk-utils \
        libglpk-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /chemocalib

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Default command
CMD ["python", "scripts/run_pipeline.py"]
