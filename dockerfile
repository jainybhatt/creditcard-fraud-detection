FROM python:3.11-slim

WORKDIR /credit_card

# Install dependencies (cached layer)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY fraud_detection/app.py .

# Copy model artifact (this is the handoff — model comes from DS, gets packaged here)

COPY models/fraud_model.pkl ./models
COPY models/fraud_detection_pipeline.pkl ./models


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# use path of the file if it is in another folder instead of '.' shown below : 
#docker build -t fraud-api path/to/folder