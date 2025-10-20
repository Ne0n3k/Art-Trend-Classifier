# AWS Lambda Container Image for Art Classification
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for image processing
RUN yum -y install mesa-libGL libXrender libXext libX11 \
    && yum clean all && rm -rf /var/cache/yum

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    mangum==0.17.0 \
    uvicorn==0.35.0 \
    python-multipart==0.0.20 \
    slowapi==0.1.9 \
    pydantic==2.11.7 \
    pillow==11.3.0 \
    numpy==2.2.6 \
    boto3==1.35.36

# Install PyTorch CPU-only (Python 3.11, aarch64)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.0 \
    torchvision==0.24.0

# Install albumentations for image preprocessing
RUN pip install --no-cache-dir \
    albumentations==2.0.8 \
    opencv-python-headless==4.12.0.88

# Copy application code
COPY image_app/ ${LAMBDA_TASK_ROOT}/

# Lambda handler
CMD ["main.handler"]