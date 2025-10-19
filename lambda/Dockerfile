# AWS Lambda Container Image for Art Classification
FROM public.ecr.aws/lambda/python:3.11

# Minimal system libs for image-related ops; keep image small
RUN yum -y install mesa-libGL libXrender libXext libX11 \
    && yum clean all && rm -rf /var/cache/yum

# App deps (general)
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    mangum==0.17.0 \
    uvicorn==0.30.6 \
    python-multipart==0.0.20 \
    pillow==11.0.0 \
    numpy==1.26.4 \
    boto3==1.35.36

# CPU-only PyTorch (Python 3.11, aarch64) from PyTorch CPU index
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchvision==0.17.2

# App code
COPY image_app/main.py ${LAMBDA_TASK_ROOT}/main.py

# Lambda handler
CMD ["main.handler"]


