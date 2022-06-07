FROM europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest
 
# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy source code to image
COPY src/ /root/demo2/src/
COPY setup.py /root/demo2/setup.py

# Install custom package
RUN pip3 install /root/demo2/