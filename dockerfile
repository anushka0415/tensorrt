# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Configure the production repository for NVIDIA Container Toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Optionally, configure the repository to use experimental packages
RUN sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list from the repository
RUN apt-get update

# Install the NVIDIA Container Toolkit packages
RUN apt-get install -y nvidia-container-toolkit

# Install NVIDIA Docker 2
RUN apt-get install -y nvidia-docker2

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /workspace
WORKDIR /workspace

# CMD or ENTRYPOINT to your application if needed
