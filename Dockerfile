FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    swig \
    python3-dev \
    python3-pip \
    libeigen3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Spectra (header-only library)
RUN git clone --depth 1 --branch v1.0.1 https://github.com/yixuan/spectra.git /tmp/spectra \
    && cd /tmp/spectra \
    && cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --install build \
    && rm -rf /tmp/spectra

# Clone FEMNet (develop branch)
RUN git clone --depth 1 --branch develop https://github.com/ymzkd/FEMNet.git /femnet

WORKDIR /femnet

# Build C++ library
RUN cmake -B build -DBUILD_CSHARP=OFF -DBUILD_PYTHON=OFF \
    && cmake --build build

# Build Python bindings
RUN cmake -B build-python -DBUILD_CSHARP=OFF -DBUILD_PYTHON=ON \
    && cmake --build build-python

# Install FEMNet Python package
RUN pip install -e python/ --break-system-packages

# Install Python dependencies for Streamlit app
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --break-system-packages

# Copy application code
COPY app/ /app/
WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
