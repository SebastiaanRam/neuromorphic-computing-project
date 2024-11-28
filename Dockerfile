FROM python:3.9-slim
# Or any preferred Python version.
ADD test.py .

RUN apt-get update && apt-get install -y \
  python3-pip \
  python3-dev \
  libgl1-mesa-glx \
  libxrender1 \
  libxext6 \
  libsm6 \
  xvfb \
  && apt-get clean


# Install Python dependencies
RUN pip install python-dotenv gymnasium swig gymnasium[box2d] pyvirtualdisplay
CMD ["python3", "./test.py"]

