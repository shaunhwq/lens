FROM python:3.9

# Install OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ADD requirements.txt .
RUN pip install -r requirements.txt

# Copy server related files to docker container
COPY ./server ./server
COPY ./lens ./lens
ADD run.py .

EXPOSE 4051
CMD [ "python3", "run.py"]