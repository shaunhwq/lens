FROM python:3.9

# Copy server related files to docker container
COPY ./server ./server
COPY ./lens ./lens
ADD run.py .
ADD requirements.txt .

# Install OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

EXPOSE 4051
CMD [ "python3", "run.py"]