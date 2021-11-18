FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
ADD . /
ENV FLASH_APP=app.py
WORKDIR .
RUN pip install -r /requirements.txt
ENTRYPOINT ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]
