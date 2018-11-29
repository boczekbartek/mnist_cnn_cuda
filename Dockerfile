FROM samuelcolvin/tensorflow-gpu-py36

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt