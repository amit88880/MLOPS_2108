# FROM ubuntu:latest
FROM python:3.9
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./serve.py /exp/serve.py
COPY ./svm_gamma_0.001_C_0.5.joblib /exp/svm_gamma_0.001_C_0.5.joblib
RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
# CMD ["python3", "./plot_graph.py"]
CMD ["python3","./serve.py"]
# for build: docker build -t exp:v1 -f docker/Dockerfile .
# for running a container: docker run -p 5000:5000 -it exp:v1