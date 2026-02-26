FROM apache/spark:3.5.3
USER root
RUN pip install numpy
RUN pip install matplotlib
RUN pip install seaborn
USER spark