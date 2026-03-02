![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)
 
# **Project Name: ECF3 -MACHINE LEARNING SUPERVISÉ**  
![Version](https://img.shields.io/badge/version-v1.0-blue)

 
## Table of Contents
 
1. [About the Project](#about-the-project)
2. [Built With](#built-with)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Project structure](#project-structure)
4. [Installation and configuration](#installation-and-configuration)
   - [Installation](#installation)
   - [Configuration](#configuration)
        -   [Docker-compose](#docker-compose)
5. [Usage](#usage)
   - [Notebooks](#notebooks)
   - [Uninstall](#uninstall)
6. [Authors](#authors)
7. [License](#license)
 
---
 
## About the Project

This project focuses on predicting customer churn for TeleCom+ using machine learning. The goal is to identify high-risk customers in order to optimize retention strategies and minimize revenue loss. The project compares two approaches: **Scikit-learn** for small-to-medium scale datasets and **Spark MLlib** for large-scale, distributed data processing.

Data preparation includes cleaning, encoding categorical variables, scaling numerical features, and handling class imbalance. Multiple models are trained, evaluated, and compared to select the best approach for production deployment.

### Comparison Between Scikit-learn and Spark MLlib

- **Scikit-learn**: Efficient for single-machine processing, easier to prototype and debug, with rich model evaluation tools. Best suited for datasets that fit into memory.  
- **Spark MLlib**: Optimized for distributed computing on large datasets, scalable, handles big data seamlessly, but training can take longer and debugging is more complex.  
- Both approaches were applied to the same dataset to compare **performance metrics (Accuracy, Precision, Recall, F1-score)** and **execution time**, helping to select the most appropriate method for production.

### **Project goals include:**  

- Implement and compare predictive models using **Scikit-learn** and **Spark MLlib**.  
- Evaluate models based on **accuracy, precision, recall, F1-score**, and execution time.  
- Identify the most important features driving customer churn for business insights.  
- Demonstrate the ability to scale machine learning pipelines from single-machine to distributed environments.  
- Provide actionable recommendations to the Customer Success team based on model results.
 
 
## Built With
 
The following technologies are used in this project:
 
https://github.com/inttter/md-badges


- ![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) **Version 3.11** – Main programming language used for data processing and analysis.  
- ![PySpark](https://img.shields.io/badge/PySpark-F26207?logo=replit&logoColor=fff) **Version 3.5.7** – Used for large-scale data processing and cleaning.  
- ![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white) Used to **containerize** the application and ensure a consistent runtime environment.  
- ![GitHub](https://img.shields.io/badge/GitHub-24292F?style=for-the-badge&logo=github&logoColor=white) Used for **version control and collaboration**, keeping the project organized and maintainable.  
- ![VSCode](https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white) **IDE** used for development, debugging, and running the project locally.  
- ![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff) Used to **create plots and visualizations** for data exploration and reporting.  
- ![Seaborn](https://img.shields.io/badge/Seaborn-EC407A?logo=seaborn&logoColor=fff) Used for **statistical data visualization**, especially heatmaps, boxplots, and regression plots.  
- ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff) Used for **data manipulation and analysis**, working with CSV, Parquet, and enriched datasets.  
- ![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff) Used for **numerical computing and array operations**, supporting analysis and calculations.
- ![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white) – A Python library for machine learning, providing tools for classification, regression, clustering, and model evaluation.
- ![PySpark ML](https://img.shields.io/badge/-PySpark%20ML-%23F26207?logo=apache-spark&logoColor=white) – PySpark's machine learning library for scalable ML pipelines, classification, regression, and feature engineering.


 
# Getting Started
 
## Prerequisites
 
Before you can install the project, make sure you have the following installed on your machine and to have basic knowledge of the following technologies:
 
- **Docker** – For containerization.  
  - [Install Docker](https://www.docker.com/products/docker-desktop/)  

- **Python 3.10+**  
  - [Install Python](https://www.python.org/downloads/)  

- **Git** – Version control.  
  - [Install Git](https://git-scm.com/downloads)  

- **Java 11 or 17** installed on your system.  
  - Set **JAVA_HOME** environment variable to your Java installation directory.  
  - [Install Java](https://www.oracle.com/java/technologies/javase-jdk11-archive-downloads.html)  

- **HADOOP_HOME** environment variable pointing to your Hadoop installation folder.  
  - [Download Hadoop](https://hadoop.apache.org/releases.html)  

## Project structure

Complete project structure (Finished) :

```bash

.
├── .venv                                # Virtual environment for Python dependencies
├── .gitignore                           # Git ignore file
├── docker-compose.yml                    # (OPTIONAL) Docker Compose setup for PySpark 3.5.7
├── Dockerfile                            # (OPTIONAL) Dockerfile for environment setup
├── README.md                             # Project README
├── requirements.txt                      # Python dependencies
├── apps
│   ├── 01_sklearn.ipynb                 # Notebook: scikit-learn analysis
│   ├── 02_spark.ipynb                   # Notebook: PySpark analysis
│   ├── 02_spark.py                      # OPTIONAL (can be launch with docker-compose)
│   └── 03_bonus.ipynb                    # Notebook: graphics & scikit-learn vs PySpark comparison
└── data
    ├── 03_DONNEES.csv                    # Raw dataset
    ├── feature_importance.csv            # Top 10 features from scikit-learn
    ├── model_metrics_sklearn_best_model_.json   # Best model metrics (scikit-learn)
    ├── model_metrics_sklearn_comparaison.json  # Comparison metrics (scikit-learn)
    ├── model_results_pyspark_best_model.json   # Best model metrics (PySpark)
    ├── model_results_pyspark_comparaison.json  # Comparison metrics (PySpark)
    ├── predictions_test.csv               # Test set predictions (churn probability)
    └── rapport.md                         # Markdown report


```

# Installation and configuration

## Installation
 
To install and set up the project, follow these steps:
 
1. Clone the repository:
```bash
git clone https://github.com/titibemo/ecf-3-machine-learning-supervise
```

Once the project has been cloned, open a terminal in your IDE and run the following command to navigate into the project directory:
```bash
cd ecf-3-machine-learning-supervise
```

2. Create a virtual environment to isolate dependencies : Navigate to the project root, open a terminal and run the following commands:
```bash
# On windows
python -m venv venv 
venv\Scripts\activate         
```

Install the required packages :
```bash
pip install -r requirements.txt  
```
 
## Configuration

### Docker-compose

```yml
services:
  spark-master:
    build: .
    container_name: spark-master
    hostname: spark-master
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master
    environment:
      - SPARK_MASTER_HOST=spark-master
      - SPARK_MASTER_PORT=7077
      - SPARK_MASTER_WEBUI_PORT=8080
    ports:
      - "8080:8080"
      - "7077:7077"
      - "4040:4040"
    volumes:
      - ./data:/data
      - ./apps:/apps
    networks:
      - spark-network

  spark-worker-1:
    build: .
    container_name: spark-worker-1
    hostname: spark-worker-1
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
    environment:
      - SPARK_WORKER_CORES=2
      - SPARK_WORKER_MEMORY=2g
    depends_on:
      - spark-master
    volumes:
      - ./data:/data
      - ./apps:/apps
    networks:
      - spark-network

  spark-worker-2:
    build: .
    container_name: spark-worker-2
    hostname: spark-worker-2
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
    environment:
      - SPARK_WORKER_CORES=2
      - SPARK_WORKER_MEMORY=2g
    depends_on:
      - spark-master
    volumes:
      - ./data:/data
      - ./apps:/apps
    networks:
      - spark-network

  spark-worker-3:
    build: .
    container_name: spark-worker-3
    hostname: spark-worker-3
    command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
    environment:
      - SPARK_WORKER_CORES=2
      - SPARK_WORKER_MEMORY=2g
    depends_on:
      - spark-master
    volumes:
      - ./data:/data
      - ./apps:/apps
    networks:
      - spark-network

  jupyter:
    image: jupyter/pyspark-notebook:latest
    container_name: spark-jupyter
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - SPARK_MASTER=spark://spark-master:7077
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    depends_on:
      - spark-master
    networks:
      - spark-network

networks:
  spark-network:
    driver: bridge


```
## Usage

# Notebooks

**NOTE:** For additional information or a global summary, please refer to the report **rapport.md**.

## 01_sklearn.ipynb

Once completed, you can run all the cells in this notebook. This will generate several files in the `data` folder:

- **model_results_sklearn_best_model.json** – contains the results of the best model in JSON format.  
- **model_results_sklearn_comparaison.json** – contains comparison metrics of all evaluated models in JSON format.  
- **feature_importance.csv** – lists the top 10 most important features from the best model.  
- **predictions_test.csv** – contains the test predictions of the best model to determine whether customers are likely to **churn**.

In this notebook, multiple models are used (Logistic Regression, Random Forest, and Gradient Boosting) to identify the best-performing model.

## 02_spark.ipynb

### **IMPORTANT NOTE:**

 For this notebook, you will need to reinstall the required packages globally, not in the `.venv`, because the Python interpreter requires **JAVA**. If you prefer not to reinstall the packages or encounter issues, you can proceed directly to the next step: **02_spark.py (Optional)**.

First, deactivate your virtual environment by running:
```bash
deactivate
```
then reinstall all required packages:
```bash
pip install -r requirements.txt     
```

## 02_spark.py (Optional)

**NOTE**: This step is only necessary if you want to run the backup script **02_spark.py** with Docker, for example if you don’t want to reinstall the dependencies globally and want to use the .venv unstall previously.

From the root of your project, open a terminal and run the following command to start the services defined in Docker Compose.  


```bash
docker compose up -d
```

Once the application is running, open a terminal and execute the following command to launch the script:

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /apps/02_spark.py
```

The generated information is identical to the information in **02_spark.ipynb**.

Once done, you can run all the cells in the notebook. This will generate two files in the `data` folder:

- **model_results_pyspark_best_model.json** – contains the results of the best PySpark model in JSON format.  
- **model_results_pyspark_comparaison.json** – contains comparison metrics of all evaluated PySpark models in JSON format.

## bonus.ipynb (Comparaison des modèles entre scikit-learn et )

This notebook is mainly used to compare the different models and their statistics (**Accuracy**, **Precision**, **Recall**, **F1-score**, **Time**) across all models.  
It also allows comparison of the best model selected by **scikit-learn** and **PySpark**.

---
## Uninstall

To uninstall the project and remove all associated volumes, run the following command:

```bash
docker compose down -v
```
---
 
## Authors
 
- [GitHub Profile](https://github.com/titibemo)
 
---
 
## License
 
This project is open-source and can be freely copied, modified, and distributed by anyone. No specific license is provided, but contributions and usage are welcome.

