FROM python:3.12-slim
WORKDIR /app
COPY app.py model.joblib
RUN pip install flask scikit-learn joblib 
CMD ["python", "app.py"]