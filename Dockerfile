# use an official lightweight python image
FROM python:3.12-slim

# set the working directory in the container
WORKDIR /app

# install system dependencies required for matplotlib and numerical operations
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy the requirements file and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application code
COPY . .

# expose the port the app runs on
EXPOSE 8000

# command to execute the fast api server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]