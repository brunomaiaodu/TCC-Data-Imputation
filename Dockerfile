# Multi-stage build for backend (FastAPI) and frontend (Vite)

#######################
# Backend build stage #
#######################
FROM python:3.11-slim AS backend-base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libfreetype6-dev libpng-dev libjpeg-dev libopenblas-dev \
    libproj-dev proj-data proj-bin libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY preprocessing/processed /app/preprocessing/processed
COPY backend/app/data /app/backend/app/data

########################
# Backend runtime stage#
########################
FROM python:3.11-slim AS backend-run
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# System dependencies needed at runtime for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 libpng16-16 libjpeg62-turbo libopenblas0 \
    libproj-dev proj-data proj-bin libgeos-dev \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

COPY --from=backend-base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=backend-base /usr/local/bin /usr/local/bin
COPY --from=backend-base /app/backend /app/backend
COPY --from=backend-base /app/preprocessing /app/preprocessing

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

########################
# Frontend build stage #
########################
FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend ./
RUN npm run build

#########################
# Frontend runtime stage#
#########################
FROM node:20-alpine AS frontend-run
WORKDIR /app
RUN npm install -g serve
COPY --from=frontend-build /app/dist ./dist

EXPOSE 4173
CMD ["serve", "-s", "dist", "-l", "4173"]
