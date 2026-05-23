FROM node:20-alpine

WORKDIR /app

# Copy package files first
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the application
COPY . .

# Start the application in development mode
CMD ["npm", "run", "dev"]
