FROM node:18-alpine

WORKDIR /app

# Copy package files first
COPY package*.json ./

# Install dependencies using npm ci (faster than install)
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start in development mode
CMD ["npm", "run", "dev"]
