FROM node:18-alpine

WORKDIR /app

# Copy the package.json file first to optimize caching
COPY ./package.json .

# Install dependencies
RUN npm install

# Copy the rest of the frontend application
COPY ./ . 

CMD ["npm", "start"]
