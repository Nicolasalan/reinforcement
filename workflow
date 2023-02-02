name: dev-env

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2
      - name: Build Docker
        run: make build
    
      - name: Run Simulations
        run: make setup
      
      - name: Run Tests
        run: make integration
      
