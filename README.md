# MongoDB vs FerretDB Comparison Tool

This project provides a comprehensive comparison between MongoDB and FerretDB, focusing on performance metrics.

## About FerretDB

FerretDB is an open-source MongoDB alternative that uses PostgreSQL as its storage engine. It implements the MongoDB wire protocol, aiming to be a drop-in replacement for MongoDB with an open-source license.

### How FerretDB Works

FerretDB acts as a proxy between MongoDB clients and PostgreSQL:

1. **Client Connection**: Applications connect to FerretDB using the MongoDB protocol (via MongoDB drivers like pymongo)
2. **Protocol Translation**: FerretDB translates MongoDB wire protocol commands into PostgreSQL queries
3. **Storage**: Data is stored in PostgreSQL database tables

This architecture allows you to use MongoDB client libraries while storing your data in PostgreSQL. In this project, we connect to FerretDB using the pymongo client with a MongoDB URI format, even though FerretDB is using PostgreSQL internally.

## Features

- Performance comparison for CRUD operations (Create, Read, Update, Delete)
- Query performance testing
- Aggregation framework performance testing
- Visualization of performance results

## Requirements

- Python 3.11 or higher
- Docker compose

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   uv sync
   ```
3. Start MongoDB and FerretDB:
     - Start the services with `docker compose up -d`

4. Import Intel Geti sample data into both MongoDB and FerretDB:
   - `mongorestore --uri="mongodb://albert:vanhouten@localhost:27018" --archive < mongo.dump`
   - `mongorestore --uri="mongodb://albert:vanhouten@localhost:27017" --archive < mongo.dump`

## Usage

Run the comparison tool:

```
python mongodb_vs_ferretdb.py
```

The script will:
1. Connect to both MongoDB and FerretDB
2. Run performance tests for various operations
3. Display results in the console
4. Generate a bar chart visualization saved as `mongodb_vs_ferretdb.png`

## Customization

You can adjust the test parameters in `mongodb_vs_ferretdb.py`:

- `NUM_DOCUMENTS`: Number of documents to use in tests
- `NUM_QUERIES`: Number of queries to perform in each test
- `NUM_RUNS`: Number of test runs for averaging results
