"""
MongoDB vs FerretDB Comparison

This script compares MongoDB and FerretDB performance and compatibility
across various operations including CRUD operations, queries, and aggregations.

FerretDB is an open-source MongoDB alternative that uses PostgreSQL as its storage engine.
It implements the MongoDB wire protocol, aiming to be a drop-in replacement.
"""

import os
import random
import statistics
import string
import time
import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo

# Connection parameters
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://albert:vanhouten@localhost:27017")
# FerretDB implements the MongoDB wire protocol, so clients connect to it using the MongoDB protocol.
# However, FerretDB uses PostgreSQL as its storage engine internally.
# This is why we connect to FerretDB using the MongoDB URI format, even though it's backed by PostgreSQL.
FERRETDB_URI = os.getenv("FERRETDB_URI", "mongodb://albert:vanhouten@localhost:27018")

# Database and collection names
DB_NAME = "comparison_db"
COLLECTION_NAME = "test_collection"

# Test parameters
NUM_QUERIES = 100
NUM_RUNS = 10  # Number of test runs for averaging results


class DatabaseComparison:
    """Class to compare MongoDB and FerretDB performance."""

    def __init__(self):
        """Initialize connections to MongoDB and FerretDB."""
        # MongoDB connection
        self.mongo_client = pymongo.MongoClient(MONGODB_URI, uuidRepresentation="standard")
        self.mongo_db = self.mongo_client[DB_NAME]
        self.mongo_collection = self.mongo_db[COLLECTION_NAME]

        # FerretDB connection
        # We use pymongo to connect to FerretDB because FerretDB implements the MongoDB wire protocol
        # Even though FerretDB uses PostgreSQL internally as its storage engine, clients connect to it
        # using the MongoDB protocol, which is why we use pymongo here
        self.ferret_client = pymongo.MongoClient(FERRETDB_URI, uuidRepresentation="standard")
        self.ferret_db = self.ferret_client[DB_NAME]
        self.ferret_collection = self.ferret_db[COLLECTION_NAME]

        self.results = {
            "insert": {"mongodb": [], "ferretdb": []},
            "read": {"mongodb": [], "ferretdb": []},
            "update": {"mongodb": [], "ferretdb": []},
            "delete": {"mongodb": [], "ferretdb": []},
            "query": {"mongodb": [], "ferretdb": []},
            "aggregation": {"mongodb": [], "ferretdb": []},
            "big_aggregation": {"mongodb": [], "ferretdb": []},
        }

    def cleanup(self):
        """Drop collections and close connections.

        Note: dataset_storage_filter_data collection is not dropped as per requirements.
        """
        self.mongo_db.drop_collection(COLLECTION_NAME)
        self.ferret_db.drop_collection(COLLECTION_NAME)
        self.mongo_client.close()
        self.ferret_client.close()

    def test_big_aggregation(self):
        """Test big aggregation performance for both databases on some sample dataset filter data from Intel Geti"""
        print("Testing complicated aggregation performance...")

        query = [{'$match': {'$and': [{'creation_date': {
            '$gt': datetime.datetime(2025, 5, 4, tzinfo=datetime.timezone(datetime.timedelta(seconds=7200))),
            '$exists': True}}, {'media_identifier.type': {'$eq': 'image', '$exists': True}},
                                      {'media_height': {'$gt': 25, '$exists': True}},
                                      {'annotations.shape.type': {'$nin': ['ELLIPSE'], '$exists': True}},
                                      {'annotations.shape.area_pixel': {'$gt': 4, '$exists': True}}, {'upload_date': {
                '$lt': datetime.datetime(2025, 5, 29, tzinfo=datetime.timezone(datetime.timedelta(seconds=7200))),
                '$exists': True}}]}}, {'$facet': {'paginated_results': [{'$group': {'_id': '$media_identifier.media_id',
                                                                                    'media_identifier': {
                                                                                        '$first': '$media_identifier'},
                                                                                    'media_name': {
                                                                                        '$first': '$media_name'},
                                                                                    'media_extension': {
                                                                                        '$first': '$media_extension'},
                                                                                    'upload_date': {
                                                                                        '$first': '$upload_date'},
                                                                                    'uploader_id': {
                                                                                        '$first': '$uploader_id'},
                                                                                    'media_height': {
                                                                                        '$first': '$media_height'},
                                                                                    'media_width': {
                                                                                        '$first': '$media_width'},
                                                                                    'annotation_scene_id': {
                                                                                        '$first': '$annotation_scene_id'},
                                                                                    'user_name': {
                                                                                        '$first': '$user_name'},
                                                                                    'media_annotation_state': {
                                                                                        '$first': '$media_annotation_state'},
                                                                                    'matched_frames': {'$sum': {
                                                                                        '$ifNull': [
                                                                                            '$unannotated_frames', 1]}},
                                                                                    'size': {'$first': '$size'},
                                                                                    'frame_rate': {
                                                                                        '$first': '$frame_rate'},
                                                                                    'frame_stride': {
                                                                                        '$first': '$frame_stride'},
                                                                                    'frame_count': {
                                                                                        '$first': '$frame_count'},
                                                                                    'preprocessing': {
                                                                                        '$first': '$preprocessing'}}},
                                                                        {'$sort': {'upload_date': -1, '_id': -1}},
                                                                        {'$skip': 0}, {'$limit': 100}],
                                                  'image_count': [{'$match': {'media_identifier.type': 'image'}},
                                                                  {'$count': 'count'}], 'video_count': [
                {'$match': {'media_identifier.type': {'$in': ['video', 'video_frame']}}},
                {'$group': {'_id': '$media_identifier.media_id'}}, {'$count': 'count'}], 'video_frame_count': [
                {'$match': {'media_identifier.type': {'$in': ['video', 'video_frame']}}},
                {'$group': {'_id': 1, 'count': {'$sum': {'$ifNull': ['$unannotated_frames', 1]}}}}]}}]

        ds_filter_collection_mongo = self.mongo_db["dataset_storage_filter_data"]
        ds_filter_collection_ferret = self.ferret_db["dataset_storage_filter_data"]

        for run in range(NUM_RUNS):
            # Test MongoDB complicated aggregation
            mongo_total_time = 0
            for i in range(NUM_QUERIES):
                start_time = time.time()
                ds_filter_collection_mongo.aggregate(query, allowDiskUse=True)
                mongo_total_time += time.time() - start_time

            # Store average time per query
            self.results["big_aggregation"]["mongodb"].append(mongo_total_time / NUM_QUERIES)

            # Test FerretDB complicated aggregation
            ferret_total_time = 0
            for i in range(NUM_QUERIES):
                start_time = time.time()
                ds_filter_collection_ferret.aggregate(query, allowDiskUse=True)
                ferret_total_time += time.time() - start_time

            # Store average time per query
            self.results["big_aggregation"]["ferretdb"].append(ferret_total_time / NUM_QUERIES)

            mongo_avg = mongo_total_time / NUM_QUERIES
            ferret_avg = ferret_total_time / NUM_QUERIES
            print(
                f"Run {run + 1}: MongoDB: {mongo_avg * 1000:.2f}ms per query, FerretDB: {ferret_avg * 1000:.2f}ms per query")

    def generate_random_document(self) -> Dict[str, Any]:
        """Generate a random document for testing."""
        return {
            "user_id": ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)),
            "name": ''.join(random.choices(string.ascii_lowercase, k=10)),
            "email": f"{''.join(random.choices(string.ascii_lowercase, k=8))}@example.com",
            "age": random.randint(18, 80),
            "score": random.randint(0, 100),
            "is_active": random.choice([True, False]),
            "tags": random.sample(
                ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"],
                random.randint(1, 5)
            ),
            "created_at": datetime.datetime.now(),
            "data": {
                "field1": random.randint(1, 1000),
                "field2": ''.join(random.choices(string.ascii_lowercase, k=20)),
                "field3": [random.random() for _ in range(5)]
            }
        }

    def test_insert_performance(self):
        """Test insert performance for both databases."""
        print("Testing insert performance...")

        for run in range(NUM_RUNS):
            # Test MongoDB insert
            mongo_total_time = 0
            for _ in range(NUM_QUERIES):
                document = self.generate_random_document()
                start_time = time.time()
                self.mongo_collection.insert_one(document)
                mongo_total_time += time.time() - start_time

            # Store average time per query
            self.results["insert"]["mongodb"].append(mongo_total_time / NUM_QUERIES)

            # Test FerretDB insert
            ferret_total_time = 0
            for _ in range(NUM_QUERIES):
                document = self.generate_random_document()
                start_time = time.time()
                self.ferret_collection.insert_one(document)
                ferret_total_time += time.time() - start_time

            # Store average time per query
            self.results["insert"]["ferretdb"].append(ferret_total_time / NUM_QUERIES)

            mongo_avg = mongo_total_time / NUM_QUERIES
            ferret_avg = ferret_total_time / NUM_QUERIES
            print(
                f"Run {run + 1}: MongoDB: {mongo_avg * 1000:.2f}ms per query, FerretDB: {ferret_avg * 1000:.2f}ms per query")

            # Clean up for next run
            if run < NUM_RUNS - 1:  # Don't clean up after the last run
                self.mongo_collection.delete_many({})
                self.ferret_collection.delete_many({})

    def test_read_performance(self):
        """Test read performance for both databases."""
        print("Testing read performance...")

        for run in range(NUM_RUNS):
            # Get all document IDs
            mongo_ids = [doc["_id"] for doc in self.mongo_collection.find({}, {"_id": 1})]
            ferret_ids = [doc["_id"] for doc in self.ferret_collection.find({}, {"_id": 1})]

            # Select random IDs for testing
            mongo_sample = random.sample(mongo_ids, min(NUM_QUERIES, len(mongo_ids)))
            ferret_sample = random.sample(ferret_ids, min(NUM_QUERIES, len(ferret_ids)))

            # Test MongoDB read
            start_time = time.time()
            for doc_id in mongo_sample:
                self.mongo_collection.find_one({"_id": doc_id})
            mongo_time = time.time() - start_time
            # Store average time per query
            self.results["read"]["mongodb"].append(mongo_time / len(mongo_sample))

            # Test FerretDB read
            start_time = time.time()
            for doc_id in ferret_sample:
                self.ferret_collection.find_one({"_id": doc_id})
            ferret_time = time.time() - start_time
            # Store average time per query
            self.results["read"]["ferretdb"].append(ferret_time / len(ferret_sample))

            print(
                f"Run {run + 1}: MongoDB: {(mongo_time / len(mongo_sample)) * 1000:.2f}ms per query, FerretDB: {(ferret_time / len(ferret_sample)) * 1000:.2f}ms per query")

    def test_update_performance(self):
        """Test update performance for both databases."""
        print("Testing update performance...")

        for run in range(NUM_RUNS):
            # Get all document IDs
            mongo_ids = [doc["_id"] for doc in self.mongo_collection.find({}, {"_id": 1})]
            ferret_ids = [doc["_id"] for doc in self.ferret_collection.find({}, {"_id": 1})]

            # Select random IDs for testing
            mongo_sample = random.sample(mongo_ids, min(NUM_QUERIES, len(mongo_ids)))
            ferret_sample = random.sample(ferret_ids, min(NUM_QUERIES, len(ferret_ids)))

            # Test MongoDB update
            start_time = time.time()
            for doc_id in mongo_sample:
                self.mongo_collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"score": random.randint(0, 100), "updated_at": datetime.datetime.now()}}
                )
            mongo_time = time.time() - start_time
            # Store average time per query
            self.results["update"]["mongodb"].append(mongo_time / len(mongo_sample))

            # Test FerretDB update
            start_time = time.time()
            for doc_id in ferret_sample:
                self.ferret_collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"score": random.randint(0, 100), "updated_at": datetime.datetime.now()}}
                )
            ferret_time = time.time() - start_time
            # Store average time per query
            self.results["update"]["ferretdb"].append(ferret_time / len(ferret_sample))

            print(
                f"Run {run + 1}: MongoDB: {(mongo_time / len(mongo_sample)) * 1000:.2f}ms per query, FerretDB: {(ferret_time / len(ferret_sample)) * 1000:.2f}ms per query")

    def test_delete_performance(self):
        """Test delete performance for both databases."""
        print("Testing delete performance...")

        for run in range(NUM_RUNS):
            # First, add documents to delete
            mongo_docs = [self.generate_random_document() for _ in range(NUM_QUERIES)]
            ferret_docs = [self.generate_random_document() for _ in range(NUM_QUERIES)]

            self.mongo_collection.insert_many(mongo_docs)
            self.ferret_collection.insert_many(ferret_docs)

            # Get IDs of the newly inserted documents
            mongo_ids = [doc["_id"] for doc in mongo_docs]
            ferret_ids = [doc["_id"] for doc in ferret_docs]

            # Test MongoDB delete
            start_time = time.time()
            for doc_id in mongo_ids:
                self.mongo_collection.delete_one({"_id": doc_id})
            mongo_time = time.time() - start_time
            # Store average time per query
            self.results["delete"]["mongodb"].append(mongo_time / len(mongo_ids))

            # Test FerretDB delete
            start_time = time.time()
            for doc_id in ferret_ids:
                self.ferret_collection.delete_one({"_id": doc_id})
            ferret_time = time.time() - start_time
            # Store average time per query
            self.results["delete"]["ferretdb"].append(ferret_time / len(ferret_ids))

            print(
                f"Run {run + 1}: MongoDB: {(mongo_time / len(mongo_ids)) * 1000:.2f}ms per query, FerretDB: {(ferret_time / len(ferret_ids)) * 1000:.2f}ms per query")

    def test_query_performance(self):
        """Test query performance for both databases."""
        print("Testing query performance...")

        # Define query types
        queries = [
            # Simple equality
            {"age": {"$gt": 30}},
            # Range query
            {"score": {"$gte": 50, "$lte": 80}},
            # Compound query
            {"age": {"$gt": 25}, "is_active": True},
            # Array query
            {"tags": {"$in": ["tag1", "tag3"]}},
            # Text search (if supported)
            {"name": {"$regex": "^a"}},
        ]

        for run in range(NUM_RUNS):
            # Test MongoDB queries
            mongo_total_time = 0
            num_mongo_queries = 0
            for query in queries:
                start_time = time.time()
                list(self.mongo_collection.find(query))
                query_time = time.time() - start_time
                mongo_total_time += query_time
                num_mongo_queries += 1

            # Store average time per query
            self.results["query"]["mongodb"].append(
                mongo_total_time / num_mongo_queries if num_mongo_queries > 0 else 0)

            # Test FerretDB queries
            ferret_total_time = 0
            num_ferret_queries = 0
            for query in queries:
                start_time = time.time()
                try:
                    list(self.ferret_collection.find(query))
                    query_time = time.time() - start_time
                    ferret_total_time += query_time
                    num_ferret_queries += 1
                except Exception as e:
                    print(f"FerretDB query error: {e}")
                    # Skip this query

            # Store average time per query
            self.results["query"]["ferretdb"].append(
                ferret_total_time / num_ferret_queries if num_ferret_queries > 0 else 0)

            mongo_avg = mongo_total_time / num_mongo_queries if num_mongo_queries > 0 else 0
            ferret_avg = ferret_total_time / num_ferret_queries if num_ferret_queries > 0 else 0
            print(
                f"Run {run + 1}: MongoDB: {mongo_avg * 1000:.2f}ms per query, FerretDB: {ferret_avg * 1000:.2f}ms per query")

    def test_aggregation_performance(self):
        """Test aggregation performance for both databases."""
        print("Testing aggregation performance...")

        # Define aggregation pipelines
        pipelines = [
            # Simple group by
            [
                {"$group": {"_id": "$is_active", "count": {"$sum": 1}, "avg_score": {"$avg": "$score"}}}
            ],
            # More complex aggregation
            [
                {"$match": {"age": {"$gt": 30}}},
                {"$group": {"_id": "$is_active", "count": {"$sum": 1}, "avg_score": {"$avg": "$score"}}},
                {"$sort": {"count": -1}}
            ],
            # Project and group
            [
                {"$project": {"age_group": {"$floor": {"$divide": ["$age", 10]}}, "score": 1}},
                {"$group": {"_id": "$age_group", "avg_score": {"$avg": "$score"}, "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]
        ]

        for run in range(NUM_RUNS):
            # Test MongoDB aggregations
            mongo_total_time = 0
            num_mongo_aggs = 0
            for pipeline in pipelines:
                start_time = time.time()
                list(self.mongo_collection.aggregate(pipeline))
                agg_time = time.time() - start_time
                mongo_total_time += agg_time
                num_mongo_aggs += 1

            # Store average time per aggregation
            self.results["aggregation"]["mongodb"].append(
                mongo_total_time / num_mongo_aggs if num_mongo_aggs > 0 else 0)

            # Test FerretDB aggregations
            ferret_total_time = 0
            num_ferret_aggs = 0
            for pipeline in pipelines:
                start_time = time.time()
                try:
                    list(self.ferret_collection.aggregate(pipeline))
                    agg_time = time.time() - start_time
                    ferret_total_time += agg_time
                    num_ferret_aggs += 1
                except Exception as e:
                    print(f"FerretDB aggregation error: {e}")
                    # Skip this aggregation

            # Store average time per aggregation
            self.results["aggregation"]["ferretdb"].append(
                ferret_total_time / num_ferret_aggs if num_ferret_aggs > 0 else 0)

            mongo_avg = mongo_total_time / num_mongo_aggs if num_mongo_aggs > 0 else 0
            ferret_avg = ferret_total_time / num_ferret_aggs if num_ferret_aggs > 0 else 0
            print(
                f"Run {run + 1}: MongoDB: {mongo_avg * 1000:.2f}ms per query, FerretDB: {ferret_avg * 1000:.2f}ms per query")

    def calculate_average_results(self) -> Dict[str, Dict[str, float]]:
        """Calculate average results across all runs."""
        avg_results = {}

        for operation, db_results in self.results.items():
            avg_results[operation] = {
                "mongodb": statistics.mean(db_results["mongodb"]),
                "ferretdb": statistics.mean(db_results["ferretdb"]),
                "ratio": statistics.mean(db_results["ferretdb"]) / statistics.mean(db_results["mongodb"])
                if statistics.mean(db_results["mongodb"]) > 0 else float('inf')
            }

        return avg_results

    def plot_results(self, avg_results: Dict[str, Dict[str, float]]):
        """Plot the comparison results."""
        operations = list(avg_results.keys())
        # Convert times from seconds to milliseconds
        mongodb_times = [avg_results[op]["mongodb"] * 1000 for op in operations]
        ferretdb_times = [avg_results[op]["ferretdb"] * 1000 for op in operations]

        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Operation': operations,
            'MongoDB': mongodb_times,
            'FerretDB': ferretdb_times
        })

        # Create the plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)

        # Set the positions for the bars
        bar_width = 0.35
        x = np.arange(len(operations))

        # Create grouped bar chart with bars next to each other for each operation
        mongo_bars = ax.bar(x - bar_width / 2, mongodb_times, bar_width, color='blue', label='MongoDB')
        ferret_bars = ax.bar(x + bar_width / 2, ferretdb_times, bar_width, color='green', label='FerretDB')

        # Add labels and title
        ax.set_title('MongoDB vs FerretDB Performance Comparison (Time per Query)', fontsize=16)
        ax.set_ylabel('Time per Query (milliseconds)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(operations, rotation=45, ha='right')

        # Add a legend using the labels from the bar chart
        ax.legend()

        plt.tight_layout()
        plt.savefig('mongodb_vs_ferretdb.png')
        plt.close()

        print(f"Results plot saved as 'mongodb_vs_ferretdb.png'")

    def print_results(self, avg_results: Dict[str, Dict[str, float]]):
        """Print the comparison results in a table format."""
        print("\n" + "=" * 80)
        print("MongoDB vs FerretDB Performance Comparison (Time per Query)")
        print("=" * 80)
        print(
            f"{'Operation':<15} {'MongoDB (ms/query)':<15} {'FerretDB (ms/query)':<15} {'Ratio (FerretDB/MongoDB)':<25}")
        print("-" * 80)

        for operation, results in avg_results.items():
            print(
                f"{operation:<15} {results['mongodb'] * 1000:<15.2f} {results['ferretdb'] * 1000:<15.2f} {results['ratio']:<25.2f}")

        print("=" * 80)
        print("Note: Ratio > 1 means FerretDB is slower than MongoDB")
        print("=" * 80)

    def run_all_tests(self):
        """Run all performance tests."""
        try:
            print("\nStarting MongoDB vs FerretDB comparison tests...")
            print(f"MongoDB URI: {MONGODB_URI}")
            print(f"FerretDB URI: {FERRETDB_URI}")
            print(f"Number of queries per test: {NUM_QUERIES}")
            print(f"Number of test runs: {NUM_RUNS}")
            print(f"Note: Results will show average time per query/document")
            print("-" * 50)

            # Run tests
            self.test_insert_performance()
            self.test_read_performance()
            self.test_update_performance()
            self.test_delete_performance()
            self.test_query_performance()
            self.test_aggregation_performance()
            self.test_big_aggregation()

            # Calculate and display results
            avg_results = self.calculate_average_results()
            self.print_results(avg_results)
            self.plot_results(avg_results)

            print("\nTests completed successfully!")

        except Exception as e:
            print(f"Error during tests: {e}")
        finally:
            self.cleanup()


def main():
    """Main function to run the comparison."""
    print("MongoDB vs FerretDB Comparison Tool")
    print("-" * 40)

    # Check if MongoDB and FerretDB are accessible
    try:
        mongo_client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print(f"Please ensure MongoDB is running at {MONGODB_URI}")
        return

    try:
        # Connect to FerretDB using pymongo because FerretDB implements the MongoDB wire protocol
        # FerretDB internally uses PostgreSQL as its storage engine, but clients connect to it
        # using the MongoDB protocol
        ferret_client = pymongo.MongoClient(FERRETDB_URI, serverSelectionTimeoutMS=5000)
        ferret_client.server_info()
        print("✅ FerretDB connection successful")
    except Exception as e:
        print(f"❌ FerretDB connection failed: {e}")
        print(f"Please ensure FerretDB is running at {FERRETDB_URI}")
        print(f"Note: FerretDB uses PostgreSQL internally, but clients connect to it using the MongoDB protocol")
        return

    # Run performance comparison
    comparison = DatabaseComparison()
    comparison.run_all_tests()


if __name__ == "__main__":
    main()
