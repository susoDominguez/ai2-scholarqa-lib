#!/usr/bin/env python3
"""
Test script to send queries to the Scholar QA API and measure performance
"""

import json
import time
import requests
from datetime import datetime


def test_query(query_text, user_id="test-performance"):
    """Send a query to the API and return the task ID"""
    url = "http://localhost:8000/api/query_corpusqa"

    payload = {"query": query_text, "user_id": user_id, "opt_in": True}

    try:
        print(f" Sending query: {query_text}")
        start_time = time.time()

        response = requests.post(url, json=payload, timeout=10)

        elapsed = time.time() - start_time
        print(f" Request completed in {elapsed:.2f}s")

        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f" Query accepted, task_id: {task_id}")
            return task_id
        else:
            print(f" Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f" Exception: {e}")
        return None


def main():
    """Test different types of queries"""

    test_queries = [
        "What are the latest advances in transformer architectures?",
        "How do neural networks work for image recognition?",
        "What is the current state of reinforcement learning research?",
        "Recent developments in natural language processing",
        "Computer vision applications in autonomous vehicles",
    ]

    print(" Starting Scholar QA Performance Tests")
    print(f" Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    results = {}

    for i, query in enumerate(test_queries, 1):
        print(f"\n Test {i}/{len(test_queries)}")
        print("-" * 40)

        task_id = test_query(query, f"test-user-{i}")
        if task_id:
            results[f"query_{i}"] = {
                "query": query,
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
            }

        # Small delay between queries
        time.sleep(2)

    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)

    for key, result in results.items():
        print(f"  {key}: {result['task_id']} - {result['query'][:50]}...")

    print(f"\n Total queries sent: {len(results)}")
    print(" Check the async_state logs to monitor query progress")

    # Save results
    with open("/api/test_query_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(" Results saved to test_query_results.json")


if __name__ == "__main__":
    main()
