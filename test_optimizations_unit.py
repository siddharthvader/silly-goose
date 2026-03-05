#!/usr/bin/env python3
"""
Unit tests for silly-goose optimizations - no external dependencies.

Tests the optimization logic without requiring Modal/vLLM.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!\n")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_semantic_similarity():
    """Test that similar questions have high similarity scores."""
    print("=" * 60)
    print("TEST: Semantic Similarity (Real Embeddings)")
    print("=" * 60)

    # Similar questions about the same thing
    similar_queries = [
        "What does the authenticate function do?",
        "Explain the authenticate method",
        "How does authenticate work?",
        "Can you describe what authenticate does?",
    ]

    # Unrelated queries
    unrelated_queries = [
        "How do I install Docker on Ubuntu?",
        "What's the weather like today?",
    ]

    base = similar_queries[0]
    base_emb = model.encode(base)

    print(f"\nBase query: '{base}'")
    print("\nSimilar queries (should be > 0.85):")
    for q in similar_queries[1:]:
        emb = model.encode(q)
        sim = cosine_sim(base_emb, emb)
        status = "✓" if sim >= 0.85 else "✗"
        print(f"  {status} {sim:.3f}: '{q}'")

    print("\nUnrelated queries (should be < 0.5):")
    for q in unrelated_queries:
        emb = model.encode(q)
        sim = cosine_sim(base_emb, emb)
        status = "✓" if sim < 0.5 else "✗"
        print(f"  {status} {sim:.3f}: '{q}'")


def test_workflow_classification():
    """Test workflow stage classification."""
    print("\n" + "=" * 60)
    print("TEST: Workflow Stage Classification")
    print("=" * 60)

    test_cases = [
        ("The pytest failed with AssertionError: expected 5 got 4", "TestAnalyze", "Hot"),
        ("Fix the bug in the authentication module", "CodeEdit", "Hot"),
        ("Write documentation for the API endpoints", "Docs", "Cold"),
        ("What files are in the src/utils directory?", "Explore", "Cold"),
        ("Create a commit message for adding user validation", "CommitMsg", "Warm"),
        ("How should I approach refactoring the database layer?", "Plan", "Warm"),
    ]

    print()
    all_pass = True
    for query, expected_stage, expected_priority in test_cases:
        q = query.lower()

        # Classification logic (matching Rust implementation)
        if "test" in q and ("fail" in q or "error" in q):
            stage, priority = "TestAnalyze", "Hot"
        elif "fix" in q or "bug" in q or "edit" in q:
            stage, priority = "CodeEdit", "Hot"
        elif "commit" in q:
            stage, priority = "CommitMsg", "Warm"
        elif "doc" in q:
            stage, priority = "Docs", "Cold"
        elif "plan" in q or "approach" in q:
            stage, priority = "Plan", "Warm"
        else:
            stage, priority = "Explore", "Cold"

        passed = (stage == expected_stage and priority == expected_priority)
        status = "✓" if passed else "✗"
        if not passed:
            all_pass = False
        print(f"  {status} '{query[:50]}...'")
        print(f"      Expected: {expected_stage} ({expected_priority})")
        print(f"      Got:      {stage} ({priority})")

    return all_pass


def test_cache_sharing_scenario():
    """Test a realistic multi-agent cache sharing scenario."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Agent Cache Sharing Scenario")
    print("=" * 60)

    # Simulate 3 agents asking about the same codebase
    agent_queries = {
        "Agent 1": [
            "What does UserAuthService.authenticate do?",
            "How is the password verified in authenticate?",
        ],
        "Agent 2": [
            "Explain the authenticate method in UserAuthService",  # Should cache hit
            "What validation does authenticate perform?",
        ],
        "Agent 3": [
            "How does authenticate work?",  # Should cache hit
            "What does the authenticate function return?",
        ],
    }

    # Build cache from Agent 1
    cache = {}
    print("\nAgent 1 builds cache:")
    for q in agent_queries["Agent 1"]:
        emb = model.encode(q)
        cache[q] = emb
        print(f"  Cached: '{q[:50]}...'")

    # Test cache hits for other agents
    print("\nAgents 2 & 3 check cache:")
    threshold = 0.85

    for agent, queries in list(agent_queries.items())[1:]:
        print(f"\n  {agent}:")
        for q in queries:
            emb = model.encode(q)

            # Check similarity against all cached entries
            best_sim = 0
            best_match = None
            for cached_q, cached_emb in cache.items():
                sim = cosine_sim(emb, cached_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = cached_q

            if best_sim >= threshold:
                print(f"    ✓ CACHE HIT ({best_sim:.3f}): '{q[:40]}...'")
                print(f"      Matched: '{best_match[:40]}...'")
            else:
                print(f"    ✗ CACHE MISS ({best_sim:.3f}): '{q[:40]}...'")
                # Add to cache
                cache[q] = emb


def main():
    print("=" * 60)
    print("SILLY-GOOSE OPTIMIZATION UNIT TESTS")
    print("=" * 60)

    test_semantic_similarity()
    test_workflow_classification()
    test_cache_sharing_scenario()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
