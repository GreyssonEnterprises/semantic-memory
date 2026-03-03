#!/usr/bin/env python3
"""
Semantic Memory Retrieval Eval — Automated test harness.

Runs the eval queries from retrieval-rubric.md against the retrieval CLI
and scores results. Designed to run once Bob's pai-memory-recall CLI is ready.

Usage:
    python eval_runner.py --agent dean --cli-path <path-to-cli>
    python eval_runner.py --agent dean --dry-run  # print queries only
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class EvalQuery:
    """Single evaluation query with expected behavior."""
    category: str       # specific, pattern, blended, negative, time, isolation
    query: str
    mode: str           # specific, pattern, blended
    expected: str       # what we expect to see
    pass_criteria: str  # how we judge pass/fail
    score: float = 0.0  # filled in after eval
    results: list = field(default_factory=list)
    notes: str = ""


# ─── TEST QUERIES ───────────────────────────────────────────────────────

EVAL_QUERIES = [
    # Test 1: Specific Recall
    EvalQuery(
        category="specific",
        query="vinyl supplier pricing for Cricut sheets",
        mode="specific",
        expected="Segment where specific supplier prices were discussed",
        pass_criteria="Top-3 results contain vinyl pricing segment",
    ),
    EvalQuery(
        category="specific",
        query="class list how many students",
        mode="specific",
        expected="Segment where class size was confirmed",
        pass_criteria="Top-3 results contain class list segment",
    ),
    EvalQuery(
        category="specific",
        query="product label design for the shop",
        mode="specific",
        expected="Segment about shop label/branding work",
        pass_criteria="Top-3 results contain label design segment",
    ),

    # Test 2: Pattern Matching
    EvalQuery(
        category="pattern",
        query="overwhelmed logistics spiral juggling too many things",
        mode="pattern",
        expected="Crisis/overwhelm sessions",
        pass_criteria="Top-5 majority are overwhelm/crisis sessions",
    ),
    EvalQuery(
        category="pattern",
        query="excited about discovering a great deal",
        mode="pattern",
        expected="High-excitement deal/book/breakthrough sessions",
        pass_criteria="Top-5 show excited emotional tenor",
    ),
    EvalQuery(
        category="pattern",
        query="calm focused single topic planning",
        mode="pattern",
        expected="Structured planning sessions",
        pass_criteria="Top-5 are structured, not scattered",
    ),

    # Test 3: Blended Retrieval
    EvalQuery(
        category="blended",
        query="Valentine's Day gift bag preparations",
        mode="blended",
        expected="Planning segments AND past Valentine's sessions",
        pass_criteria="Mix of segment-level and summary-level results",
    ),
    EvalQuery(
        category="blended",
        query="what supplies do we need to restock",
        mode="blended",
        expected="Supply discussions across domains",
        pass_criteria="Cross-domain results (crafting + business + household)",
    ),

    # Test 4: Negative Results
    EvalQuery(
        category="negative",
        query="quantum physics research papers",
        mode="specific",
        expected="Nothing relevant",
        pass_criteria="Top results have similarity < 0.5",
    ),
    EvalQuery(
        category="negative",
        query="the user's favorite programming language",
        mode="specific",
        expected="Nothing relevant (she's not technical)",
        pass_criteria="Top results have similarity < 0.5",
    ),
    EvalQuery(
        category="negative",
        query="stock market investment portfolio",
        mode="specific",
        expected="Nothing relevant",
        pass_criteria="Top results have similarity < 0.5",
    ),

    # Test 6: Cross-Agent Isolation
    EvalQuery(
        category="isolation",
        query="what CVEs did Shale work on in PSIRT",
        mode="specific",
        expected="Zero results from Dean's store (this is Patterson's domain)",
        pass_criteria="No PSIRT/CVE segments returned — privacy absolute",
    ),
    EvalQuery(
        category="isolation",
        query="TAM customer escalation procedure",
        mode="specific",
        expected="Zero results from Dean's store (this is Bob's domain)",
        pass_criteria="No TAM/customer segments returned — privacy absolute",
    ),
    EvalQuery(
        category="isolation",
        query="NAS infrastructure server configuration",
        mode="specific",
        expected="May return cross-agent coordination segments (shared topic) but NOT Bob's private infra sessions",
        pass_criteria="Only segments from Dean's own sessions about NAS/infra",
    ),
]


def run_query(cli_path: str, agent: str, mode: str, query: str, top_k: int = 5) -> list[dict]:
    """Run a query against the retrieval CLI and parse results."""
    cmd = [
        cli_path,
        "--agent", agent,
        "--mode", mode,
        "--top", str(top_k),
        "--json",
        query,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return [{"error": result.stderr.strip()}]
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return [{"error": str(e)}]


def run_eval(agent: str, cli_path: str, dry_run: bool = False) -> dict:
    """Run all eval queries and collect results."""
    results = {
        "agent": agent,
        "timestamp": datetime.now().isoformat(),
        "cli_path": cli_path,
        "queries": [],
        "summary": {},
    }

    for eq in EVAL_QUERIES:
        entry = {
            "category": eq.category,
            "query": eq.query,
            "mode": eq.mode,
            "expected": eq.expected,
            "pass_criteria": eq.pass_criteria,
        }

        if dry_run:
            entry["status"] = "dry_run"
            print(f"  [{eq.category}] ({eq.mode}) \"{eq.query}\"")
            print(f"    Expected: {eq.expected}")
            print(f"    Criteria: {eq.pass_criteria}")
            print()
        else:
            print(f"  Running: [{eq.category}] \"{eq.query[:50]}...\"", end=" ")
            query_results = run_query(cli_path, agent, eq.mode, eq.query)
            entry["results"] = query_results
            entry["result_count"] = len(query_results)

            # Auto-score what we can
            if eq.category == "negative":
                # Check if top results have low similarity (our JSON uses "score" not "similarity")
                scores = [r.get("score", r.get("similarity", 1.0)) for r in query_results if "error" not in r]
                if scores and max(scores) < 0.3:
                    entry["auto_score"] = 5.0
                    print("✓ (low scores)")
                elif scores and max(scores) < 0.45:
                    entry["auto_score"] = 3.0
                    print("~ (moderate scores)")
                else:
                    entry["auto_score"] = 1.0
                    print(f"✗ (high score on irrelevant query: {max(scores) if scores else 'N/A'})")

            elif eq.category == "isolation":
                # For isolation: ANY result that's not from the target agent is a failure
                agent_violations = [r for r in query_results if r.get("agent") and r["agent"] != agent]
                if not agent_violations:
                    entry["auto_score"] = 5.0
                    print("✓ (isolated)")
                else:
                    entry["auto_score"] = 1.0
                    print(f"✗ ISOLATION FAILURE — {len(agent_violations)} cross-agent results!")

            else:
                entry["auto_score"] = None  # needs manual review
                print(f"→ {len(query_results)} results (manual review needed)")

        results["queries"].append(entry)

    # Compute category averages
    categories = {}
    for q in results["queries"]:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = {"scores": [], "count": 0}
        categories[cat]["count"] += 1
        if q.get("auto_score") is not None:
            categories[cat]["scores"].append(q["auto_score"])

    results["summary"] = {
        cat: {
            "query_count": data["count"],
            "auto_scored": len(data["scores"]),
            "avg_auto_score": round(sum(data["scores"]) / len(data["scores"]), 2) if data["scores"] else None,
        }
        for cat, data in categories.items()
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic Memory Retrieval Eval")
    parser.add_argument("--agent", default="dean", help="Agent to eval")
    parser.add_argument("--cli-path", default="pai-memory-recall", help="Path to retrieval CLI")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without running")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SEMANTIC MEMORY — Retrieval Eval")
    print(f"{'='*60}")
    print(f"Agent: {args.agent}")
    print(f"CLI: {args.cli_path}")
    print(f"Queries: {len(EVAL_QUERIES)}")
    print(f"{'='*60}\n")

    results = run_eval(args.agent, args.cli_path, dry_run=args.dry_run)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cat, data in results.get("summary", {}).items():
        avg = data.get("avg_auto_score")
        avg_str = f"{avg:.1f}" if avg is not None else "manual"
        print(f"  {cat:20s}  {data['query_count']} queries, avg: {avg_str}")
    print(f"{'='*60}")

    # Save results
    output_path = args.output or f"eval/results/eval-{args.agent}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
