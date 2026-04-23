import json
import os
from analyzer import analyze_job, analyze_multiple


def print_report(result, rank=None):
    header = f"JOB #{rank} " if rank else ""
    print(f"\n{'='*55}")
    print(f"  {header}JOB ANALYSIS REPORT")
    print(f"{'='*55}")
    print(f"  Title     : {result.job_title}")
    print(f"  Company   : {result.company.name} ({result.company.size or 'unknown size'})")
    print(f"  Industry  : {result.company.industry}")
    print(f"  Seniority : {result.seniority.value.upper()}")
    print(f"  Remote    : {'Yes' if result.remote_ok else 'No'}")

    if result.salary_min and result.salary_max:
        print(f"  Salary    : ${result.salary_min:,} - ${result.salary_max:,}")
    elif result.salary_min:
        print(f"  Salary    : ${result.salary_min:,}+")
    else:
        print(f"  Salary    : Not mentioned")

    print(f"\n  Required Skills:")
    for skill in result.required_skills:
        print(f"    • {skill}")

    if result.nice_to_have:
        print(f"\n  Nice to Have:")
        for skill in result.nice_to_have:
            print(f"    ◦ {skill}")

    if result.gap_skills:
        print(f"\n  Your Skill Gaps:")
        for skill in result.gap_skills:
            print(f"    ✗ {skill}")

    score_bar = "█" * (result.match_score // 10) + "░" * (10 - result.match_score // 10)
    print(f"\n  Match Score : {result.match_score}/100  [{score_bar}]")
    print(f"  Why        : {result.match_reason}")
    print(f"\n  Summary:")
    print(f"  {result.summary}")
    print(f"{'='*55}")


def save_results(results, filename="results.json"):
    data = [r.model_dump() for r in results]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {filename}")


def load_sample_jobs():
    jobs = []
    sample_dir = "sample_jobs"
    if os.path.exists(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.endswith(".txt"):
                with open(os.path.join(sample_dir, fname)) as f:
                    jobs.append(f.read())
    return jobs


def main():
    print("\nAI Job Analyzer — Structured Outputs Demo")
    print("==========================================")
    print("1. Analyze a single job (paste text)")
    print("2. Analyze all sample jobs (ranked by match)")
    print("3. Exit")

    choice = input("\nChoose (1/2/3): ").strip()

    if choice == "1":
        print("\nPaste job description (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        job_text = "\n".join(lines)
        print("\nAnalyzing...")
        result = analyze_job(job_text)
        print_report(result)
        save_results([result], "single_result.json")

    elif choice == "2":
        jobs = load_sample_jobs()
        if not jobs:
            print("No sample jobs found in sample_jobs/ folder")
            return
        print(f"\nFound {len(jobs)} jobs. Analyzing and ranking...")
        results = analyze_multiple(jobs)
        for i, result in enumerate(results, 1):
            print_report(result, rank=i)
        save_results(results, "all_results.json")
        print(f"\nBest Match: {results[0].job_title} at {results[0].company.name} ({results[0].match_score}/100)")

    elif choice == "3":
        print("Goodbye!")


if __name__ == "__main__":
    main()
