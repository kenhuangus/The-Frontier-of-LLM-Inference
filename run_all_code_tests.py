import os
import subprocess
import sys
import time

def main():
    code_root = os.path.dirname(os.path.abspath(__file__))
    print("=" * 80)
    print("  RUNNING ALL PRODUCTION CODE TESTS: THE FRONTIER OF LLM INFERENCE")
    print(f"  Root: {code_root}")
    print("=" * 80)
    
    total = 0
    passed = 0
    failed = []
    
    start_time = time.time()
    
    for dirpath, _, filenames in sorted(os.walk(code_root)):
        for f in sorted(filenames):
            if f.endswith(".py") and f != "run_all_code_tests.py":
                total += 1
                script_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(script_path, code_root)
                
                res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, encoding="utf-8")
                
                if res.returncode == 0:
                    passed += 1
                    print(f"  [PASS] {rel_path}")
                else:
                    failed.append((rel_path, res.stderr))
                    print(f"  [FAIL] {rel_path}")
                    print(f"         Error: {res.stderr[:300]}")
                    
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"  TEST SUITE RESULTS: {passed} / {total} PASSED (in {elapsed:.2f}s)")
    if failed:
        print(f"  FAILED SCRIPTS ({len(failed)}):")
        for p, err in failed:
            print(f"    - {p}: {err.strip()[:100]}")
        return 1
    else:
        print("  ALL CHAPTER SCRIPTS EXECUTED SUCCESSFULLY WITH ZERO ERRORS!")
        print("=" * 80)
        return 0

if __name__ == "__main__":
    sys.exit(main())
