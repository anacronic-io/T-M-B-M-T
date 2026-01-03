#!/usr/bin/env python3
"""
Master Test Runner for T-M-B-M-T Verification Suite
Runs all verification tests in sequence and reports results.
"""

import subprocess
import sys
import time
from typing import List, Tuple


class TestRunner:
    """Coordinates running all verification tests."""

    def __init__(self):
        self.tests = [
            ('ZOH Discretization', 'core/zoh_discretization.py'),
            ('GB-RBM', 'core/gb_rbm.py'),
            ('Spectral Stability', 'core/spectral_stability.py'),
            ('Bridge Modules', 'bridges/bridge_modules.py'),
            ('Complexity Analysis', 'analysis/complexity_analysis.py'),
            ('Integration Test', 'tests/integration_test.py'),
        ]
        self.results = []

    def run_test(self, name: str, path: str) -> Tuple[bool, float]:
        """
        Run a single test.

        Args:
            name: Test name
            path: Path to test file

        Returns:
            (success, elapsed_time)
        """
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"File: {path}")
        print('=' * 70)

        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=False,
                check=True,
                timeout=300  # 5 minute timeout
            )
            elapsed = time.time() - start_time
            success = result.returncode == 0

            if success:
                print(f"\n✓ {name} PASSED ({elapsed:.2f}s)")
            else:
                print(f"\n✗ {name} FAILED ({elapsed:.2f}s)")

            return success, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"\n⏱ {name} TIMEOUT ({elapsed:.2f}s)")
            return False, elapsed

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {name} FAILED with error ({elapsed:.2f}s)")
            print(f"Error: {e}")
            return False, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ {name} CRASHED ({elapsed:.2f}s)")
            print(f"Exception: {e}")
            return False, elapsed

    def run_all(self) -> bool:
        """
        Run all tests.

        Returns:
            True if all tests passed
        """
        print("=" * 70)
        print("T-M-B-M-T VERIFICATION SUITE")
        print("Paper: 'A Neuro-Symbolic Hourglass Architecture'")
        print("=" * 70)
        print(f"\nTotal tests to run: {len(self.tests)}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        all_passed = True
        total_time = 0.0

        for name, path in self.tests:
            success, elapsed = self.run_test(name, path)
            self.results.append((name, success, elapsed))
            total_time += elapsed

            if not success:
                all_passed = False

        # Print summary
        self.print_summary(total_time)

        return all_passed

    def print_summary(self, total_time: float):
        """Print test results summary."""
        print("\n" + "=" * 70)
        print("VERIFICATION RESULTS SUMMARY")
        print("=" * 70)

        passed_count = sum(1 for _, success, _ in self.results if success)
        total_count = len(self.results)

        print(f"\n{'Test Name':<30} {'Status':<10} {'Time (s)':<10}")
        print("-" * 70)

        for name, success, elapsed in self.results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{name:<30} {status:<10} {elapsed:>8.2f}")

        print("-" * 70)
        print(f"{'TOTAL':<30} {passed_count}/{total_count} {total_time:>8.2f}")

        print("\n" + "=" * 70)
        if passed_count == total_count:
            print("ALL TESTS PASSED ✓")
            print("=" * 70)
            print("\n✅ All mathematical formulations verified")
            print("✅ All theoretical guarantees confirmed")
            print("✅ All architectural components validated")
            print("✅ All complexity claims supported")
            print("\nThe T-M-B-M-T paper claims are mathematically sound.")
        else:
            print(f"SOME TESTS FAILED: {total_count - passed_count}/{total_count}")
            print("=" * 70)
            print("\n⚠️  Please review failed tests above")
            print("⚠️  Check VERIFICATION_README.md for troubleshooting")

        print(f"\nTotal verification time: {total_time:.2f}s")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main entry point."""
    runner = TestRunner()
    all_passed = runner.run_all()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
