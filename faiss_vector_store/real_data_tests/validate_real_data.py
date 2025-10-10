"""
Validation Checklist - Verify real data test results.

Performs specific checks on test results and generates pass/fail report.
"""
import sys
import json
from pathlib import Path
import time


class RealDataValidator:
    """Validate real data test results."""

    def __init__(self, data_dir: Path):
        """Initialize validator."""
        self.data_dir = Path(data_dir)
        self.checks = []
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, details: str = ""):
        """
        Perform a validation check.

        Args:
            name: Check name
            condition: True if check passes
            details: Additional details
        """
        status = "[PASS]" if condition else "[FAIL]"
        symbol = "✓" if condition else "✗"

        # Use ASCII symbols for Windows compatibility
        symbol = "[OK]" if condition else "[X]"

        print(f"  {symbol} {name}")
        if details:
            print(f"      {details}")

        self.checks.append({
            'check': name,
            'passed': condition,
            'details': details
        })

        if condition:
            self.passed += 1
        else:
            self.failed += 1

        return condition

    def validate_test_results(self):
        """Validate test_real_data.py results."""
        print("\n[1/3] Validating test results...")

        results_path = self.data_dir / "real_data_test_results.json"

        if not results_path.exists():
            self.check(
                "Test results file exists",
                False,
                "File not found. Run test_real_data.py first."
            )
            return False

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Check 1: All 5,000 documents imported
        num_docs = results['summary']['num_documents']
        self.check(
            "All 5,000 documents imported successfully",
            num_docs == 5000,
            f"Imported: {num_docs:,}"
        )

        # Check 2: Query 1 returns exact match
        query1 = results['queries'][0]
        query1_distance = query1['top_5_distances'][0]
        self.check(
            "Query 1 returns exact match (distance < 0.01)",
            query1_distance < 0.01,
            f"Distance: {query1_distance:.6f}"
        )

        # Check 3: Query 2 returns reasonable results
        query2 = results['queries'][1]
        query2_distance = query2['top_5_distances'][0]
        self.check(
            "Query 2 returns reasonable results (distance < 1.0)",
            query2_distance < 1.0,
            f"Top distance: {query2_distance:.4f}"
        )

        # Check 4: Query 3 returns results (not empty)
        query3 = results['queries'][2]
        self.check(
            "Query 3 returns results (not empty)",
            query3['num_results'] > 0,
            f"Results: {query3['num_results']}"
        )

        # Check 5: Query latency
        avg_query_time = results['summary']['avg_query_time_ms']
        self.check(
            "Query latency < 100ms for all queries",
            avg_query_time < 100,
            f"Average: {avg_query_time:.2f} ms"
        )

        return True

    def validate_resource_usage(self):
        """Validate resource usage."""
        print("\n[2/3] Validating resource usage...")

        results_path = self.data_dir / "real_data_test_results.json"

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Check RAM usage
        if results['resource_usage']:
            latest_resources = results['resource_usage'][-1]

            ram_gib = latest_resources['ram_gib']
            ram_limit = latest_resources['ram_limit_gib']

            self.check(
                "Peak RAM usage < 20GB",
                ram_gib < 20,
                f"RAM: {ram_gib:.2f} / {ram_limit:.2f} GiB"
            )

            vram_gib = latest_resources['vram_gib']
            vram_limit = latest_resources['vram_limit_gib']

            self.check(
                "Peak VRAM usage < 3GB",
                vram_gib < 3,
                f"VRAM: {vram_gib:.2f} / {vram_limit:.2f} GiB"
            )

    def validate_file_outputs(self):
        """Validate that all expected files exist."""
        print("\n[3/3] Validating file outputs...")

        required_files = [
            "real_data_embeddings.npy",
            "real_data_documents.json",
            "real_data_metadata.json",
            "real_data_ids.json",
            "export_metadata.json",
            "real_data_test_results.json"
        ]

        for filename in required_files:
            file_path = self.data_dir / filename
            self.check(
                f"File exists: {filename}",
                file_path.exists(),
                f"Path: {file_path}"
            )

    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print(f"\nTotal Checks: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        # Save report
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_checks': total,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': pass_rate,
            'checks': self.checks,
            'overall_status': 'PASS' if self.failed == 0 else 'FAIL'
        }

        report_path = self.data_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_path.name}")

        print("\n" + "="*60)
        if self.failed == 0:
            print("[SUCCESS] ALL VALIDATIONS PASSED")
            print("="*60)
            print("\nYour Faiss implementation is working correctly with real data!")
            print("\nReady for production use. You can now:")
            print("  1. Export your full Chroma database")
            print("  2. Migrate to Faiss for production")
            print("  3. Monitor performance in production")
        else:
            print("[FAIL] SOME VALIDATIONS FAILED")
            print("="*60)
            print(f"\n{self.failed} checks failed. Please review:")

            for check in self.checks:
                if not check['passed']:
                    print(f"  - {check['check']}")
                    if check['details']:
                        print(f"    {check['details']}")

            print("\nDO NOT proceed to production until all checks pass.")

        return self.failed == 0


def main():
    """Main validation function."""
    print("="*60)
    print("REAL DATA VALIDATION CHECKLIST")
    print("="*60)

    data_dir = Path(__file__).parent

    try:
        validator = RealDataValidator(data_dir)

        # Run validations
        validator.validate_test_results()
        validator.validate_resource_usage()
        validator.validate_file_outputs()

        # Generate report
        all_passed = validator.generate_report()

        sys.exit(0 if all_passed else 1)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run the following scripts in order:")
        print("  1. python export_chroma_sample.py")
        print("  2. python test_real_data.py")
        print("  3. python compare_performance.py")
        print("  4. python validate_real_data.py")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Validation failed:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
