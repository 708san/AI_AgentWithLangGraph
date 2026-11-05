"""Command-line entry points for repository and experiment operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_ops.catalog import build_catalog, write_catalog
from project_ops.compatibility import compare_artifact_directories
from project_ops.execution import parse_output_spec, run_managed_command
from project_ops.evaluation import analyze_run
from project_ops.legacy_import import import_legacy_root
from project_ops.reporting import generate_project_report
from project_ops.profiles import list_profiles
from project_ops.profile_execution import run_profile
from project_ops.run_management import (
    attach_output_validation,
    initialize_run,
    list_runs,
    update_run_status,
)
from project_ops.output_validation import (
    validate_external_response,
    validation_filename,
    write_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m project_ops",
        description="Catalog repository files, scripts, datasets, and artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command")
    inventory = subparsers.add_parser(
        "inventory",
        help="write a machine-readable project catalog",
    )
    inventory.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root; defaults to the current Git repository",
    )
    inventory.add_argument(
        "--output",
        type=Path,
        default=None,
        help="catalog JSON path; defaults to local_artifacts/catalog/project_catalog.json",
    )
    inventory.add_argument(
        "--hash",
        action="store_true",
        help="include SHA-256 for files up to the configured size limit",
    )
    inventory.add_argument(
        "--max-hash-mb",
        type=float,
        default=50.0,
        help="maximum file size to hash when --hash is used (default: 50 MB)",
    )
    init_run = subparsers.add_parser(
        "run-init",
        help="initialize a managed run and register its conditions",
    )
    init_run.add_argument("--run-id", required=True)
    init_run.add_argument("--kind", required=True)
    init_run.add_argument("--purpose")
    init_run.add_argument("--entrypoint")
    init_run.add_argument("--dataset-id")
    init_run.add_argument("--dataset-version")
    init_run.add_argument("--model-name")
    init_run.add_argument("--evidence-mode")
    init_run.add_argument("--root", type=Path, default=None)
    init_run.add_argument("--registry", type=Path, default=None)
    init_run.add_argument(
        "--metadata-json",
        type=Path,
        help="optional JSON object with additional run metadata",
    )
    init_run.add_argument(
        "--command",
        dest="run_command",
        nargs=argparse.REMAINDER,
        required=True,
        help="exact command that will execute the run",
    )
    register_run_parser = subparsers.add_parser(
        "run-status",
        help="update a managed run status and refresh the registry",
    )
    register_run_parser.add_argument("--run-dir", type=Path, required=True)
    register_run_parser.add_argument("--status", required=True)
    register_run_parser.add_argument("--registry", type=Path, default=None)
    register_run_parser.add_argument("--error")
    list_run_parser = subparsers.add_parser(
        "run-list",
        help="list registered runs",
    )
    list_run_parser.add_argument("--registry", type=Path, default=None)
    list_run_parser.add_argument("--status")
    validate_parser = subparsers.add_parser(
        "validate-output",
        help="validate and attach an external output to a managed Run",
    )
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.add_argument("--response", type=Path, required=True)
    validate_parser.add_argument("--output-kind", default="external_response")
    validate_parser.add_argument("--case-id")
    validate_parser.add_argument("--schema-version")
    validate_parser.add_argument("--registry", type=Path, default=None)
    validate_parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="return failure for partial validation as well as failed validation",
    )
    managed_parser = subparsers.add_parser(
        "run-managed",
        help="run an existing repository command with Run lifecycle management",
    )
    managed_parser.add_argument("--run-id", required=True)
    managed_parser.add_argument("--kind", required=True)
    managed_parser.add_argument("--purpose")
    managed_parser.add_argument("--entrypoint")
    managed_parser.add_argument("--dataset-id")
    managed_parser.add_argument("--dataset-version")
    managed_parser.add_argument("--model-name")
    managed_parser.add_argument("--evidence-mode")
    managed_parser.add_argument("--root", type=Path, default=None)
    managed_parser.add_argument("--registry", type=Path, default=None)
    managed_parser.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="validate and attach an output after the command completes; repeatable",
    )
    managed_parser.add_argument(
        "--command",
        dest="run_command",
        nargs=argparse.REMAINDER,
        required=True,
        help="exact command to execute",
    )
    analyze_parser = subparsers.add_parser(
        "analyze-run",
        help="register saved evaluation metrics and optional baseline comparison",
    )
    analyze_parser.add_argument("--run-dir", type=Path, required=True)
    analyze_parser.add_argument("--report", type=Path, required=True)
    analyze_parser.add_argument("--baseline-report", type=Path)
    analyze_parser.add_argument("--registry", type=Path, default=None)
    report_parser = subparsers.add_parser(
        "report-project",
        help="generate project overview HTML/JSON/CSV from saved catalogs and Runs",
    )
    report_parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("local_artifacts/catalog/project_catalog.json"),
    )
    report_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("local_artifacts/catalog/run_registry.json"),
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_artifacts/reports/project_overview"),
    )
    import_parser = subparsers.add_parser(
        "import-legacy",
        help="register existing local execution artifacts without rerunning them",
    )
    import_parser.add_argument("--source-root", type=Path, required=True)
    import_parser.add_argument("--root", type=Path, default=None)
    import_parser.add_argument("--registry", type=Path, default=None)
    import_parser.add_argument("--limit", type=int)
    profiles_parser = subparsers.add_parser(
        "list-profiles",
        help="list canonical existing execution and analysis profiles",
    )
    profiles_parser.add_argument("--root", type=Path, default=None)
    profile_run_parser = subparsers.add_parser(
        "run-profile",
        help="run an existing entrypoint through its canonical profile and Run manager",
    )
    profile_run_parser.add_argument("--profile-id", required=True)
    profile_run_parser.add_argument("--run-id", required=True)
    profile_run_parser.add_argument("--root", type=Path, default=None)
    profile_run_parser.add_argument("--registry", type=Path, default=None)
    profile_run_parser.add_argument("--dataset-id")
    profile_run_parser.add_argument("--dataset-version")
    profile_run_parser.add_argument("--model-name")
    profile_run_parser.add_argument("--evidence-mode")
    profile_run_parser.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="validate and attach an output after the command completes; repeatable",
    )
    profile_run_parser.add_argument(
        "--script-args",
        dest="script_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="arguments passed unchanged to the profile entrypoint",
    )
    compare_parser = subparsers.add_parser(
        "compare-artifacts",
        help="compare two saved execution artifact directories without rerunning APIs",
    )
    compare_parser.add_argument("--profile-id")
    compare_parser.add_argument("--baseline-dir", type=Path, required=True)
    compare_parser.add_argument("--candidate-dir", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.add_argument(
        "--attach-run-dir",
        type=Path,
        help="optional managed Run directory to attach the compatibility report to",
    )
    compare_parser.add_argument("--registry", type=Path, default=None)
    compare_parser.add_argument(
        "--assert-equivalent",
        action="store_true",
        help="return exit code 2 unless the saved artifacts are equivalent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "inventory":
        root = (args.root or Path.cwd()).expanduser().resolve()
        output = args.output or root / "local_artifacts/catalog/project_catalog.json"
        catalog = build_catalog(
            root,
            include_hash=args.hash,
            max_hash_bytes=max(1, int(args.max_hash_mb * 1024 * 1024)),
        )
        write_catalog(output, catalog)
        summary = catalog["summary"]
        print(f"Catalog: {output}")
        print(f"Files: {summary['file_count']}")
        print(f"Scripts: {summary['script_count']}")
        print(f"Data files: {summary['data_file_count']}")
        print(f"Artifact roots: {summary['artifact_root_count']}")
        return 0
    if args.command == "run-init":
        root = (args.root or Path.cwd()).expanduser().resolve()
        metadata = {}
        if args.metadata_json:
            metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                parser.error("--metadata-json must contain a JSON object")
        manifest = initialize_run(
            root=root,
            run_id=args.run_id,
            kind=args.kind,
            command=args.run_command,
            registry_path=args.registry,
            purpose=args.purpose,
            entrypoint=args.entrypoint,
            dataset={
                key: value for key, value in {
                    "dataset_id": args.dataset_id,
                    "version": args.dataset_version,
                }.items() if value
            },
            model={"name": args.model_name} if args.model_name else {},
            pipeline={"evidence_mode": args.evidence_mode} if args.evidence_mode else {},
            metadata=metadata,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-status":
        manifest = update_run_status(
            args.run_dir,
            args.status,
            registry_path=args.registry,
            error=args.error,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-list":
        registry = args.registry or Path.cwd() / "local_artifacts/catalog/run_registry.json"
        print(json.dumps(list_runs(registry, args.status), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-output":
        run_dir = args.run_dir.expanduser().resolve()
        response_path = args.response.expanduser().resolve()
        report = validate_external_response(
            response_path,
            schema_version=args.schema_version,
            output_kind=args.output_kind,
        )
        report["run_id"] = run_dir.name
        if args.case_id:
            report["case_id"] = args.case_id
        report_path = run_dir / "validation" / validation_filename(args.output_kind, args.case_id)
        write_validation_report(report_path, report)
        output_record = {
            "kind": args.output_kind,
            "case_id": args.case_id,
            "path": str(response_path),
            "schema_version": report.get("schema_version"),
            "sha256": report.get("sha256"),
            "validation_status": report.get("status"),
            "validation_report": str(report_path),
        }
        attach_output_validation(run_dir, output_record, registry_path=args.registry)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] == "failed" or (args.fail_on_warnings and report["status"] == "partial"):
            return 2
        return 0
    if args.command == "run-managed":
        root = (args.root or Path.cwd()).expanduser().resolve()
        try:
            output_specs = [parse_output_spec(value) for value in args.output]
        except ValueError as exc:
            parser.error(str(exc))
        result = run_managed_command(
            root=root,
            run_id=args.run_id,
            kind=args.kind,
            command=args.run_command,
            purpose=args.purpose,
            entrypoint=args.entrypoint,
            dataset={
                key: value for key, value in {
                    "dataset_id": args.dataset_id,
                    "version": args.dataset_version,
                }.items() if value
            },
            model={"name": args.model_name} if args.model_name else {},
            pipeline={"evidence_mode": args.evidence_mode} if args.evidence_mode else {},
            output_specs=output_specs,
            registry_path=args.registry,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1
    if args.command == "analyze-run":
        result = analyze_run(
            args.run_dir,
            args.report,
            baseline_report_path=args.baseline_report,
            registry_path=args.registry,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report-project":
        result = generate_project_report(
            catalog_path=args.catalog,
            registry_path=args.registry,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "import-legacy":
        root = (args.root or Path.cwd()).expanduser().resolve()
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        result = import_legacy_root(
            root=root,
            source_root=args.source_root,
            registry_path=args.registry,
            limit=args.limit,
        )
        print(json.dumps({"imported_count": len(result), "runs": result}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "list-profiles":
        root = (args.root or Path.cwd()).expanduser().resolve()
        profiles = list_profiles(root)
        print(json.dumps({"count": len(profiles), "profiles": profiles}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-profile":
        root = (args.root or Path.cwd()).expanduser().resolve()
        try:
            output_specs = [parse_output_spec(value) for value in args.output]
            result = run_profile(
                root=root,
                profile_id=args.profile_id,
                run_id=args.run_id,
                script_args=args.script_args,
                dataset={
                    key: value for key, value in {
                        "dataset_id": args.dataset_id,
                        "version": args.dataset_version,
                    }.items() if value
                },
                model={"name": args.model_name} if args.model_name else {},
                pipeline={"evidence_mode": args.evidence_mode} if args.evidence_mode else {},
                output_specs=output_specs,
                registry_path=args.registry,
            )
        except (KeyError, FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1
    if args.command == "compare-artifacts":
        root = Path.cwd().expanduser().resolve()
        profile = None
        if args.profile_id:
            from project_ops.profiles import get_profile

            try:
                profile = get_profile(root, args.profile_id)
            except KeyError as exc:
                parser.error(str(exc))
        try:
            result = compare_artifact_directories(
                args.baseline_dir,
                args.candidate_dir,
                profile=profile,
                report_path=args.report,
                attach_run_dir=args.attach_run_dir,
                registry_path=args.registry,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.assert_equivalent and result["status"] != "equivalent":
            return 2
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2

