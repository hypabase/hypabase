"""Persistence utilities for hypergraph save/load.

Supports both single-store and multi-namespace database persistence
in JSON and HIF formats.

Directory structure for multi-namespace save:
    output_path/
    |-- manifest.json           # {version, format, namespaces: [...]}
    |-- default.json            # Flat namespaces as files
    |-- financial/              # Hierarchical -> subdirectories
        |-- entities.json
        |-- themes.json

Security:
    Path validation is performed to prevent path traversal attacks.
    All paths are resolved to absolute paths and validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .core import HypergraphCore

FormatType = Literal["json", "hif"]

MANIFEST_VERSION = "1.0"


def _validate_path(path: str, base_dir: Path | None = None) -> Path:
    """Validate and resolve a file path.

    Performs security checks to prevent path traversal attacks.

    Args:
        path: The path to validate
        base_dir: Optional base directory that the path must be within

    Returns:
        Resolved absolute Path

    Raises:
        ValueError: If path is invalid or attempts path traversal
    """
    # Check for null bytes before any path operations (common attack vector)
    if "\x00" in path:
        raise ValueError(f"Invalid path (contains null bytes): {path!r}")

    resolved = Path(path).resolve()

    # If base_dir is specified, ensure the path is within it
    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: {path} is outside base directory {base_dir}"
            )

    return resolved


def _validate_namespace_path(namespace: str, base_dir: Path) -> Path:
    """Validate a namespace-derived path is within the base directory.

    Args:
        namespace: Namespace name (may contain '/')
        base_dir: Base directory for the database

    Returns:
        Validated absolute path

    Raises:
        ValueError: If namespace path attempts directory traversal
    """
    rel_path = _namespace_to_path(namespace)
    full_path = (base_dir / rel_path).resolve()

    # Ensure the resolved path is within base_dir
    base_resolved = base_dir.resolve()
    try:
        full_path.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Namespace '{namespace}' resolves to path outside base directory")

    return full_path


def save_store(
    store: HypergraphCore,
    path: str,
    format: FormatType = "json",
) -> None:
    """Save a single HypergraphCore to a file.

    Args:
        store: The store to save
        path: Output file path
        format: "json" for internal dict format, "hif" for HIF standard

    Raises:
        ValueError: If path is invalid or format is unknown
    """
    validated_path = _validate_path(path)

    if format == "json":
        data = store.to_dict()
    elif format == "hif":
        data = store.to_hif()
    else:
        raise ValueError(f"Unknown format: {format!r}")

    # Ensure parent directory exists
    validated_path.parent.mkdir(parents=True, exist_ok=True)

    with open(validated_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_store(
    path: str,
    format: FormatType = "json",
) -> HypergraphCore:
    """Load a single HypergraphCore from a file.

    Args:
        path: Input file path
        format: "json" for internal dict format, "hif" for HIF standard

    Returns:
        Loaded HypergraphCore

    Raises:
        ValueError: If path is invalid or format is unknown
        FileNotFoundError: If file does not exist
    """
    validated_path = _validate_path(path)

    with open(validated_path, encoding="utf-8") as f:
        data = json.load(f)

    if format == "json":
        return HypergraphCore.from_dict(data)
    elif format == "hif":
        return HypergraphCore.from_hif(data)
    else:
        raise ValueError(f"Unknown format: {format!r}")


def _namespace_to_path(namespace: str) -> str:
    """Convert namespace name to relative file path.

    "default" -> "default.json"
    "financial/entities" -> "financial/entities.json"
    """
    return f"{namespace}.json"


def _path_to_namespace(path: str) -> str:
    """Convert relative file path to namespace name.

    "default.json" -> "default"
    "financial/entities.json" -> "financial/entities"
    """
    return path.removesuffix(".json")


def save_db(
    namespaces: dict[str, HypergraphCore],
    path: str,
    format: FormatType = "json",
) -> None:
    """Save multiple namespaces to a directory.

    Creates a directory structure with manifest and individual
    namespace files. Hierarchical namespace names become subdirectories.

    Args:
        namespaces: Dict mapping namespace names to stores
        path: Output directory path
        format: "json" for internal dict format, "hif" for HIF standard

    Raises:
        ValueError: If path is invalid or namespace paths attempt traversal
    """
    base_path = _validate_path(path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Validate all namespace paths before writing anything
    for namespace in namespaces:
        _validate_namespace_path(namespace, base_path)

    # Build manifest
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "format": format,
        "namespaces": sorted(namespaces.keys()),
    }

    # Write manifest
    manifest_path = base_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write each namespace
    for namespace, store in namespaces.items():
        file_path = _validate_namespace_path(namespace, base_path)

        # Create subdirectory if needed (for hierarchical namespaces)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = store.to_dict()
        else:
            data = store.to_hif()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_db(
    path: str,
    format: FormatType | None = None,
) -> dict[str, HypergraphCore]:
    """Load multiple namespaces from a directory.

    Args:
        path: Directory path containing manifest.json
        format: Override format from manifest (optional)

    Returns:
        Dict mapping namespace names to stores

    Raises:
        ValueError: If path is invalid or namespace paths attempt traversal
        FileNotFoundError: If manifest or namespace files don't exist
    """
    base_path = _validate_path(path)
    manifest_path = base_path / "manifest.json"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Use manifest format unless overridden
    actual_format = format or manifest.get("format", "json")

    namespaces: dict[str, HypergraphCore] = {}

    for namespace in manifest.get("namespaces", []):
        file_path = _validate_namespace_path(namespace, base_path)

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if actual_format == "json":
            store = HypergraphCore.from_dict(data)
        else:
            store = HypergraphCore.from_hif(data)

        namespaces[namespace] = store

    return namespaces


def get_manifest(path: str) -> dict[str, Any]:
    """Read the manifest from a saved database directory.

    Args:
        path: Directory path

    Returns:
        Manifest dict

    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If manifest doesn't exist
    """
    base_path = _validate_path(path)
    manifest_path = base_path / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result
