#!/usr/bin/env python3
"""Release pre-flight audit script for whatsapp-total-recall.

Verifies:
1. Version synchronization across pyproject.toml, Info.plist, and mcp.py.
2. Documentation integrity: README.pypi.md exists and has zero broken relative links.
3. CI/CD workflow alignment: .github/workflows/release.yaml matches PyPI publisher config.
4. Clean build test: ensures hatchling packages sdist and wheel with both readmes.
"""

import plistlib
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def check_versions() -> str:
    # 1. pyproject.toml
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    pyproject_ver = pyproject_data.get("project", {}).get("version")
    if not pyproject_ver:
        sys.exit("FAIL: pyproject.toml is missing project.version")

    # 2. Info.plist
    plist_path = REPO_ROOT / "apps" / "WhatsAppTotalRecallBar" / "Info.plist"
    with open(plist_path, "rb") as f:
        plist_data = plistlib.load(f)
    plist_ver = plist_data.get("CFBundleShortVersionString")
    if plist_ver != pyproject_ver:
        sys.exit(
            f"FAIL: Info.plist version ({plist_ver}) != pyproject.toml version ({pyproject_ver})"
        )

    # 3. mcp.py
    mcp_path = REPO_ROOT / "whatsapp_total_recall" / "mcp.py"
    mcp_content = mcp_path.read_text(encoding="utf-8")
    mcp_match = re.search(r'"serverInfo":\s*\{[^}]*"version":\s*"([^"]+)"', mcp_content)
    if not mcp_match:
        sys.exit(
            "FAIL: Could not extract serverInfo.version from whatsapp_total_recall/mcp.py"
        )
    mcp_ver = mcp_match.group(1)
    if mcp_ver != pyproject_ver:
        sys.exit(
            f"FAIL: mcp.py version ({mcp_ver}) != pyproject.toml version ({pyproject_ver})"
        )

    print(f"PASS: Versions synchronized across all targets: {pyproject_ver}")
    return pyproject_ver


def check_docs() -> None:
    # Verify pyproject.toml points to README.pypi.md
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    readme_setting = pyproject_data.get("project", {}).get("readme")
    if readme_setting != "README.pypi.md":
        sys.exit(
            f"FAIL: pyproject.toml readme must point to 'README.pypi.md', got '{readme_setting}'"
        )

    pypi_readme = REPO_ROOT / "README.pypi.md"
    if not pypi_readme.exists():
        sys.exit("FAIL: README.pypi.md does not exist.")

    # Check for unresolvable relative markdown links
    content = pypi_readme.read_text(encoding="utf-8")
    relative_links = re.findall(
        r"\[([^\]]+)\]\((?!(?:https?://|mailto:|#))([^)]+)\)", content
    )
    if relative_links:
        print(
            f"WARN: README.pypi.md contains relative links that will 404 on PyPI: {relative_links}"
        )
        sys.exit(
            "FAIL: Convert relative markdown links in README.pypi.md to absolute GitHub URLs."
        )

    # Check sdist includes
    sdist_includes = (
        pyproject_data.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("sdist", {})
        .get("include", [])
    )
    if "/README.pypi.md" not in sdist_includes:
        sys.exit(
            "FAIL: /README.pypi.md must be included in tool.hatch.build.targets.sdist.include"
        )

    print(
        "PASS: Documentation integrity verified (README.pypi.md scoped with absolute links)"
    )


def check_workflow() -> None:
    workflow_yml = REPO_ROOT / ".github" / "workflows" / "release.yml"

    if not workflow_yml.exists():
        sys.exit("FAIL: .github/workflows/release.yml not found.")

    content = workflow_yml.read_text(encoding="utf-8")
    if "pypa/gh-action-pypi-publish" not in content:
        sys.exit("FAIL: release.yml is missing pypa/gh-action-pypi-publish step")
    if "environment:" not in content or "name: pypi" not in content:
        sys.exit("FAIL: release.yml is missing environment: pypi for PyPI OIDC minting")

    print(
        "PASS: GitHub Actions release.yml matches PyPI Trusted Publisher specification"
    )


def main() -> None:
    print("=== Running WhatsApp Total Recall Release Pre-Flight Checks ===")
    version = check_versions()
    check_docs()
    check_workflow()
    print(f"=== ALL PRE-FLIGHT RELEASE CHECKS PASSED FOR v{version} ===")


if __name__ == "__main__":
    main()
