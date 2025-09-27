#!/usr/bin/env python3
"""Audit repo imports and surface Clean-Core inventory."""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOTS = [
    "mghd_public/infer.py",
    "mghd_public/model_v2.py",
    "mghd_public/features_v2.py",
    "mghd_clustered/clustered_primary.py",
    "mghd_clustered/cluster_core.py",
    "teachers/ensemble.py",
    "teachers/mwpf_ctx.py",
    "teachers/mwpm_ctx.py",
    "cudaq_backend/backend_api.py",
    "cudaq_backend/syndrome_gen.py",
    "cudaq_backend/circuits.py",
    "cudaq_backend/garnet_noise.py",
    "cudaq_backend/constants.py",
    "training/cluster_crops_train.py",
    "tools/make_cluster_crops.py",
    "tools/bench_clustered_sweep_surface.py",
]
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "env",
    "build",
    "dist",
}
V1_NAME_TOKENS = ("model.py", "features.py", "cluster_proxy", "poc_")
REPORT_PATH = ROOT / "results" / "clean_core_inventory.json"


class ModuleIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path_to_module: Dict[Path, str] = {}
        self.module_to_path: Dict[str, Path] = {}
        self.is_package: Dict[Path, bool] = {}

    def build(self) -> None:
        for path in self.root.rglob("*.py"):
            if any(part in SKIP_DIR_NAMES for part in path.relative_to(self.root).parts):
                continue
            rel_path = path.relative_to(self.root)
            parts = list(rel_path.parts)
            if not parts:
                continue
            if parts[-1] == "__init__.py":
                module_parts = parts[:-1]
                is_package = True
            else:
                module_parts = parts[:-1] + [parts[-1][:-3]]
                is_package = False
            module_name = ".".join(module_parts)
            rel_posix = rel_path.as_posix()
            self.path_to_module[path] = module_name
            self.module_to_path[module_name] = path
            self.is_package[path] = is_package

    def module_for_path(self, path: Path) -> str:
        return self.path_to_module.get(path, "")

    def path_for_module(self, module_name: str) -> Path | None:
        return self.module_to_path.get(module_name)

    def package_for_module(self, path: Path) -> str:
        module = self.module_for_path(path)
        if self.is_package.get(path, False):
            return module
        if not module:
            return ""
        if "." not in module:
            return ""
        return module.rsplit(".", 1)[0]


class ImportGraph:
    def __init__(self, index: ModuleIndex) -> None:
        self.index = index
        self.graph: Dict[str, Set[str]] = defaultdict(set)

    def add_dependency(self, src: Path, module_name: str) -> None:
        if not module_name:
            return
        target_path = self.index.path_for_module(module_name)
        if not target_path:
            return
        src_rel = src.relative_to(self.index.root).as_posix()
        tgt_rel = target_path.relative_to(self.index.root).as_posix()
        if src_rel == tgt_rel:
            return
        self.graph[src_rel].add(tgt_rel)

    def build(self) -> None:
        for path in self.index.path_to_module:
            self._process_file(path)

    def _process_file(self, path: Path) -> None:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            return
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return
        current_package = self.index.package_for_module(path)
        current_module = self.index.module_for_path(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._handle_absolute_import(path, alias.name)
            elif isinstance(node, ast.ImportFrom):
                self._handle_from_import(path, node, current_package)

    def _handle_absolute_import(self, src: Path, name: str) -> None:
        candidates = self._candidate_modules(name)
        for candidate in candidates:
            self.add_dependency(src, candidate)

    def _handle_from_import(self, src: Path, node: ast.ImportFrom, current_package: str) -> None:
        base_module: str | None
        if node.level == 0:
            base_module = node.module
        else:
            package = current_package
            steps = node.level - 1
            while steps > 0 and package:
                if "." in package:
                    package = package.rsplit(".", 1)[0]
                else:
                    package = ""
                steps -= 1
            if node.module:
                if package:
                    base_module = f"{package}.{node.module}"
                else:
                    base_module = node.module
            else:
                base_module = package
        for alias in node.names:
            names_to_try: List[str] = []
            if alias.name == "*":
                if base_module:
                    names_to_try.append(base_module)
            else:
                if base_module:
                    names_to_try.append(f"{base_module}.{alias.name}")
                    names_to_try.append(base_module)
                names_to_try.append(alias.name)
            for candidate in names_to_try:
                self._handle_absolute_import(src, candidate)

    def _candidate_modules(self, dotted: str) -> List[str]:
        parts = dotted.split(".")
        candidates: List[str] = []
        while parts:
            name = ".".join(parts)
            if self.index.path_for_module(name):
                candidates.append(name)
                break
            parts.pop()
        return candidates


def discover_runtime_roots(index: ModuleIndex) -> List[Path]:
    roots: List[Path] = []
    missing: List[str] = []
    for rel in RUNTIME_ROOTS:
        path = ROOT / rel
        if path.exists():
            roots.append(path)
        else:
            missing.append(rel)
    if missing:
        print(
            "Warning: missing runtime roots:" + ", ".join(missing),
            file=sys.stderr,
        )
    return roots


def bfs_used_modules(graph: ImportGraph, roots: Iterable[Path]) -> Set[str]:
    used: Set[str] = set()
    queue: deque[str] = deque()
    for root_path in roots:
        rel = root_path.relative_to(graph.index.root).as_posix()
        used.add(rel)
        queue.append(rel)
    while queue:
        current = queue.popleft()
        for neighbor in graph.graph.get(current, set()):
            if neighbor not in used:
                used.add(neighbor)
                queue.append(neighbor)
    return used


def find_v1_candidates(paths: Iterable[Path]) -> List[str]:
    candidates: Set[str] = set()
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        lower = rel.lower()
        if any(token in lower for token in V1_NAME_TOKENS):
            candidates.add(rel)
    return sorted(candidates)


def main() -> None:
    index = ModuleIndex(ROOT)
    index.build()
    graph = ImportGraph(index)
    graph.build()
    roots = discover_runtime_roots(index)
    used_modules = bfs_used_modules(graph, roots)

    all_paths = {
        path.relative_to(ROOT).as_posix()
        for path in index.path_to_module.keys()
    }
    v1_candidates = find_v1_candidates(index.path_to_module.keys())
    orphans = sorted(all_paths - used_modules)
    used_sorted = sorted(used_modules)

    report = {
        "used_modules": used_sorted,
        "v1_candidates": v1_candidates,
        "orphans": orphans,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Inventory: used={len(used_sorted)} v1_candidates={len(v1_candidates)} orphans={len(orphans)}"
    )


if __name__ == "__main__":
    main()
