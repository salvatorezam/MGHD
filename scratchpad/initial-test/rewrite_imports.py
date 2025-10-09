#!/usr/bin/env python3
"""Script to rewrite imports from old structure to new mghd package structure."""
import os
import re
from pathlib import Path

# Define the import rewrite rules
REWRITES = [
    # mghd_main -> mghd.core
    (r'\bfrom\s+mghd_main\.', r'from mghd.core.'),
    (r'\bimport\s+mghd_main\.', r'import mghd.core.'),
    (r'\bfrom\s+mghd_main\s+import\b', r'from mghd.core import'),
    (r'\bimport\s+mghd_main\b', r'import mghd.core'),
    
    # teachers -> mghd.decoders
    (r'\bfrom\s+teachers\.', r'from mghd.decoders.'),
    (r'\bimport\s+teachers\.', r'import mghd.decoders.'),
    (r'\bfrom\s+teachers\s+import\b', r'from mghd.decoders import'),
    (r'\bimport\s+teachers\b', r'import mghd.decoders'),
    
    # tad_rl -> mghd.tad.rl
    (r'\bfrom\s+tad_rl\.', r'from mghd.tad.rl.'),
    (r'\bimport\s+tad_rl\.', r'import mghd.tad.rl.'),
    (r'\bfrom\s+tad_rl\s+import\b', r'from mghd.tad.rl import'),
    (r'\bimport\s+tad_rl\b', r'import mghd.tad.rl'),
    
    # cudaq_backend -> mghd.samplers.cudaq_backend
    (r'\bfrom\s+cudaq_backend\.', r'from mghd.samplers.cudaq_backend.'),
    (r'\bimport\s+cudaq_backend\.', r'import mghd.samplers.cudaq_backend.'),
    (r'\bfrom\s+cudaq_backend\s+import\b', r'from mghd.samplers.cudaq_backend import'),
    (r'\bimport\s+cudaq_backend\b', r'import mghd.samplers.cudaq_backend'),
    
    # codes_registry -> mghd.codes.registry
    (r'\bfrom\s+codes_registry\s+import\b', r'from mghd.codes.registry import'),
    (r'\bimport\s+codes_registry\b', r'import mghd.codes.registry'),
    
    # samplers -> mghd.samplers (if any top-level imports exist)
    (r'\bfrom\s+samplers\.', r'from mghd.samplers.'),
    (r'\bimport\s+samplers\.', r'import mghd.samplers.'),
]

def rewrite_file(filepath):
    """Rewrite imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        for pattern, replacement in REWRITES:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return False

def main():
    """Find and rewrite all Python files."""
    repo_root = Path(__file__).parent
    changed_files = []
    
    # Process files in mghd/, tests/, and any remaining top-level scripts
    patterns = [
        'mghd/**/*.py',
        'tests/**/*.py',
        'tools/**/*.py',  # Any remaining tools
        '*.py',  # Top-level scripts
    ]
    
    all_files = set()
    for pattern in patterns:
        all_files.update(repo_root.glob(pattern))
    
    for filepath in sorted(all_files):
        if filepath.is_file() and not str(filepath).startswith('.'):
            if rewrite_file(filepath):
                changed_files.append(filepath)
                print(f"✓ Updated: {filepath.relative_to(repo_root)}")
    
    print(f"\n{len(changed_files)} files updated")
    return len(changed_files)

if __name__ == '__main__':
    main()
