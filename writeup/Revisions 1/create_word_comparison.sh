#!/bin/bash
# Script to create a Word document showing changes between branches
# Uses pandoc to convert markdown diff to Word format

echo "Creating Word document comparison..."

# Get the repository root and change to it
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
WRITEUP_DIR="$REPO_ROOT/writeup"
OUTPUT_DIR="$WRITEUP_DIR/Revisions 1"

# Get the old version from main branch
echo "Extracting version from main branch..."
git show main:writeup/writeup.md > "$OUTPUT_DIR/version_main.md"

# Get current version
echo "Using current version..."
cp "$WRITEUP_DIR/writeup.md" "$OUTPUT_DIR/version_current.md"

# Create a markdown document with the comparison
cat > "$OUTPUT_DIR/comparison_document.md" << 'HEREDOC'
---
title: "Document Comparison: Main Branch vs Current Branch"
subtitle: "Changes to writeup.md"
author: Revision Tracking
date: 2025
---

# Overview

This document shows the changes made to `writeup.md` between the main branch and the current branch (redo_resample).

## Summary of Changes

```
HEREDOC

# Add git diff stats (ignore whitespace)
git diff --stat --ignore-all-space main..redo_resample -- writeup/writeup.md >> "$OUTPUT_DIR/comparison_document.md"

cat >> "$OUTPUT_DIR/comparison_document.md" << 'HEREDOC'
```

## Detailed Changes

Lines starting with:
- `-` (minus) were **removed** from the original
- `+` (plus) were **added** in the revision

**Note:** Whitespace-only changes are ignored to focus on substantive changes.

```diff
HEREDOC

# Add the actual diff (ignore whitespace)
git diff --unified=3 --ignore-all-space main..redo_resample -- writeup/writeup.md >> "$OUTPUT_DIR/comparison_document.md"

echo '```' >> "$OUTPUT_DIR/comparison_document.md"

# Convert to Word document
echo "Converting to Word document..."
cd "$WRITEUP_DIR"

# Use pandoc to create Word doc (no template needed)
pandoc "$OUTPUT_DIR/comparison_document.md" \
    -o "$OUTPUT_DIR/comparison_document.docx" \
    2>&1 | grep -v "Missing character" || true

# Clean up temporary markdown file
rm "$OUTPUT_DIR/version_main.md" "$OUTPUT_DIR/version_current.md"

if [ -f "$OUTPUT_DIR/comparison_document.docx" ]; then
    echo ""
    echo "✓ SUCCESS!"
    echo "Word document created at:"
    echo "  $OUTPUT_DIR/comparison_document.docx"
    echo ""
    echo "You can open this in Microsoft Word or LibreOffice Writer."
    echo "The diff is shown in code blocks with +/- indicators."
    echo ""
    echo "File size: $(du -h "$OUTPUT_DIR/comparison_document.docx" | cut -f1)"
else
    echo "ERROR: Word document creation failed"
    echo "Markdown source available at: $OUTPUT_DIR/comparison_document.md"
    exit 1
fi
