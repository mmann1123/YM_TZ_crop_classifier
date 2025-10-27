# Revision Tools

This folder contains scripts to help compare document versions between git branches, useful for journal revisions.

## Quick Start

**For journal submission, use this:**
```bash
cd writeup
./Revisions\ 1/create_simple_comparison_v2.sh
```

Then open `comparison_document.html` in your browser and print to PDF (Ctrl+P → Save as PDF).

## Scripts

### 1. `compare_branches.sh` - Quick Comparison

Shows what changed between branches using multiple methods.

**Usage:**
```bash
cd writeup
./Revisions\ 1/compare_branches.sh
```

**What it does:**
- Shows summary of all changed files
- Shows number of lines changed in writeup.md
- Saves full diff to `writeup_diff.txt`
- Provides GitHub URL to view changes in browser

### 2. `create_simple_comparison.sh` - HTML Comparison ⭐ **RECOMMENDED**

Creates an HTML document with perfect word wrapping that you can view in browser and print to PDF.

**Usage:**
```bash
cd writeup
./Revisions\ 1/create_simple_comparison.sh
```

**What it does:**
- Extracts version from main branch
- Compares to current working version
- Creates HTML showing line-by-line changes with proper word wrapping
- Output: `comparison_document.html`

**How to use the output:**
1. Open `comparison_document.html` in your browser:
   ```bash
   firefox "writeup/Revisions 1/comparison_document.html"
   ```
2. Print to PDF from browser (Ctrl+P → Save as PDF)
3. Adjust settings as needed (margins, page size, etc.)

**Perfect for:**
- Journal submission (best readability with word wrap)
- Reviewing what changed
- Printing with custom formatting

### 3. `create_readable_comparison.sh` - Summary Document

Creates a human-readable PDF summarizing changes in plain language.

**Usage:**
```bash
cd writeup
./Revisions\ 1/create_readable_comparison.sh
```

**What it does:**
- Summarizes statistics (lines changed, additions, deletions)
- Lists modified sections
- Describes key changes in plain language
- Shows word-level differences
- Output: `readable_comparison.pdf`

**Note:** May have long line issues in code blocks. Use HTML version for better formatting.

## Manual Git Commands

### View changes in terminal:
```bash
# See summary of changes
git diff --stat main..redo_resample

# See full diff for writeup.md
git diff main..redo_resample -- writeup/writeup.md

# See word-by-word changes (easier to read)
git diff --word-diff main..redo_resample -- writeup/writeup.md

# See changes with context (10 lines before/after)
git diff -U10 main..redo_resample -- writeup/writeup.md
```

### View on GitHub:
```bash
# Get the compare URL
echo "https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/compare/main...$(git rev-parse --abbrev-ref HEAD)"
```

### Extract specific versions:
```bash
# Get version from main branch
git show main:writeup/writeup.md > writeup_main.md

# Get version from another branch
git show redo_resample:writeup/writeup.md > writeup_branch.md
```

## Workflow for Journal Revision

1. **Make your revisions** on your branch (e.g., `redo_resample`)

2. **Review what changed:**
   ```bash
   ./Revisions\ 1/compare_branches.sh
   ```

3. **Create comparison document for reviewers:**
   ```bash
   ./Revisions\ 1/create_simple_comparison_v2.sh
   # Then open comparison_document.html in browser
   # Print to PDF using browser (Ctrl+P)
   ```

4. **Create clean revised manuscript:**
   ```bash
   pandoc writeup.md --template=mytemplate.tex -o output_IEEE.pdf \
     --bibliography=refs.bib --pdf-engine=xelatex
   ```

5. **Submit to journal:**
   - Upload clean PDF as "Main Document"
   - Upload comparison PDF (from browser print) as "Supporting Document - Annotated Changes"
   - Upload reviewer response PDF as "Supporting Document - Response Letter"

## Files for Submission

Essential files ready for journal submission:

- `comments.md` / `reviewer_response.md` - Formatted response to reviewers
- `reviewer_response_JSTARS-2025-00807.pdf` - Compiled response letter (52KB)
- `comparison_document.html` - Changes document (print to PDF from browser)

## Deprecated Scripts

The following scripts are kept for reference but not recommended:

- `create_simple_comparison.sh` - PDF version with word wrap issues
- `create_tracked_changes.sh` - LaTeX tracked changes (compilation errors)
- `create_readable_comparison.sh` - Summary PDF with word wrap issues

## Troubleshooting

**Problem:** Long lines extend off the page in PDF

**Solution:** Use the HTML version instead:
```bash
./Revisions\ 1/create_simple_comparison_v2.sh
firefox "writeup/Revisions 1/comparison_document.html"
# Then Ctrl+P to print to PDF
```

**Problem:** Git can't find main branch version
```bash
# Make sure main branch exists
git branch -a | grep main

# Or use a different base branch
git show <branch-name>:writeup/writeup.md > writeup_old.md
```

**Problem:** Want side-by-side visual comparison
```bash
# Use meld or another visual diff tool
meld writeup/writeup.md <(git show main:writeup/writeup.md)
```
