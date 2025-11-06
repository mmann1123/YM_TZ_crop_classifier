#!/bin/bash
# Script to compare branches using various methods
# Useful for seeing what changed between main and current branch

BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
MAIN_BRANCH="main"

echo "=== Comparing branches: $MAIN_BRANCH vs $BRANCH_NAME ==="
echo ""

# Method 1: Show summary of changed files
echo "1. FILES CHANGED:"
echo "----------------------------------------"
git diff --stat $MAIN_BRANCH..$BRANCH_NAME
echo ""

# Method 2: Show changes to specific file (writeup.md)
echo "2. CHANGES TO writeup/writeup.md:"
echo "----------------------------------------"
if git diff --quiet $MAIN_BRANCH..$BRANCH_NAME -- writeup/writeup.md; then
    echo "No changes to writeup.md"
else
    echo "Number of lines changed:"
    git diff --numstat $MAIN_BRANCH..$BRANCH_NAME -- writeup/writeup.md
    echo ""
    echo "Use 'git diff $MAIN_BRANCH..$BRANCH_NAME -- writeup/writeup.md' to see full diff"
fi
echo ""

# Method 3: Save diff to file
echo "3. SAVING DIFF TO FILE:"
echo "----------------------------------------"
git diff $MAIN_BRANCH..$BRANCH_NAME -- writeup/writeup.md > "Revisions 1/writeup_diff.txt"
echo "✓ Full diff saved to: Revisions 1/writeup_diff.txt"
echo ""

# Method 4: Create side-by-side comparison files
echo "4. CREATING SIDE-BY-SIDE FILES:"
echo "----------------------------------------"
# Get the repo root directory
REPO_ROOT=$(git rev-parse --show-toplevel)
git show $MAIN_BRANCH:writeup/writeup.md > "$REPO_ROOT/writeup/Revisions 1/writeup_main_version.md"
cp "$REPO_ROOT/writeup/writeup.md" "$REPO_ROOT/writeup/Revisions 1/writeup_current_version.md"
echo "✓ Main branch version saved to: Revisions 1/writeup_main_version.md"
echo "✓ Current version saved to: Revisions 1/writeup_current_version.md"
echo ""

# Method 5: GitHub compare URL
echo "5. GITHUB WEB COMPARISON:"
echo "----------------------------------------"
REPO_URL=$(git config --get remote.origin.url | sed 's/\.git$//' | sed 's/git@github.com:/https:\/\/github.com\//')
echo "View changes in browser:"
echo "$REPO_URL/compare/$MAIN_BRANCH...$BRANCH_NAME"
echo ""

echo "=== COMPARISON COMPLETE ==="
echo ""
echo "To create a tracked-changes PDF (with red deletions, blue additions):"
echo "  ./create_tracked_changes.sh"
echo ""
echo "To view diff with word-level highlighting:"
echo "  git diff --word-diff $MAIN_BRANCH..$BRANCH_NAME -- writeup/writeup.md"
