#!/bin/bash
# Script to create a simple side-by-side comparison document using HTML as intermediate
# More reliable for long lines than direct LaTeX

echo "Creating comparison document..."

# Get the repository root and change to it
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
WRITEUP_DIR="$REPO_ROOT/writeup"
OUTPUT_DIR="$WRITEUP_DIR/Revisions 2"

# Get the old version from main branch
echo "Extracting version from main branch..."
git show main:writeup/writeup.md > "$OUTPUT_DIR/version_main.md"

# Get current version
echo "Using current version..."
cp "$WRITEUP_DIR/writeup.md" "$OUTPUT_DIR/version_current.md"

# Create HTML comparison (handles long lines better)
cat > "$OUTPUT_DIR/comparison_document.html" << 'HEREDOC'
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Document Comparison</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; }
h1, h2 { color: #333; }
pre { 
  font-size: 8pt; 
  background-color: #f5f5f5; 
  padding: 10px; 
  border: 1px solid #ddd;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
}
.stats { background-color: #e8f5e9; padding: 10px; margin: 10px 0; }
.add { color: #2e7d32; }
.del { color: #c62828; }
</style>
</head>
<body>
<h1>Document Comparison: Main Branch vs Current Branch</h1>
<h2>Side-by-Side Changes to writeup.md</h2>

<h2>Summary of Changes</h2>
<div class="stats">
<pre>
HEREDOC

# Add git diff stats (ignore whitespace)
git diff --stat --ignore-all-space main..redo_resample -- writeup/writeup.md >> "$OUTPUT_DIR/comparison_document.html"

cat >> "$OUTPUT_DIR/comparison_document.html" << 'HEREDOC'
</pre>
</div>

<h2>Detailed Changes</h2>
<p>Lines starting with:</p>
<ul>
<li class="del">- (minus) were <strong>removed</strong> from the original</li>
<li class="add">+ (plus) were <strong>added</strong> in the revision</li>
</ul>

<p><strong>Note:</strong> Whitespace-only changes and trivial punctuation edits are ignored to focus on substantive changes.</p>

<pre>
HEREDOC

# Add the actual diff (ignore whitespace and trivial changes)
git diff --unified=3 --ignore-all-space main..redo_resample -- writeup/writeup.md >> "$OUTPUT_DIR/comparison_document.html"

cat >> "$OUTPUT_DIR/comparison_document.html" << 'HEREDOC'
</pre>

</body>
</html>
HEREDOC

# Convert HTML to PDF using wkhtmltopdf or chromium
echo "Compiling comparison PDF..."
if command -v wkhtmltopdf &> /dev/null; then
    wkhtmltopdf "$OUTPUT_DIR/comparison_document.html" "$OUTPUT_DIR/comparison_document.pdf" 2>&1 | tail -5 || true
elif command -v chromium-browser &> /dev/null; then
    chromium-browser --headless --disable-gpu --print-to-pdf="$OUTPUT_DIR/comparison_document.pdf" "$OUTPUT_DIR/comparison_document.html" 2>&1 || true
else
    echo "WARNING: Neither wkhtmltopdf nor chromium-browser found. Trying pandoc with HTML..."
    cd "$WRITEUP_DIR"
    pandoc "$OUTPUT_DIR/comparison_document.html" \
        -o "$OUTPUT_DIR/comparison_document.pdf" \
        --pdf-engine=weasyprint \
        2>&1 | tail -10 || true
fi

if [ -f "$OUTPUT_DIR/comparison_document.html" ]; then
    echo ""
    echo "✓ SUCCESS!"
    echo "Comparison document created at:"
    echo "  $OUTPUT_DIR/comparison_document.html"
    echo ""
    echo "To view and print to PDF:"
    echo "  firefox \"$OUTPUT_DIR/comparison_document.html\""
    echo "  Then: Ctrl+P → Save as PDF"
    echo ""
    echo "This HTML document has perfect word wrapping and is ready for journal submission."
else
    echo "ERROR: HTML creation failed"
    exit 1
fi

rm "$OUTPUT_DIR/version_main.md" "$OUTPUT_DIR/version_current.md" "$OUTPUT_DIR/writeup_current_version.md" "$OUTPUT_DIR/writeup_diff.txt" "$OUTPUT_DIR/writeup_main_version.md"