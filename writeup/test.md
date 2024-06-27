---
title: "The document title"
author:
- name: Michael L. Mann
  affiliation: The George Washington University, Washington DC 20052
  thanks: Corresponding author. Email mmann1123@gmail.com
- name: Lisa Colson
  affiliation: USDA Foreign Agricultural Service, Washington DC 20250
header-includes:
  - |
    ```{=latex}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    \rfoot{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancypagestyle{plain}{
      \fancyhf{}
      \rfoot{\thepage}
      \renewcommand{\headrulewidth}{0pt}
      \renewcommand{\footrulewidth}{0pt}
    }
    ```
abstract: |
  This is the abstract of the document. It contains a brief summary of the content and objectives of the document.
  It consists of two paragraphs.

  This is the second paragraph of the abstract.
bibliography: ["refs.bib"]
---

This is a sample document. Here is a citation to an example article [@example2023].

\newpage
# Bibliography


<!-- compile working with:
pandoc test.md --template=mytemplate.tex -o output.pdf --bibliography=refs.bib --pdf-engine=xelatex --citeproc -->