# Maintenance Task-to-Manual Matcher

A high-performance data processing pipeline that extracts maintenance tasks from structured task definitions and intelligently matches them with the corresponding technical manuals (PDFs). This system includes fuzzy matching and page-level indexing, making it ideal for automating document processing in industrial and maintenance environments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [User Guide](#user Guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

The **Maintenance Task-to-Manual Matcher** is designed to automate the process of associating maintenance tasks (extracted from task definition spreadsheets) with the corresponding sections in technical manuals (provided as PDFs). The system uses:

- **Optimized data loading**: Fast CSV import with vectorized operations.
- **Parallel PDF text extraction**: Uses `ThreadPoolExecutor` for concurrent extraction of text from PDFs.
- **Advanced text matching**: Combines exact matching (via regex) with fuzzy matching (using RapidFuzz or difflib).
- **Detailed output**: Exports a consolidated report with task details, matched manuals, and page references.

---

## Features

- **CSV Data Loading & Normalization**
  - Reads document metadata and task definitions.
  - Normalizes file paths and filters manuals based on document type.

- **Optimized PDF Text Extraction**
  - Extracts text from PDFs using both `pdfplumber` and `pypdf`.
  - Precomputes full text with page-level indexing for later matching.

- **Task Extraction and Cleaning**
  - Splits and cleans task definitions using precompiled regex patterns.
  - Handles both numbered tasks (e.g., `1. Replace filter`) and unnumbered tasks.

- **Manual-to-Task Matching**
  - Uses a combination of exact regex matching (with caching) and fuzzy matching.
  - Matches tasks to manuals and determines relevant page numbers.

- **Concurrency for Performance**
  - Utilizes `ThreadPoolExecutor` and `ProcessPoolExecutor` for parallel processing.
  - Designed to scale with large datasets and multiple documents.

- **Export Capabilities**
  - Outputs results in CSV and Excel formats with complete task and manual references.

---

## Project Structure

```plaintext
.
├── src/
│   ├── __init__.py
│   └── processor.py          # Core application code (DocumentProcessor, TaskManager, TaskManualMatcher, etc.)
├── tests/
│   ├── unit/                 # Component-level tests
│   │   ├── document_processor/  # Tests for data loading & PDF processing
│   │   ├── task_manager/        # Tests for task extraction logic
│   │   └── matcher/             # Tests for manual-task matching
│   ├── integration/          # End-to-end workflow tests
│   ├── utils/                # Infrastructure tests
|   ├── __init__.py
|   └── conftest.py
├── data/                     # Data files and outputs (CSV/Excel)
├── README.md                 # This documentation file
├── requirements.txt          # List of dependencies
└── main.py                   # Entry point for running the full pipeline
```

---

## User Guide 📖

### Input Requirements
#### Documents CSV
| Column | Type | Description |
|--------|------|-------------|
| IDENTIFIER* | String | Unique document ID |
| FOLDER* | String | Must be '1001-Manual' for processing |
| CODE* | Integer | System identifier (1-9999) |
| file path | String | Absolute path to PDF file |
| DOCUMENT NAME* | String | Human-readable document name |

#### Output Columns
- **attachment**: Matching manual IDs
- **DOC_NAME_PAGE**: Formatted page references
- **match_score**: Fuzzy match percentage

---

## Troubleshooting
```text
ERROR: InvalidSystemCode → Verify CSV CODE* columns contain integers
WARNING: NoTextExtracted → Check PDF is text-based (not scanned)