"""
11_multimodal_anthropic.py
==========================
Concept: Multimodal Inputs
--------------------------
Claude can read images and PDFs alongside text. You base64-encode the file
and include it in the messages array. The model sees the visual content
and can reason about it.

Data engineering use cases:
- Parse PDF invoices/reports into structured JSON
- Read a Tableau/QuickSight screenshot and describe the trend
- Extract table data from scanned documents
- Read an ERD diagram screenshot and write CREATE TABLE statements
- Analyse dashboard screenshots for anomalies

Install: pip install anthropic Pillow requests
Set env:  export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic
import base64
import json
import io
from pathlib import Path

client = anthropic.Anthropic()


def encode_image_file(path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    suffix = Path(path).suffix.lower()
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_types.get(suffix, "image/png")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


def encode_pdf_file(path: str) -> str:
    """Read a PDF file and return base64_data."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


# ── Demo 1: Analyse a chart/dashboard image ───────────────────────────────────
def analyse_chart_demo(image_path: str):
    """
    Pass a screenshot of a dashboard chart to Claude.
    It can describe trends, identify anomalies, and extract data points.

    Usage: provide a path to any chart PNG/JPEG on your machine.
    """

    print("=== Chart Analysis Demo ===\n")

    image_data, media_type = encode_image_file(image_path)

    response = client.messages.create(
        model="claude-sonnet-4-6",  # vision tasks work better with Sonnet
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """You are a data analyst. Analyse this chart and provide:
1. What metric is being shown
2. The overall trend (up/down/flat/volatile)
3. Any anomalies or notable data points
4. A one-sentence summary for a standup report

Respond as JSON: {"metric": "", "trend": "", "anomalies": [], "summary": ""}"""
                }
            ]
        }]
    )

    result = json.loads(response.content[0].text)
    print(json.dumps(result, indent=2))


# ── Demo 2: Extract data from a PDF report ────────────────────────────────────
def extract_pdf_data_demo(pdf_path: str):
    """
    Pass a PDF report to Claude and extract structured data.
    Works on invoices, financial reports, scanned tables, etc.
    """

    print("=== PDF Data Extraction Demo ===\n")

    pdf_data = encode_pdf_file(pdf_path)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {
                    "type": "text",
                    "text": """Extract all tables from this document as structured JSON.
For each table, return:
{
  "tables": [
    {
      "title": "table heading if present",
      "columns": ["col1", "col2", ...],
      "rows": [["val1", "val2", ...], ...]
    }
  ]
}
Return ONLY valid JSON."""
                }
            ]
        }]
    )

    try:
        result = json.loads(response.content[0].text)
        print(f"Extracted {len(result.get('tables', []))} table(s)")
        for i, table in enumerate(result.get("tables", []), 1):
            print(f"\nTable {i}: {table.get('title', 'Untitled')}")
            print(f"Columns: {table.get('columns', [])}")
            print(f"Rows: {len(table.get('rows', []))}")
    except json.JSONDecodeError:
        print("Raw output:", response.content[0].text[:500])


# ── Demo 3: Read an ERD image and generate CREATE TABLE statements ─────────────
def erd_to_sql_demo(image_path: str):
    """
    Screenshot your ERD diagram and have Claude write the SQL DDL.
    Useful for documenting legacy systems or onboarding.
    """

    print("=== ERD to SQL Demo ===\n")

    image_data, media_type = encode_image_file(image_path)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """You are a data engineer. Look at this ERD diagram and:
1. Write Snowflake CREATE TABLE statements for each entity you can see
2. Include primary keys, foreign keys, and data types (infer from column names)
3. Add comments explaining each table

Return only valid SQL."""
                }
            ]
        }]
    )

    print(response.content[0].text)


# ── Demo 4: URL-based image (no file upload needed) ───────────────────────────
def url_image_demo():
    """
    You can also pass image URLs directly — no base64 encoding needed.
    Useful for images already hosted in S3, GCS, or public URLs.
    """

    print("=== URL-based Image Demo ===\n")

    # Use a public sample chart image
    # In production: use your S3/GCS presigned URLs
    sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bimodal_bivariate_small.png/220px-Bimodal_bivariate_small.png"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": sample_image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image in 2 sentences."
                }
            ]
        }]
    )

    print(response.content[0].text)


# ── Demo 5: Multi-image comparison ────────────────────────────────────────────
def compare_dashboards_demo(image_path_1: str, image_path_2: str):
    """
    Compare two dashboard screenshots — e.g. today vs yesterday.
    Claude can spot differences, regressions, and anomalies.
    """

    print("=== Dashboard Comparison Demo ===\n")

    img1_data, type1 = encode_image_file(image_path_1)
    img2_data, type2 = encode_image_file(image_path_2)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Dashboard from yesterday:"},
                {"type": "image", "source": {"type": "base64", "media_type": type1, "data": img1_data}},
                {"type": "text", "text": "Dashboard from today:"},
                {"type": "image", "source": {"type": "base64", "media_type": type2, "data": img2_data}},
                {"type": "text", "text": "What are the key differences between these two dashboards? Flag any potential data quality issues."}
            ]
        }]
    )

    print(response.content[0].text)


# ── Main: run the demos that don't need local files ───────────────────────────

if __name__ == "__main__":
    # Demo 4 works without local files
    url_image_demo()

    print("\n--- Other demos require local files ---")
    print("To use them:")
    print("  analyse_chart_demo('path/to/chart.png')")
    print("  extract_pdf_data_demo('path/to/report.pdf')")
    print("  erd_to_sql_demo('path/to/erd_screenshot.png')")
    print("  compare_dashboards_demo('yesterday.png', 'today.png')")
