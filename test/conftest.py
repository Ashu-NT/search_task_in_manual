import pytest
import pandas as pd
from unittest.mock import MagicMock
from io import BytesIO
import pypdf
import pdfplumber

@pytest.fixture
def mock_pdf():
    def _create_mock_pdf(text="", pages=1):
        pdf_bytes = BytesIO()
        writer = pypdf.PdfWriter()
        for _ in range(pages):
            page = pypdf.PageObject.create_blank_page(None, 612, 792)
            page.extract_text = lambda: text
            writer.add_page(page)
        writer.write(pdf_bytes)
        pdf_bytes.seek(0)
        return pdf_bytes
    return _create_mock_pdf

@pytest.fixture
def sample_document_processor():
    from src.task_def_v4 import DocumentProcessor  
    processor = DocumentProcessor("dummy.csv", "dummy_tasks.csv")
    processor.df_doc = pd.DataFrame({
        'IDENTIFIER*': ['id01','id02','id03'],
        'DOCUMENT NAME*': ['doc1','doc2','doc3'],
        'FOLDER*': ['1001-Manual', 'Other', '1001-Manual'],
        'CODE*': ['1', '2', '3'],
        'file path': ['/valid/path1.pdf', '/valid/path2.pdf', '/valid/path3.pdf']
    })
    
    processor.manuals_df = processor.df_doc[processor.df_doc['FOLDER*'] == '1001-Manual']
    return processor