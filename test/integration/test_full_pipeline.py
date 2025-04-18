from reportlab.pdfgen import canvas
from io import BytesIO
import pandas as pd

def test_full_pipeline(tmpdir):
    from src.task_def_v4 import MaintenanceProcessor
    
    # Generate PDF with extractable text using reportlab
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    text = "The following is a maintenance task: greasing maintenance, There checking oil"
    c.drawString(50, 700, text)  # Add text at position (50, 700)
    c.save()
    pdf_bytes = pdf_buffer.getvalue()
    
    temp_file = tmpdir.join("test.pdf")
    temp_file.write_binary(pdf_bytes)
    
    # Create document CSV
    temp_doc_csv = tmpdir.join("documents.csv")
    doc_df = pd.DataFrame({
        'IDENTIFIER*': ['id01', 'id02', 'id03'],
        'FOLDER*': ['1001-Manual', '1001-Manual', 'other'],
        'CODE*': ['1', '1', '2'],
        'file path': [str(temp_file)] * 3,
        'DOCUMENT NAME*': ['Test Manual.pdf', 'Manual.pdf', 'Other.pdf']
    })
    doc_df.to_csv(temp_doc_csv, index=False)
    
    # Create tasks CSV with matching code and task
    temp_tasks_csv = tmpdir.join("tasks.csv")
    tasks_df = pd.DataFrame({
        'CODE*': ['A10-1-H'],  # Extracts to system code 1
        'DESCRIPTION': ['1) greasing maintenance']  # Exact match in PDF
    })
    tasks_df.to_csv(temp_tasks_csv, index=False)
    
    # Execute pipeline
    processor = MaintenanceProcessor(str(temp_doc_csv), str(temp_tasks_csv))
    result_df = processor.process()
    
    # Validate results
    assert not result_df.empty
    assert 'attachment' in result_df.columns
    assert 'DOC_NAME_PAGE' in result_df.columns
    
    attachment_value = result_df['attachment'].iloc[0]
    assert 'id01' in attachment_value and 'id02' in attachment_value, "Expected manuals not attached."
    
    doc_entry = result_df['DOC_NAME_PAGE'].iloc[0]
    assert 'Test Manual.pdf' in doc_entry and 'Manual.pdf' in doc_entry, "Document names missing from DOC_NAME_PAGE"
    assert 'Pages: 1' in doc_entry, "Page numbers not correctly identified"