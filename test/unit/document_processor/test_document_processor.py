import pytest
import pandas as pd
from unittest.mock import patch

def test_document_processor_initialization(sample_document_processor):
    assert sample_document_processor is not None
    assert isinstance(sample_document_processor.df_doc, pd.DataFrame)

def test_load_data(sample_document_processor):
    with patch('pandas.read_csv') as mock_read:
        mock_read.side_effect = [
            pd.DataFrame({'IDENTIFIER*':["id01"],'DOCUMENT NAME*': ['doc1'],'FOLDER*': ['1001-Manual'], 'CODE*': ['1'], 'file path': ['test.pdf']}),
            pd.DataFrame({'CODE*': ['01-1-S'], 'DESCRIPTION': ['Test task']})
        ]
        sample_document_processor.load_data()
        
    assert sample_document_processor.manuals_df is not None
    assert 'CODE*' in sample_document_processor.manuals_df.columns

def test_pdf_text_extraction(sample_document_processor, mock_pdf, tmpdir, mocker):
    test_text = "Sample PDF Content"
    pdf_bytes = mock_pdf(text=test_text)
    temp_file = tmpdir.join("test.pdf")
    temp_file.write_binary(pdf_bytes.getvalue())

    # Mock PDF parsing libraries to return expected text
    mock_page = mocker.MagicMock()
    mock_page.extract_text.return_value = test_text
    mock_pdf_plumber = mocker.patch('pdfplumber.open')
    mock_pdf_plumber.return_value.__enter__.return_value.pages = [mock_page]

    mock_pypdf_page = mocker.MagicMock()
    mock_pypdf_page.extract_text.return_value = test_text
    mock_pypdf_reader = mocker.patch('pypdf.PdfReader')
    mock_pypdf_reader.return_value.pages = [mock_pypdf_page]

    # Update paths and process
    sample_document_processor.manuals_df.loc[:,'file path'] = str(temp_file)
    sample_document_processor.extract_manual_texts()

    manual_id = sample_document_processor.manuals_df['IDENTIFIER*'].iloc[0]
    extracted_text = sample_document_processor.manual_full_texts[manual_id]
    
    # Normalize whitespace and check containment
    assert test_text.lower() in " ".join(extracted_text.split())

def test_folder_filtering(sample_document_processor):
    # Store original FOLDER* values for verification
    original_folders = sample_document_processor.df_doc['FOLDER*'].copy()
    
    # Perform filtering
    sample_document_processor._filter_manuals()
    
    # Verify filtering using the original data
    expected_count = sum(original_folders == '1001-Manual')
    
    # Check 1: Verify correct number of manuals
    assert len(sample_document_processor.manuals_df) == expected_count
    
    # Check 2: Verify filtered IDs match original data
    original_manual_ids = sample_document_processor.df_doc.loc[
        original_folders == '1001-Manual', 'IDENTIFIER*'
    ].tolist()
    
    assert all(
        manual_id in original_manual_ids
        for manual_id in sample_document_processor.manuals_df['IDENTIFIER*']
    )