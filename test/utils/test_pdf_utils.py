import pytest
from src.task_def_v4 import DocumentProcessor 

class TestPDFUtils:
    def test_page_indexing(self, mock_pdf):
        processor = DocumentProcessor("dummy.csv", "dummy_tasks.csv")
        processor.page_index = {'test': {1: "page content"}}
        assert 1 in processor.page_index['test']