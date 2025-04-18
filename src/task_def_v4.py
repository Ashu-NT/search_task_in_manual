"""
OPTIMIZED VERSION WITH SAME STRUCTURE (TO BE USED)
Changes focused on performance-critical areas while maintaining original class/method structure
"""

import os
import re
import difflib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import pypdf

from pypdf.generic import NullObject
import pdfplumber
import io
import concurrent.futures
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document type constant
MANUAL = '1001-Manual'
THRESHOLD_PERCENTAGE = 0.82
DEBUG = False

# RapidFuzz integration (if installed)
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
    logger.info("Rapid Fuzz Installed")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# =============================================================================
# Optimized Parallel Worker
# =============================================================================

def _parallel_worker(args):
    """Optimized with early exit and chunk size tuning"""
    start, end, manual_text, task, threshold, task_len, step = args
    for i in range(start, min(end, len(manual_text) - task_len + 1), step):
        window = manual_text[i:i + task_len]
        if RAPIDFUZZ_AVAILABLE:
            if fuzz.ratio(window, task) >= threshold * 100:
                return True
        else:
            if difflib.SequenceMatcher(None, window, task).quick_ratio() >= threshold:
                if difflib.SequenceMatcher(None, window, task).ratio() >= threshold:
                    return True
    return False

# =============================================================================
# Optimized DocumentProcessor
# =============================================================================

class DocumentProcessor:
    def __init__(self, doc_path: str, task_def_path: str):
        self.doc_path = doc_path
        self.task_def_path = task_def_path
        self.df_doc = None
        self.df_task_def = None
        self.manuals_df = None
        self.manual_texts: Dict[str, List[Tuple[str, str]]] = {}
        self.manual_full_texts: Dict[str, str] = {}  # Precomputed full texts
        self.page_index: Dict[str, Dict[int, str]] = {}  # Page number to text mapping

    def load_data(self) -> None:
        """Optimized data loading with vectorized operations"""
        logger.info("Running: %s.load_data", self.__class__.__name__)
        try:
            self.df_doc = pd.read_csv(self.doc_path, dtype={'CODE*': str})
            self.df_task_def = pd.read_csv(self.task_def_path, dtype={'CODE*': str})
        except Exception as e:
            logger.error("Error loading CSV files: %s", str(e))
            raise
        
        if self.df_doc.empty or self.df_task_def.empty:
            raise ValueError("Input CSV files contain no data")
        
        self._normalize_paths()
        self._filter_manuals()
        self._modify_code()

    def _normalize_paths(self) -> None:
        """Vectorized path normalization"""
        self.df_doc['normalized_path'] = self.df_doc['file path'].apply(
            lambda p: str(Path(os.path.expanduser(str(p))).resolve()) if pd.notnull(p) else None
        )
        self.df_doc.dropna(subset=['normalized_path'], inplace=True)

    def _filter_manuals(self) -> None:
        """Boolean indexing for faster filtering"""
        mask = self.df_doc['FOLDER*'] == MANUAL
        self.manuals_df = self.df_doc.loc[mask, ['IDENTIFIER*', 'CODE*', 'file path', 'DOCUMENT NAME*']].copy()
        self.manuals_df["CODE*"] = self.manuals_df["CODE*"].astype(int)
        logger.info(f"🔎 Data type of system codes in doc_df: {self.manuals_df['CODE*'].dtype}")

    def _modify_code(self) -> None:
        """Vectorized code modification"""
        self.df_task_def["code"] = self.df_task_def["CODE*"].str.split("-").str[1]
        self.df_task_def["code"] = pd.to_numeric(self.df_task_def["code"], errors='coerce').astype('Int64')
        logger.info(f"🔎 Data type of system codes in task_df: {self.df_task_def['code'].dtype}")

    @staticmethod
    def _extract_manual_text(row: pd.Series) -> Tuple[str, List[Tuple[str, str]], str]:
        """Optimized text extraction with combined processing"""
        manual_id = row['IDENTIFIER*']
        file_path = row['file path']
        pages_dict = {}
        full_text = []
        page_index = {}

        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text_parts = []
                    
                    # pypdf extraction
                    try:
                        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                        pdf_page = pdf_reader.pages[page_num-1]
                        text_py = pdf_page.extract_text() or ""
                    except Exception:
                        text_py = ""
                    
                    # pdfplumber extraction
                    text_pl = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_text = "\n".join("\t".join(filter(None, map(str, row))) for table in tables for row in table) if tables else ""
                    
                    # Combine text sources
                    combined = f"{text_py} {text_pl} {table_text}".lower().replace('\n', ' ').strip()
                    if combined:
                        pages_dict.setdefault(combined, set()).add(page_num)
                        full_text.append(combined)
                        page_index[page_num] = combined

            final_pages = sorted(
                [(", ".join(map(str, sorted(pages))), text) 
                 for text, pages in pages_dict.items()],
                key=lambda x: int(x[0].split(",")[0])
            )
            return (manual_id, final_pages, " ".join(full_text), page_index)

        except Exception as e:
            logger.error("Error processing %s: %s", manual_id, e)
            return (manual_id, [], "", {})

    def extract_manual_texts(self) -> None:
        """Optimized text extraction with full text precomputation"""
        logger.info("Running: %s.extract_manual_texts", self.__class__.__name__)
        logging.getLogger("pdfminer").setLevel(logging.ERROR)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for _, row in self.manuals_df.iterrows():
                future = executor.submit(self._extract_manual_text, row)
                futures[future] = row['IDENTIFIER*']

            for future in concurrent.futures.as_completed(futures):
                manual_id, pages, full_text, page_index = future.result()
                self.manual_texts[manual_id] = pages
                self.manual_full_texts[manual_id] = full_text
                self.page_index[manual_id] = page_index

    @staticmethod
    def _parallel_character_based_match(manual_text: str, task: str, threshold: float, step: int) -> bool:
        """Optimized parallel matching with chunk size tuning"""
        task_len = len(task)
        if task_len == 0 or len(manual_text) < task_len:
            return False

        num_workers = min(os.cpu_count() or 4, 8)  # Cap workers
        chunk_size = max(1024, len(manual_text) // (num_workers * 2))
        args_list = []

        for i in range(0, len(manual_text), chunk_size):
            args_list.append((
                i, 
                min(i + chunk_size, len(manual_text)),
                manual_text,
                task,
                threshold,
                task_len,
                step
            ))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_parallel_worker, args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    executor.shutdown(cancel_futures=True)
                    return True
        return False

    @staticmethod
    def fuzzy_match_exists(manual_text: str, task: str, threshold: float = THRESHOLD_PERCENTAGE, 
                           step_size: Optional[int] = None, use_rapidfuzz: bool = True) -> bool:
        """Optimized matching with RapidFuzz prioritization"""
        if RAPIDFUZZ_AVAILABLE and use_rapidfuzz:
            return fuzz.partial_ratio(manual_text, task) >= threshold * 100

        task_len = len(task)
        if task_len == 0:
            return False

        step = step_size or max(1, task_len // 4)
        
        return DocumentProcessor._parallel_character_based_match(manual_text, task, threshold, step)
        

# =============================================================================
# Optimized Task Processing
# =============================================================================


class TaskManager:
    # Enhanced task splitting regex for numbered tasks
    _task_split_re = re.compile(
        r'(?:^|(?<=\s))'                      # Start of string or after whitespace
        r'(?:\d+[\.\)]|step\s+\d+[:\.]?)'       # Task markers: e.g., "1.", "2)", "step 3:" 
        r'\s*'                                # Optional whitespace
        r'(.*?)'                              # Lazily capture the task text
        r'(?=\s*(?:\d+[\.\)]|step\s+\d+|$))',   # Lookahead for the next marker or end of string
        re.IGNORECASE | re.DOTALL
    )
    
    # Expanded reference removal regex
    _page_refs_re = re.compile(
        r'\b(?:page|sec|section|chap|chapter|see|fig|figure|tab|table)\b[\d\s,a-zA-Z\-]*\)?'
        r'|\([^)]*\)'       # Any parenthetical content
        r'|\{[^}]*\}'       # Curly brace content
        r'|\[.*?\]'         # Square bracket content
        r'|\\u00b0C'        # Degree Celsius symbol
        r'|\S+°\S+'         # Any degree symbol usage
        r'|\s{2,}',         # Multiple whitespace
        re.IGNORECASE
    )

    @staticmethod
    @lru_cache(maxsize=1000)
    def extract_tasks(maintenance_str: str) -> List[str]:
        """Enhanced task extraction with comprehensive cleanup"""
        # Convert to lower case
        maintenance_str = str(maintenance_str).lower()
        # If the string does not start with a recognized task marker, prepend a marker.
        if not re.match(r'^\s*(\d+[\.\)]|step\s+\d+)', maintenance_str):
            maintenance_str = "1. " + maintenance_str
        
        # Remove control characters and normalize the input.
        cleaned_str = re.sub(r'[\x00-\x1F\x7F]', ' ', maintenance_str)
        
        # Use the task splitting regex to find all tasks.
        raw_tasks = TaskManager._task_split_re.findall(cleaned_str)
        tasks = []
        for task in raw_tasks:
            # task is the captured text from our regex.
            task_text = task.strip()
            # Remove technical references and normalize whitespace.
            task_clean = TaskManager._page_refs_re.sub(' ', task_text)
            task_clean = re.sub(r'\s+', ' ', task_clean)
            task_clean = re.sub(r'^\W+|\W+$', '', task_clean)
            if task_clean:
                tasks.append(task_clean)
                
        return tasks


class TaskManualMatcher:
    def __init__(self, document_processor: DocumentProcessor):
        self.dp = document_processor
        self.df_task_def = self.dp.df_task_def
        self.regex_cache = {}

    def process_tasks(self) -> pd.DataFrame:
        """Optimized processing pipeline"""
        self._add_task_list()
        self._find_attachments()
        self._add_doc_name_page()
        return self.df_task_def

    def _add_task_list(self) -> None:
        """Vectorized task extraction"""
        self.df_task_def['tasks'] = self.df_task_def['DESCRIPTION'].apply(TaskManager.extract_tasks)

    def _find_attachments(self) -> None:
        """Optimized with system code grouping and batched processing"""
        system_groups = self.df_task_def.groupby('code', sort=False)
        results = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._process_system_group, code, group.copy()): code
                for code, group in system_groups
            }
            
            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                try:
                    results.update(future.result())
                except Exception as e:
                    logger.error(f"Error processing system code {code}: {e}")

        self.df_task_def['attachment'] = self.df_task_def.index.map(results)

    def _process_system_group(self, system_code: int, group: pd.DataFrame) -> Dict[int, str]:
        """Batch process tasks for a system code"""
        manuals = self.dp.manuals_df[self.dp.manuals_df['CODE*'] == system_code]
        if manuals.empty:
            return {idx: '' for idx in group.index}

        manual_data = {
            manual_id: self.dp.manual_full_texts.get(manual_id, '')
            for manual_id in manuals['IDENTIFIER*']
        }

        results = {}
        for idx, row in group.iterrows():
            tasks = [t for t in row['tasks'] if t]
            if not tasks:
                results[idx] = ''
                continue

            matches = set()
            for manual_id, text in manual_data.items():
                if all(self._task_match(text, t) for t in tasks):
                    matches.add(manual_id)
                elif any(self._task_match(text, t) for t in tasks):
                    matches.add(manual_id)

            results[idx] = ';'.join(sorted(matches)) if matches else ''
        
        return results

    def _task_match(self, text: str, task: str) -> bool:
        """Optimized task matching with regex caching"""
        if task in self.regex_cache:
            regex = self.regex_cache[task]
        else:
            # Split task into words, escape each, and join with \s+
            words = [re.escape(word) for word in task.split()]
            pattern = r'\b' + r'\s+'.join(words) + r'\b'
            regex = re.compile(pattern, re.IGNORECASE)
            self.regex_cache[task] = regex

        return bool(regex.search(text)) or DocumentProcessor.fuzzy_match_exists(text, task)

    def _add_doc_name_page(self) -> None:
        """
        Corrected version that properly associates tasks with their pages
        """
        logger.info("Running: %s._add_doc_name_page", self.__class__.__name__)
        
        # Precache manual names and IDs
        manual_info = self.dp.manuals_df.set_index('IDENTIFIER*')[['DOCUMENT NAME*']].to_dict('index')
        
        doc_name_pages = []
        
        for _, row in self.df_task_def.iterrows():
            entries = []
            manual_ids = row['attachment'].split(';') if row['attachment'] else []
            tasks = [t for t in row['tasks'] if t]  # Filter empty tasks
            
            for manual_id in manual_ids:
                # Get manual metadata
                if manual_id not in manual_info:
                    continue
                doc_name = manual_info[manual_id]['DOCUMENT NAME*']
                
                # Get pages for this manual
                pages_data = self.dp.manual_texts.get(manual_id, [])
                page_index = self.dp.page_index.get(manual_id, {})
                
                # Find pages for each task in this manual
                for task in tasks:
                    task_pages = set()
                    
                    # Check each page for this specific task
                    for page_num_str, page_text in pages_data:
                        if (re.search(rf'\b{re.escape(task)}\b', page_text, re.IGNORECASE) or 
                            DocumentProcessor.fuzzy_match_exists(page_text, task, THRESHOLD_PERCENTAGE)):
                            # Convert page string to individual numbers
                            page_nums = [int(n) for n in page_num_str.split(', ')]
                            task_pages.update(page_nums)
                    
                    # Alternative method using page index
                    if not task_pages:
                        for page_num, text in page_index.items():
                            if (re.search(rf'\b{re.escape(task)}\b', text, re.IGNORECASE) or 
                                DocumentProcessor.fuzzy_match_exists(text, task, THRESHOLD_PERCENTAGE)):
                                task_pages.add(page_num)
                    
                    if task_pages:
                        sorted_pages = sorted(task_pages)
                        pages_str = ', '.join(map(str, sorted_pages))
                        entries.append(
                            f"Task: '{task}' - Manual: {doc_name} (ID: {manual_id}) - Pages: {pages_str}"
                        )

            # Remove duplicate entries while preserving order
            seen = set()
            unique_entries = [e for e in entries if not (e in seen or seen.add(e))]
            doc_name_pages.append('\n'.join(unique_entries))

        self.df_task_def['DOC_NAME_PAGE'] = doc_name_pages

# =============================================================================
# Main Maintenance Processing Pipeline (Unchanged Structure)
# =============================================================================

class MaintenanceProcessor:
    def __init__(self, doc_path: str, task_def_path: str):
        self.doc_processor = DocumentProcessor(doc_path, task_def_path)
        self.matcher = None

    def process(self) -> pd.DataFrame:
        self.doc_processor.load_data()
        self.doc_processor.extract_manual_texts()
        self.matcher = TaskManualMatcher(self.doc_processor)
        return self.matcher.process_tasks()

if __name__ == "__main__":
    try:
        processor = MaintenanceProcessor(
            doc_path=r'C:\Users\ashu\Desktop\Python Workspace\training_model\data\training\batc_2_cosmos_ABB.csv',
            task_def_path=r"C:\Users\ashu\Downloads\Import_Sheet_Batch 2_V02.2_ABB.xlsx - Task definition.csv"
        )

        result_df = processor.process()
        result_df = result_df[["NAME*", 'CODE*', 'code', 'DESCRIPTION','tasks', 'attachment', 'DOC_NAME_PAGE']].copy()
        result_df.to_csv("../data/Batch_2_output.csv", index=False)
        result_df.to_excel("../data/Batch_2_output.xlsx", index=False)
        logger.info("Processing complete. Results:")
        print(result_df[["NAME*", 'CODE*', 'code', 'DESCRIPTION', 'attachment', 'DOC_NAME_PAGE']])
    except Exception as e:
        logger.error("An error occurred during processing: %s", str(e))