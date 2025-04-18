"""
    Before importing the two data frames, they should have the following columns (headings):
    1) DOCUMENT: ["IDENTIFIER*"] | ["FOLDER*"] | ["file path"] | ["CODE*"] | ["DOCUMENT NAME*"]
    2) TASK DEFINITION: ["DESCRIPTION"] | ["CODE*"]

    IMPORTANT!!!!
    CHECK THE FOLLOWING LINES BEFORE RUNNING:
    LINES 198 --> 198 | 292 --> 294  FOR DEBUG PURPOSES
    LINES: 31 ---> 41 | 94,106 | 413| 441 --> 459
"""

import os
import re
import difflib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import pandas as pd
import PyPDF2
from PyPDF2.generic import NullObject  # used to check for NullObject responses
import pdfplumber
import io

import concurrent.futures
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document type constant
MANUAL = '1001-Manual'
THRESHOLD_PERCENTAGE = 0.55

# For debuging change to TRUE
debug = False

# =============================================================================
# Parallel Worker for Fuzzy Matching
# =============================================================================

def _parallel_worker(args):
    """
    Worker function for parallel character-based fuzzy matching.
    
    Args:
        args: A tuple containing:
            start (int): Starting index for this chunk.
            end (int): Ending index for this chunk.
            manual_text (str): The text in which to search.
            task (str): The phrase to search for.
            threshold (float): The similarity threshold.
            task_len (int): Length of the task.
            step (int): Sliding window step size.
    
    Returns:
        bool: True if a similar match is found in this chunk; otherwise, False.
    """
    start, end, manual_text, task, threshold, task_len, step = args
    for i in range(start, end, step):
        if i + task_len > len(manual_text):
            break
        window = manual_text[i:i + task_len]
        similarity = difflib.SequenceMatcher(None, window, task).ratio()
        if similarity >= threshold:
            return True  # Early exit: match found in this chunk
    return False

# =============================================================================
# Documents Processing Components
# =============================================================================

class DocumentProcessor:
    def __init__(self, doc_path: str, task_def_path: str):
        """
        :param doc_path: Path to the documents CSV file.
        :param task_def_path: Path to the task definition CSV file.
        """
        self.doc_path = doc_path
        self.task_def_path = task_def_path
        self.df_doc = None
        self.df_task_def = None
        self.manuals_df = None
        # Dictionary to store manual texts {manual_id: List of (page_number, page_text)}
        self.manual_texts: Dict[str, List[Tuple[int, str]]] = {}

    def load_data(self) -> None:
        """Load and preprocess input data."""
        logger.info("Running: %s.load_data", self.__class__.__name__)
        try:
            self.df_doc = pd.read_csv(self.doc_path)
            self.df_task_def = pd.read_csv(self.task_def_path)
        except Exception as e:
            logger.error("Error loading CSV files: %s", str(e))
            raise

        self._normalize_paths()
        self._filter_manuals()
        self._modify_code()

    def _normalize_paths(self) -> None:
        """Normalize file paths in the documents DataFrame."""
        logger.info("Running: %s._normalize_paths", self.__class__.__name__)

        def normalize_path(path):
            try:
                return Path(os.path.normpath(os.path.expanduser(str(path)))).resolve().as_posix()
            except Exception as e:
                logger.error("Invalid path: %s - %s", path, str(e))
                return None

        self.df_doc['normalized_path'] = self.df_doc['file path'].apply(normalize_path)
        self.df_doc = self.df_doc.dropna(subset=['normalized_path'])

    def _filter_manuals(self) -> None:
        """Filter documents to only include manuals."""
        logger.info("Running: %s._filter_manuals", self.__class__.__name__)
        self.manuals_df = self.df_doc[self.df_doc['FOLDER*'] == MANUAL].copy()
        # Keep necessary columns for later processing
        self.manuals_df = self.manuals_df[['IDENTIFIER*', 'CODE*', 'file path', 'DOCUMENT NAME*']]
        logger.info(f"🔎 Data type of system codes in doc_df: {self.manuals_df['CODE*'].dtype}")

    def _modify_code(self) -> None:
        """
        Modify the task definition 'CODE*' field.
        Splits the code on '-' and uses the second part if available.
        """
        logger.info("Running: %s._modify_code", self.__class__.__name__)
        self.df_task_def.loc[:, "code"] = self.df_task_def["CODE*"].apply(
            lambda x: str(x).split("-")[1] if len(str(x).split("-")) >= 1 else None
        )
        self.df_task_def['code'] = self.df_task_def['code'].astype(int)
        logger.info(f"🔎 Data type of system codes in task_df: {self.df_task_def['code'].dtype}")

    @staticmethod
    def _extract_manual_text(row: pd.Series) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract text (including table content) from a single manual.
        Returns a tuple of (manual_id, final_pages) where final_pages is a list
        of tuples (pages_str, text).
        """
        manual_id = row['IDENTIFIER*']
        file_path = row['file path']
        pages_dict = {}  # key: cleaned text, value: set of page numbers
        try:
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data), strict=False)
            except Exception as e:
                logger.error("Error reading PDF via PyPDF2 for %s: %s", manual_id, e)
                pdf_reader = None

            with pdfplumber.open(file_path) as pdf_plumber_obj:
                for page_num, page in enumerate(pdf_plumber_obj.pages, start=1):
                    combined_text = ""
                    text_py = ""
                    if pdf_reader is not None and len(pdf_reader.pages) >= page_num:
                        try:
                            text_py = pdf_reader.pages[page_num - 1].extract_text() or ""
                            if isinstance(text_py, NullObject):
                                text_py = ""
                        except Exception as e:
                            logger.error("Error extracting PyPDF2 text on page %d for %s: %s", page_num, manual_id, e)
                    combined_text += text_py + " "
                    try:
                        text_pl = page.extract_text() or ""
                        combined_text += text_pl + " "
                        table = page.extract_table()
                        if table:
                            table_text = "\n".join(
                                "\t".join(filter(None, map(str, row_data)))
                                for row_data in table if any(row_data)
                            )
                            combined_text += table_text + " "
                    except Exception as e:
                        logger.error("Error processing pdfplumber page %d for %s: %s", page_num, manual_id, e)
                    
                    cleaned_text = combined_text.lower().replace('\n', ' ').strip()
                    if cleaned_text:
                        if cleaned_text in pages_dict:
                            pages_dict[cleaned_text].add(page_num)
                        else:
                            pages_dict[cleaned_text] = {page_num}
            final_pages = []
            for text, page_nums in pages_dict.items():
                sorted_pages = sorted(page_nums)
                pages_str = ", ".join(str(num) for num in sorted_pages)
                final_pages.append((pages_str, text))
            final_pages = sorted(final_pages, key=lambda tup: int(tup[0].split(",")[0]))
            return (manual_id, final_pages)
        except Exception as e:
            logger.error("Error opening or processing %s: %s", manual_id, e)
            return (manual_id, [])

    def extract_manual_texts(self) -> None:
        """
        Extract text (including table content) from PDF manuals using both PyPDF2 and pdfplumber.
        For each manual, text is extracted via both methods, combined, and cleaned.
        Duplicate page texts are removed by merging their page numbers (e.g. "1, 3").
        The final output is stored as a list of tuples (pages_str, text) for each manual.
        This version uses parallel processing to speed up extraction.
        """
        logger.info("Running: %s.extract_manual_texts", self.__class__.__name__)
        
        # Suppress pdfminer warnings (CropBox missing)
        logging.getLogger("pdfminer.pdfpage").setLevel(logging.ERROR)
        
        # Use ThreadPoolExecutor to process each manual in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(DocumentProcessor._extract_manual_text, row): row 
                       for _, row in self.manuals_df.iterrows()}
            for future in concurrent.futures.as_completed(futures):
                manual_id, final_pages = future.result()
                self.manual_texts[manual_id] = final_pages
        
        # For debugging
        if debug:        
            print(self.manual_texts.keys())
            for manual_id, pages in self.manual_texts.items():
                for page_num, text in pages:
                    print(f"Manual: {manual_id}, Page {page_num}: {text[:500]}")  # Print first 500 chars    

    @staticmethod
    def _character_based_match(manual_text: str, task: str, threshold: float, step: int) -> bool:
        """
        Perform a sliding-window character-based fuzzy match.
        The loop returns immediately after a match is found.
        
        Args:
            manual_text: The text in which to search.
            task: The substring or phrase to search for.
            threshold: The similarity threshold.
            step: The step size for sliding the window.
        
        Returns:
            bool: True if a similar match is found; otherwise, False.
        """
        task_len = len(task)
        for i in range(0, len(manual_text) - task_len + 1, step):
            window = manual_text[i:i + task_len]
            similarity = difflib.SequenceMatcher(None, window, task).ratio()
            if similarity >= threshold:
                return True  # Early exit: match found
        return False

    @staticmethod
    def _parallel_character_based_match(manual_text: str, task: str, threshold: float, step: int) -> bool:
        """
        Parallelized sliding-window character-based fuzzy match using multiple processes.
        Returns as soon as any worker finds a match.
        
        Args:
            manual_text: The text in which to search.
            task: The substring or phrase to search for.
            threshold: The similarity threshold.
            step: The step size for sliding the window.
        
        Returns:
            bool: True if a similar match is found; otherwise, False.
        """
        task_len = len(task)
        total_length = len(manual_text) - task_len + 1
        if total_length <= 0:
            return False

        num_workers = os.cpu_count() or 1
        iterations = (total_length + step - 1) // step  # ceiling division
        chunk_size = max(1, iterations // num_workers)

        args_list = []
        for chunk in range(num_workers):
            start_idx = chunk * chunk_size * step
            end_idx = min(len(manual_text), start_idx + chunk_size * step)
            if start_idx >= total_length:
                break
            args_list.append((start_idx, end_idx, manual_text, task, threshold, task_len, step))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_parallel_worker, args) for args in args_list]
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        executor.shutdown(cancel_futures=True)
                        return True  # Early exit: match found in one of the parallel chunks
                except Exception as e:
                    logger.error(f"Error in parallel worker: {e}")
        return False

    @staticmethod
    def _word_based_match(manual_text: str, task: str, threshold: float) -> bool:
        """
        Fallback word-based fuzzy match.
        
        Args:
            manual_text: The text to search within.
            task: The phrase to search for.
            threshold: The similarity threshold.
        
        Returns:
            bool: True if a similar match is found; otherwise, False.
        """
        clean_text = re.sub(r'\W+', ' ', manual_text).strip()
        words = clean_text.split()
        task_words = task.split()
        window_size = len(task_words)
        if window_size > 0 and len(words) >= window_size:
            word_step = max(1, window_size // 2)
            for i in range(0, len(words) - window_size + 1, word_step):
                word_window = ' '.join(words[i:i + window_size])
                similarity = difflib.SequenceMatcher(None, word_window, task).ratio()
                if similarity >= threshold:
                    return True
        return False

    @staticmethod
    def fuzzy_match_exists(manual_text: str, task: str, threshold: float = THRESHOLD_PERCENTAGE, 
                           step_size: Optional[int] = None, use_rapidfuzz: bool = True,
                           parallel: bool = True) -> bool:
        """
        Check if a given task string exists in manual_text with a similarity above threshold.
        This method slides a window over the text to find a fuzzy match.
        It attempts to use RapidFuzz for performance, falling back to custom matching if necessary.
        The character-based matching is optimized for parallel processing if enabled.
        
        Args:
            manual_text: Text to search within.
            task: Phrase to search for.
            threshold: Similarity threshold (0-1).
            step_size: Window sliding step size (None to compute dynamically).
            use_rapidfuzz: Whether to try using RapidFuzz for matching.
            parallel: Whether to use parallel processing for character-based matching.
        
        Returns:
            bool: True if a similar match is found; otherwise, False.
        """
        manual_text = manual_text.lower()
        task = task.lower()
        task_len = len(task)
        if task_len == 0 or len(manual_text) < task_len:
            return False

        # Quick exact word check using simple substring search
        if manual_text.startswith(task + ' ') or manual_text.endswith(' ' + task) or f' {task} ' in manual_text:
            return True

        # Attempt to use RapidFuzz if enabled
        if use_rapidfuzz:
            try:
                from rapidfuzz import fuzz
                if fuzz.partial_ratio(manual_text, task) >= threshold * 100:
                    return True
            except ImportError:
                logger.warning("RapidFuzz not installed; falling back to custom fuzzy matching.")

        # Calculate dynamic step size if not provided
        step = step_size or max(1, task_len // 4)

        # Use parallel or sequential character-based matching based on flag
        if parallel:
            if DocumentProcessor._parallel_character_based_match(manual_text, task, threshold, step):
                return True
        else:
            if DocumentProcessor._character_based_match(manual_text, task, threshold, step):
                return True

        # Fallback to word-based matching
        return DocumentProcessor._word_based_match(manual_text, task, threshold)

# =============================================================================
# Task Processing Components
# =============================================================================

class TaskManager:
    _task_split_re = re.compile(r'(?:(?:\d+[\.\)])\s*)?(.*?)(?=\s*\d+[\.\)]|$)', re.DOTALL)
    _page_refs_re = re.compile(
        r'\(page \d+\)|'
        r'see page \d+|'
        r'\(see chapter \d+(\.\d+)*, page \d+\)|'
        r'chapter \d+(\.\d+)*|'
        r'\(chapter \d+(\.\d+)*( \/ page \d+(-\d+)*)?\)|'
        r'\(section \d+( page \d+)?(, function description)?\)|'
        r'\(\s*3a[\w\d-]*\s*\)?|'
        r'\(abb\)|'
        r'(\([^)]+\)?)(\s*(\([^)]+\)?))+',  # Consecutive parentheses pattern
        re.IGNORECASE
    )

    @staticmethod
    @lru_cache(maxsize=1000)
    def extract_tasks(maintenance_str: str) -> List[str]:
        """Optimized with regex precompilation and caching"""
        tasks = [
            re.sub(r'^\d+[\.\)]\s*', '', task).strip()
            for task in TaskManager._task_split_re.findall(str(maintenance_str).lower())
        ]
        return [
            TaskManager._page_refs_re.sub('', task).strip()
            for task in tasks
            if task.strip()
        ]


class TaskManualMatcher:
    def __init__(self, document_processor: DocumentProcessor):
        """
        :param document_processor: Instance of DocumentProcessor with loaded data and extracted texts.
        """
        self.dp = document_processor
        self.df_task_def = self.dp.df_task_def

    def process_tasks(self) -> pd.DataFrame:
        """Main processing pipeline for task-manual matching."""
        self._add_task_list()
        self._find_attachments()
        self._add_doc_name_page()
        return self.df_task_def

    def _add_task_list(self) -> None:
        """Add extracted task list to the task definitions DataFrame."""
        logger.info("Running: %s._add_task_list", self.__class__.__name__)
        self.df_task_def.loc[:, 'tasks'] = self.df_task_def['DESCRIPTION'].apply(TaskManager.extract_tasks)

    def _find_attachments(self) -> None:
        """Match tasks to manuals and populate the attachment column."""
        logger.info("Running: %s._find_attachments", self.__class__.__name__)
        
        if debug:
            logger.info(f"🔎 Data type of system codes in manuals_df: {self.dp.manuals_df['CODE*'].dtype}")
            logger.info(f"🔎 Data type of system codes in task_df: {self.dp.df_task_def['code'].dtype}")
                    
        # Use ThreadPoolExecutor to parallelize the process of matching tasks to manuals
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_row = {executor.submit(self._find_manual_matches_for_task, row): row
                             for _, row in self.df_task_def.iterrows()}

            for future in concurrent.futures.as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    attachment = future.result()
                    self.df_task_def.at[row.name, 'attachment'] = attachment
                except Exception as exc:
                    logger.error(f"Task matching failed for task {row['NAME*']}: {exc}")
                    self.df_task_def.at[row.name, 'attachment'] = ''

    def _find_manual_matches_for_task(self, row: pd.Series) -> str:
        """Find matching manuals for a given task row."""
        system_code = row['code']
        tasks = row['tasks']

        # Filter manuals based on system code
        system_manuals = self.dp.manuals_df[self.dp.manuals_df['CODE*'] == system_code]
        if system_manuals.empty or not tasks:
            if debug: logger.warning(f"No manuals found for system code: {system_code} or tasks list is empty.")
            return ''
        
        tasks = [task for task in tasks if task != '']

        if not tasks:
            return ''

        complete_matches = self._find_complete_matches(system_manuals, tasks)
        if complete_matches:
            if debug: logger.info(f"Complete matches found for {system_code}: {complete_matches}")
            return ';'.join(sorted(complete_matches))

        partial_matches = self._find_partial_matches(system_manuals, tasks)
        if partial_matches:
            if debug: logger.info(f"Partial matches found for {system_code}: {partial_matches}")
            return ';'.join(sorted(partial_matches))

        if debug:
            logger.warning(f"No match found for system code: {system_code}")
            logger.warning(f"Tasks: {tasks}")
            logger.warning(f"Available manuals: {system_manuals[['DOCUMENT NAME*', 'IDENTIFIER*']].to_dict(orient='records')}")

        return ''

    def _find_complete_matches(self, system_manuals: pd.DataFrame, tasks: List[str],
                               threshold: float = THRESHOLD_PERCENTAGE) -> List[str]:
        """
        Find manuals that contain all tasks with fuzzy matching.
        The generator expression in 'all()' short-circuits if any task is not found.
        """
        matches = []
        for manual_id in system_manuals['IDENTIFIER*']:
            pages = self.dp.manual_texts.get(manual_id, [])
            manual_text = ' '.join([text for _, text in pages])
            if all(
                re.search(rf'\b{re.escape(task)}\b', manual_text) or 
                DocumentProcessor.fuzzy_match_exists(manual_text, task, threshold)
                for task in tasks if task != ''
            ):
                matches.append(manual_id)
        return matches

    def _find_partial_matches(self, system_manuals: pd.DataFrame, tasks: List[str],
                              threshold: float = THRESHOLD_PERCENTAGE) -> List[str]:
        """
        Find manuals that contain any of the tasks using fuzzy matching.
        For each manual, as soon as one task is found, the search stops for that manual.
        """
        matches = set()
        for manual_id in system_manuals['IDENTIFIER*']:
            pages = self.dp.manual_texts.get(manual_id, [])
            manual_text = ' '.join([text for _, text in pages])
            for task in tasks:
                if task != '' and (re.search(rf'\b{re.escape(task)}\b', manual_text) or 
                                   DocumentProcessor.fuzzy_match_exists(manual_text, task, threshold)):
                    matches.add(manual_id)
                    break  # Early exit: match found for this manual
        return list(matches)

    def _add_doc_name_page(self) -> None:
        """
        Add a DOC_NAME_PAGE column to the task definitions DataFrame that aggregates
        document names, identifiers, task names, and the page numbers where the tasks were found.
        """
        logger.info("Running: %s._add_doc_name_page", self.__class__.__name__)
        doc_name_pages = []
        
        for _, row in self.df_task_def.iterrows():
            manual_ids = row['attachment'].split(';') if row['attachment'] else []
            entries = []
            tasks = row['tasks']

            for manual_id in manual_ids:
                manual_info = self.dp.manuals_df[self.dp.manuals_df['IDENTIFIER*'] == manual_id]
                if manual_info.empty:
                    continue

                doc_name = manual_info['DOCUMENT NAME*'].iloc[0]
                pages = self.dp.manual_texts.get(manual_id, [])
                found_pages = set()

                for page_num, page_text in pages:
                    for task in tasks:
                        if re.search(rf'\b{re.escape(task)}\b', page_text, re.IGNORECASE):
                            found_pages.add(page_num)
                            break  # Early exit: task found on this page, move to next page

                if found_pages:
                    sorted_pages = sorted(found_pages)
                    pages_str = ', '.join(str(p) for p in sorted_pages)
                    for task in tasks:
                        entries.append(f"Task: '{task}' - (Manual ID: {manual_id}) {doc_name} - Page(s): {pages_str}")

            unique_entries = list(set(entries))
            doc_name_pages.append('\n'.join(unique_entries))

        self.df_task_def['DOC_NAME_PAGE'] = doc_name_pages

# =============================================================================
# Main Maintenance Processing Pipeline
# =============================================================================

class MaintenanceProcessor:
    def __init__(self, doc_path: str, task_def_path: str):
        self.doc_processor = DocumentProcessor(doc_path, task_def_path)
        self.matcher = None

    def process(self) -> pd.DataFrame:
        """Execute the full processing pipeline."""
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
