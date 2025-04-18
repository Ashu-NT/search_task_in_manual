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
THRESHOLD_PERCENTAGE = 0.8

# For debuging change to TRUE
debug = False

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

    def extract_manual_texts(self) -> None:
        """
        Extract text (including table content) from PDF manuals using both PyPDF2 and pdfplumber.
        For each page, text is extracted via both methods, combined, and cleaned.
        Duplicate page texts are removed by merging their page numbers (e.g. "1, 3").
        The final output is stored as a list of tuples (page_numbers, text) for each manual.
        """
        logger.info("Running: %s.extract_manual_texts", self.__class__.__name__)
        
        # Suppress pdfminer warnings (CropBox missing)
        logging.getLogger("pdfminer.pdfpage").setLevel(logging.ERROR)
        
        for _, row in self.manuals_df.iterrows():
            manual_id = row['IDENTIFIER*']
            file_path = row['file path']
            pages_dict = {}  # key: cleaned text, value: set of page numbers
            
            try:
                # Read the file into memory for PyPDF2 so that the stream remains open.
                try:
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data), strict=False)
                except Exception as e:
                    logger.error("Error reading PDF via PyPDF2 for %s: %s", manual_id, e)
                    pdf_reader = None

                # Open the file with pdfplumber for text and table extraction.
                with pdfplumber.open(file_path) as pdf_plumber:
                    for page_num, page in enumerate(pdf_plumber.pages, start=1):
                        combined_text = ""
                        
                        # Attempt text extraction using PyPDF2
                        text_py = ""
                        if pdf_reader is not None and len(pdf_reader.pages) >= page_num:
                            try:
                                text_py = pdf_reader.pages[page_num - 1].extract_text() or ""
                                # If text_py is a NullObject, replace it with an empty string.
                                if isinstance(text_py, NullObject):
                                    text_py = ""
                            except Exception as e:
                                logger.error("Error extracting PyPDF2 text on page %d for %s: %s",
                                            page_num, manual_id, e)
                        combined_text += text_py + " "
                        
                        # Extract text and table content using pdfplumber
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
                            logger.error("Error processing pdfplumber page %d for %s: %s",
                                        page_num, manual_id, e)
                        
                        # Clean the combined text: lowercase, remove newlines, and trim whitespace.
                        cleaned_text = combined_text.lower().replace('\n', ' ').strip()
                        
                        if cleaned_text:
                            # Merge duplicate text by collecting page numbers.
                            if cleaned_text in pages_dict:
                                pages_dict[cleaned_text].add(page_num)
                            else:
                                pages_dict[cleaned_text] = {page_num}
                
                # Convert pages_dict into a sorted list of tuples (pages, text).
                final_pages = []
                for text, page_nums in pages_dict.items():
                    sorted_pages = sorted(page_nums)
                    pages_str = ", ".join(str(num) for num in sorted_pages)
                    final_pages.append((pages_str, text))
                # Sort the final pages by the first page number.
                final_pages = sorted(final_pages, key=lambda tup: int(tup[0].split(",")[0]))
                self.manual_texts[manual_id] = final_pages
            
            except Exception as e:
                logger.error("Error opening or processing %s: %s", manual_id, e)
                self.manual_texts[manual_id] = []
        
        # For debuging
        if debug:        
            print(self.manual_texts.keys())
            for manual_id, pages in self.manual_texts.items():
                for page_num, text in pages:
                    print(f"Manual: {manual_id}, Page {page_num}: {text[:500]}")  # Print first 500 chars    
       
        
    @staticmethod
    def _character_based_match(manual_text: str, task: str, threshold: float, step: int) -> bool:
        """
        Perform a sliding-window character-based fuzzy match.

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
                return True
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
                           step_size: Optional[int] = None, use_rapidfuzz: bool = True) -> bool:
        """
        Check if a given task string exists in manual_text with a similarity above threshold.
        This method slides a window over the text to find a fuzzy match.
        It attempts to use RapidFuzz for performance, falling back to custom matching if necessary.

        Args:
            manual_text: Text to search within.
            task: Phrase to search for.
            threshold: Similarity threshold (0-1).
            step_size: Window sliding step size (None to compute dynamically).
            use_rapidfuzz: Whether to try using RapidFuzz for matching.

        Returns:
            bool: True if a similar match is found; otherwise, False.
        """
        manual_text = manual_text.lower()
        task = task.lower()
        task_len = len(task)
        if task_len == 0 or len(manual_text) < task_len:
            return False

        # Attempt to use RapidFuzz if enabled
        if use_rapidfuzz:
            try:
                from rapidfuzz import fuzz
                # RapidFuzz works with percentage thresholds
                if fuzz.partial_ratio(manual_text, task) >= threshold * 100:
                    return True
            except ImportError:
                logger.warning("RapidFuzz not installed; falling back to custom fuzzy matching.")

        # Calculate dynamic step size if not provided
        step = step_size or max(1, task_len // 4)

        # Try character-based matching first
        if DocumentProcessor._character_based_match(manual_text, task, threshold, step):
            return True

        # Fallback to word-based matching
        return DocumentProcessor._word_based_match(manual_text, task, threshold)

# =============================================================================
# Task Processing Components
# =============================================================================

class TaskManager:
    @staticmethod
    def extract_tasks(maintenance_str: str) -> List[str]:
        """
        Extract maintenance tasks from the DESCRIPTION string while removing page references.
        
        :param maintenance_str: The DESCRIPTION string containing tasks.
        :return: List of cleaned task strings.
        """
        maintenance_str = str(maintenance_str).lower()
        # Extract tasks based on numbering
        tasks = [
            re.sub(r'^\d+[\.\)]\s*', '', task).strip()
            for task in re.findall(r'\d+[\.\)]\s*(.*?)(?=\s*\d+[\.\)]|$)', maintenance_str)
        ]

        # Remove common page/chapter references from each task
        cleaned_tasks = []
        for task in tasks:
            cleaned_task = re.sub(
                r'\(page \d+\)|see page \d+|\(see chapter \d+(\.\d+)*, page \d+\)|chapter \d+(\.\d+)*|'
                r'\(chapter \d+(\.\d+)*( \/ page \d+(-\d+)*)?\)|'
                r'\(section \d+( page \d+)?(, function description)?\)|'
                r'\(\s*3a[\w\d-]*\s*\)?|\(abb\)',
                '', task,
                flags=re.IGNORECASE
            ).strip()
            
            #print(f"Original Task: {task} -> Cleaned Task: {cleaned_task}")  # Debug print
            
            cleaned_tasks.append(cleaned_task)

        return cleaned_tasks


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
            # Partially apply the function to include the current row
            future_to_row = {executor.submit(self._find_manual_matches_for_task, row): row
                             for _, row in self.df_task_def.iterrows()}

            # Update the 'attachment' column once futures are completed
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
            if debug:logger.warning(f"No manuals found for system code: {system_code} or tasks list is empty.")
            return ''
        
        # Filter out empty tasks from the list
        tasks = [task for task in tasks if task != '']

        if not tasks:
            return ''  # Return empty if all tasks are empty

        # First, try to find complete matches
        complete_matches = self._find_complete_matches(system_manuals, tasks)
        if complete_matches:
            if debug:logger.info(f"Complete matches found for {system_code}: {complete_matches}")
            return ';'.join(sorted(complete_matches))

        # If no complete matches, try partial matches
        partial_matches = self._find_partial_matches(system_manuals, tasks)
        if partial_matches:
            if debug:logger.info(f"Partial matches found for {system_code}: {partial_matches}")
            return ';'.join(sorted(partial_matches))

        # Debugging: No matches found at all
        if debug:
            logger.warning(f"No match found for system code: {system_code}")
            logger.warning(f"Tasks: {tasks}")
            logger.warning(f"Available manuals: {system_manuals[['DOCUMENT NAME*', 'IDENTIFIER*']].to_dict(orient='records')}")

        return ''  # Return empty string if no matches are found

    def _find_complete_matches(self, system_manuals: pd.DataFrame, tasks: List[str],
                               threshold: float = THRESHOLD_PERCENTAGE) -> List[str]:
        """
        Find manuals that contain all tasks with fuzzy matching.
        """
        matches = []
        for manual_id in system_manuals['IDENTIFIER*']:
            pages = self.dp.manual_texts.get(manual_id, [])
            # Combine texts from all pages
            manual_text = ' '.join([text for _, text in pages])
            if all(
                re.search(rf'\b{re.escape(task)}\b', manual_text) or 
                DocumentProcessor.fuzzy_match_exists(manual_text, task, threshold)
                for task in tasks if task != ''  # Skip empty tasks
            ):
                matches.append(manual_id)
        return matches

    def _find_partial_matches(self, system_manuals: pd.DataFrame, tasks: List[str],
                              threshold: float = THRESHOLD_PERCENTAGE) -> List[str]:
        """
        Find manuals that contain any of the tasks using fuzzy matching.
        """
        matches = set()
        for manual_id in system_manuals['IDENTIFIER*']:
            pages = self.dp.manual_texts.get(manual_id, [])
            manual_text = ' '.join([text for _, text in pages])
            for task in tasks:
                if task != '' and re.search(rf'\b{re.escape(task)}\b', manual_text) or DocumentProcessor.fuzzy_match_exists(manual_text, task, threshold):
                    matches.add(manual_id)
                    break  # Found a match for this manual; no need to check further
        return list(matches)

    def _add_doc_name_page(self) -> None:
        """
        Add a DOC_NAME_PAGE column to the task definitions DataFrame that aggregates
        document names, identifiers, task names, and the page numbers where the tasks were found.
        """
        logger.info("Running: %s._add_doc_name_page", self.__class__.__name__)
        doc_name_pages = []
        
        # Loop through each task in the task definitions DataFrame
        for _, row in self.df_task_def.iterrows():
            manual_ids = row['attachment'].split(';') if row['attachment'] else []
            entries = []
            tasks = row['tasks']

            for manual_id in manual_ids:
                # Retrieve manual info from the manuals DataFrame
                manual_info = self.dp.manuals_df[self.dp.manuals_df['IDENTIFIER*'] == manual_id]
                if manual_info.empty:
                    continue

                doc_name = manual_info['DOCUMENT NAME*'].iloc[0]
                # Retrieve pages for this manual
                pages = self.dp.manual_texts.get(manual_id, [])
                found_pages = set()

                for page_num, page_text in pages:
                    for task in tasks:
                        # Check if the task name appears in the page text
                        if re.search(rf'\b{re.escape(task)}\b', page_text, re.IGNORECASE):
                            found_pages.add(page_num)
                            break  # Move to next page after a match

                if found_pages:
                    sorted_pages = sorted(found_pages)
                    pages_str = ', '.join(str(p) for p in sorted_pages)
                    # Include the task name and manual information in the entry
                    for task in tasks:
                        entries.append(f"Task: '{task}' - (Manual ID: {manual_id}) {doc_name} - Page(s): {pages_str}")

            # Remove duplicates and join multiple entries
            unique_entries = list(set(entries))
            doc_name_pages.append('\n'.join(unique_entries))

        # Update the DataFrame with the aggregated DOC_NAME_PAGE information
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
    # Example usage; update the paths accordingly.
    try:
        processor = MaintenanceProcessor(
            doc_path=r'C:\Users\ashu\Desktop\Python Workspace\training_model\data\training\batc_2_cosmos_ABB.csv',  # Change to document file
            task_def_path=r"C:\Users\ashu\Downloads\Import_Sheet_Batch 2_V02.2_ABB.xlsx - Task definition.csv"   # Change to task definition file
        )

        result_df = processor.process()
        # Reorder and select desired columns
        result_df = result_df[["NAME*", 'CODE*', 'code', 'DESCRIPTION','tasks', 'attachment', 'DOC_NAME_PAGE']].copy()
        # Output to CSV and Excel
        result_df.to_csv("../data/Batch_2_output.csv", index=False)
        result_df.to_excel("../data/Batch_2_output.xlsx", index=False)
        logger.info("Processing complete. Results:")
        print(result_df[["NAME*", 'CODE*', 'code', 'DESCRIPTION', 'attachment', 'DOC_NAME_PAGE']])
    except Exception as e:
        logger.error("An error occurred during processing: %s", str(e))
