# CODE TO WORK WITH THE "TASK DEFINITION" TAB OF THE IMPORT TEMPLATE
import pandas as pd
import PyPDF2 
import re
import os
from pathlib import Path

#-----------Normal structure of DOCUMENT TAB----------------------------------------------------
# IDENTIFIER,DOCUMENT NAME*,DOCUMENT CATEGORY*,FOLDER*,SYSTEM,VERSION* --------------(IMPORTANT)
# HOLDER,RESPONSIBILITY,ISSUANCE DATE,EXPIRATION DATE,EXPIRING SOON
# FILE -----------------------------------------------------------------------------(IMPORTANT)

#-----------Normal structure of TASK DEFINITION TAB--------------------------------------------
# IDENTIFIER*, NAME*, CODE*,---------------------------------------------------------(IMPORTANT)
# ASSIGNEES,DUE DATE,
# DESCRIPTION -----------------------------------------------------------------------(IMPORTANT)
# TASK GROUPING
# ATTACHMENT,SYSTEM

# --------------Load data (replace with actual paths)--------------------------------------------------------
df_doc = pd.read_csv("files.csv")  # Contains: identifier, document type, system code,file path
df_task_def = pd.read_csv("components.csv")  # Contains: system code,component,Maintenance


# --------------Normalize file paths-------------------------------------------------------------------------
def normalize_path(path):
    try:
        return Path(os.path.normpath(os.path.expanduser(str(path)))).resolve().as_posix()
    except Exception as e:
        print(f"Invalid path: {path} - {str(e)}")
        return None
    
df_doc['normalized_path'] = df_doc['file path'].apply(normalize_path) 
df_doc = df_doc.dropna(subset=['normalized_path'])

#----------------- Filter manuals ----------------------------------------------------------------------------
manuals_df = df_doc[df_doc['document type'] == 'Manual']
manuals_df = manuals_df[['identifier', 'system code', 'file path']].copy() #system code here refers to "SYSTEM" in document tab

#----------------- Extract text from PDFs with improved error handling ---------------------------------------
manual_texts = {}
for _, row in manuals_df.iterrows():
    try:
        with open(row['file path'], 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            text = " ".join([
                page.extract_text() 
                for page in pdf.pages
                if page.extract_text()
            ]).lower().replace('\n', ' ')  # Normalize text

            manual_texts[row['identifier']] = text
    except Exception as e:
        print(f"Error processing {row['identifier']}: {str(e)}")
        manual_texts[row['identifier']] = ""

#-------------- Split maintenance tasks using robust regex ---------------------------------------------------
def extract_tasks(maintenance_str):
    maintenance_str = str(maintenance_str).lower()
    return [
        re.sub(r'^\d+[\.\)]\s*', '', task).strip()  # <-- THIS LINE
        for task in re.findall(                      # <-- SOURCE OF "task"
            r'\d+[\.\)]\s*(.*?)(?=\s*\d+[\.\)]|$)', 
            maintenance_str
        )
    ]

df_task_def['tasks'] = df_task_def['DESCRIPTION'].apply(extract_tasks)

#-------------- Match tasks to manuals ---------------------------------------------------------------------
def find_attachments(row):
    system_code = row['system code']
    tasks = row['tasks']
    
    # Get relevant manuals for this system code
    system_manuals = manuals_df[manuals_df['system code'] == system_code]
    if system_manuals.empty or not tasks:
        return ''
    
    # Phase 1: Check for manuals containing ALL tasks
    complete_matches = []
    for manual_id in system_manuals['identifier']:
        text = manual_texts.get(manual_id, '')
        
        # Check if ALL tasks exist in this manual
        if all(re.search(rf'\b{re.escape(task)}\b', text) for task in tasks):
            complete_matches.append(manual_id)
    
    if complete_matches:
        return ';'.join(sorted(complete_matches))  # Return ALL complete matches
    
    # Phase 2: Only if no complete matches, check for partial matches
    partial_matches = set()
    for task in tasks:
        for manual_id in system_manuals['identifier']:
            text = manual_texts.get(manual_id, '')
            if re.search(rf'\b{re.escape(task)}\b', text):
                partial_matches.add(manual_id)
    
    return ';'.join(sorted(partial_matches)) if partial_matches else ''

df_task_def['attachment'] = df_task_def.apply(find_attachments, axis=1)

# Show results
print(df_task_def[['system code', 'component', 'Maintenance', 'attachment']])