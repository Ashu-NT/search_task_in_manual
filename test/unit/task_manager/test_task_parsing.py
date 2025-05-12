import pytest
from src.task_def_v4 import TaskManager

class TestTaskParsing:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            # Standard numbered tasks
            ("1. Calibrate sensor 2) Check alignment", 
             ["calibrate sensor", "check alignment"]),
            
            # Mixed numbering formats
            ("Step 3: Clean filters (page 12)\n4) Inspect valves", 
             ["clean filters", "inspect valves"]),
            
            # Page reference handling
            ("Check pressure (see page 15) 2. Verify temperature", 
             ["check pressure", "verify temperature"]),
            
            # Non-numbered items
            ("No numbered tasks here", ["no numbered tasks here"]),
            
            # Complex whitespace
            ("  1.   Remove  debris  \n 2) \tLubricate joints  ", 
             ["remove debris", "lubricate joints"]),
            
            # Special characters
            ("1. Check temp (100°C) 2) Validate JSON {format}", 
             ["check temp", "validate json"]),
            
            # Special characters
            ("• Check temp (100°C) ► Validate JSON {format}", 
             ["check temp", "validate json"]),
            
            # Nested parentheses
            ("1. (Main) System check (see chapter 3)", 
             ["system check"]),
            
            # Multiple page references
            ("1. Component test (page 5, section 2a)", 
             ["component test"])
        ]
    )
    def test_task_extraction(self, input_str, expected):
        """Test task extraction with enhanced regex patterns"""
        result = TaskManager.extract_tasks(input_str)
        assert result == expected, f"Failed for input: '{input_str}'"

    def test_empty_input(self):
        """Test empty string handling"""
        assert TaskManager.extract_tasks("") == []