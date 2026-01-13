#!/usr/bin/env python3
"""
Batch Reasoning Path Processor
Reads Excel files from input directory and analyzes model responses using LLM APIs.
Supports true concurrent processing across multiple models.

This module processes Excel files containing model predictions and extracts
reasoning pathway information using the reasoning_analyzer module.
"""

import os
import json
import csv
import argparse
import time
import multiprocessing as mp
import pandas as pd
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from colorama import init, Fore, Back, Style
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue

from reasoning_analyzer import analyze_text_with_retry

# Initialize colorama
init(autoreset=True)

class ReasoningPathProcessor:
    def __init__(self, result_dir: Optional[str] = None, output_dir: Optional[str] = None, 
                 max_workers: int = 8, question_file: Optional[str] = None,
                 api_base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize the batch processor
        
        Args:
            result_dir: Input directory containing Excel files (default: script_dir/output_data)
            output_dir: Output directory for CSV files (default: script_dir/processed_output)
            max_workers: Number of concurrent processes
            question_file: Path to Excel file containing question index mapping (optional)
            api_base_url: API base URL (optional, uses reasoning_analyzer default)
            api_key: API key (optional, uses reasoning_analyzer default)
            model_name: Model name (optional, uses reasoning_analyzer default)
        """
        script_dir = Path(__file__).parent
        
        # Set input directory
        if result_dir is None:
            result_dir = str(script_dir / "input_data")
        self.result_dir = Path(result_dir)
        
        # Set output directory
        if output_dir is None:
            output_dir = str(script_dir / "processed_output")
        elif not Path(output_dir).is_absolute():
            output_dir = str(script_dir / output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_workers = max_workers
        
        # Store API configuration for passing to analyzer
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        
        # CSV column definition (must match the fields returned by the analyzer)
        self.csv_columns = [
            'model_name', 'index', 'prediction',
            'modality_order', 'feature_order', 'conclusion_order', 'others_order',
            'modality_subs', 'feature_subs', 'conclusion_subs', 'others_subs',
            'parse_status'
        ]
        
        # Thread locks for CSV file writing protection
        self.csv_locks = {}
        
        # Load question index file, create index -> question mapping
        self.question_map = {}
        if question_file:
            qa_file = Path(question_file)
        else:
            qa_file = script_dir / "question_index.xlsx"  # Default name
        
        if qa_file.exists():
            try:
                print(f"{Fore.CYAN}📖 Loading question index file: {qa_file}")
                qa_df = pd.read_excel(qa_file)
                if 'index' in qa_df.columns and 'question' in qa_df.columns:
                    for _, row in qa_df.iterrows():
                        if not pd.isna(row['index']):
                            idx = int(row['index'])
                            question = str(row['question']).strip() if not pd.isna(row['question']) else ''
                            self.question_map[idx] = question
                    print(f"{Fore.GREEN}✅ Successfully loaded {len(self.question_map)} question indices")
                else:
                    print(f"{Fore.YELLOW}⚠️  Question index file missing required columns: index or question")
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️  Failed to load question index file: {e}, will proceed without question context")
        else:
            print(f"{Fore.YELLOW}⚠️  Question index file not found: {qa_file}, will proceed without question context")
    
    def find_excel_files(self, model_name: Optional[str] = None) -> List[Tuple[str, Path]]:
        """
        Find all Excel files to process
        
        Returns:
            List[Tuple[model_name, excel_file_path]]
        """
        excel_files = []
        
        if not self.result_dir.exists():
            print(f"{Fore.RED}❌ Directory does not exist: {self.result_dir}")
            return excel_files
        
        # Traverse all subdirectories in result_dir
        for subdir in self.result_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # If model name is specified, only process matching directories
            if model_name and subdir.name != model_name:
                continue
            
            # Find *_cot.xlsx files in this directory
            cot_files = list(subdir.glob("*_cot.xlsx"))
            
            if cot_files:
                # Usually each directory has only one *_cot.xlsx file
                excel_file = cot_files[0]
                model_name_from_file = excel_file.stem  # Remove .xlsx extension
                excel_files.append((model_name_from_file, excel_file))
                print(f"{Fore.BLUE}📄 Found file: {excel_file}")
            else:
                print(f"{Fore.YELLOW}⚠️  No *_cot.xlsx file found in directory {subdir.name}")
        
        return excel_files
    
    def get_all_tasks(self, model_name: Optional[str] = None) -> List[Tuple]:
        """Get all tasks that need to be processed"""
        excel_files = self.find_excel_files(model_name)
        
        if not excel_files:
            print(f"{Fore.RED}❌ No Excel files found")
            return []
        
        all_tasks = []
        
        for model_name_from_file, excel_file in excel_files:
            csv_file = self.output_dir / f"{model_name_from_file}.csv"
            
            # Create CSV template
            self.create_csv_template(model_name_from_file, excel_file)
            
            # Load existing data
            existing_data = self.load_existing_data(csv_file)
            
            # Read Excel data
            try:
                print(f"{Fore.CYAN}📖 Reading Excel file: {excel_file}")
                df = pd.read_excel(excel_file)
                
                # Check if required columns exist
                if 'index' not in df.columns or 'prediction' not in df.columns:
                    print(f"{Fore.RED}❌ Excel file missing required columns: index or prediction")
                    continue
                
                # Prepare tasks and collect empty predictions to update
                empty_predictions_to_update = {}
                
                for _, row in df.iterrows():
                    # Check if index is empty
                    if pd.isna(row['index']):
                        continue
                    
                    index = int(row['index'])
                    
                    # Process prediction field
                    if pd.isna(row['prediction']):
                        prediction = ''
                    else:
                        prediction = str(row['prediction']).strip()
                    
                    # If prediction is empty
                    if not prediction:
                        # If CSV already has this index record, and previously had prediction, now empty
                        if index in existing_data:
                            old_prediction = existing_data[index].get('prediction', '')
                            if old_prediction:  # Previously had prediction, now empty
                                print(f"{Fore.YELLOW}⚠️  {model_name_from_file} index {index}: prediction changed from value to empty, will mark as failed")
                                # Record empty prediction to update
                                empty_predictions_to_update[index] = {
                                    'model_name': model_name_from_file,
                                    'index': str(index),
                                    'prediction': '',
                                    'modality_order': '',
                                    'feature_order': '',
                                    'conclusion_order': '',
                                    'others_order': '',
                                    'modality_subs': '',
                                    'feature_subs': '',
                                    'conclusion_subs': '',
                                    'others_subs': '',
                                    'parse_status': 'false'
                                }
                        # Skip empty prediction, don't add to task list (cannot analyze)
                        continue
                    
                    # Check if processing is needed
                    # If already processed and prediction is the same (case-insensitive), skip
                    if index in existing_data:
                        csv_status = existing_data[index].get('parse_status', '').upper()
                        csv_prediction = existing_data[index].get('prediction', '').strip()
                        # Case-insensitive comparison, as CSV may have case changes after Windows editing
                        if (csv_status == 'TRUE' and 
                            csv_prediction.upper() == prediction.upper()):
                            continue
                    
                    # If prediction changed, log it
                    if (index in existing_data and 
                        existing_data[index].get('parse_status') == 'true' and
                        existing_data[index].get('prediction') != prediction):
                        print(f"{Fore.YELLOW}⚠️  {model_name_from_file} index {index}: prediction changed, will reprocess")
                    
                    # If it's a failed record (parse_status == 'false'), log it
                    if (index in existing_data and 
                        existing_data[index].get('parse_status') == 'false'):
                        print(f"{Fore.CYAN}🔄 {model_name_from_file} index {index}: detected failed record, will reprocess")
                    
                    # Get corresponding question
                    question = self.question_map.get(index, '')
                    
                    # Add task: (model_name, index, prediction, question, csv_file_path)
                    all_tasks.append((model_name_from_file, index, prediction, question, str(csv_file)))
                
                # If there are empty predictions to update, save immediately
                if empty_predictions_to_update:
                    existing_data.update(empty_predictions_to_update)
                    csv_file_path = self.output_dir / f"{model_name_from_file}.csv"
                    self.save_to_csv(csv_file_path, existing_data)
                    print(f"{Fore.BLUE}💾 Updated {len(empty_predictions_to_update)} empty prediction records for {model_name_from_file}")
                
                print(f"{Fore.GREEN}✅ Loaded {len([t for t in all_tasks if t[0] == model_name_from_file])} tasks from {model_name_from_file}")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to read Excel file {excel_file}: {e}")
                continue
        
        return all_tasks
    
    def get_failed_tasks(self, model_name: Optional[str] = None) -> List[Tuple]:
        """
        Get only failed tasks (parse_status == 'false')
        Used for incremental processing of failed records
        """
        excel_files = self.find_excel_files(model_name)
        
        if not excel_files:
            print(f"{Fore.RED}❌ No Excel files found")
            return []
        
        failed_tasks = []
        
        for model_name_from_file, excel_file in excel_files:
            csv_file = self.output_dir / f"{model_name_from_file}.csv"
            
            # If CSV file doesn't exist, no failed records
            if not csv_file.exists():
                continue
            
            # Load existing data
            existing_data = self.load_existing_data(csv_file)
            
            # Read Excel data
            try:
                df = pd.read_excel(excel_file)
                
                # Check if required columns exist
                if 'index' not in df.columns or 'prediction' not in df.columns:
                    continue
                
                # Find failed tasks
                for _, row in df.iterrows():
                    if pd.isna(row['index']):
                        continue
                    
                    index = int(row['index'])
                    
                    # Process prediction field
                    if pd.isna(row['prediction']):
                        prediction = ''
                    else:
                        prediction = str(row['prediction']).strip()
                    
                    # Skip empty predictions
                    if not prediction:
                        continue
                    
                    # Only process failed records (parse_status == 'false')
                    if (index in existing_data and 
                        existing_data[index].get('parse_status') == 'false'):
                        # Get corresponding question
                        question = self.question_map.get(index, '')
                        failed_tasks.append((model_name_from_file, index, prediction, question, str(csv_file)))
                        print(f"{Fore.CYAN}🔄 Added failed retry task: {model_name_from_file} index {index}")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to read Excel file {excel_file}: {e}")
                continue
        
        return failed_tasks
    
    def create_csv_template(self, model_name: str, excel_file: Path, max_index: int = 1079):
        """Create CSV template file"""
        csv_file = self.output_dir / f"{model_name}.csv"
        
        # If file doesn't exist, create template
        if not csv_file.exists():
            print(f"{Fore.BLUE}📝 Creating CSV template: {csv_file}")
            
            # Try to get max index from Excel file
            try:
                df = pd.read_excel(excel_file)
                if 'index' in df.columns:
                    max_index = int(df['index'].max())
            except:
                pass
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
                
                # Create template rows for all indices
                for i in range(1, max_index + 1):
                    row = {
                        'model_name': model_name,
                        'index': str(i),
                        'prediction': '',
                        'modality_order': '',
                        'feature_order': '',
                        'conclusion_order': '',
                        'others_order': '',
                        'modality_subs': '',
                        'feature_subs': '',
                        'conclusion_subs': '',
                        'others_subs': '',
                        'parse_status': 'false'
                    }
                    writer.writerow(row)
    
    def load_existing_data(self, csv_file: Path) -> Dict[int, Dict]:
        """Load existing CSV data (handles encoding errors)"""
        existing_data = {}
        
        if csv_file.exists():
            # Handle common issues with CSV files edited in Windows:
            # 1. UTF-8 BOM (Excel may add)
            # 2. Invalid UTF-8 bytes (special characters incorrectly encoded)
            # 3. Different encoding formats
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk']
            
            for encoding in encodings:
                try:
                    with open(csv_file, 'r', encoding=encoding, errors='replace') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                index = int(row['index'])
                                existing_data[index] = row
                            except (ValueError, KeyError):
                                continue
                    # If successfully read, break loop
                    if encoding != 'utf-8':
                        print(f"{Fore.YELLOW}⚠️  {csv_file.name} read using {encoding} encoding (may have been edited in Windows)")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    # If current encoding fails, try next
                    continue
                except Exception as e:
                    # Other errors (e.g., CSV format errors), also try next encoding
                    if encoding == encodings[-1]:
                        # All encodings failed
                        print(f"{Fore.RED}❌ {csv_file.name} read failed: {e}")
                    continue
        
        return existing_data
    
    def save_to_csv(self, csv_file: Path, data: Dict[int, Dict]):
        """Save data to CSV file (using temporary file for safety)"""
        # Use temporary file to ensure save process safety
        temp_file = csv_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
                
                # Write sorted by index
                for index in sorted(data.keys()):
                    writer.writerow(data[index])
            
            # Atomic replacement: write to temp file first, then replace original
            temp_file.replace(csv_file)
            
        except Exception as e:
            # If save fails, delete temp file
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def process_single_item_true_concurrent(self, args: Tuple[str, int, str, str, str]) -> Tuple[str, int, Dict, bool]:
        """
        Process a single data item (true concurrent version)
        
        Args:
            args: (model_name, index, prediction, question, csv_file_path)
        
        Returns:
            (model_name, index, result_data, success)
        """
        model_name, index, prediction, question, csv_file_path = args
        
        print(f"{Fore.YELLOW}[Process {os.getpid()}] Processing {model_name} index {index}...")
        
        # Analyze text - call reasoning_analyzer function, pass question info and API config
        result = analyze_text_with_retry(
            prediction, 
            question=question if question else None,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        if result['success']:
            data = result['data']
            
            # Convert lists to strings (for CSV storage)
            def list_to_str(lst):
                if isinstance(lst, list):
                    return ' | '.join(str(item) for item in lst)
                return str(lst)
            
            result_data = {
                'model_name': model_name,
                'index': str(index),
                'prediction': prediction,
                'modality_order': str(data.get('modality_order', '')),
                'feature_order': str(data.get('feature_order', '')),
                'conclusion_order': str(data.get('conclusion_order', '')),
                'others_order': str(data.get('others_order', '')),
                'modality_subs': list_to_str(data.get('modality_subs', [])),
                'feature_subs': list_to_str(data.get('feature_subs', [])),
                'conclusion_subs': list_to_str(data.get('conclusion_subs', [])),
                'others_subs': list_to_str(data.get('others_subs', [])),
                'parse_status': 'true'
            }
            
            print(f"{Fore.GREEN}[Process {os.getpid()}] ✅ {model_name} index {index} processed successfully")
            return model_name, index, result_data, True
            
        else:
            # Processing failed
            result_data = {
                'model_name': model_name,
                'index': str(index),
                'prediction': prediction,
                'modality_order': '',
                'feature_order': '',
                'conclusion_order': '',
                'others_order': '',
                'modality_subs': '',
                'feature_subs': '',
                'conclusion_subs': '',
                'others_subs': '',
                'parse_status': 'false'
            }
            
            print(f"{Fore.RED}[Process {os.getpid()}] ❌ {model_name} index {index} processing failed: {result.get('error', 'Unknown error')}")
            return model_name, index, result_data, False
    
    def process_all_models_true_concurrent(self, model_name: Optional[str] = None, retry_failed_only: bool = False):
        """
        True concurrent processing of all models (supports interrupt recovery)
        
        Args:
            model_name: Specify model name to process (directory name)
            retry_failed_only: If True, only process failed tasks (parse_status == 'false')
        """
        print(f"{Fore.CYAN}{'='*60}")
        if retry_failed_only:
            print(f"{Fore.CYAN}🔄 Reasoning Path Processor - Failed Retry Mode")
        else:
            print(f"{Fore.CYAN}🚀 Reasoning Path Processor - True Concurrent Processing Started")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}📁 Input Directory: {self.result_dir}")
        print(f"{Fore.CYAN}📁 Output Directory: {self.output_dir}")
        print(f"{Fore.YELLOW}💡 Tip: Press Ctrl+C to safely interrupt, processed data will be automatically saved")
        
        # Get tasks: if retry_failed_only is True, only get failed tasks
        if retry_failed_only:
            all_tasks = self.get_failed_tasks(model_name)
            if not all_tasks:
                print(f"{Fore.GREEN}✅ No failed tasks found to retry")
                return
            print(f"{Fore.YELLOW}🔄 Failed retry mode: only processing failed tasks")
        else:
            all_tasks = self.get_all_tasks(model_name)
        
        if not all_tasks:
            print(f"{Fore.GREEN}✅ All tasks completed, no tasks need processing")
            print(f"{Fore.CYAN}💡 Tip: If there were failed records before, they will be automatically reprocessed")
            return
        
        # Count failed retries
        failed_retry_count = 0
        if all_tasks:
            # Check first task's CSV file to count failed retries
            first_task = all_tasks[0]
            csv_file_check = Path(first_task[4])  # csv_file_path is 5th element
            if csv_file_check.exists():
                existing_data_check = self.load_existing_data(csv_file_check)
                for task in all_tasks:
                    task_index = task[1]  # index is 2nd element
                    if task_index in existing_data_check:
                        if existing_data_check[task_index].get('parse_status') == 'false':
                            failed_retry_count += 1
        
        new_task_count = len(all_tasks) - failed_retry_count
        
        print(f"{Fore.GREEN}📊 Total records to process: {len(all_tasks)}")
        if failed_retry_count > 0:
            print(f"{Fore.YELLOW}   - Failed retries: {failed_retry_count}")
        if new_task_count > 0:
            print(f"{Fore.CYAN}   - New tasks: {new_task_count}")
        print(f"{Fore.CYAN}💡 Supports incremental processing: completed records automatically skipped, failed records automatically retried")
        print(f"{Fore.CYAN}🔧 Using {self.max_workers} processes for true concurrent processing")
        print(f"{Fore.CYAN}⚡ All models' data will be processed simultaneously!")
        
        start_time = time.time()
        
        # Statistics
        processed_count = 0
        success_count = 0
        error_count = 0
        
        # Dictionary for saving results
        results_by_model = {}
        
        # Interrupt flag
        interrupted = threading.Event()
        
        # Define interrupt handler
        def signal_handler(sig, frame):
            print(f"\n{Fore.YELLOW}⚠️  Interrupt signal received, saving processed data...")
            interrupted.set()  # Set interrupt flag
        
        # Register signal handlers (Unix systems only)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Use process pool for true concurrent processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(self.process_single_item_true_concurrent, task): task for task in all_tasks}
                
                # Process completed tasks
                try:
                    for future in as_completed(future_to_task):
                        # Check interrupt flag
                        if interrupted.is_set():
                            print(f"\n{Fore.YELLOW}⚠️  Interrupt request detected, canceling remaining tasks and saving data...")
                            # Cancel all unfinished tasks
                            for f in future_to_task:
                                if not f.done():
                                    f.cancel()
                            break
                        
                        try:
                            model_name, index, result_data, success = future.result()
                            
                            # Group results by model for saving
                            if model_name not in results_by_model:
                                results_by_model[model_name] = {}
                            
                            results_by_model[model_name][index] = result_data
                            
                            if success:
                                success_count += 1
                            else:
                                error_count += 1
                            
                            processed_count += 1
                            
                            # Save every 50 records
                            if processed_count % 50 == 0:
                                self.save_all_results(results_by_model)
                                print(f"{Fore.BLUE}💾 Saved {processed_count}/{len(all_tasks)} records")
                            
                        except Exception as e:
                            print(f"{Fore.RED}❌ Error processing task: {e}")
                            error_count += 1
                except KeyboardInterrupt:
                    # Catch KeyboardInterrupt in loop
                    interrupted.set()
                    print(f"\n{Fore.YELLOW}⚠️  Interrupt request detected, canceling remaining tasks and saving data...")
                    # Cancel all unfinished tasks
                    for f in future_to_task:
                        if not f.done():
                            f.cancel()
            
            # Final save
            if interrupted.is_set():
                print(f"{Fore.YELLOW}⚠️  Saving processed data...")
            self.save_all_results(results_by_model)
            
            if interrupted.is_set():
                print(f"{Fore.GREEN}✅ Data saved, program exiting")
                sys.exit(0)
            
        except KeyboardInterrupt:
            # Handle keyboard interrupt (Windows and Unix)
            print(f"\n{Fore.YELLOW}⚠️  Keyboard interrupt signal received, saving processed data...")
            self.save_all_results(results_by_model)
            print(f"{Fore.GREEN}✅ Data saved, program exiting")
            sys.exit(0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}📈 Reasoning Path Processor Completion Statistics:")
        print(f"{Fore.GREEN}   - Total tasks: {len(all_tasks)}")
        print(f"{Fore.GREEN}   - Successfully processed: {success_count}")
        print(f"{Fore.GREEN}   - Processing failed: {error_count}")
        print(f"{Fore.GREEN}   - Total time: {duration:.2f} seconds")
        if duration > 0:
            print(f"{Fore.GREEN}   - Average speed: {len(all_tasks)/duration:.2f} records/second")
        print(f"{Fore.GREEN}{'='*60}")
    
    def save_all_results(self, results_by_model: Dict[str, Dict]):
        """Save all model results (with error handling)"""
        for model_name, results in results_by_model.items():
            csv_file = self.output_dir / f"{model_name}.csv"
            
            try:
                # Load existing data
                existing_data = self.load_existing_data(csv_file)
                
                # Update data
                existing_data.update(results)
                
                # Save
                self.save_to_csv(csv_file, existing_data)
                print(f"{Fore.GREEN}💾 Saved {model_name}: {len(results)} records")
                
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to save {model_name}: {e}")
                # Continue processing other models, don't interrupt entire flow

def main():
    parser = argparse.ArgumentParser(description='Reasoning Path Processor - Read and analyze data from Excel files')
    parser.add_argument('--model', type=str, help='Specify model name to process (directory name)')
    parser.add_argument('--result-dir', type=str, default=None, help='Excel file directory (default: script_dir/input_data)')
    parser.add_argument('--output-dir', type=str, default=None, help='CSV output directory (default: script_dir/processed_output)')
    parser.add_argument('--question-file', type=str, default=None, help='Question index Excel file (optional)')
    parser.add_argument('--workers', type=int, default=8, help='Number of concurrent processes')
    parser.add_argument('--retry-failed', action='store_true', help='Only process failed tasks (parse_status == false)')
    parser.add_argument('--api-base-url', type=str, default=None, help='API base URL (optional)')
    parser.add_argument('--api-key', type=str, default=None, help='API key (optional)')
    parser.add_argument('--model-name', type=str, default=None, help='Model name for API (optional)')
    
    args = parser.parse_args()
    
    processor = ReasoningPathProcessor(
        args.result_dir, 
        args.output_dir, 
        args.workers,
        args.question_file,
        args.api_base_url,
        args.api_key,
        args.model_name
    )
    processor.process_all_models_true_concurrent(args.model, retry_failed_only=args.retry_failed)

if __name__ == "__main__":
    main()

