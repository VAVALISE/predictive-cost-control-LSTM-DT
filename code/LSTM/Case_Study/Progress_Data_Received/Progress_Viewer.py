"""
Progress Viewer - Flexible Progress Inspection Tool
===================================================
Dynamically detects available progress models and allows flexible viewing and comparison.

Features:
- Automatically detect all available progress models
- Display models sorted by time
- View the latest progress
- View progress at a specified time point
- Compare progress trends across multiple time points
- Export progress data

"""


import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

# Import required modules
try:
    from forge_config import load_forge_config, validate_config
    from ACC_File_Tool import ForgeClient
    from urn_mapper import URNMapper
    from Progress_Adapter import ForgeProgressService, ProgressAccuracyEvaluator
except ImportError as e:
    print(f"Error: Missing required modules: {e}")
    print("Please ensure Progress_Adapter.py is in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ProgressViewer")


class ProgressViewer:
    """
    Flexible progress viewer with dynamic model management.
    """

    def __init__(self):
        """Initialize the progress viewer."""
        # Initialize URN mapper
        self.urn_mapper = URNMapper("urn_mapping.json")

        # Initialize Forge service
        config = load_forge_config()
        if not validate_config(config):
            raise ValueError("Forge configuration invalid")

        client = ForgeClient(config['client_id'], config['client_secret'])
        self.forge_service = ForgeProgressService(client)

        logger.info("ProgressViewer initialized")

    def get_available_models(self) -> List[Dict]:
        """
        Get all available progress models (sorted by time).

        Returns:
            List of dicts with model info:
            [{
                'filename': 'M01_Foundation.rvt',
                'urn': 'urn:adsk...',
                'size_mb': 21.85,
                'uploaded_at': '2024-11-12T15:00:29',
                'month': 1
            }]
        """
        files = self.urn_mapper.list_files()

        if not files:
            logger.warning("No models found in URN mapping")
            return []

        models = []
        for filename in files:
            mapping = self.urn_mapper.get_mapping(filename)
            if mapping and 'urn' in mapping:
                # Extract month number from filename (e.g., M01 -> 1)
                month = self._extract_month_from_filename(filename)

                models.append({
                    'filename': filename,
                    'urn': mapping['urn'],
                    'size_mb': mapping.get('file_size_mb', 0),
                    'uploaded_at': mapping.get('last_updated', ''),
                    'month': month
                })

        # Sort by month or upload time
        models.sort(key=lambda x: (x['month'] if x['month'] else 999, x['uploaded_at']))

        return models

    def _extract_month_from_filename(self, filename: str) -> Optional[int]:
        """Extract month index from filename."""
        import re
        match = re.search(r'M(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def display_available_models(self, models: List[Dict]):
        """
        Display the list of available models.

        Args:
            models: List of model dicts
        """
        print("\n" + "="*70)
        print(f"Available progress models (total {len(models)})")
        print("="*70)

        if not models:
            print("No models found.")
            return

        print(f"\n{'Index':<6} {'Filename':<35} {'Size':<10} {'Uploaded at':<20}")
        print("-"*70)

        for i, model in enumerate(models, 1):
            filename = model['filename']
            size = f"{model['size_mb']:.2f} MB" if model['size_mb'] else "N/A"
            upload_time = model['uploaded_at'][:19] if model['uploaded_at'] else "N/A"

            # Highlight the latest model
            prefix = "→ " if i == len(models) else"  "

            print(f"{prefix}{i:<4} {filename:<35} {size:<10} {upload_time:<20}")

        print("="*70)
        print(f"\n→ Latest model: {models[-1]['filename']}")
        print(f"  Total time points: {len(models)}")

    def get_latest_progress(self) -> Optional[Dict]:
        """
        Get the latest progress data.

        Returns:
            Progress dict or None
        """
        models = self.get_available_models()

        if not models:
            print("\nNo progress models found.")
            return None

        latest_model = models[-1]

        print(f"\nExtracting latest progress: {latest_model['filename']}")
        print("-"*70)

        progress, error = self.forge_service.extract_progress_data(latest_model['urn'])

        if error:
            print(f"Extraction failed: {error}")
            return None

        # Add model info
        progress['model_info'] = latest_model

        self._display_progress_summary(progress, latest_model)

        return progress

    def get_specific_progress(self, model_index: int = None, month: int = None) -> Optional[Dict]:
        """
        Get progress at a specific time point.

        Args:
            model_index: Model index (1-based)
            month: Month index (1–24)

        Returns:
            Progress dict or None
        """
        models = self.get_available_models()

        if not models:
            print("\nNo progress models found.")
            return None

        # Select model
        if month:
            # Find by month number
            model = next((m for m in models if m['month'] == month), None)
            if not model:
                print(f"\nNo model found for month {month}.")
                return None
        elif model_index:
            # Find by index
            if 1 <= model_index <= len(models):
                model = models[model_index - 1]
            else:
                print(f"\nInvalid index: {model_index} (valid range: 1-{len(models)})")
                return None
        else:
            print("\nYou must specify either model_index or month.")
            return None

        print(f"\nExtracting progress: {model['filename']}")
        print("-"*70)

        progress, error = self.forge_service.extract_progress_data(model['urn'])

        if error:
            print(f"Extraction failed: {error}")
            return None

        # Add model info
        progress['model_info'] = model

        self._display_progress_summary(progress, model)

        return progress

    def _display_progress_summary(self, progress: Dict, model_info: Dict):
        """Display progress summary (incremental detection version)."""
        print("\n" + "=" * 70)
        print("Progress summary (Incremental Detection)")
        print("=" * 70)
        print(f"Model file: {model_info['filename']}")
        print(f"Uploaded at: {model_info['uploaded_at'][:19]}")
        print(f"File size: {model_info['size_mb']:.2f} MB")
        print("-" * 70)
        print(f"Total components: {progress.get('total_components', 0):,}")
        print(f"Delta components: {progress.get('delta_components', 0):+,} (vs last detection)")
        print(f"Current stage: {progress.get('current_stage', 'N/A')}")
        if progress.get('previous_stage'):
            print(f"Previous stage: {progress.get('previous_stage')}")
        print("-" * 70)
        print(f"Matched month: {progress.get('matched_month', 'N/A')}")
        print(f"Match confidence: {progress.get('confidence', 0):.2f}")
        print("-" * 70)
        print(f"Completion (contract/REVIT weighted): {progress['completion_percentage']:.2f}%")
        print("=" * 70)

        if 'category_counts' in progress:
            print("\nComponent category distribution:")
            for cat, count in progress['category_counts'].items():
                if count > 0:
                    print(f"  {cat}: {count:,} components")

        print()

    def compare_progress_trend(self, model_indices: List[int] = None) -> Dict:
        """
        Compare progress trends across multiple time points.

        Args:
            model_indices: Indices of models to compare (if None, use all models).

        Returns:
            Dict with trend data
        """
        models = self.get_available_models()

        if not models:
            print("\nNo progress models found.")
            return {}

        # Select models to compare
        if model_indices:
            selected_models = [models[i-1] for i in model_indices if 1 <= i <= len(models)]
        else:
            selected_models = models

        print(f"\nExtracting progress data for {len(selected_models)} time points...")
        print("="*70)

        trend_data = []

        for i, model in enumerate(selected_models, 1):
            print(f"\n[{i}/{len(selected_models)}] {model['filename']}")

            progress, error = self.forge_service.extract_progress_data(model['urn'], model['filename'])

            if error:
                print(f"  ✗ Failed: {error}")
                continue

            completion = progress['completion_percentage']
            stage = progress.get('current_stage', 'N/A')
            delta = progress.get('delta_components', 0)
            geometry = progress.get('geometry_pct')

            if geometry is None:
                geometry_text = "N/A"
            else:
                try:
                    import math
                    if math.isnan(geometry):
                        geometry_text = "N/A"
                    else:
                        geometry_text = f"{geometry:.2f}%"
                except (TypeError, ValueError):
                    geometry_text = "N/A"

            print(f"  ✓ Completion: {completion:.2f}% | Geometry: {geometry_text} | Stage: {stage} | Delta: {delta:+d}")

            trend_data.append({
                'month': model['month'] if model['month'] else i,
                'filename': model['filename'],
                'uploaded_at': model['uploaded_at'],
                'completion_percentage': completion,
                'matched_month': progress.get('matched_month', f"M{model['month']:02d}"),
                'confidence': progress.get('confidence', 0),
                'current_stage': stage,
                'delta_total': delta,
                'total_components': progress.get('total_components', 0)
            })

        print("\n" + "="*70)
        print("Progress trend analysis")
        print("="*70)

        if trend_data:
            self._display_trend_analysis(trend_data)

            # Accuracy evaluation
            print("\n" + "="*70)
            print("Accuracy evaluation (vs real project)")
            print("="*70)
            evaluator = ProgressAccuracyEvaluator()
            for data in trend_data:
                matched_month = data.get('matched_month', f"M{data['month']:02d}")
                evaluator.add_extracted_progress(matched_month, data['completion_percentage'])

            metrics = evaluator.calculate_metrics()
            if metrics:
                print(f"RMSE (Root Mean Squared Error):  {metrics['rmse']:.2f}%")
                print(f"MAE  (Mean Absolute Error): {metrics['mae']:.2f}%")
                print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
                print(f"Sample size: {metrics['sample_count']}")
                print("\nExporting results...")
                print("✓ Exported to progress_output/ folder")
                print("  - progress_trend_auto.csv")
                print("  - progress_accuracy_metrics.csv")
                print("  - progress_accuracy_plot.png")
            else:
                print("Cannot compute accuracy metrics (real project CSV may be missing)")

        return {'trend_data': trend_data, 'models': selected_models}

    def _display_trend_analysis(self, trend_data: List[Dict]):
        """Display trend analysis (incremental version)"""
        print(f"\n{'Month':<6} {'Stage':<25} {'Delta':<10} {'Completion':<10} {'Change':<10}")
        print("-"*80)

        prev_completion = 0
        for i, data in enumerate(trend_data):
            month = data['month']
            stage = data.get('current_stage', 'N/A')[:23]
            delta = data.get('delta_total', 0)
            completion = data['completion_percentage']

            if i == 0:
                change = "-"
                delta_str = "-"
            else:
                change = f"+{completion - prev_completion:.2f}%"
                delta_str = f"{delta:+d}"

            print(f"M{month:02d}    {stage:<25} {delta_str:<10} {completion:>6.2f}%    {change:<10}")
            prev_completion = completion

        print("-"*80)

        if len(trend_data) >= 2:
            first = trend_data[0]
            last = trend_data[-1]
            total_change = last['completion_percentage'] - first['completion_percentage']
            avg_change = total_change / (len(trend_data) - 1)

            print(f"\nTotal change: {total_change:.2f}%")
            print(f"Average change: {avg_change:.2f}% per time point")

            if avg_change > 0:
                remaining = 100 - last['completion_percentage']
                estimated_periods = remaining / avg_change
                print(f"Estimated remaining time points to reach 100%: {estimated_periods:.1f}")

    def export_to_csv(self, trend_data: List[Dict], output_path: str = "progress_output/progress_trend.csv"):
        """
        Export progress trend data to CSV.

        Args:
            trend_data: Trend data from compare_progress_trend()
            output_path: Output CSV file path
        """
        if not trend_data:
            print("\nNo data to export.")
            return

        # Prepare data for CSV
        df = pd.DataFrame(trend_data)

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_file, index=False)

        print(f"\n✓ Data has been exported to: {output_path}")
        print(f"  Contains {len(df)} time points")
        print(f"  Columns: {', '.join(df.columns)}")


def interactive_menu():
    """Interactive command-line menu"""
    viewer = ProgressViewer()

    while True:
        print("\n" + "="*70)
        print("Progress Viewer - Incremental Detection (Stage-Aware)")
        print("=" * 70)
        print("\nPlease select an option:")
        print("1. List all available progress models")
        print("2. View latest progress")
        print("3. View progress at a specific time point")
        print("4. Compare progress trend (all time points) + stage analysis")
        print("5. Compare progress trend (selected time points) + stage analysis")
        print("6. Export progress trend data to CSV")
        print("7. View detection history")
        print("8. Clear detection history (use with caution)")
        print("0. Exit")
        print("=" * 70)

        choice = input("\nEnter an option (0-8): ").strip()

        try:
            if choice == "0":
                print("\nGoodbye!")
                break

            elif choice == "1":
                models = viewer.get_available_models()
                viewer.display_available_models(models)

            elif choice == "2":
                viewer.get_latest_progress()

            elif choice == "3":
                models = viewer.get_available_models()
                if not models:
                    continue

                viewer.display_available_models(models)
                print("\nPlease choose how to select the model:")
                print("1. By index")
                print("2. By month index")

                sub_choice = input("\nSelect (1-2): ").strip()

                if sub_choice == "1":
                    index = int(input(f"Enter index (1-{len(models)}): ").strip())
                    viewer.get_specific_progress(model_index=index)
                elif sub_choice == "2":
                    month = int(input("Enter month index: ").strip())
                    viewer.get_specific_progress(month=month)

            elif choice == "4":
                result = viewer.compare_progress_trend()

                if result.get('trend_data'):
                    export = input("\nExport to CSV? (y/n): ").strip().lower()
                    if export == 'y':
                        viewer.export_to_csv(result['trend_data'])

            elif choice == "5":
                models = viewer.get_available_models()
                if not models:
                    continue

                viewer.display_available_models(models)

                print("\nEnter the indices of time points to compare (comma-separated)")
                print("Example: 1,5,10,15,20")
                indices_str = input("\nIndices: ").strip()
                indices = [int(x.strip()) for x in indices_str.split(',')]

                result = viewer.compare_progress_trend(model_indices=indices)

                if result.get('trend_data'):
                    export = input("\nExport to CSV? (y/n): ").strip().lower()
                    if export == 'y':
                        viewer.export_to_csv(result['trend_data'])

            elif choice == "6":
                # Export all available progress
                result = viewer.compare_progress_trend()
                if result.get('trend_data'):
                    output_path = input("\nOutput file path (default: progress_output/progress_trend.csv): ").strip()
                    if not output_path:
                        output_path = "progress_output/progress_trend.csv"
                    viewer.export_to_csv(result['trend_data'], output_path)

            elif choice == "7":
                # View detection history
                history_df = viewer.forge_service.get_history_summary()
                if history_df.empty:
                    print("\nNo detection history yet.")
                else:
                    print("\n" + "="*70)
                    print("Detection history")
                    print("="*70)
                    print(history_df.to_string(index=False))
                    print("="*70)
                    print(f"\nTotal {len(history_df)} records")

            elif choice == "8":
                # Clear history
                confirm = input("\nConfirm clearing all detection history? This cannot be undone! (yes/no): ").strip().lower()
                if confirm == 'yes':
                    viewer.forge_service.clear_history()
                    print("✓ Detection history cleared.")
                else:
                    print("✗ Operation cancelled.")

            else:
                print("\nInvalid option, please try again.")

        except Exception as e:
            logger.error(f"Operation failed: {e}", exc_info=True)
            print(f"\nError: {e}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    """
    Main entry point
    """
    print("\n" + "="*70)
    print("Progress Viewer - Flexible Progress Management System")
    print("=" * 70)
    print("\nFeatures:")
    print("✓ Automatically detect available progress models")
    print("✓ View latest progress")
    print("✓ View progress at any time point")
    print("✓ Compare progress trends")
    print("✓ Export progress data")
    print("="*70)

    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user, goodbye!")
    except Exception as e:
        logger.error(f"Program error: {e}", exc_info=True)
        print(f"\nProgram error: {e}")