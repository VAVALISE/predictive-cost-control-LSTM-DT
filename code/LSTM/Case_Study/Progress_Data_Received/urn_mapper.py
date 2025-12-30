"""
URN Mapping Manager
===================
Manages the mapping between uploaded files and their URNs.
Saves to JSON file for persistence across sessions.

Author: Enhanced version
Date: 2024-11-12
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger("URNMapper")


class URNMapper:
    """
    Manages URN mappings for uploaded files
    """

    def __init__(self, db_path: str = "urn_mapping.json"):
        """
        Initialize URN mapper

        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = Path(db_path)
        self.mappings = self._load_mappings()
        logger.info(f"URNMapper initialized with {len(self.mappings)} existing mappings")

    def _load_mappings(self) -> Dict[str, Dict]:
        """
        Load mappings from JSON file

        Returns:
            Dictionary of file mappings
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} URN mappings from {self.db_path}")
                    return data
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted URN mapping file: {e}. Starting fresh.")
                return {}
            except Exception as e:
                logger.error(f"Error loading URN mappings: {e}")
                return {}
        else:
            logger.info("No existing URN mapping file found. Starting fresh.")
            return {}

    def _save_mappings(self) -> bool:
        """
        Save mappings to JSON file

        Returns:
            True if saved successfully
        """
        try:
            # Create parent directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            if self.db_path.exists():
                backup_path = self.db_path.with_suffix('.json.bak')
                if backup_path.exists():
                    backup_path.unlink()
                self.db_path.rename(backup_path)

            # Save with pretty formatting
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.mappings, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.mappings)} URN mappings to {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save URN mappings: {e}")
            return False

    def add_mapping(self,
                    file_name: str,
                    urn: str,
                    item_id: str = None,
                    version_id: str = None,
                    file_path: str = None,
                    file_size_mb: float = None) -> bool:
        """
        Add or update a file URN mapping

        Args:
            file_name: Name of the file
            urn: URN from ACC/Forge
            item_id: ACC item ID (optional)
            version_id: ACC version ID (optional)
            file_path: Full path to the file (optional)
            file_size_mb: File size in MB (optional)

        Returns:
            True if saved successfully
        """
        mapping_data = {
            "urn": urn,
            "last_updated": datetime.now().isoformat(),
        }

        if item_id:
            mapping_data["item_id"] = item_id
        if version_id:
            mapping_data["version_id"] = version_id
        if file_path:
            mapping_data["file_path"] = file_path
        if file_size_mb:
            mapping_data["file_size_mb"] = round(file_size_mb, 2)

        self.mappings[file_name] = mapping_data

        # Save to file
        success = self._save_mappings()

        if success:
            logger.info(f"Saved URN mapping for: {file_name}")
            logger.info(f"URN: {urn}")
        else:
            logger.warning(f"✗ Failed to save URN mapping for: {file_name}")

        return success

    def get_urn(self, file_name: str) -> Optional[str]:
        """
        Get URN for a file

        Args:
            file_name: Name of the file

        Returns:
            URN if found, None otherwise
        """
        mapping = self.mappings.get(file_name)
        if mapping:
            return mapping.get("urn")
        return None

    def get_mapping(self, file_name: str) -> Optional[Dict]:
        """
        Get complete mapping data for a file

        Args:
            file_name: Name of the file

        Returns:
            Mapping dictionary if found, None otherwise
        """
        return self.mappings.get(file_name)

    def get_all_urns(self) -> Dict[str, str]:
        """
        Get all filename to URN mappings

        Returns:
            Dictionary mapping filenames to URNs
        """
        return {
            filename: data["urn"]
            for filename, data in self.mappings.items()
            if "urn" in data
        }

    def list_files(self) -> List[str]:
        """
        List all files with URN mappings

        Returns:
            List of filenames
        """
        return list(self.mappings.keys())

    def has_file(self, file_name: str) -> bool:
        """
        Check if file has a URN mapping

        Args:
            file_name: Name of the file

        Returns:
            True if mapping exists
        """
        return file_name in self.mappings

    def remove_mapping(self, file_name: str) -> bool:
        """
        Remove a URN mapping

        Args:
            file_name: Name of the file

        Returns:
            True if removed successfully
        """
        if file_name in self.mappings:
            del self.mappings[file_name]
            return self._save_mappings()
        return False

    def get_statistics(self) -> Dict:
        """
        Get statistics about URN mappings

        Returns:
            Dictionary with statistics
        """
        total = len(self.mappings)
        with_item_id = sum(1 for m in self.mappings.values() if "item_id" in m)
        with_version_id = sum(1 for m in self.mappings.values() if "version_id" in m)

        return {
            "total_mappings": total,
            "with_item_id": with_item_id,
            "with_version_id": with_version_id,
        }

    def print_summary(self):
        """
        Print a summary of URN mappings to console
        """
        print("\n" + "=" * 70)
        print("URN MAPPING SUMMARY")
        print("=" * 70)

        if not self.mappings:
            print("No URN mappings found.")
            return

        stats = self.get_statistics()
        print(f"Total files: {stats['total_mappings']}")
        print(f"With Item ID: {stats['with_item_id']}")
        print(f"With Version ID: {stats['with_version_id']}")
        print("\n" + "=" * 70)
        print("FILES:")
        print("=" * 70)

        for filename, data in sorted(self.mappings.items()):
            urn = data.get("urn", "N/A")
            size = data.get("file_size_mb", "N/A")
            updated = data.get("last_updated", "N/A")

            print(f"\n{filename}")
            print(f"  URN: {urn}")
            if size != "N/A":
                print(f"  Size: {size} MB")
            print(f"  Updated: {updated}")

        print("\n" + "=" * 70)


# Convenience function for quick URN lookup
def get_urn_for_file(file_name: str, db_path: str = "urn_mapping.json") -> Optional[str]:
    """
    Quick function to get URN for a file

    Args:
        file_name: Name of the file
        db_path: Path to URN mapping database

    Returns:
        URN if found, None otherwise
    """
    mapper = URNMapper(db_path)
    return mapper.get_urn(file_name)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("\n" + "="*70)
    print("URN MAPPER - EXAMPLE USAGE")
    print("="*70)

    # Initialize mapper
    mapper = URNMapper("urn_mapping.json")

    # Print current mappings
    mapper.print_summary()

    # Example: Add a mapping (for testing)
    # mapper.add_mapping(
    #     file_name="test_model.rvt",
    #     urn="dXJuOmFkc2sud2lwcHJvZDpmcy5maWxlOnZmLk1nMVdxVFdMVEtPc1NHcENFOUJxQUE_dmVyc2lvbj0x",
    #     item_id="urn:adsk.wipprod:dm.lineage:abc123",
    #     version_id="urn:adsk.wipprod:fs.file:vf.xyz789?version=1",
    #     file_size_mb=45.3
    # )

    print("\nURN Mapper ready for use")
