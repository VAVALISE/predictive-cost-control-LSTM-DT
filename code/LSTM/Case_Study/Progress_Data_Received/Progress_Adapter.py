"""
Progress Adapter - Incremental Detection Version
================================================
Tracks component changes over time to determine current construction stage.

Key Features:
- Stores historical detection records
- Calculates component delta (new vs previous)
- Analyzes component type distribution in delta
- Determines current construction stage
- Matches with Preview_progress_fusion.csv for weighted progress

Date: 2024-11-14 (Updated for fusion CSV)
"""

import os
import base64
import requests
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import pandas as pd

try:
    from forge_config import load_forge_config, validate_config
    from ACC_File_Tool import ForgeClient
    from urn_mapper import URNMapper
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()


DEBUG_UNCATEGORIZED_SUMMARY = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProgressAdapter")


def encode_urn_for_api(urn: str) -> str:
    """Encode URN to Base64 URL-safe format"""
    urn_bytes = urn.encode('utf-8')
    encoded = base64.urlsafe_b64encode(urn_bytes).decode('utf-8')
    return encoded.rstrip('=')


class ComponentIncrementalAnalyzer:
    """
    Incremental Component Analyzer

    Tracks component changes over time and determines current stage.
    """

    def __init__(
    self,
    lookup_csv: str = None,
):
    if lookup_csv is None:
        project_root = Path(__file__).parent
        lookup_csv = project_root / "data" / "Preview_progress_fusion.csv"
    self.lookup_csv = str(lookup_csv)

        self.history_file = "progress_history.json"
        self.history = []  # List of {month, total, delta, stage, ...}

        # Component category keywords (enhanced: Foundation–Interior)
        self.category_keywords = {
            'Foundation': [
                # English keywords
                'foundation', 'footing', 'strip footing', 'pad footing', 'raft', 'raft slab',
                'raft foundation', 'mat foundation', 'pile', 'pile cap', 'pilecap', 'pile-cap',
                'pile foundation', 'pile group', 'piled raft', 'ground beam', 'grade beam',
                'tie beam', 'blinding', 'lean concrete', 'base slab', 'basement slab',
                'basement', 'substructure', 'retaining wall', 'retaining', 'earthwork',
                'excavat', 'excavation', 'backfill', 'earth', 'soil',
                # Chinese keywords
                '基础', '条基', '承台', '筏板', '桩基', '独立基础', '地梁', '承台梁',
                '地基', '垫层', '基坑', '围护', '支护', '止水', '抗浮', '底板', '圆形桩', '桩'
            ],
            'MEP': [
                # Ductwork & air side
                'duct', 'ductwork', 'ducting', 'flex duct', 'grille', 'diffuser', 'nozzle',
                'hvac', 'ahu', 'air handling unit', 'vav', 'fan coil', 'fcu', 'exhaust fan',
                'return fan', 'supply fan', 'vent', 'ventilation',
                # Piping & fittings
                'pipe', 'piping', 'pipework', 'pipeline', 'chilled water', 'chws', 'chwr',
                'heating water', 'hw', 'water pipe', 'gas pipe', 'medical gas',
                'drain', 'floor drain', 'gully', 'waste water', 'storm water', 'rainwater',
                'pipe fitting', 'fitting', 'valve', 'ball valve', 'gate valve',
                'control valve', 'check valve', 'elbow', 'tee', 'flange', 'union',
                'coupling', 'reducer', 'strainer', 'strainer basket', 'flex', 'flexible joint',
                'riser', 'pipe riser',
                # Fire services
                'sprinkler', 'sprinkler head', 'spray head', 'fire hose', 'hydrant',
                'fire pump', 'jockey pump', 'fire fighting', 'fire protection',
                'smoke detector', 'heat detector', 'alarm bell', 'fire alarm',
                # Electrical distribution & lighting
                'cable', 'cable tray', 'tray', 'ladder tray', 'bus duct', 'busbar',
                'conduit', 'trunking', 'raceway', 'wireway',
                'panel', 'panelboard', 'distribution board', 'db', 'switchboard',
                'switchgear', 'mccb', 'rcbo', 'breaker',
                'transformer', 'generator', 'ups',
                'socket', 'outlet', 'receptacle', 'switch', 'light switch',
                'lighting', 'light fixture', 'luminaire', 'downlight', 'fitting light',
                # Controls & devices
                'bms', 'control panel', 'controller', 'sensor', 'thermostat', 'detector', 'device',
                # Generic service equipment
                'mechanical', 'plumbing', 'electrical', 'equipment', 'terminal',
                # Chinese keywords
                '管道', '管线', '配管', '风管', '水管', '喷淋', '喷头', '阀门', '弯头', '三通',
                '法兰', '接头', '支架', '吊架', '桥架', '线槽', '电缆', '配电箱',
                '照明', '灯具', '风机', '风口', '机房', '泵', '地漏', '排水', '给排水',
                '暖通', '空调', '机械设备', 'pvc-u', 'pvc'
            ],
            'Interior': [
                # Openings & partitions
                'door', 'door frame', 'door leaf', 'door set',
                'window', 'window frame', 'skylight',
                'curtain', 'curtain wall panel', 'glazing panel',
                'partition', 'stud wall', 'drywall partition', 'gypsum partition',
                # Ceilings & linings
                'ceiling', 'suspended ceiling', 'false ceiling', 'ceiling tile',
                'bulkhead', 'soffit', 'lining', 'wall lining', 'paneling',
                # Finishes
                'finish', 'finishing', 'tile', 'floor tile', 'wall tile',
                'stone finish', 'marble', 'granite', 'terrazzo',
                'paint', 'coating', 'plaster', 'skim coat', 'render',
                'drywall', 'gypsum', 'gyprock',
                'flooring', 'vinyl', 'linoleum', 'timber floor', 'wood floor',
                'carpet', 'carpet tile',
                # Joinery & fittings
                'railing', 'handrail', 'balustrade', 'guardrail',
                'trim', 'moulding', 'cornice', 'skirting', 'baseboard',
                'joinery', 'cabinet', 'cupboard', 'wardrobe',
                'counter', 'countertop', 'benchtop', 'vanity',
                # Chinese keywords
                '门', '门套', '窗', '幕墙', '隔墙', '内墙',
                '吊顶', '天花', '顶棚', '装修', '装饰', '饰面',
                '瓷砖', '地砖', '墙砖', '抹灰', '批荡',
                '踢脚线', '地毯', '木地板', '石材',
                '护栏', '栏杆', '扶手', '橱柜', '柜', '衣柜', '台面', '竖梃'
            ],
            'Superstructure': [
                # English keywords
                'column', 'beam', 'girder', 'joist', 'brace', 'bracing',
                'floor', 'floor slab', 'slab', 'deck', 'decking',
                'roof', 'roof slab', 'roof beam', 'roof deck',
                'wall', 'shear wall', 'core wall', 'core', 'lift core', 'lift shaft',
                'elevator shaft', 'stair', 'staircase', 'stair flight', 'landing',
                'ramp', 'parapet', 'parapet wall',
                'structural', 'structure', 'frame', 'framing',
                'concrete', 'reinforced concrete', 'rc',
                'steel', 'steelwork', 'steel frame',
                'level', 'storey', 'story',
                # Chinese keywords
                '柱', '梁', '楼板', '板', '屋面', '屋顶', '墙', '剪力墙',
                '核心筒', '楼梯', '梯段', '平台', '坡道', '结构', '钢结构',
                '混凝土', '圈梁', '楼层', '层'
            ],
            'Outdoor': [
                'road', 'pavement', 'landscape', 'fence', 'gate',
                'lighting pole', 'bollard', 'furniture', 'signage',
                # Chinese keywords
                '雨棚', '玻璃轻钢雨棚'
            ]
        }

        self.category_category_keywords = {
            'Foundation': [
                'structural foundations', 'footings', 'pile foundations', 'foundation',
                '基礎', '基础', '桩基', '承台'
            ],
            'Superstructure': [
                'structural columns', 'structural framing', 'structural walls',
                'columns', 'frames', 'framing',
                'walls', 'floors', 'roofs', 'stairs', 'ramps', 'railings',
                '楼板', '柱', '梁', '剪力墙', '楼梯', '坡道'
            ],
            'MEP': [
                'ducts', 'duct fittings', 'duct accessories',
                'pipes', 'pipe fittings', 'pipe accessories',
                'cable trays', 'cable tray fittings',
                'conduits', 'conduit fittings',
                'mechanical equipment', 'plumbing fixtures',
                'air terminals', 'sprinklers',
                'lighting fixtures', 'electrical fixtures',
                'data devices', 'communication devices', 'security devices',
                '机械设备', '给排水', '风管', '喷淋', '照明', '电气'
            ],
            'Interior': [
                'ceilings', 'doors', 'windows',
                'generic models', 'casework',
                'furniture', 'furniture systems',
                'specialty equipment', 'entourage',
                '室内', '吊顶', '天花', '门', '窗', '家具', '橱柜'
            ],
            'Outdoor': [
                'topography', 'planting', 'site', 'parking', 'roads',
                'pads', 'plaza', '景观', '绿化', '场地', '道路'
            ]
        }


        self.exclude_items = [
            'view', 'sheet', 'schedule', 'legend', 'annotation',
            'dimension', 'text', 'detail', 'callout', 'section',
            'elevation', 'level', 'grid', 'reference', 'camera'
        ]

        self._load_lookup_table()
        self._load_history()

    def _load_lookup_table(self):
        """
        Load lookup table for progress matching.

        Supports two formats:
        1. Legacy Preview_progress.csv:
           - Month, total, Cumulative_%_Weighted6, Foundation, Superstructure, MEP, Interior, Outdoor_proxy, Handover_proxy
        2. New Preview_progress_fusion.csv:
           - Month
           - Foundation_revit, Superstructure_revit, MEP_revit, Interior_revit, Outdoor_proxy, Handover_proxy
           - Total_revit, Total_APS
           - Cumulative_%_Weighted6, P_APS_scaled_0_90
           and other related columns.
        """
        try:
            lookup_path = Path(self.lookup_csv)
            if not lookup_path.exists():
                logger.error(f"Lookup table not found: {self.lookup_csv}")
                return

            df = pd.read_csv(lookup_path)

            if "Month" not in df.columns:
                logger.error("Lookup table must contain 'Month' column.")
                return

            # ------------------------------------------------------------------
            # 1) Normalize "total" column:
            #    Prefer APS Total_APS, fall back to Total_revit when missing.
            # ------------------------------------------------------------------
            if "Total_APS" in df.columns:
                df["total"] = pd.to_numeric(df["Total_APS"], errors="coerce")
                if "Total_revit" in df.columns:
                    total_revit_values = pd.to_numeric(df["Total_revit"], errors="coerce")
                    df["total"] = df["total"].fillna(total_revit_values)
            elif "total" in df.columns:
                df["total"] = pd.to_numeric(df["total"], errors="coerce")
            elif "Total_revit" in df.columns:
                df["total"] = pd.to_numeric(df["Total_revit"], errors="coerce")
            else:
                logger.error("Lookup table must contain 'total' / 'Total_APS' / 'Total_revit' column.")
                return

            # For any remaining NaN in total, forward-fill from the previous row.
            df["total"] = df["total"].ffill()

            # ------------------------------------------------------------------
            # 2) Parse progress percentage column
            # ------------------------------------------------------------------
            # Revit weight curve
            if "REVIT_CumWeighted6" in df.columns:
                if df["REVIT_CumWeighted6"].dtype == "object":
                    df["REVIT_CumWeighted6"] = (
                        df["REVIT_CumWeighted6"]
                        .astype(str)
                        .str.rstrip("%")
                        .replace("", np.nan)
                        .astype(float)
                    )
            elif "Cumulative_%_Weighted6" in df.columns:
                # Compatible with old column names
                if df["Cumulative_%_Weighted6"].dtype == "object":
                    df["REVIT_CumWeighted6"] = (
                        df["Cumulative_%_Weighted6"]
                        .astype(str)
                        .str.rstrip("%")
                        .replace("", np.nan)
                        .astype(float)
                    )
                else:
                    df["REVIT_CumWeighted6"] = df["Cumulative_%_Weighted6"]
            else:
                df["REVIT_CumWeighted6"] = np.nan

            # APS geometry progress
            if "APS_geometry_pct" in df.columns:
                if df["APS_geometry_pct"].dtype == "object":
                    df["APS_geometry_pct"] = (
                        df["APS_geometry_pct"]
                        .astype(str)
                        .str.rstrip("%")
                        .replace("", np.nan)
                        .astype(float)
                    )
            elif "P_APS_scaled_0_90" in df.columns:
                if df["P_APS_scaled_0_90"].dtype == "object":
                    df["APS_geometry_pct"] = (
                        df["P_APS_scaled_0_90"]
                        .astype(str)
                        .str.rstrip("%")
                        .replace("", np.nan)
                        .astype(float)
                    )
                else:
                    df["APS_geometry_pct"] = df["P_APS_scaled_0_90"]
            else:
                df["APS_geometry_pct"] = np.nan

            # ------------------------------------------------------------------
            # 3) Build DT_hybrid_pct (contract/REVIT weighted progress)
            # ------------------------------------------------------------------
            if "REVIT_CumWeighted6" in df.columns:
                df["DT_hybrid_pct"] = (
                    df["REVIT_CumWeighted6"]
                    .astype(float)
                    .ffill()
                    .fillna(0.0)
                )
            else:
                df["DT_hybrid_pct"] = 0.0


            # ------------------------------------------------------------------
            # 4) Generate stage_label (based on cumulative values)
            # ------------------------------------------------------------------
            df["stage_label"] = "Unknown"

            for idx in df.index:
                # Read the cumulative value
                f_val = float(df.loc[idx, "Foundation_revit"]) if pd.notna(df.loc[idx, "Foundation_revit"]) else 0
                s_val = float(df.loc[idx, "Superstructure_revit"]) if pd.notna(df.loc[idx, "Superstructure_revit"]) else 0
                mep_val = float(df.loc[idx, "MEP_revit"]) if pd.notna(df.loc[idx, "MEP_revit"]) else 0
                int_val = float(df.loc[idx, "Interior_revit"]) if pd.notna(df.loc[idx, "Interior_revit"]) else 0
                outdoor_proxy = int(df.loc[idx, "Outdoor_proxy"]) if pd.notna(df.loc[idx, "Outdoor_proxy"]) else 0
                handover_proxy = int(df.loc[idx, "Handover_proxy"]) if pd.notna(df.loc[idx, "Handover_proxy"]) else 0

                # Judgment Phase
                has_f = f_val > 0
                has_s = s_val > 0
                has_m = mep_val > 0
                has_i = int_val > 0
                has_out = outdoor_proxy > 0
                has_h = handover_proxy > 0

                if has_h and has_out:
                    stage = "Outdoor & Handover"
                elif has_h:
                    stage = "Handover"
                elif has_out and has_m and has_i:
                    stage = "MEP & Interior & Outdoor"
                elif has_out and has_i:
                    stage = "Interior & Outdoor"
                elif has_out:
                    stage = "Outdoor"
                elif has_m and has_i and (has_s or has_f):
                    stage = "Superstructure & MEP & Interior"
                elif has_m and has_i:
                    stage = "MEP & Interior"
                elif has_i:
                    stage = "Interior"
                elif has_m and has_s:
                    stage = "Superstructure & MEP"
                elif has_m:
                    stage = "MEP"
                elif has_s:
                    stage = "Superstructure"
                elif has_f:
                    stage = "Foundation"
                else:
                    stage = "Unknown"

                df.loc[idx, "stage_label"] = stage

            # ------------------------------------------------------------------
            # 5) Phased Unification
            # ------------------------------------------------------------------
            if "Foundation" not in df.columns and "Foundation_revit" in df.columns:
                df["Foundation"] = df["Foundation_revit"]
            if "Superstructure" not in df.columns and "Superstructure_revit" in df.columns:
                df["Superstructure"] = df["Superstructure_revit"]
            if "MEP" not in df.columns and "MEP_revit" in df.columns:
                df["MEP"] = df["MEP_revit"]
            if "Interior" not in df.columns and "Interior_revit" in df.columns:
                df["Interior"] = df["Interior_revit"]

            for col in ["Outdoor_proxy", "Handover_proxy"]:
                if col not in df.columns:
                    df[col] = np.nan

            self.lookup_data = df
            logger.info(f"✓ Loaded lookup table from: {self.lookup_csv}")
            logger.info(f"  {len(df)} months available")

        except Exception as e:
            logger.error(f"Failed to load lookup table: {e}")

    def _load_history(self):
        """Load historical detection records"""
        try:
            if Path(self.history_file).exists():
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"✓ Loaded history: {len(self.history)} records")
            else:
                logger.info("No history file found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
            self.history = []

    def _save_history(self):
        """Save detection history"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"✓ Saved history: {len(self.history)} records")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def detect_components_by_category(self, metadata: Dict, manifest: Dict) -> Dict[str, List[str]]:
        """
        Detect components and categorize them

        FIX: Uses guid/objectid for unique identification to avoid duplicate counting

        Returns:
            Dict with component lists by category
        """
        categorized = {
            'Foundation': [],
            'Superstructure': [],
            'MEP': [],
            'Interior': [],
            'Outdoor': [],
            'Uncategorized': []
        }

        detected_items = {}  #Changed from list to dict: {guid: name}

        # Extract from metadata
        if metadata:
            try:
                if "data" in metadata and "metadata" in metadata["data"]:
                    for item in metadata["data"]["metadata"]:
                        item_name = item.get("name", "").lower()
                        item_role = item.get("role", "").lower()
                        item_guid = item.get("guid", "") or item.get("objectid", "")

                        if self._is_valid_component(item_name) or self._is_valid_component(item_role):
                            # Use guid as key to ensure uniqueness
                            if item_guid:
                                detected_items[item_guid] = item_name or item_role or "component"
                            else:
                                # Fallback: use name+index if no guid available
                                unique_key = f"{item_name or item_role}_{len(detected_items)}"
                                detected_items[unique_key] = item_name or item_role
            except Exception as e:
                logger.warning(f"Error extracting from metadata: {e}")

        # Extract from manifest
        if manifest:
            try:
                if "derivatives" in manifest:
                    for derivative in manifest["derivatives"]:
                        if "children" in derivative:
                            self._extract_from_children(derivative["children"], detected_items)
            except Exception as e:
                logger.warning(f"Error extracting from manifest: {e}")

        # Categorize components (now detected_items is a dict)
        for guid, item_name in detected_items.items():
            categorized_flag = False
            for category, keywords in self.category_keywords.items():
                if any(keyword in item_name for keyword in keywords):
                    categorized[category].append(guid)
                    categorized_flag = True
                    break

            if not categorized_flag:
                categorized['Uncategorized'].append(guid)

        # Already unique by guid, but keep set() for safety
        for category in categorized:
            categorized[category] = list(set(categorized[category]))

        total = sum(len(categorized[cat]) for cat in categorized)
        logger.info(f"✓ Categorized {total} unique components")

        return categorized

    def _extract_from_children(self, children: List, detected_items: Dict):
        """
        FIX: Modified to work with dict instead of list
        Recursively extract components from manifest children
        """
        for child in children:
            child_type = child.get("type", "").lower()
            child_name = child.get("name", "").lower()
            child_guid = child.get("guid", "") or child.get("objectid", "")

            if child_type == "geometry" or self._is_valid_component(child_name):
                # Use guid if available, otherwise create unique key
                if child_guid:
                    detected_items[child_guid] = child_name or child_type
                else:
                    # Use name+type+index as fallback unique key
                    unique_key = f"{child_name or child_type}_{len(detected_items)}"
                    detected_items[unique_key] = child_name or child_type

            if "children" in child:
                self._extract_from_children(child["children"], detected_items)


    def detect_components_from_properties(self, properties_data: Dict) -> Dict[str, List[str]]:
        """
        NEW: Extract components from Properties API data.
        The Properties API contains the full object list and attributes.
        This is the correct source for obtaining the real component count.
        """
        categorized = {
            'Foundation': [],
            'Superstructure': [],
            'MEP': [],
            'Interior': [],
            'Outdoor': [],
            'Uncategorized': []
        }

        if not properties_data or "data" not in properties_data:
            logger.warning("Properties data is empty or invalid")
            return categorized

        collection = properties_data["data"].get("collection", [])
        logger.info(f"Processing {len(collection)} objects from Properties API")

        detected_items = {}

        for obj in collection:
            objectid = obj.get("objectid", "")
            name = obj.get("name", "").lower()

            # Get Category property (supporting APS nested structure + displayValue).
            props = obj.get("properties", {})
            category = ""

            # Typical structures (two common formats):
            # 1) { "Identity Data": { "Category": "Walls", ... }, ... }
            # 2) { "Identity Data": { "Category": { "displayValue": "Walls", ... }, ... }, ... }
            for group in props.values():
                if isinstance(group, dict):
                    for key, val in group.items():
                        key_l = str(key).lower()
                        if key_l in ["category", "类别", "revit category"]:
                            if isinstance(val, dict):
                                raw_val = (
                                        val.get("displayValue")
                                        or val.get("value")
                                        or ""
                                )
                            else:
                                raw_val = val
                            category = str(raw_val).lower()
                            break
                if category:
                    break


            # Exclude non-component items.
            exclude_categories = ['views', 'sheets', 'schedules', 'legends']
            is_excluded = any(ex in category or ex in name for ex in exclude_categories)

            if not is_excluded and (name or category):
                identifier = name or category or "component"
                detected_items[str(objectid)] = {
                    "identifier": identifier,
                    "category": category
                }

        # Categorization: first by Revit Category, then fall back to name keywords.
        for objectid, info in detected_items.items():
            item_name = info.get("identifier", "") or ""
            cat_str = info.get("category", "") or ""

            categorized_flag = False

            # 1) Matching based on the Revit Category field
            for stage, cat_keywords in self.category_category_keywords.items():
                if any(kw in cat_str for kw in cat_keywords):
                    categorized[stage].append(objectid)
                    categorized_flag = True
                    break

            # 2) If the Category is not matched, then use name keywords for matching.
            if not categorized_flag:
                for stage, keywords in self.category_keywords.items():
                    if any(kw in item_name for kw in keywords):
                        categorized[stage].append(objectid)
                        categorized_flag = True
                        break

            # 3) If it is still not recognized, go to Uncategorized.
            if not categorized_flag:
                categorized['Uncategorized'].append(objectid)

        total = sum(len(categorized[cat]) for cat in categorized)
        logger.info(f"✓ Categorized {total} unique components from Properties API")

        # === Debug: report most frequent categories and identifiers in Uncategorized ===
        try:
            from collections import Counter

            unc_cat = []
            unc_name = []
            for oid in categorized['Uncategorized']:
                info = detected_items.get(str(oid)) or detected_items.get(oid)
                if not info:
                    continue
                cat = info.get("category") or ""
                name = info.get("identifier") or ""
                if cat:
                    unc_cat.append(cat)
                if name:
                    unc_name.append(name)

            if DEBUG_UNCATEGORIZED_SUMMARY and (unc_cat or unc_name):
                top_cats = Counter(unc_cat).most_common(10)
                top_names = Counter(unc_name).most_common(10)

                logger.info("Top 10 Uncategorized categories:")
                for c, cnt in top_cats:
                    logger.info(f"  {cnt:5d} × {c}")

                logger.info("Top 10 Uncategorized identifiers:")
                for n, cnt in top_names:
                    logger.info(f"  {cnt:5d} × {n}")
        except Exception as debug_e:
            logger.warning(f"Debug summary for Uncategorized failed: {debug_e}")
        # === End of debug block ===

        return categorized


    def _is_valid_component(self, item_name: str) -> bool:
        """Check if valid component"""
        if not item_name:
            return False

        for exclude_term in self.exclude_items:
            if exclude_term in item_name.lower():
                return False

        return True

    def analyze_incremental(self, categorized_components: Dict, model_name: str = "") -> Dict:
        """
        Analyze with incremental detection

        Change: first match progress, then determine stage based on the matched month.
        """
        # Count total
        total_components = sum(len(comp_list) for comp_list in categorized_components.values())

        # Get previous state
        previous_total = 0
        previous_stage = None
        if self.history:
            last_record = self.history[-1]
            previous_total = last_record['total']
            previous_stage = last_record.get('current_stage', None)

        delta_total = total_components - previous_total

        category_counts = {cat: len(comp_list) for cat, comp_list in categorized_components.items()}

        match_result = self._match_progress(total_components, "", model_name)
        matched_month = match_result['matched_month']

        current_stage = self._determine_current_stage(category_counts, previous_stage,
                                                      delta_total, matched_month)

        # Build analysis result - NOW INCLUDE DBID LISTS
        analysis = {
            'model_name': model_name,
            'total': total_components,
            'previous_total': previous_total,
            'delta_total': delta_total,
            'category_counts': category_counts,
            'categorized_dbids': categorized_components,  # STORE ACTUAL DBID LISTS
            'current_stage': current_stage,
            'previous_stage': previous_stage,
            'matched_month': match_result['matched_month'],
            'progress_percentage': match_result['progress_percentage'],
            'geometry_pct': match_result.get('geometry_pct', 0.0),  # APS几何进度
            'confidence': match_result['confidence'],
            'timestamp': datetime.now().isoformat()
        }

        # Add to history
        self.history.append(analysis)
        self._save_history()

        logger.info(f"✓ Analysis: Stage={current_stage}, Delta={delta_total:+d}, " +
                    f"Progress={match_result['progress_percentage']:.2f}%")

        return analysis

    def _determine_current_stage(
        self,
        category_counts: Dict,
        previous_stage: Optional[str],
        delta: int,
        matched_month: str = None,
    ) -> str:
        """
        Determine current construction stage using the precomputed stage_label in the lookup table.
        """
        if self.lookup_data is None or not matched_month:
            return "Unknown"

        matched_rows = self.lookup_data[self.lookup_data["Month"] == matched_month]
        if matched_rows.empty:
            return "Unknown"

        row = matched_rows.iloc[0]

        if "stage_label" in row.index and pd.notna(row["stage_label"]):
            return str(row["stage_label"])

        return "Unknown"




    def _match_progress(self, total_components: int, current_stage: str, model_name: str = "") -> Dict:
        """
        Match with Preview_progress_fusion.csv (with tolerance and filename fallback).

        Rules:
        1. If total_components < 10, infer the month directly from the filename.
        2. Otherwise, find the row in lookup_data['total'] with the largest value
           that is still <= total_components.
        3. Once a row is matched:
           - If P_APS_scaled_0_90 exists and is not NaN, it can be used as an APS-based curve;
           - In this revised version, DT_hybrid_pct (based on REVIT_CumWeighted6) is used
             as the main completion_percentage, and APS_geometry_pct is stored separately
             as geometry_pct for diagnostics.
        """
        if self.lookup_data is None:
            return {
                "matched_month": None,
                "progress_percentage": 0.0,
                "confidence": 0.0,
            }

        # Safeguard: if there are no components, treat it as M01.
        if total_components == 0:
            base_row = self.lookup_data[self.lookup_data["Month"] == "M01"]
            if not base_row.empty:
                row = base_row.iloc[0]
                progress_pct = float(row.get("DT_hybrid_pct", 0.0))

                raw_geometry = row.get("APS_geometry_pct", np.nan)
                if pd.isna(raw_geometry) or raw_geometry == "":
                    geometry_pct = None
                else:
                    geometry_pct = float(raw_geometry)
            else:
                progress_pct = 0.0
                geometry_pct = None

            return {
                "matched_month": "M01",
                "progress_percentage": progress_pct,
                "geometry_pct": geometry_pct,
                "confidence": 0.0,
            }

        # Safeguard: if component count is too low, fall back to filename.
        if total_components < 10:
            logger.warning(f"⚠ Component count ({total_components}) too low, using filename fallback")
            return self._fallback_to_filename(model_name)

        # Use the normalized total column (prefer APS Total_APS).
        reached_rows = self.lookup_data[self.lookup_data["total"] <= total_components]

        if reached_rows.empty:
            logger.warning(f"⚠ Component count ({total_components}) below M01, using filename fallback")
            return self._fallback_to_filename(model_name)

        # Find the maximum total value that is <= total_components.
        max_total = reached_rows["total"].max()
        candidates = reached_rows[reached_rows["total"] == max_total]

        if len(candidates) == 1:
            matched_row = candidates.iloc[0]
        else:
            month_from_filename = None
            import re
            m = re.search(r"M(\d{2})", model_name or "", re.IGNORECASE)
            if m:
                month_from_filename = f"M{int(m.group(1)):02d}"

            if month_from_filename and month_from_filename in candidates["Month"].values:
                matched_row = candidates[candidates["Month"] == month_from_filename].iloc[0]
                logger.info(
                    f"Multiple months share total={int(max_total)}, using filename-based match: {month_from_filename}"
                )
            else:
                matched_row = candidates.iloc[-1]
                logger.warning(
                    f"Multiple months share total={int(max_total)}, filename match failed, using last candidate {matched_row.get('Month', 'UNKNOWN')}"
                )

        logger.info(
            f"✓ Matched {total_components} components to {matched_row['Month']} "
            f"(table: {int(matched_row['total'])} components)"
        )

        progress_pct = float(matched_row.get("DT_hybrid_pct", 0.0))

        raw_geometry = matched_row.get("APS_geometry_pct", np.nan)
        if pd.isna(raw_geometry) or raw_geometry == "":
            geometry_pct = None
        else:
            geometry_pct = float(raw_geometry)

        return {
            "matched_month": matched_row["Month"],
            "progress_percentage": round(progress_pct, 2),
            "geometry_pct": round(geometry_pct, 2) if geometry_pct is not None else None,
            "confidence": 1.0,
        }


    def _fallback_to_filename(self, model_name: str) -> Dict:
        """
        Use the month encoded in the filename as a fallback, e.g.:
        M16_MEP&Construction.rvt → M16

        Completion is taken from DT_hybrid_pct in the lookup table
        (equivalent to REVIT_CumWeighted6 in this setup), and
        APS_geometry_pct is stored separately as geometry_pct for display
        and diagnostic purposes only.
        """
        if not model_name or self.lookup_data is None:
            return {
                "matched_month": None,
                "progress_percentage": 0.0,
                "confidence": 0.0,
            }

        import re

        match = re.search(r"M(\d{2})", model_name, re.IGNORECASE)
        if not match:
            logger.error(f"✗ Could not extract month from filename: {model_name}")
            return {
                "matched_month": None,
                "progress_percentage": 0.0,
                "confidence": 0.0,
            }

        month_num = int(match.group(1))
        matched_month = f"M{month_num:02d}"

        matched_rows = self.lookup_data[self.lookup_data["Month"] == matched_month]
        if matched_rows.empty:
            logger.error(f"✗ Month {matched_month} not found in lookup table")
            return {
                "matched_month": None,
                "progress_percentage": 0.0,
                "confidence": 0.0,
            }

        row = matched_rows.iloc[0]
        progress_pct = float(row.get("DT_hybrid_pct", 0.0))

        raw_geometry = row.get("APS_geometry_pct", np.nan)
        if pd.isna(raw_geometry) or raw_geometry == "":
            geometry_pct = None
        else:
            geometry_pct = float(raw_geometry)

        logger.info(f"✓ Filename fallback: {model_name} → {matched_month} → {progress_pct:.2f}%")

        return {
            "matched_month": matched_month,
            "progress_percentage": round(progress_pct, 2),
            "geometry_pct": round(geometry_pct, 2) if geometry_pct is not None else None,
            "confidence": 0.8,
        }


    def get_history_summary(self) -> pd.DataFrame:
        """Get history as DataFrame"""
        if not self.history:
            return pd.DataFrame()

        df = pd.DataFrame(self.history)
        return df[['model_name', 'total', 'delta_total', 'current_stage',
                   'progress_percentage', 'matched_month']]


class ForgeProgressService:
    """Forge service with incremental detection"""

    def __init__(self, client: ForgeClient = None):
        if client:
            self.client = client
        else:
            config = load_forge_config()
            if not validate_config(config):
                raise ValueError("Forge credentials not configured")
            self.client = ForgeClient(config['client_id'], config['client_secret'])

        self.urn_mapper = URNMapper("urn_mapping.json")
        self.incremental_analyzer = ComponentIncrementalAnalyzer()

    def get_urn_from_mapping(self, file_name: str) -> Optional[str]:
        return self.urn_mapper.get_urn(file_name)

    def get_token(self) -> Optional[str]:
        try:
            return self.client.authenticate()
        except Exception as e:
            logger.error(f"Auth failed: {e}")
            return None

    def get_model_metadata(self, model_urn: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            token = self.get_token()
            if not token:
                return None, "Auth failed"

            encoded_urn = encode_urn_for_api(model_urn)
            url = f"https://developer.api.autodesk.com/modelderivative/v2/designdata/{encoded_urn}/metadata"
            headers = {"Authorization": f"Bearer {token}"}

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json(), None

        except requests.RequestException as e:
            return None, str(e)

    def get_model_manifest(self, model_urn: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            token = self.get_token()
            if not token:
                return None, "Auth failed"

            encoded_urn = encode_urn_for_api(model_urn)
            url = f"https://developer.api.autodesk.com/modelderivative/v2/designdata/{encoded_urn}/manifest"
            headers = {"Authorization": f"Bearer {token}"}

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json(), None

        except requests.RequestException as e:
            return None, str(e)

    def get_model_properties(self, model_urn: str, guid: str = None, force: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Get model properties via the Properties API, with optional force parameter.

        Args:
            model_urn: Model URN
            guid: View GUID; if not provided, the 3D view GUID will be auto-detected
            force: Whether to force retrieval (by appending ?forceget=true), useful for large models

        Returns:
            Tuple[properties_data, error_message]
        """
        try:
            token = self.get_token()
            if not token:
                return None, "Auth failed"

            # Get guid if not provided
            if not guid:
                metadata, error = self.get_model_metadata(model_urn)
                if error or not metadata:
                    return None, "Failed to get metadata for guid"

                if "data" in metadata and "metadata" in metadata["data"]:
                    for item in metadata["data"]["metadata"]:
                        if item.get("role") == "3d":
                            guid = item.get("guid")
                            if guid:
                                logger.info(f"Found 3D view guid: {guid[:20]}...")
                                break

                if not guid:
                    logger.warning("No 3D view found, using first available guid")
                    if "data" in metadata and "metadata" in metadata["data"]:
                        metadata_list = metadata["data"]["metadata"]
                        if metadata_list:
                            guid = metadata_list[0].get("guid")

            if not guid:
                return None, "No guid available"

            # Build URL; if force=True, append ?forceget=true.
            encoded_urn = encode_urn_for_api(model_urn)
            url = f"https://developer.api.autodesk.com/modelderivative/v2/designdata/{encoded_urn}/metadata/{guid}/properties"

            if force:
                url += "?forceget=true"
                logger.info(f"Fetching properties with force=true (this may take longer)...")
            else:
                logger.info(f"Fetching properties (this may take a moment)...")

            headers = {"Authorization": f"Bearer {token}"}

            response = requests.get(url, headers=headers, timeout=120)  # 增加timeout到120秒
            response.raise_for_status()

            properties_data = response.json()

            if "data" in properties_data and "collection" in properties_data["data"]:
                object_count = len(properties_data["data"]["collection"])
                logger.info(f"✓ Fetched {object_count} objects from Properties API")

            return properties_data, None

        except requests.RequestException as e:
            logger.error(f"Properties API request failed: {e}")
            return None, str(e)


    def extract_progress_data(self, model_urn: str, filename: str = "") -> Tuple[Optional[Dict], Optional[str]]:
        """
        Extract progress using incremental detection with Properties API

        MODIFIED:
        - Now uses Properties API to get real component data
        - Auto-retry with force=true when encountering 413 error
        """
        try:
            logger.info(f"Extracting (Properties API + Incremental): {filename or 'Unknown'}")

            # Use Properties API to obtain real component data.
            properties, error = self.get_model_properties(model_urn)

            # If a 413 error occurs, retry once with force=True.
            if error and "413" in str(error):
                logger.warning(f"Received 413 error, retrying with force=true...")
                properties, error = self.get_model_properties(model_urn, force=True)

            if error:
                logger.error(f"Failed to get properties: {error}")
                # Fallback to old method
                logger.warning("Falling back to metadata/manifest method")
                metadata, _ = self.get_model_metadata(model_urn)
                manifest, _ = self.get_model_manifest(model_urn)
                categorized = self.incremental_analyzer.detect_components_by_category(metadata, manifest)
            else:
                # Extract components from Properties API data.
                categorized = self.incremental_analyzer.detect_components_from_properties(properties)

            # Incremental analysis (same as before)
            analysis = self.incremental_analyzer.analyze_incremental(categorized, filename)

            progress_data = {
                "completion_percentage": analysis['progress_percentage'],
                "geometry_pct": analysis.get('geometry_pct', None),
                "matched_month": analysis['matched_month'],
                "confidence": analysis['confidence'],
                "total_components": analysis['total'],
                "delta_components": analysis['delta_total'],
                "current_stage": analysis['current_stage'],
                "previous_stage": analysis['previous_stage'],
                "category_counts": analysis['category_counts'],
                "analysis_method": "Properties API + Incremental Detection",
                "model_name": filename,
                "timestamp": datetime.now().isoformat()
            }

            # Safely format geometry progress to avoid errors when None.
            geom_val = progress_data["geometry_pct"]
            if geom_val is None:
                geom_text = "N/A"
            else:
                try:
                    geom_text = f"{float(geom_val):.2f}%"
                except (TypeError, ValueError):
                    geom_text = "N/A"

            # Enhanced logging: show contract/REVIT weighted progress and APS geometry progress.
            logger.info(
                f"✓ Progress: {progress_data['completion_percentage']:.2f}% (Revit / contract) | "
                f"Geometry: {geom_text} (APS) | "
                f"Stage: {analysis['current_stage']} | Delta: {analysis['delta_total']:+d}"
            )

            return progress_data, None

        except Exception as e:
            return None, str(e)


    def get_history_summary(self) -> pd.DataFrame:
        """Get detection history summary"""
        return self.incremental_analyzer.get_history_summary()

    def clear_history(self):
        """Clear detection history (use with caution)"""
        self.incremental_analyzer.history = []
        self.incremental_analyzer._save_history()
        logger.info("✓ History cleared")


class ProgressAccuracyEvaluator:
    """Evaluates accuracy"""

    def __init__(self, real_csv: str = None):
        """
        real_csv:
            - If None, defaults to LSTM_DT/input_csv data/real_project/Chengbei_24m_work.csv
            - If you want to use another real-progress file, pass its absolute path explicitly.
        """
        if real_csv is None:
            base = Path(__file__).resolve().parents[3]
            real_csv = base / "input_csv data" / "real_project" / "Chengbei_24m_work.csv"

        self.real_csv = str(real_csv)
        self.real_data = None
        self.extracted_data = []

        if Path(self.real_csv).exists():
            self._load_real_data()
        else:
            logger.warning(f"Real progress CSV not found: {self.real_csv}")

    def _load_real_data(self):
        """Load the real project's progress curve."""
        try:
            df = pd.read_csv(self.real_csv, encoding='utf-8-sig')
            # month_index (1–24), cumulative_share_pct (0–100 percentage).
            self.real_data = df[['month_index', 'cumulative_share_pct']].copy()
            self.real_data.columns = ['month', 'actual_progress']

            logger.info(f"Loaded real data: {len(self.real_data)} months from {self.real_csv}")
        except Exception as e:
            logger.error(f"Load real progress CSV failed: {e}")
            self.real_data = None

    def add_extracted_progress(self, month_str: str, progress_pct: float, confidence: float = 1.0):
        """Record one extracted APS/DT progress value for later comparison with the real project."""
        try:
            month_num = int(str(month_str).replace('M', '').replace('m', ''))
            self.extracted_data.append({
                'month': month_num,
                'predicted_progress': progress_pct,
                'confidence': confidence
            })
        except Exception as e:
            logger.warning(f"Failed to add extracted progress for {month_str}: {e}")

    def calculate_metrics(self) -> Dict:
        if self.real_data is None or not self.extracted_data:
            return {}

        extracted_df = pd.DataFrame(self.extracted_data)
        merged = pd.merge(extracted_df, self.real_data, on='month', how='inner')

        if len(merged) == 0:
            return {}

        pred = merged['predicted_progress'].values
        actual = merged['actual_progress'].values

        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        mae = np.mean(np.abs(pred - actual))
        mape = np.mean(np.abs((pred - actual) / (actual + 1e-6))) * 100

        return {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'sample_count': len(merged)
        }

    def export_results(self, output_dir: str = "progress_output"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.extracted_data:
            df = pd.DataFrame(self.extracted_data)
            df.to_csv(output_path / "progress_trend_incremental.csv", index=False)

        metrics = self.calculate_metrics()
        if metrics:
            pd.DataFrame([metrics]).to_csv(output_path / "progress_accuracy_metrics.csv", index=False)

        self.plot_comparison(output_path / "progress_accuracy_plot.png")
        logger.info(f"✓ Exported accuracy results to {output_dir}")

    def plot_comparison(self, output_path: str):
        if self.real_data is None or not self.extracted_data:
            return

        try:
            extracted_df = pd.DataFrame(self.extracted_data)
            merged = pd.merge(extracted_df, self.real_data, on='month', how='outer').sort_values('month')

            plt.figure(figsize=(12, 6))
            plt.plot(merged['month'], merged['actual_progress'], 'o-', label='Real', linewidth=2, markersize=8)
            plt.plot(merged['month'], merged['predicted_progress'], 's--',
                     label='Extracted (Incremental)', linewidth=2, markersize=8, alpha=0.7)

            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Progress (%)', fontsize=12)
            plt.title('Incremental Detection vs Real Project', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300)
            plt.close()

        except Exception as e:
            logger.error(f"Plot comparison failed: {e}")


class ProgressAdapter:
    """Progress Adapter with incremental detection"""

    def __init__(self):
        self.forge = ForgeProgressService()

    def cleanup(self):
        pass


def export_viewer_coloring(month: int, output_path: str, lookup_csv: str = None):
    """
    Export dbId lists for completed vs incomplete elements at a given month,
    for use in Forge/APS Viewer theming.

    Args:
        month: Month number (e.g., 15 for M15)
        output_path: Output JSON file path
        lookup_csv: Path to Preview_progress_fusion.csv (optional)
    """
    if lookup_csv is None:
        project_root = Path(__file__).parent
        lookup_csv = project_root / "data" / "Preview_progress_fusion.csv"

    analyzer = ComponentIncrementalAnalyzer(lookup_csv=lookup_csv)

    if not analyzer.history:
        logger.error("No detection history found. Please run incremental detection first.")
        logger.info("Run Progress_Viewer.py option 4 to generate detection history for all months.")
        return

    month_str = f"M{month:02d}"
    matched_record = None

    for record in analyzer.history:
        if record.get('matched_month') == month_str:
            matched_record = record
            break

    if matched_record is None:
        logger.error(f"No detection record found for month {month_str}")
        available = [r.get('matched_month') for r in analyzer.history]
        logger.info(f"Available months in history: {available}")
        logger.info("Run Progress_Viewer.py option 4 to detect all months first.")
        return

    # Check if we have the dbId lists stored
    categorized_dbids = matched_record.get('categorized_dbids', None)

    if categorized_dbids is None:
        logger.error(f"Month {month_str} record does not contain dbId lists!")
        logger.error("The history was generated with an old version. Please re-run detection.")
        logger.info("Delete progress_history.json and run Progress_Viewer.py option 4 again.")
        return

    # Get all dbIds from all categories
    all_dbids = []
    for category, dbid_list in categorized_dbids.items():
        if category != 'Uncategorized':  # Optionally exclude uncategorized
            all_dbids.extend(dbid_list)

    # Read lookup table to determine which categories are "completed" at this month
    lookup_data = analyzer.lookup_data
    if lookup_data is None:
        logger.error("Lookup table not loaded")
        return

    matched_rows = lookup_data[lookup_data["Month"] == month_str]
    if matched_rows.empty:
        logger.error(f"Month {month_str} not found in lookup table")
        return

    row = matched_rows.iloc[0]

    # Get current stage to determine completion logic
    current_stage = matched_record.get('current_stage', 'Unknown')

    # Determine which categories are completed vs incomplete based on stage
    completed_dbids = []
    incomplete_dbids = []

    # Stage-based completion logic
    if 'Foundation' in current_stage:
        # Foundation stage: Foundation components are in progress
        completed_dbids.extend(categorized_dbids.get('Foundation', []))
        incomplete_dbids.extend(categorized_dbids.get('Superstructure', []))
        incomplete_dbids.extend(categorized_dbids.get('MEP', []))
        incomplete_dbids.extend(categorized_dbids.get('Interior', []))
        incomplete_dbids.extend(categorized_dbids.get('Outdoor', []))

    elif 'Superstructure' in current_stage and 'MEP' not in current_stage and 'Interior' not in current_stage:
        # Pure Superstructure stage
        completed_dbids.extend(categorized_dbids.get('Foundation', []))
        completed_dbids.extend(categorized_dbids.get('Superstructure', []))
        incomplete_dbids.extend(categorized_dbids.get('MEP', []))
        incomplete_dbids.extend(categorized_dbids.get('Interior', []))
        incomplete_dbids.extend(categorized_dbids.get('Outdoor', []))

    elif 'Superstructure' in current_stage and 'MEP' in current_stage:
        # Superstructure & MEP stage
        completed_dbids.extend(categorized_dbids.get('Foundation', []))
        completed_dbids.extend(categorized_dbids.get('Superstructure', []))
        # MEP is partially complete
        mep_dbids = categorized_dbids.get('MEP', [])
        split_point = int(len(mep_dbids) * 0.5)
        completed_dbids.extend(mep_dbids[:split_point])
        incomplete_dbids.extend(mep_dbids[split_point:])
        incomplete_dbids.extend(categorized_dbids.get('Interior', []))
        incomplete_dbids.extend(categorized_dbids.get('Outdoor', []))

    elif 'MEP' in current_stage and 'Interior' in current_stage:
        # MEP & Interior stage
        completed_dbids.extend(categorized_dbids.get('Foundation', []))
        completed_dbids.extend(categorized_dbids.get('Superstructure', []))
        # MEP mostly complete, Interior in progress
        mep_dbids = categorized_dbids.get('MEP', [])
        interior_dbids = categorized_dbids.get('Interior', [])

        split_mep = int(len(mep_dbids) * 0.7)
        completed_dbids.extend(mep_dbids[:split_mep])
        incomplete_dbids.extend(mep_dbids[split_mep:])

        split_interior = int(len(interior_dbids) * 0.3)
        completed_dbids.extend(interior_dbids[:split_interior])
        incomplete_dbids.extend(interior_dbids[split_interior:])

        incomplete_dbids.extend(categorized_dbids.get('Outdoor', []))

    elif 'Interior' in current_stage and 'Outdoor' in current_stage:
        # Interior & Outdoor stage
        completed_dbids.extend(categorized_dbids.get('Foundation', []))
        completed_dbids.extend(categorized_dbids.get('Superstructure', []))
        completed_dbids.extend(categorized_dbids.get('MEP', []))
        # Interior mostly done, outdoor in progress
        interior_dbids = categorized_dbids.get('Interior', [])
        outdoor_dbids = categorized_dbids.get('Outdoor', [])

        split_interior = int(len(interior_dbids) * 0.8)
        completed_dbids.extend(interior_dbids[:split_interior])
        incomplete_dbids.extend(interior_dbids[split_interior:])
        incomplete_dbids.extend(outdoor_dbids)

    elif 'Outdoor' in current_stage or 'Handover' in current_stage:
        # Final stages: most complete
        completed_dbids.extend(categorized_dbids.get('Foundation', []))
        completed_dbids.extend(categorized_dbids.get('Superstructure', []))
        completed_dbids.extend(categorized_dbids.get('MEP', []))
        completed_dbids.extend(categorized_dbids.get('Interior', []))

        outdoor_dbids = categorized_dbids.get('Outdoor', [])
        if 'Handover' in current_stage:
            # Everything complete
            completed_dbids.extend(outdoor_dbids)
        else:
            # Outdoor in progress
            split_outdoor = int(len(outdoor_dbids) * 0.5)
            completed_dbids.extend(outdoor_dbids[:split_outdoor])
            incomplete_dbids.extend(outdoor_dbids[split_outdoor:])

    else:
        # Fallback: use progress percentage
        progress_pct = matched_record.get('progress_percentage', 0.0)
        split_point = int(len(all_dbids) * (progress_pct / 100.0))
        completed_dbids = all_dbids[:split_point]
        incomplete_dbids = all_dbids[split_point:]

    # Remove duplicates and sort
    completed_dbids = sorted(list(set(completed_dbids)))
    incomplete_dbids = sorted(list(set(incomplete_dbids)))

    # Build output payload
    payload = {
        "month": month,
        "month_str": month_str,
        "completed": completed_dbids,
        "incomplete": incomplete_dbids,
        "total_components": matched_record.get('total', 0),
        "progress_percentage": matched_record.get('progress_percentage', 0.0),
        "stage": current_stage,
        "model_name": matched_record.get('model_name', ''),
        "timestamp": matched_record.get('timestamp', '')
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"✓ Exported viewer coloring JSON for {month_str}")
    logger.info(f"  - Completed: {len(payload['completed'])} components (green)")
    logger.info(f"  - Incomplete: {len(payload['incomplete'])} components (grey)")
    logger.info(f"  - Stage: {current_stage}")
    logger.info(f"  - Progress: {payload['progress_percentage']:.2f}%")
    logger.info(f"  - Output: {output_path}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Progress Adapter - Incremental Detection")
    parser.add_argument(
        "--export-viewer-month",
        type=int,
        help="Export Forge viewer coloring JSON for the given month (e.g. 15 for M15)"
    )
    parser.add_argument(
        "--viewer-json-output",
        type=str,
        default=None,
        help="Output path for viewer coloring JSON"
    )
    parser.add_argument(
        "--lookup-csv",
        type=str,
        default=None,
        help="Path to Preview_progress_fusion.csv"
    )

    args = parser.parse_args()

    if args.export_viewer_month is not None:
        month = args.export_viewer_month

        if args.viewer_json_output:
            output_path = args.viewer_json_output
        else:
            PROJECT_ROOT = Path(__file__).resolve().parent
            output_path = os.path.join(
                PROJECT_ROOT, "static", "progress", f"viewer_coloring_m{month:02d}.json"
            )

        export_viewer_coloring(month, output_path, args.lookup_csv)
        print(f"[INFO] Viewer coloring JSON exported to: {output_path}")
        sys.exit(0)

    print("\n" + "="*70)
    print("INCREMENTAL DETECTION - Stage-Aware Progress Tracking")
    print("="*70)
    print("Features:")
    print("• Tracks component changes over time")
    print("• Determines current construction stage")
    print("• Analyzes component delta and distribution")
    print("• Matches with Preview_progress_fusion.csv")
    print("• Export viewer coloring JSON with --export-viewer-month")
    print("="*70)