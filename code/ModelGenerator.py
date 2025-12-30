import os
import json
import numpy as np
import pandas as pd

# ===========================
# Project Type Definitions
# ===========================
PROJECT_TYPES = {
    "residential": {
        "name": "Residential Building",
        "weight": 0.35,  # 35% of projects
        "budget_range": (8e6, 40e6),
        "duration_range": (18, 30),
        "mat_ratio": (0.40, 0.50),
        "lab_ratio": (0.28, 0.38),
        "equip_ratio": (0.10, 0.14),
        "admin_ratio": (0.06, 0.10)
    },
    "commercial": {
        "name": "Commercial Building",
        "weight": 0.25,  # 25% of projects
        "budget_range": (15e6, 60e6),
        "duration_range": (20, 32),
        "mat_ratio": (0.38, 0.48),
        "lab_ratio": (0.26, 0.36),
        "equip_ratio": (0.12, 0.16),
        "admin_ratio": (0.06, 0.09)
    },
    "municipal": {
        "name": "Municipal Engineering",
        "weight": 0.20,  # 20% of projects
        "budget_range": (10e6, 80e6),
        "duration_range": (22, 36),
        "mat_ratio": (0.42, 0.52),
        "lab_ratio": (0.25, 0.35),
        "equip_ratio": (0.12, 0.18),
        "admin_ratio": (0.05, 0.08)
    },
    "industrial": {
        "name": "Industrial Building",
        "weight": 0.15,  # 15% of projects
        "budget_range": (20e6, 100e6),
        "duration_range": (24, 36),
        "mat_ratio": (0.35, 0.45),
        "lab_ratio": (0.25, 0.35),
        "equip_ratio": (0.15, 0.20),
        "admin_ratio": (0.05, 0.08)
    },
    "infrastructure": {
        "name": "Infrastructure",
        "weight": 0.05,  # 5% of projects
        "budget_range": (30e6, 150e6),
        "duration_range": (26, 36),
        "mat_ratio": (0.45, 0.55),
        "lab_ratio": (0.22, 0.32),
        "equip_ratio": (0.12, 0.18),
        "admin_ratio": (0.04, 0.07)
    }
}


# ===========================
# Configuration File Loading
# ===========================
def load_config(country="CN"):
    """
    Load industry configuration from JSON file with multi-path search support

    Args:
        country: Country code (default: "CN")

    Returns:
        dict: Configuration dictionary for the specified country

    Raises:
        FileNotFoundError: Configuration file not found
    """
    # Define possible configuration file locations
    possible_paths = [
        "industry_config.json",
        "./industry_config.json",
        "./generator/industry_config.json"
    ]

    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break

    # Raise error if configuration file not found
    if config_path is None:
        raise FileNotFoundError(
            f"Configuration file 'industry_config.json' not found in any of these locations:\n"
            + "\n".join(f"  - {p}" for p in possible_paths)
        )

    # Load and parse JSON configuration
    with open(config_path, "r", encoding="utf-8") as f:
        all_configs = json.load(f)

    # Extract country-specific configuration
    if country not in all_configs:
        raise ValueError(f"Country '{country}' not found in configuration file")

    cfg = all_configs[country]

    # Validate and set safe defaults for missing fields
    for index_name in ["CPI", "PPI", "MPPI", "Labor"]:
        if index_name not in cfg:
            cfg[index_name] = {}

        cfg[index_name].setdefault("base_value", 1.0)
        cfg[index_name].setdefault("avg_growth", 0.0)
        cfg[index_name].setdefault("volatility", 0.01)

    return cfg


# ===========================
# Index Series Generation
# ===========================
def generate_index_series(base_value=1.0, avg_growth=0.0, volatility=0.01, months=24, **kwargs):
    """
    Generate time series index with trend and volatility

    Args:
        base_value: Initial index value (default: 1.0)
        avg_growth: Average monthly growth rate (default: 0.0)
        volatility: Standard deviation of growth rate (default: 0.01)
        months: Number of months to generate (default: 24)
        **kwargs: Additional parameters (retained for compatibility)

    Returns:
        np.array: Generated index series
    """
    series = [base_value]

    # Generate series with compounding growth and random volatility
    for _ in range(1, months):
        growth = np.random.normal(avg_growth, volatility)
        next_val = series[-1] * (1 + growth)
        series.append(next_val)

    return np.array(series)


# ===========================
# S-Curve Progress Function (Normalized Version)
# ===========================
def s_curve_normalized(t, k=0.3, t0=None, total_months=24):
    """
    Generate normalized S-shaped project progress curve, ensuring 100% at final month

    Args:
        t: Time array (months)
        k: Steepness parameter (default: 0.3)
        t0: Curve midpoint (default: total_months/2)
        total_months: Total project duration in months

    Returns:
        np.array: Progress values (between 0 and 1)
    """
    if t0 is None:
        t0 = total_months / 2

    # Standard S-curve
    raw_curve = 1 / (1 + np.exp(-k * (t - t0)))

    # Normalize: ensure final month reaches exactly 1.0
    max_val = 1 / (1 + np.exp(-k * (total_months - t0)))
    normalized_curve = raw_curve / max_val

    return normalized_curve


# ===========================
# Project Type Assignment
# ===========================
def assign_project_types(n_projects):
    """
    Assign project types based on weights

    Args:
        n_projects: Total number of projects

    Returns:
        list: List of project types
    """
    types = list(PROJECT_TYPES.keys())
    weights = [PROJECT_TYPES[t]["weight"] for t in types]

    # Randomly assign based on weights
    assigned_types = np.random.choice(types, size=n_projects, p=weights)

    return assigned_types


# ===========================
# Single Project Generation (Enhanced)
# ===========================
def generate_project(pid, project_type, config, max_months=36):
    """
    Generate synthetic cost and progress data for a single project,
    including realistic project completion phase

    Args:
        pid: Project ID string
        project_type: Project type (residential/commercial/etc.)
        config: Configuration dictionary
        max_months: Maximum months (for index generation)

    Returns:
        pd.DataFrame: Project data with monthly cost breakdown
    """
    # Set random seed based on project ID for reproducibility
    np.random.seed(int(pid[-3:]) * 13)

    # ===========================
    # Get Project Type Configuration
    # ===========================
    type_config = PROJECT_TYPES[project_type]

    # Project duration: random within type-defined range
    project_months = np.random.randint(*type_config["duration_range"])

    # Project total budget: random within type-defined range
    total_budget = np.random.uniform(*type_config["budget_range"])

    # Cost component ratios: random within type-defined ranges
    mat_ratio = np.random.uniform(*type_config["mat_ratio"])
    lab_ratio = np.random.uniform(*type_config["lab_ratio"])
    equip_ratio = np.random.uniform(*type_config["equip_ratio"])
    admin_ratio = np.random.uniform(*type_config["admin_ratio"])

    # ===========================
    # Generate Price Indices (use max_months to ensure sufficient length)
    # ===========================
    mppi_idx = generate_index_series(**config["MPPI"], months=max_months)
    ppi_idx = generate_index_series(**config["PPI"], months=max_months)
    lab_idx = generate_index_series(**config["Labor"], months=max_months)
    cpi_idx = generate_index_series(**config["CPI"], months=max_months)

    # Combined material index: MPPI 70%, PPI 30%
    mat_idx = 0.7 * mppi_idx + 0.3 * ppi_idx

    # Truncate to actual project duration
    mat_idx = mat_idx[:project_months]
    lab_idx = lab_idx[:project_months]
    cpi_idx = cpi_idx[:project_months]

    # ===========================
    # Generate Progress Curve (ensure 100% completion at final month)
    # ===========================
    months_arr = np.arange(1, project_months + 1)

    # Randomize S-curve parameters
    k = np.random.uniform(0.15, 0.40)  # Steepness
    t0 = np.random.uniform(project_months * 0.4, project_months * 0.6)  # Midpoint

    # Generate normalized progress curve
    progress = s_curve_normalized(months_arr, k, t0, project_months)
    delta_prog = np.diff(np.insert(progress, 0, 0))

    # ===========================
    # Define Project Completion Phase Logic
    # ===========================
    # Last 2-3 months are completion phase
    completion_phase_months = min(3, max(2, int(project_months * 0.1)))
    completion_start = project_months - completion_phase_months

    # ===========================
    # Calculate Monthly Costs
    # ===========================
    data = []
    for i, m in enumerate(months_arr):
        # Check if in completion phase
        in_completion_phase = (i >= completion_start)

        if in_completion_phase:
            # Completion phase: material and equipment costs significantly reduced
            # Final month: almost no material and equipment costs (acceptance only)
            if i == project_months - 1:
                # Final month: only labor and admin costs (acceptance and handover)
                mat_multiplier = 0.0
                equip_multiplier = 0.0
                lab_multiplier = 0.3  # Only 30% labor (acceptance personnel)
                admin_multiplier = 0.5  # 50% admin (documentation and handover)
            else:
                # 2nd-3rd to last months: gradual reduction in materials and equipment
                remaining_ratio = (project_months - i - 1) / completion_phase_months
                mat_multiplier = 0.2 + 0.3 * remaining_ratio  # 20%-50%
                equip_multiplier = 0.1 + 0.4 * remaining_ratio  # 10%-50%
                lab_multiplier = 0.6 + 0.3 * remaining_ratio  # 60%-90%
                admin_multiplier = 0.7 + 0.2 * remaining_ratio  # 70%-90%
        else:
            # Normal construction phase: full costs
            mat_multiplier = 1.0
            equip_multiplier = 1.0
            lab_multiplier = 1.0
            admin_multiplier = 1.0

        # Calculate each cost component
        mat_cost = total_budget * mat_ratio * delta_prog[i] * mat_idx[i] * mat_multiplier
        lab_cost = total_budget * lab_ratio * delta_prog[i] * lab_idx[i] * lab_multiplier
        equip_cost = total_budget * equip_ratio * delta_prog[i] * equip_multiplier
        admin_cost = total_budget * admin_ratio * delta_prog[i] * admin_multiplier

        # Total cost with 3% random noise
        total_cost = (mat_cost + lab_cost + equip_cost + admin_cost)
        total_cost *= np.random.normal(1, 0.03)

        # Store monthly record
        data.append({
            "project_id": pid,
            "project_type": project_type,
            "project_type_name": type_config["name"],
            "total_duration_months": project_months,
            "month": m,
            "progress_pct": progress[i] * 100,
            "is_completion_phase": in_completion_phase,
            "mat_index": mat_idx[i],
            "lab_index": lab_idx[i],
            "cpi_index": cpi_idx[i],
            "material_cost": mat_cost,
            "labour_cost": lab_cost,
            "equip_cost": equip_cost,
            "admin_cost": admin_cost,
            "total_cost": total_cost
        })

    # Convert to DataFrame and ensure data quality
    df = pd.DataFrame(data)

    # Replace NaN and inf values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df


# ===========================
# Batch Project Generation
# ===========================
def generate_all_projects(n_projects=20, country="CN", output_path="data/generated/"):
    """
    Generate synthetic data for multiple construction projects

    Args:
        n_projects: Number of projects to generate (default: 20)
        country: Country code (default: "CN")
        output_path: Output directory path (default: "data/generated/")

    Returns:
        pd.DataFrame: Combined dataset of all projects
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load country configuration
    config = load_config(country)

    # Assign project types
    project_types = assign_project_types(n_projects)

    # Generate all projects
    all_projects = []
    type_counts = {}

    for i in range(1, n_projects + 1):
        pid = f"P{i:03d}"
        ptype = project_types[i - 1]

        # Count project types
        type_counts[ptype] = type_counts.get(ptype, 0) + 1

        # Generate project data (max 36 months for index generation)
        df = generate_project(pid, ptype, config, max_months=36)
        all_projects.append(df)

    # Combine all project data
    dataset = pd.concat(all_projects, ignore_index=True)

    # Final data quality check
    dataset = dataset.replace([np.inf, -np.inf], 0)
    dataset = dataset.fillna(0)

    # Save to CSV
    output_file = os.path.join(output_path, f"synthetic_{country}_projects.csv")
    dataset.to_csv(output_file, index=False)

    # Print detailed summary
    print(f"✅ Project generation completed!")
    print(f"=" * 60)
    print(f"Total projects: {n_projects}")
    print(f"Total records: {len(dataset)}")
    print(f"\nProject type distribution:")
    for ptype, count in sorted(type_counts.items()):
        pct = count / n_projects * 100
        type_name = PROJECT_TYPES[ptype]["name"]
        print(f"  • {type_name:25s} ({ptype:15s}): {count:2d} projects ({pct:.1f}%)")

    # Duration statistics
    duration_stats = dataset.groupby('project_id')['total_duration_months'].first()
    print(f"\nProject duration statistics:")
    print(f"  • Shortest duration: {duration_stats.min()} months")
    print(f"  • Longest duration: {duration_stats.max()} months")
    print(f"  • Average duration: {duration_stats.mean():.1f} months")

    print(f"\nOutput file: {output_file}")
    print(f"=" * 60)

    return dataset


# ===========================
# Main Program Entry
# ===========================
if __name__ == "__main__":
    # Generate 30 projects with automatic type and duration assignment
    generate_all_projects(n_projects=30, country="CN")