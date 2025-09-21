def nice_format(experiment_dir_name: str) -> str:
    """
    Get the human preference from the experiment directory name.
    """
    if "safe" in experiment_dir_name:
        return "Safe"
    elif "risky" in experiment_dir_name:
        return "Risky"
    elif "now" in experiment_dir_name:
        return "Now"
    elif "later" in experiment_dir_name:
        return "Later"
    else:
        return "Unknown"