"""Names of the original analysis code that the parity tests use.

The parity tests call the original functions by these names and pass or read
the listed values. This is the only
file of the repository in which those names are written; tests/test_naming.py
exempts it, and only it, from the naming rule.

Layout: module of the original code -> {name in this package: name in the
original code}. A function that is not listed has the same name in both.
"""

ORIGINAL_PACKAGE = "wavlm_add"

ORIGINAL_NAMES = {
    "metrics": {"eer_roc_corners": "eer_v1_roc"},
    "audio": {"padded_window": "fit_window"},
    "features": {"shard_lengths": "v3_shard_lengths"},
    "analysis": {"selected_configs": "phase5_configs", "replicate_seeds": "phase6_replicate_seeds",
                 "layer_configs": "phase9_configs", "threshold_configs": "phase8_configs"},
}


# Values and column names of the original code that the parity tests pass or read.
ORIGINAL_VALUES = {
    "unpadded_feature_set": "v3-unpadded",
    "best_dev_eer_column": "best_dev_eer_v1_roc",
}


def original_name(module: str, name: str) -> str:
    """Name of ``name`` of ``module`` in the original code."""
    return ORIGINAL_NAMES.get(module, {}).get(name, name)
