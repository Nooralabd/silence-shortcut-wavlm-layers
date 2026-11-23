"""EER and tandem t-DCF, computed from saved scores.

Convention of this package: label 0 = bonafide, 1 = spoof; score = P(spoof),
so a higher score means more spoof-like.

Official code
-------------
``obtain_asv_error_rates``, ``compute_det_curve``, ``compute_eer`` and
``compute_tDCF_legacy`` are ported from

    https://github.com/asvspoof-challenge/2021
    file   eval-package/eval_metrics.py
    commit 7926f45ac44324768fa6c492b43d00214367e714

That file states that its functions are imported from eval_metrics.py of the
official t-DCF package of the ASVspoof consortium (asvspoof.org/resources).
``compute_tDCF_legacy`` is the ASVspoof 2019 formulation (Kinnunen et al.,
Odyssey 2018; ASVspoof 2019 evaluation plan). The default cost parameters and
the ASV operating point (the ASV EER threshold) follow
evaluate_tDCF_asvspoof19.py of the same package.

Differences from the official file, none of which changes a returned number:
``sys.exit(msg)`` is ``raise ValueError(msg)``, and the built-in ``sum`` over
boolean arrays is ``np.sum``.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

from . import config as cfg


# Official ASVspoof functions (ported) ---------------------------------------

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """ASV error rates at a fixed threshold, as fractions.

    Returns (Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv).
    """

    # False alarm and miss rates for ASV
    Pfa_asv = np.sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = np.sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    """DET curve; higher score = more target-like.

    Returns (frr, far, thresholds), each of length N + 1.
    """

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Equal error rate (a fraction, not a percentage) and its threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF_legacy(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost):
    """ASVspoof 2019 tandem detection cost function for a fixed ASV system.

    Higher CM scores must support the bonafide hypothesis, the opposite of
    this package's convention; ``tandem_min_tdcf`` handles the sign.

    Args:
        bonafide_score_cm: CM scores of the bonafide trials.
        spoof_score_cm: CM scores of the spoof trials.
        Pfa_asv: false acceptance rate of the ASV system (fraction).
        Pmiss_asv: miss rate of the ASV system (fraction).
        Pmiss_spoof_asv: fraction of spoof trials rejected by the ASV system.
        cost_model: dict with Ptar, Pnon, Pspoof, Cmiss_asv, Cfa_asv,
            Cmiss_cm, Cfa_cm.
        print_cost: print a summary of the cost model.

    Returns:
        (tDCF_norm, CM_thresholds): the normalised t-DCF curve over the CM
        operating points and the matching thresholds. min(tDCF_norm) is the
        ASVspoof 2019 minimum t-DCF.
    """

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        raise ValueError('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        raise ValueError('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        raise ValueError('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        raise ValueError('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        raise ValueError('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds


# EER -------------------------------------------------------------------------

def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal error rate in percent: the official ASVspoof ``compute_eer``.

    labels (N,) in {0, 1}; scores (N,) = P(spoof). The official code expects
    higher = more bonafide, so the scores are negated.
    """
    labels = np.asarray(labels)
    cm = -np.asarray(scores, dtype=np.float64)
    value, _ = compute_eer(cm[labels == cfg.LABEL_BONAFIDE], cm[labels == cfg.LABEL_SPOOF])
    return float(value * 100)


def eer_roc_corners(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-based EER in percent on the corner points of the ROC curve.

    Used only to select the checkpoint during training; never reported.
    labels (N,) in {0, 1}; scores (N,) = P(spoof), positive class = spoof.
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2 * 100


# Tandem t-DCF ----------------------------------------------------------------

def default_cost_model() -> dict:
    """Default ASVspoof 2019 t-DCF parameters."""
    Pspoof = 0.05
    return {
        'Pspoof': Pspoof,              # prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,   # prior probability of a target speaker
        'Pnon': (1 - Pspoof) * 0.01,   # prior probability of a nontarget speaker
        'Cmiss_asv': 1,   # cost of the ASV system rejecting a target speaker
        'Cfa_asv': 10,    # cost of the ASV system accepting a nontarget speaker
        'Cmiss_cm': 1,    # cost of the CM rejecting a target speaker
        'Cfa_cm': 10,     # cost of the CM accepting a spoof
    }


@dataclass(frozen=True)
class AsvErrorRates:
    """Error rates of the fixed ASV system at its EER threshold (fractions)."""

    Pfa_asv: float
    Pmiss_asv: float
    Pmiss_spoof_asv: float
    eer_asv: float
    threshold: float


def asv_error_rates(tar_asv: np.ndarray, non_asv: np.ndarray,
                    spoof_asv: np.ndarray) -> AsvErrorRates:
    """ASV error rates with the operating point fixed at the ASV EER threshold."""
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, _ = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold)
    if Pmiss_spoof_asv is None:
        raise ValueError("the ASV score file contains no spoof trials")
    return AsvErrorRates(float(Pfa_asv), float(Pmiss_asv), float(Pmiss_spoof_asv),
                         float(eer_asv), float(asv_threshold))


def load_asv_error_rates(path: str) -> AsvErrorRates:
    """Error rates from an official ASV score file.

    Line format: ``<source> <target|nontarget|spoof> <score>``.
    """
    asv_data = np.genfromtxt(path, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)
    return asv_error_rates(asv_scores[asv_keys == 'target'],
                           asv_scores[asv_keys == 'nontarget'],
                           asv_scores[asv_keys == 'spoof'])


def tandem_min_tdcf(labels: np.ndarray, scores: np.ndarray, asv: AsvErrorRates,
                    cost_model: dict | None = None) -> float:
    """ASVspoof 2019 minimum tandem t-DCF.

    labels (N,) in {0, 1}; scores (N,) = P(spoof); ``asv`` are the error rates
    of the official ASV system on the same partition. The scores are negated
    for the official code.
    """
    labels = np.asarray(labels)
    cm = -np.asarray(scores, dtype=np.float64)
    bona_cm = cm[labels == cfg.LABEL_BONAFIDE]
    spoof_cm = cm[labels == cfg.LABEL_SPOOF]
    tDCF_curve, _ = compute_tDCF_legacy(
        bona_cm, spoof_cm, asv.Pfa_asv, asv.Pmiss_asv, asv.Pmiss_spoof_asv,
        cost_model or default_cost_model(), print_cost=False)
    return float(tDCF_curve[np.argmin(tDCF_curve)])


def extract_asv_scores(zip_path: str, out_dir: str) -> dict[str, str]:
    """Copy the official ASV score files out of LA.zip; existing files are kept.

    Returns {"dev": path, "eval": path}.
    """
    cfg.makedirs(out_dir)
    out = {k: os.path.join(out_dir, name) for k, name in cfg.ASV_SCORE_FILENAMES.items()}
    missing = {k: p for k, p in out.items() if not os.path.exists(p)}
    if not missing:
        return out
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for k, dst in missing.items():
            wanted = cfg.ASV_SCORE_FILENAMES[k]
            members = [n for n in names if n.endswith("/" + wanted) or n == wanted]
            if len(members) != 1:
                raise FileNotFoundError(
                    f"expected exactly one '{wanted}' in {zip_path}, found {members}")
            tmp = dst + ".tmp"
            with z.open(members[0]) as src, open(tmp, "wb") as f:
                f.write(src.read())
            os.replace(tmp, dst)
    return out
