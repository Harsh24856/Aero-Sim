"""Published C-MAPSS damage-propagation constants.

EVERY value in this module is traceable to a citation. The source is:

    A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for
    Aircraft Engine Run-to-Failure Simulation", PHM'08, Denver CO, Oct 2008.

A local copy ships with the dataset at CMAPSSData/Damage Propagation Modeling.pdf
and page numbers below refer to it. Nothing here is invented or tuned: if a value
is not in the paper it does not belong in this file, it belongs in the fitted
response surface (surrogate.py) where it is learned from the released data and
labelled as such.

WHY THAT SPLIT MATTERS
----------------------
The paper publishes the damage model completely - the exponential propagation
law, its parameter ranges, the health-index definition and the failure
criterion. It does NOT publish the C-MAPSS thermodynamic model itself, i.e. the
mapping from (efficiency, flow, operating condition) to the 21 sensor outputs.
That software was never publicly released. So the replication is:

    damage layer      -> exact, from the equations below
    response surface  -> surrogate, fitted to the released NASA data

Keeping the two in separate modules keeps the claim honest: we can say the
degradation model is reproduced from published equations without implying the
engine thermodynamics were.
"""

# ---------------------------------------------------------------------------
# Damage propagation. Paper section IV.D, eq. (5)-(6), p.4-5.
#
#   d(t) = 1 - d - exp{a . t^b}
#
# applied independently to efficiency e(t) and flow f(t) (eq. 6), each with its
# own sampled (a, b) but a shared per-unit initial deterioration.
# ---------------------------------------------------------------------------

# eq. (10), p.5. "the randomly chosen direction and evolution of faults is
# constrained by" - these are the literal published bounds.
A_RANGE = (0.001, 0.003)      # exponential rate coefficient
B_RANGE = (1.4, 1.6)          # exponential shape exponent
K_VALUES = (1, 2)             # discrete fault-direction selector

# eq. (9), p.5. Initial wear from manufacturing and assembly variation. The
# paper bounds maximum initial deterioration at 1% of healthy, citing [14], so
# "each health index trajectory starts with a number between 1 and 0.99".
E0_RANGE = (0.99, 1.0)        # initial efficiency
F0_RANGE = (0.99, 1.0)        # initial flow

# eq. (10), p.5: |f_i - e_i| <= 1%. Efficiency and flow start close together;
# they are not independently drawn across the whole range.
MAX_EF_DELTA = 0.01

# ---------------------------------------------------------------------------
# Health index. Paper section IV.D eq. (7)-(8) p.5, and section V.D p.6.
#
#   H(t) = g(e(t), f(t)) = min(m_Fan, m_HPC, m_HPT, m_EGT)
#
# Each margin is normalised to [0, 1]: 1 = perfectly healthy, 0 = the margin has
# decayed by its specified limit. Failure is declared at H = 0 (step 3, p.5).
#
# NOTE ON WHICH MARGINS: eq. (8) p.5 names "fan, HPC, HPT and EGT". Section V.D
# p.6 instead sets stall-margin limits for "HPC, LPC and fan" plus EGT. The two
# lists disagree on HPT vs LPC - an inconsistency in the source, not a reading
# error. We implement the section V.D set (fan, LPC, HPC stall + EGT) because
# that is the one the paper attaches actual numeric limits to.
# ---------------------------------------------------------------------------

# Section V.D, p.6: "For the challenge data this limit was set at 15% for HPC,
# LPC and fan stall margins and about 2% for the EGT margin."
STALL_MARGIN_LIMIT = 0.15     # fractional decay that drives a stall margin to 0
EGT_MARGIN_LIMIT = 0.02       # fractional decay that drives the EGT margin to 0

MARGIN_NAMES = ("fan", "LPC", "HPC", "EGT")

FAILURE_THRESHOLD = 0.0       # step 3, p.5: "Stop when health H = 0"

# ---------------------------------------------------------------------------
# Dataset geometry. CMAPSSData/readme.txt, verified against the shipped files.
# Unit counts are exact and are asserted against the parsed data, so a silent
# mismatch here fails loudly rather than producing a subtly wrong replication.
# ---------------------------------------------------------------------------

# FD004 CAVEAT - the shipped readme.txt is wrong here, and these values follow
# the DATA, not the readme. readme.txt states "Train trajectories: 248 / Test
# trajectories: 249" for FD004, but the released files contain 249 train units
# and 248 test units: the two are transposed. The data is self-consistent
# (RUL_FD004.txt carries exactly 248 values, matching the 248 test units), so
# the error is in the documentation. Verified directly against the files.
SUBSETS = {
    "FD001": {"train_units": 100, "test_units": 100, "n_conditions": 1, "fault_modes": ("HPC",)},
    "FD002": {"train_units": 260, "test_units": 259, "n_conditions": 6, "fault_modes": ("HPC",)},
    "FD003": {"train_units": 100, "test_units": 100, "n_conditions": 1, "fault_modes": ("HPC", "Fan")},
    "FD004": {"train_units": 249, "test_units": 248, "n_conditions": 6, "fault_modes": ("HPC", "Fan")},
}

# ---------------------------------------------------------------------------
# Column schema. CMAPSSData/readme.txt p.1 gives the 26-column layout; the
# sensor symbols and units are Table 2, p.3 of the paper.
#
# The released files carry 26 columns: unit, cycle, 3 operating settings and 21
# sensors. The paper's Table 2 also lists the operability margins (SmFan, SmLPC,
# SmHPC, EGT margin) but states they "were used for health index calculation
# only and were not available to the participants explicitly" - so they are
# computed internally by our generator and never written to the sensor table.
# ---------------------------------------------------------------------------

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Table 2, p.3. Order is the file's s1..s21 order.
SENSOR_COLS = [
    "T2",         # Total temperature at fan inlet          degR
    "T24",        # Total temperature at LPC outlet         degR
    "T30",        # Total temperature at HPC outlet         degR
    "T50",        # Total temperature at LPT outlet         degR
    "P2",         # Pressure at fan inlet                   psia
    "P15",        # Total pressure in bypass-duct           psia
    "P30",        # Total pressure at HPC outlet            psia
    "Nf",         # Physical fan speed                      rpm
    "Nc",         # Physical core speed                     rpm
    "epr",        # Engine pressure ratio (P50/P2)          --
    "Ps30",       # Static pressure at HPC outlet           psia
    "phi",        # Ratio of fuel flow to Ps30              pps/psi
    "NRf",        # Corrected fan speed                     rpm
    "NRc",        # Corrected core speed                    rpm
    "BPR",        # Bypass ratio                            --
    "farB",       # Burner fuel-air ratio                   --
    "htBleed",    # Bleed enthalpy                          --
    "Nf_dmd",     # Demanded fan speed                      rpm
    "PCNfR_dmd",  # Demanded corrected fan speed            rpm
    "W31",        # HPT coolant bleed                       lbm/s
    "W32",        # LPT coolant bleed                       lbm/s
]

ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS
assert len(ALL_COLS) == 26, "C-MAPSS files carry exactly 26 columns"

# ---------------------------------------------------------------------------
# RUL labelling. NOT from the damage-propagation paper - this is the downstream
# convention established by the prognostics literature that uses C-MAPSS, and
# it is applied only when training a RUL model (A17), never by the generator.
#
# True RUL is linear in cycles-to-failure, but a healthy engine is not
# meaningfully "347 cycles from failure" in any learnable sense, so the standard
# treatment clips the label to a constant early-life ceiling. 125 is the most
# widely used value and is what we adopt, recorded here so the choice is visible
# rather than buried in a training script.
# ---------------------------------------------------------------------------
RUL_CAP = 125

# ---------------------------------------------------------------------------
# Reproducibility. Every generated artefact derives from this seed plus the
# subset name, so any run can be reproduced exactly from the config alone.
# ---------------------------------------------------------------------------
MASTER_SEED = 20080103   # the paper's publication date, for no reason but traceability


def subset_seed(subset: str, split: str) -> int:
    """Deterministic per-(subset, split) seed derived from MASTER_SEED."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {list(SUBSETS)}")
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}; expected 'train' or 'test'")
    offset = int(subset[-3:]) * 10 + (0 if split == "train" else 1)
    return MASTER_SEED + offset
