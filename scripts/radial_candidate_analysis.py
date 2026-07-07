#!/usr/bin/env python
"""DEPRECATED alias for the INJECTION-CENTRED radial analysis.

This script is injection-centred: it measures each candidate's distance from the
injection centre, which is a DIFFERENT analysis from the candidate-to-candidate
pair correlation (``run_pair_correlation.py``). It is easy to run this by mistake
when candidate-to-candidate clustering was intended, so it now refuses to run
unless ``--confirm-injection-centered`` is passed.

  * candidate-to-candidate clustering  -> scripts/run_pair_correlation.py
  * injection-centred radial analysis  -> scripts/injection_centered_radial_analysis.py
    (or this alias with --confirm-injection-centered)
"""

import sys

import _bootstrap  # noqa: F401

import injection_centered_radial_analysis as injection_centered

CONFIRM_FLAG = "--confirm-injection-centered"

_WARNING = (
    "WARNING: radial_candidate_analysis.py is INJECTION-CENTRED (distance from the\n"
    "         injection centre). This is NOT the candidate-to-candidate pair\n"
    "         correlation. For candidate-to-candidate clustering run instead:\n"
    "             python scripts/run_pair_correlation.py\n"
)


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print(_WARNING, file=sys.stderr)
    if CONFIRM_FLAG not in argv:
        print(
            f"Refusing to run injection-centred analysis without {CONFIRM_FLAG}.\n"
            f"Re-run with {CONFIRM_FLAG} to confirm you want the injection-centred\n"
            "analysis, or use scripts/run_pair_correlation.py for candidate-to-"
            "candidate pair correlation.",
            file=sys.stderr,
        )
        return 2
    # Confirmed: delegate to the real injection-centred implementation.
    forwarded = [argument for argument in argv if argument != CONFIRM_FLAG]
    return injection_centered.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
