import argparse

from correlations import run as run_correlations
from expression_analysis import run as run_expression_analysis
from figures import run as run_figures
from gee_analysis import run as run_gee
from mixed_models import run as run_mixed_models
from nonlinearity_analysis import run as run_nonlinearity
from validate_data import main as validate
from validate_stimuli import main as validate_stimuli


def main():
    parser = argparse.ArgumentParser(description="Reproduce the AI face-rating analyses.")
    parser.add_argument(
        "--skip-lmm",
        action="store_true",
        help="Skip the slower cross-classified mixed models.",
    )
    parser.add_argument(
        "--skip-expression",
        action="store_true",
        help="Skip the automated facial-expression robustness analysis.",
    )
    args = parser.parse_args()

    print("\n[1/8] Validating public data")
    validate()
    print("\n[2/8] Validating stimulus images")
    validate_stimuli()
    print("\n[3/8] Participant-wise correlations and Fisher z inference")
    run_correlations()
    print("\n[4/8] GEE models")
    run_gee()
    print("\n[5/8] Cross-classified mixed models")
    if args.skip_lmm:
        print("Skipped by --skip-lmm")
    else:
        run_mixed_models()
    print("\n[6/8] Figures")
    run_figures()
    print("\n[7/8] Non-linearity (uncanny-valley) test")
    run_nonlinearity()
    print("\n[8/8] Automated facial-expression robustness analysis")
    if args.skip_expression:
        print("Skipped by --skip-expression")
    else:
        run_expression_analysis(skip_lmm=args.skip_lmm)
    print("\nReproduction complete.")


if __name__ == "__main__":
    main()

