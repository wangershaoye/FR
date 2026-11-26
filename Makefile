.PHONY: install install-expression validate analysis fast-analysis expression extract-expression clean

install:
	python -m pip install -r requirements.txt

install-expression:
	python -m pip install -r requirements-expression.txt

validate:
	python src/validate_data.py
	python src/validate_stimuli.py

analysis: validate
	python src/run_all.py

fast-analysis: validate
	python src/run_all.py --skip-lmm

expression:
	python src/expression_analysis.py

extract-expression:
	python src/extract_expression_features.py

clean:
	rm -f outputs/tables/*.csv outputs/figures/*.png outputs/figures/*.pdf

