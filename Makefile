.PHONY: install check quality incidents backtest forecast all app
install:
	python -m pip install -e '.[all]'
check:
	python -m pytest -q
quality:
	retail-black-box quality
incidents:
	retail-black-box incidents
backtest:
	retail-black-box backtest --samples 120 --seed 42
forecast:
	retail-black-box forecast
all:
	retail-black-box all
app:
	streamlit run app.py
