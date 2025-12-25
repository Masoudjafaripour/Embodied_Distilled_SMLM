.PHONY: data train clean

data:
	python data/generate_data.py

train:
	python train.py

clean:
	rm -rf outputs __pycache__ */__pycache__
