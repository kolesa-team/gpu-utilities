.PHONY: tests requirements dist

env:
	export PYTHONPATH="$PYTHONPATH:$(PWD)"

format:
	black -l 79 . --preview

type-checking:
	mypy --ignore-missing-imports gpu_utils 

hooks:
	git config core.hooksPath .hooks/
	chmod -R +x .hooks/

requirements:
	pipreqs --encoding utf-8 ./ --savepath requirements/pipreqs.txt --force
	pip freeze > requirements/raw-pip-freeze.txt

tests:
	pytest tests/ -o log_cli=true --log-cli-level=DEBUG --pdb

build:
	docker -H $(HOST) build -t $(IMAGE_NAME) .

run: build
	docker -H $(HOST) run --rm -it $(IMAGE_NAME)

run-tests: build
	docker -H $(HOST) run --entrypoint pytest --rm -it --gpus all $(IMAGE_NAME)

run-shell: build
	docker -H $(HOST) run --entrypoint /bin/bash --rm -it --gpus all $(IMAGE_NAME)

dist:
	python3 -m pip install --upgrade build
	python3 -m build

publish-test: dist 
	python3 -m pip install --upgrade twine
	python3 -m twine upload --repository testpypi dist/* --verbose

publish: dist
	python3 -m pip install --upgrade twine
	python3 -m twine upload dist/* --verbose

clean:
	pip install --upgrade cleanpy
	cleanpy --all .
