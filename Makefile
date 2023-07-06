<<<<<<< HEAD
.PHONY: venv setup setup-dev install install-dev tidy compile clean

all: setup

venv:
	pyenv virtualenv .venv || true
	pyenv activate .venv

install:
	# pip install --upgrade poetry
	# poetry install -E all
	pip install -r requirements.txt
	# Hacky way to make sure HF cli exists.
	pip install --upgrade 'huggingface_hub[cli]'
	huggingface-cli login		# login to huggingface
	poetry install

install-dev:
	# pip install --upgrade poetry
	# poetry install --with dev -E all
	pip install -r requirements-dev.txt
	# Hacky way to make sure HF cli exists.
	pip install --upgrade 'huggingface_hub[cli]'
	huggingface-cli login		# login to huggingface
	poetry install --with dev

setup: venv install

setup-dev: venv install-dev

tidy:
	pre-commit run --all-files

compile:
	poetry export -f requirements.txt --output requirements.txt --only main --without-hashes --without-urls
	poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes --without-urls

clean:
	pyenv deactivate
||||||| e5f9959
=======
.PHONY: venv setup setup-dev install install-dev tidy compile clean

all: setup

venv:
	pyenv virtualenv .venv || true
	pyenv activate .venv

install:
	# pip install --upgrade poetry
	# poetry install -E all
	pip install -r requirements.txt
	# Hacky way to make sure HF cli exists.
	# pip install --upgrade huggingface_hub
	huggingface-cli login		# login to huggingface
	poetry install

install-dev:
	# pip install --upgrade poetry
	# poetry install --with dev -E all
	pip install -r requirements-dev.txt
	# Hacky way to make sure HF cli exists.
	# pip install --upgrade huggingface_hub
	huggingface-cli login		# login to huggingface
	poetry install --with dev

setup: venv install

setup-dev: venv install-dev

tidy:
	pre-commit run --all-files

compile:
	poetry export -f requirements.txt --output requirements.txt --only main --without-hashes --without-urls
	poetry export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes --without-urls

clean:
	pyenv deactivate
>>>>>>> a55c5363be1122a352cdb77930afef38e1cf1d83
