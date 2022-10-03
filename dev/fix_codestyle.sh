isort ./**/*.py --profile black
black --line-length 79 ./**/*.py
flake8 --ignore=E203,W503 ./**/*.py #ignore - wrong whitespaces
find . -type f -name "*.py" | xargs pylint --rcfile=./dev/.pylintrc