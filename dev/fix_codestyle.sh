isort . --profile black
black --line-length 79 . 
flake8 --ignore=E203,W503 . #ignore - wrong whitespaces
find . -type f -name "*.py" | xargs pylint --rcfile=./dev/.pylintrc