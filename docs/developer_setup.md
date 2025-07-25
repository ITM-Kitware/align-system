## Installation from source

If you're planning on working directly on the `align-system` code, we
recommend using [Poetry](https://python-poetry.org/) as that's what we
use to manage dependencies.  Once poetry is installed, you can install
the project (from inside a local clone of this repo) with `poetry
install`.  By default poetry will create a virtual environment (with
`venv`) for the project if one doesn't already exist.

Some huggingface models are 'gated' and require accepting terms and conditions (e.g. [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)). 
A gated model is indicated by errors like this:
```
Cannot access gated repo for url https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/config.json.
Access to model mistralai/Mistral-7B-Instruct-v0.3 is restricted. You must have access to it and be authenticated to access it. Please log in.
```
1. Visit the model URL while logged in to huggingface to accept the terms and conditions.
2. If not already done, create an access token on the HuggingFace website: click your profile picture > Access Tokens > Create new token > Read > Create Token > copy token 
3. Store the token on the system running align:
```
poetry run python
>>> from huggingface_hub import login
>>> login()

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

Enter your token (input will not be visible):
Add token as git credential? (Y/n) n
>>>
```
