# length-controllable-summarisation

## vi
### Setup Environment
- Python 3.8
- Create new virtual environment: `python3.8 -m venv venv`
- Update pip: `pip install -U pip`
- Install required packages: `pip install -r requirements.txt`

### BARTpho
Before quantizing the BartPho model checkpoint with ctranslate2, please ensure the following:

- Save the tokenizer in the checkpoint directory.
- Add the setting `"normalize_before": true` to the `config.json` file."