# config/loader.py

# Bring the tool PyYAML
import yaml


# A function that promises to deliver a Python dictionary whenever called
def load_settings(path: str = "config/settings.yaml") -> dict:
    """
    Opens the settings.yaml file and hands back its contents
    as a Python dictionary, so the rest of our program can use it.
    """
    with open(path, "r") as file:  # open the file to path in read mode
        settings = yaml.safe_load(
            file
        )  # take the raw yaml text and turn it into a real Python dictionary
        # We use safe_load() incase of  a bug hidding inside the YAML
    return settings  # Hand the dictionary back to whoever calls it
