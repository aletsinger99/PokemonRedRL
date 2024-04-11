


# Create environment using conda

Creates a pyboy conda environment. First, install the environment minus pytorch.

```bash
conda env create -f environment.yml
```
Then follow the instructions for your system to [install pytorch](https://pytorch.org/get-started/locally/).

# Open the ROM to play

```bash
python3 -m pyboy path/to/rom.gb
```