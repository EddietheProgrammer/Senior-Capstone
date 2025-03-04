# Capstone Project

**Make sure you have [python](https://www.python.org/downloads/) installed.**

# Installation
Right now I am using the [baseballcv](https://github.com/dylandru/BaseballCV/tree/main) package to load pre-trained models and datasets to make our lives easier. 

Here's how you should set this up (I recommend an IDE like VS Code; video files kinda crash Jupyter Notebook).

```bash
git clone https://github.com/EddietheProgrammer/Senior-Capstone.git
```

Please keep in mind Windows file system is different with \ instead of / (Good going Microsoft):
```bash
cd Senior\ Capstone/
```

I recommend creating a virtual environment so you don't run into any issues with package versions.
```bash
python -m venv myenv # myenv = my environment
```

You then activate it with
```bash
source myenv/bin/activate # Note: This may be different for Windows so let me know if this doesn't work
```

Lastly, you will need to:
```bash
pip install baseballcv
```

This will install everything you need. I will update if there's additional packages. To deactivate your environment, simply type `deactivate` in the terminal.

Also, please use a .gitignore file for files you don't want merged with the main branch. 
i.e. the `myenv` folder. To do this, create a file called .gitignore then type your environment name in the file. It should be greyed out.


## Editors Note:
You may also need to install the following:
```bash
pip install git+https://github.com/Jensen-holm/statcast-era-pitches.git
```

# Contributing
If you want to contribute, fork the repository then use the following commands:
```bash
git checkout -b feature/YourFeature # Can change the naming
git commit -m "Adding files" # This is done afer you mainipulate the repo
git push origin feature/YourFeature # Pushes changes to your master branch
```
After you push, open a pull request and I will review and change if I like it.