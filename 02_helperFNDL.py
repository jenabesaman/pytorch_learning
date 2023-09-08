import requests
from pathlib import Path
if Path("helper_functions.py").is_file():
    print("helper_functions.py is already exist , skipping download")
else:
    print("downloading helper_functions:")
    request=requests.get(url="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open(file="helper_functions.py",mode="wb") as f:
        f.write(request.content)
