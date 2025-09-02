# electbrian

<p>Implements the point electrode approximation for neuromodulation using the Brian 2 package. Works with the 
SpatialNeuron object from Brian 2.</p>

## Requirements

<p>It is recommended to install these requirements in a separate virtual environment using pip 
See for example: https://code.visualstudio.com/docs/python/environments for VS Code or 
https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html for PyCharm. </p>

**Python >=3.7**, Installation (https://www.python.org/downloads/)
<p>Although a pip based virtual environment is recommended, Python can also be obtained from an anaconda installation 
or a miniconda installation (https://www.anaconda.com/docs/getting-started/miniconda/main). </p>

**Brian2**, Installation (https://brian2.readthedocs.io/en/stable/introduction/install.html#standard-install) <br>
**Scipy**, Installation (https://scipy.org/install/) <br>

## Installation
<p>Just as it is recommended to install the above requirements in a separate virtual environment using pip, it is also
recommended to install electbrian in that same virtual environment. </p>

This can be done using the following command: <br>
``pip install electbrian`` <br>

## Theory
<p>The point electrode approximation is defined as follows:</p>
$$ \frac{I}{4\pi\sigma_ex\|r\|}$$
