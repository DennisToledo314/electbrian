# electbrian

Implements the point electrode approximation for neuromodulation using the Brian 2 package. Works with the SpatialNeuron
object from Brian 2.

## Requirements

<p>It is recommended to install these requirements in a separate virtual environment using pip 
See for example: https://code.visualstudio.com/docs/python/environments for VS Code 
Or https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html for PyCharm </p>

**Python >=3.7**, Installation (https://www.python.org/downloads/) <br>
<br>
<p>Although a pip based virtual environment is recommended, Python can also be obtained from an anaconda installation 
or a miniconda installation (https://www.anaconda.com/docs/getting-started/miniconda/main) </p>

**Brian2**, Installation (https://brian2.readthedocs.io/en/stable/introduction/install.html#standard-install) <br>
**Scipy**, Installation (https://scipy.org/install/) <br>

## Installation
<p>Just as it is recommended to install the above requirements in a separate virtual environment using pip, it is also
recommended to install electbrian in that same virtual environment. </p>

This can be done using the following command: <br>
``pip install electbrian`` <br>

## Theory
<p>For information on the point electrode approximation one may consult the following resources:</p>

<p>D. T. Brocker and W. M. Grill, "Chapter 1 - Principles of electrical stimulation of
neural tissue," in Handbook of Clinical Neurology, vol. 116, A. M. Lozano and
M. Hallett Eds.: Elsevier, 2013, pp. 3-18.</p>

<p>D. R. Merrill, M. Bikson, and J. G. Jefferys, "Electrical stimulation of excitable
tissue: design of efficacious and safe protocols," J. Neurosci. Methods, vol. 141,
no. 2, pp. 171-98, Feb 15 2005, doi: 10.1016/j.jneumeth.2004.10.020.</p>

<p>For an understanding of how this applies to neuronal cables (i.e. axons) and hence to gain insight into the 
methodolgy behind electbrian please consult the following important publication:</p>

<p>A. G. Richardson, C. C. McIntyre, and W. M. Grill, "Modelling the effects of
electric fields on nerve fibres: Influence of the myelin sheath," Med. Biol. Eng.
Comput., vol. 38, no. 4, pp. 438-46, Jul 2000, doi: 10.1007/BF02345014.</p>