# This repository is no longer actively developed. Please see [seabed](https://github.com/MCS-Quantum/seabed)



## OBE_JAX is a JAX powered package for Bayesian inference and experimental design using sequential monte carlo methods

This package originated as a fork of [OptBayesExpt](https://github.com/usnistgov/optbayesexpt).

Since the original fork, there have been breaking API changes, Class/Variable/Function name changes,
and many other significant deviations from the original software. However, the base ParticlePDF class
is very similar to the one implemented in OptBayesExpt.

OBE_JAX requires [JAX](https://github.com/google/jax) so please follow the instructions to properly install JAX for your hardware. 

To finish the installation just do:

```bash
git clone https://github.com/MCS-Quantum/OBE_JAX
cd ./OBE_JAX
pip3 install .
```

WARNINGS:

Documentation is lacking but being added and breaking changes are likely. 

The software has not been benchmarked thoroughly, either so use at your own risk. 

## Legal stuff from OptBayesExpt

### Disclaimer
Certain commercial firms and trade names are identified in this document in
order to specify the installation and usage procedures adequately. Such
identification is not intended to imply recommendation or endorsement by the
[National Institute of Standards and Technology](http://www.nist.gov), nor
is it intended to imply that related products are necessarily the best
available for the purpose.

### Terms of Use
This software was developed by employees of the National Institute of
Standards and Technology (NIST), an agency of the Federal
Government and is being made available as a public service. Pursuant to
title 17 United States Code Section 105, works of NIST employees are not
subject to copyright protection in the United States. This software may be
subject to foreign copyright. Permission in the United States and in
foreign countries, to the extent that NIST may hold copyright, to use,
copy, modify, create derivative works, and distribute this software and its
documentation without fee is hereby granted on a non-exclusive basis,
provided that this notice and disclaimer of warranty appears in all copies.

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING,
BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
