---
title: 'AutoFunc: A Python package for automating and verifying functional modeling'
tags:
  - Python
  - engineering design
  - functional modeling
  - design repository
  - data mining
authors:
  - name: Alex Mikes
    affiliation: 1
  - name: Katherine Edmonds
    affiliation: 1
  - name: Robert B. Stone
    affiliation: 1
  - name: Bryony DuPont
    affiliation: 1
affiliations:
 - name: Design Engineering Lab, Oregon State University
   index: 1
date: 24 February 2020
bibliography: paper.bib

---

# Summary

Engineering design is a multi-step process that uses various iterative tools to help improve products. Each component 
in a product performs a corresponding set of subfunctions that contribute to the overall functionality 
of the product. Designers often store this product information, including components and subfunction relationships, in
a database known as a design repository. In addition to storing product information, it also helpful to visualize it
in a graphical representation known as a functional model. Functional modeling is a popular tool in the early design
phases that helps designers ensure the product adheres to the customer requirements while maintaining the 
desired functionality. While significant work has been done to help increase consistency in the structure, syntax, 
and formatting of functional models, they are still highly subjective and time-consuming to create [@Stone2000; @Hirtz2002; @kurtoglu2005]. 
Because of the time requirements, inconsistencies, and inaccuracies involved with making them, functional models are 
often omitted from the concept generation process, despite their useful contributions to the early stages of 
engineering design [@Kurfman2003]. 

``AutoFunc`` is a Python package that automatically generates the functional representations of components based on data from 
design repositories. The functional representations of components can be connected to form a complete functional model. 
``AutoFunc`` also contains methods to validate and optimize the automation algorithm. A designer can use this software to 
input a list of components in their product, and it will automatically generate the functional representations for those 
components based on the most commonly seen functions and flows from previous products in the design repository. 
The package uses common data-mining techniques for finding information and classifying new observations based on 
that data. ``AutoFunc`` also uses the common methods of cross-validation and the F1 score to find the accuracy at 
different values for the threshold variables.

``AutoFunc`` is intended for use by engineering design researchers, students, and professionals. It has been used in 
several engineering design publications and presentations[@edmonds2020;@mikes2020]. Further development is required to 
automate a complete functional model, but this software is a significant step in that direction. Automating functional 
modeling will help standardize the format and syntax, decrease the time required to make them, and increase the 
prevalence and accuracy of functional models in engineering design and design repositories. ``AutoFunc`` has been 
archived to Zenodo with the linked DOI [@alexmikes]


# Acknowledgements

Thanks to Kyle Niemeyer for support with open software and open science.

This material is based upon work supported by the National Science Foundation under Grant No. CMMI-1826469. 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and 
do not necessarily reflect the views of the National Science Foundation.

# References