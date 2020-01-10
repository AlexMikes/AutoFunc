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
affiliations:
 - name: Design Engineering Lab, Oregon State University
   index: 1
date: 9 January 2020
bibliography: paper.bib

---

# Summary

Each component in a product performs a corresponding set of subfunctions that contribute to the overall functionality of the product. A design repository is a database of product information that includes component and subfunction relationships. Functional modeling is an engineering design tool that organizes these subfunctions into a graphical format for better understanding and visualization. Therefore, functional modeling is a popular tool in the concept generation phase that helps designers ensure the product adheres to the customer requirements while maintaining the desired functionality. While significant work has been done to help increase consistency in the structure, syntax, and formatting of functional models, they are still highly subjective and time-consuming to create [@Pearson:2017]. Because of the time requirements, inconsistencies, and inaccuracies involved with making them, functional models are often omitted from the concept generation process, despite their useful contributions to the early stages of engineering design. 

``AutoFunc`` is a Python package that automatically generates the functional chains of components based on data from design repositories. The functional chains of components can be connected to form a complete functional model. ``AutoFunc`` also contains methods to validate and optimize the automation. A designer can use this software to input a list of components in their product, and it will automatically generate the functional chains for those components based on the most commonly seen functions and flows from previous products in the design repository. The package uses common data-mining techniques for finding information and classifying new observations based on that data. ``AutoFunc`` also uses the common methods of cross-validation and the F1 score to find the accuracy at different values for the variables involved. 

``AutoFunc`` is intended for use by engineering design researchers, students, and professionals. It has been used in several engineering design publications and presentations. Further development is required to automate a complete functional model, but this software is a significant step in that direction. Automating functional modeling will help standardize the format and syntax, decrease the time required to make them, and increase the prevalence and accuracy of functional models in engineering design and design repositories. ``AutoFunc`` has been archived to Zenodo with the linked DOI: 




# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements



# References