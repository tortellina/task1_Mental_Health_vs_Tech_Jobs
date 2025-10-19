#MENTAL HEALTH STATUS IN TECH RELATED JOBS
##Machine Learning Project: Unsupervised Learning and Feature Engineering

This repository contains the source code and the data used for the analysis and segmentation of personnel in the technology sectorbased on their mental health survey responses. 
The primary goal is to support the Human Resources (HR) department in designing targeted mitigation programs.

##Project Objective
    The analysis uses Unsupervised Learning techniques to identify distinct macro-segments within the employee population. 
    By employing dimensionality reduction and clustering, the project aims to provide clear insights into risk and engagement profiles, overcoming the complexity of the raw dataset.

##Core Methodology
    - Preprocessing and Feature Engineering: The process involved rigorous data cleaning, handling missing values, normalizing categorical responses (e.g., yes/no to binary values), and creating new features.
    - Dimensionality Reduction: PCA (Principal Component Analysis) was applied to reduce the data complexity from 69 encoded features to 40 principal components, preserving 86.65% of the variance.
    - Clustering: The K-Means algorithm was used for segmentation. The optimal number of clusters was empirically determined as k=2, which successfully separated the population into two meaningful groups.
    Interpretation: Cluster composition analysis revealed that the primary differentiator was the direct experience with mental health conditions and engagement with the topic (not solely demographics or job role).


##project structure
the project include:
 - task1_Mental_Health_vs_Tech_Jobs_Analyzer.py  --> main file that has to be executed
 - mental-health-in-tech-2016_20161114.csv   --> csv survey file to be analyzed
 - requiments.txt --> requirements to be installed
 - README.md


##Execution Instructions
1. Install all necessary libraries using the provided requirements.txt file: pip install -r requirements.txt
2. clone the repository: git clone <repo-url>
3. execute task1_Mental_Health_vs_Tech_Jobs_Analyzer.py: py task1_Mental_Health_vs_Tech_Jobs_Analyzer.py

ATTENTION: if you want to perform data exploration and early statistical insight, uncomment  data_exploration(df_clean) and descriptive_analysis(df_clean) at the end of the file.

##Libraries and Technical References
The reproducibility of this analysis relies on the following key Python libraries: pandas, NumPy, scikit-learn, and matplotlib/seaborn. 
Their formal academic citations are included in the final report's BIBLIOGRAFY section.

##results
The K-Means algorithm successfully segmented the 1,433 participants into two distinct and coherent groups, with an optimal cluster number of k=2.
Cluster 0 comprises 57.1% of respondents, and Cluster 1 comprises 42.9%

The resulting profiles are:

    Cluster 0 (The 'Diagnosed' Group): This group shows a high prevalence of diagnosed or treated mental health conditions.
     They are more open to discussing and seeking help and are slightly more gender-balanced.

    Cluster 1 (The 'Non-Diagnosed' Group): This group is predominantly male, with limited or no direct experience of mental health conditions. 
     They are less likely to be engaged in related discussions.
     
The data indicate that women are approximately twice as likely to seek mental health treatment compared to men. 
This observed gender gap highlights the need for targeted interventions to encourage openness across all genders

##Contact
    For questions or suggestions, please contact:

    Student: Vittoria Tagliabue (ID: 9212846)
    Course: Machine Learning – Unsupervised Learning and Feature Engineering (DLBDSMLUSL01)