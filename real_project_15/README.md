# Project 15: Thyroid disease

## Topic: 
Thyroid disease prediction

## Summary: 
The data contains information on thyroid diseases for about 9'000 patients. It
includes demographics, diagnosis, medication status and blood markers. The 
dataset was created by reconciling thyroid disease datasets provided by the UCI 
Machine Learning Repository (see references).


## Data details: 
9172 observations x 31 columns

The diagnosis consists of a string of letters indicating diagnosed conditions.
A diagnosis "-" indicates no condition requiring comment.  A diagnosis of the
form "X|Y" is interpreted as "consistent with X, but more likely Y".  The
conditions are divided into groups where each group corresponds to a class of
comments.

    Letter  Diagnosis
    ------  ---------

    hyperthyroid conditions:

    A   hyperthyroid
    B   T3 toxic
    C   toxic goitre
    D   secondary toxic

    hypothyroid conditions:

    E   hypothyroid
    F   primary hypothyroid
    G   compensated hypothyroid
    H   secondary hypothyroid

    binding protein:

    I   increased binding protein
    J   decreased binding protein

    general health:

    K   concurrent non-thyroidal illness

    replacement therapy:

    L   consistent with replacement therapy
    M   underreplaced
    N   overreplaced

    antithyroid treatment:

    O   antithyroid drugs
    P   I131 treatment
    Q   surgery

    miscellaneous:

    R   discordant assay results
    S   elevated TBG
    T   elevated thyroid hormones

## [References](https://archive.ics.uci.edu/ml/datasets/thyroid+disease)
