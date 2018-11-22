# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# Author: Jean-Matthieu Schertzer <jmschertzer@quantmetry.com>

import pandas as pd


def _compute_dummy_dict(column_names, dummy_separator):
    """Compute dummy dictionnary with categorical variables as keys and
    lists of associated modalities as values.

    Parameters
    ----------
    column_names : list,
        Column names should be in format Variable + dummy_separator + Modality.

    dummy_separator : str,
        Separator used during dummification
        (format: Variable + dummy_separator + Modality).

    Returns
    -------
    dummy_dict : dict,
        Dictionnary of categorical variables (dummy dictionnary).
    """
    dummy_dict = {}
    for var_split in [var.split(dummy_separator) for var in column_names]:
        if (len(var_split))==2:
            if var_split[0] in dummy_dict:
                dummy_dict[var_split[0]].append(var_split[1])
            else:
                dummy_dict[var_split[0]] = [var_split[1]]
    return dummy_dict


def _compute_all_dummy_variable_names(dummy_dict, dummy_separator="__"):
    """Compute all dummy variable names from dummy dictionnary.

    Parameters
    ----------
    dummy_dict : dict,
        Dictionnary of categorical variables, with categorical variables
        as keys and lists of associated modalities as values.

    dummy_separator : str,
        Separator used during dummification
        (format: Variable + dummy_separator + Modality).

    Returns
    -------
    dummy_vars : list,
        List of dummy variable names
        (format: Variable + dummy_separator + Modality).
    """
    dummy_vars = []
    for key in dummy_dict.keys():
        for value in dummy_dict[key]:
            dummy_vars.append(key+dummy_separator+value)
    return dummy_vars


def _agg_dummy_contribution(df_contrib, dummy_dict, dummy_separator="__"):
    """Sum contributions associated with different modalities, for
    each categorical variable.

    Parameters
    ----------
    df_contrib : DataFrame,
        DataFrame of contributions.

    dummy_dict : dict,
        Dictionnary of categorical variables, with categorical variables
        as keys and lists of associated modalities as values.

    dummy_separator : str, optional
        Separator used during dummification
        (format: Variable + dummy_separator + Modality).

    Returns
    -------
    agg_df_contrib : DataFrame,
        DataFrame of aggregate contributions, where modalities have been
        summed.
    """
    # Get categorical variable names
    dummy_vars = _compute_all_dummy_variable_names(dummy_dict, dummy_separator)

    # Copy contributions from numerical variables
    df_contrib_numerical = pd.DataFrame(index=df_contrib.index)
    for var in (var for var in df_contrib.columns\
             if var not in dummy_vars\
        ):
        df_contrib_numerical[var] = df_contrib[var]

    # Aggregate contributions from categorical variables
    df_contrib_categorical = pd.DataFrame(index=df_contrib.index)
    for cat_var in dummy_dict.keys():
        df_contrib_categorical[cat_var]=0
        for dummy_var in dummy_dict[cat_var]:
            df_contrib_categorical[cat_var]+=df_contrib[cat_var+dummy_separator+dummy_var]


    # Concatenate both contributions from numerical and categorical variables
    agg_df_contrib = pd.concat([df_contrib_categorical, df_contrib_numerical], axis = 1)

    return agg_df_contrib


def _agg_X_ref(X_ref, dummy_dict, dummy_separator="__"):
    """Perform an inverse dummification transformation, based on column names
    which should be formatted as Variable + dummy_separator + Modality.

    Parameters
    ----------
    X_ref : DataFrame,
        Dummified DataFrame.

    dummy_dict : dict,
        Dictionnary of categorical variables, with categorical variables
        as keys and lists of associated modalities as values.

    dummy_separator : str, optional
        Separator used during dummification
        (format: Variable + dummy_separator + Modality).

    Returns
    -------
    agg_X_ref : DataFrame,
        DataFrame, for which an inverse dummification has been applied.
    """

    # Get categorical variable names
    dummy_vars = _compute_all_dummy_variable_names(dummy_dict, dummy_separator)

    # Copy contributions from numerical variables
    agg_X_ref_numerical = pd.DataFrame(index=X_ref.index)
    for var in (var for var in X_ref.columns\
             if var not in dummy_vars\
        ):
        agg_X_ref_numerical[var] = X_ref[var]

    # Aggregate contributions from categorical variables
    agg_X_ref_categorical = pd.DataFrame(index=X_ref.index)
    for cat_var in dummy_dict.keys():
        agg_X_ref_categorical[cat_var] = '_DEFAULT_VALUE_WHEN_NO_DUMMY_IS_SET_TO_ONE_'
        for dummy_var in dummy_dict[cat_var]:
            cat_var_index = X_ref.index[X_ref[cat_var+dummy_separator+dummy_var]==1]
            agg_X_ref_categorical.loc[cat_var_index, cat_var] = dummy_var

    # Concatenate both contributions from numerical and categorical variables
    agg_X_ref = pd.concat([agg_X_ref_categorical, agg_X_ref_numerical], axis = 1)

    return agg_X_ref


def prepare_interpretable_contribution(X_ref,
                                       f_predict,
                                       df_contrib,
                                       bias,
                                       dummy_separator="__"):
    """Post processing of shap contributions to aggregate it by categorical
    variables (equivalent to inverse dummification).

    Parameters
    ----------
    X_ref : DataFrame,
        Dummified DataFrame.

    f_predict : function,
        Prediction function that could be called as f_predict(X_ref),
        to compute predictions.

    df_contrib : DataFrame,
        DataFrame of shap contributions.

    bias : flt,
        Shared term in all predictions. The following relation should hold:
        f_predict(X_ref) == bias + df_contrib

    dummy_separator : str, optional
        Separator used during dummification
        (format: Variable + dummy_separator + Modality).

    Returns
    -------
    interpretable_dict : dict,
        Dictionnary with post-processed data and contributions:
        - agg_X_ref: DataFrame, for which an inverse dummification
        has been applied
        - contrib: DataFrame of aggregate contributions, where
        modalities have been summed
        - bias: Float of shared term in all predictions
        - pred: Series of predictions.
    """

    # Contributions
    agg_df_shap = _agg_dummy_contribution(\
        df_contrib=df_contrib,
        dummy_dict=_compute_dummy_dict(X_ref.columns, dummy_separator)\
    )

    # Predictions
    pred = f_predict(X_ref)

    # X_ref
    agg_X_ref = _agg_X_ref(\
        X_ref=X_ref,
        dummy_dict=_compute_dummy_dict(X_ref.columns, dummy_separator)\
    )

    return {
        'agg_X_ref': agg_X_ref,
        'contrib': agg_df_shap,
        'bias': bias,
        'pred': pred
    }
