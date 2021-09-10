import pandas as pd
from scipy import stats


def perform_statistical_analysis():
    path = f'../data/'  # Path to the directory with data table
    table_name = f'all_features'  # Name of xlsx table with all features
    features_file = 'biomarkers'  # Name of file with features, which will use to build the clock

    # Status column
    status_col_name = 'Group'
    status_options = ['Control', 'ESRD']

    # Select here features for statistical analysis
    regression_features = ['Age', 'DNAmAgeHannum', 'DNAmAge', 'DNAmPhenoAge', 'DNAmGrimAge']

    df_global = pd.read_excel(f'{path}/{table_name}.xlsx')

    with open(f'{path}/{features_file}.txt') as f:
        target_features = f.read().splitlines()

    # Create a dictionary with statistical results
    res_dict = {'metric': target_features}
    res_dict['mw_p_value'] = [] # Mann-Whitney p-values
    for a in regression_features:
        # Pearson's correlation for controls
        res_dict[f'{a}_pearson_r_C'] = []
        res_dict[f'{a}_pearson_p_value_C'] = []
        # Pearson's correlation for subjects with the ESRD
        res_dict[f'{a}_pearson_r_T'] = []
        res_dict[f'{a}_pearson_p_value_T'] = []

    for m_id, m in enumerate(target_features):

        # Associations with the status
        test_data = {}
        pb_x = {}
        for g_id, g in enumerate(status_options):
            test_data[g] = df_global.loc[df_global[status_col_name] == g][m].to_list()
            pb_x[g] = [g_id] * len(test_data[g])

        _, mw_p_value = stats.mannwhitneyu(
            test_data[status_options[0]],
            test_data[status_options[1]],
            alternative='two-sided'
        )
        res_dict['mw_p_value'].append(mw_p_value)

        # Correlations in Controls
        df_control = df_global.loc[df_global[status_col_name] == 'Control']
        for a in regression_features:
            pearson_r, pearson_p_value = stats.pearsonr(df_control[m].to_list(), df_control[a].to_list())
            res_dict[f'{a}_pearson_r_C'].append(pearson_r)
            res_dict[f'{a}_pearson_p_value_C'].append(pearson_p_value)

        # Correlations in ESRD group
        df_disease = df_global.loc[df_global[status_col_name] == 'ESRD']
        for a in regression_features:
            pearson_r, pearson_p_value = stats.pearsonr(df_disease[m].to_list(), df_disease[a].to_list())
            res_dict[f'{a}_pearson_r_T'].append(pearson_r)
            res_dict[f'{a}_pearson_p_value_T'].append(pearson_p_value)

    results_df = pd.DataFrame(res_dict)
    fn_save = f"{path}/statistics.xlsx"
    results_df.to_excel(fn_save, index=False)


if __name__ == "__main__":
    perform_statistical_analysis()