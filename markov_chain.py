from util import *

if __name__ == '__main__':
    transfer_matrix_bach = dir_to_interval_transition_probability(r'bach_xml_transpose_cross_validation')
    transfer_matrix_chinese = dir_to_interval_transition_probability(r'guqin_xml_cross_validation')

    labels = [str(i) for i in range(13)]
    fig1 = show_heatmap(transfer_matrix_bach, labels, filename='transition_matrix_Bach', show_text=False, cmap='gray_r')
    fig2 = show_heatmap(transfer_matrix_chinese, labels, filename='transition_matrix_Guqin', show_text=False,
                        cmap='gray_r')

    prediction_all_1, prediction_all_2 = cross_validation(r'guqin_xml_cross_validation',
                                                          r'bach_xml_transpose_cross_validation',
                                                          dir_to_interval_transition_probability)

    table = get_tt_test_table(prediction_all_1, prediction_all_2, name1='Guqin', name2='Bach')
    print(table)
    table.to_csv('results/interval_matrix_distribution.csv', encoding="utf_8")
