from util import *

if __name__ == '__main__':
    interval_probability_chinese = dir_to_interval_probability(r'bach_xml_transpose_cross_validation')
    interval_probability_bach = dir_to_interval_probability(r'guqin_xml_cross_validation')
    show_two_interval_count(interval_probability_bach, interval_probability_chinese,
                            'interval distribution of Bach and Guqin', label1='Bach', label2='Guqin',
                            filename='intv_d_bach_guqin')

    prediction_all_1, prediction_all_2 = cross_validation(r'guqin_xml_cross_validation',
                                                          r'bach_xml_transpose_cross_validation',
                                                          dir_to_interval_probability)

    table = get_tt_test_table(prediction_all_1, prediction_all_2, name1='Guqin', name2='Bach')
    print(table)
    table.to_csv('results/interval_distribution.csv', encoding="utf_8")
