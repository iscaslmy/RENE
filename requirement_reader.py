import xlrd


def read_all_labeled_stories(sample_path='story/labled_stories.xlsx',
                                   nrow=1551):
    """
    将初始样本加入数据库
    :param sample_path:
    :param nrow:
    :return:
    """
    # 初始样本的位置
    stories = []

    '''从表格中读取需要预测的数据，然后存入list'''
    sheet = xlrd.open_workbook(sample_path).sheet_by_index(0)
    # sheet rows
    for i in range(1, nrow):
        story_id = str(sheet.cell_value(i, 0))
        story_summary = str(sheet.cell_value(i, 1)).strip()
        story_summary = story_summary[:-1] if story_summary.endswith((u'。', u'；')) \
            else story_summary
        story_summary = story_summary.replace(',', '，')
        story_description = str(sheet.cell_value(i, 2))
        story_acceptance = str(sheet.cell_value(i, 3))
        ground_truths = [x for x in str(sheet.cell_value(i, 4)).split('/') if len(x) > 0]
        if len(ground_truths) > 0:
            stories.append((story_summary, ground_truths))

    return stories
