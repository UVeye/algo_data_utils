from allegroai import DataView, Dataset, Task
import tqdm
import matplotlib  # comment out for dgx
matplotlib.use('TkAgg')  # comment out for dgx
from utils_data import get_fields, plot_bars
from collections import defaultdict


def see_distributions_clearml(attr_list, clearml_datasets, frame_query=None, clearml_logger=None):
    dv = DataView(name='dv_distributions')
    for clearml_dataset, clearml_versions in clearml_datasets.items():
        if clearml_versions is None:
            clearml_versions = [v.version_name for v in Dataset.get(dataset_name=clearml_dataset).get_versions() if
                                v.version_name != 'Current']

        for clearml_version in clearml_versions:
            print('adding query', clearml_dataset, clearml_version)
            dv.add_query(dataset_name=clearml_dataset, version_name=clearml_version, frame_query=frame_query)
    num_frames = dv.get_count()[0]
    print('Number of frames: {}'.format(num_frames))

    stats = defaultdict(list)
    for frame in tqdm.tqdm(dv, total=num_frames):
        fields = get_fields(frame, attr_list)
        for field in fields:
            stats[field] += [fields[field]]

    print('\nNumber of samples: {}'.format(num_frames))
    print('\nattributes:')
    for field in stats:
        strings = stats[field]
        if isinstance(strings[0], str):
            strings = [s.strip().lower() for s in strings]

        plot_bars(strings, field, log_scale=False, x_ticks_num=False, clearml_logger=clearml_logger)
        # plot_bars(strings, field, log_scale=False, x_ticks_num='model' in field)


def main():
    project_name = 'temp_test_data_analysis'

    attr_list = [
        # 'fg_percentage',
        'body',
        'make_model',
        'make',
        'model',
        # 'siteId',
        'site',
        'cam',
        'color',
        'month',
        'hour',
        'timestamp',
        'date',
        'time',
        'frame_num',
    ]

    # clearml_datasets = {
    #     'atlas_lite__2024_Q3': None,
    #     'atlas_lite__2024_Q4': None,
    # }
    # frame_query = 'meta.frame:"frame_0000" AND meta.cam:"at_front_00"'  # ["2024-10-27T00:00:00.000Z" TO "2024-10-28T23:59:59.000Z"] AND
    # task_name = 'atlas_lite__2024_Q3_Q4'


    clearml_datasets = {
        'car_parts_granularity_2025': ['frames_for_ann__f10k_batch_0__1000',
                                       'frames_for_ann__f10k_batch_1__70']
    }
    task_name = 'testing_new_git'

    frame_query = None

    clearml_task = Task.init(project_name=project_name, task_name=task_name)
    clearml_logger = clearml_task.get_logger()
    see_distributions_clearml(attr_list=attr_list, clearml_datasets=clearml_datasets, frame_query=frame_query, clearml_logger=clearml_logger)


if __name__ == "__main__":
    main()


