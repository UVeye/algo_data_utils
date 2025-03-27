import os

from allegroai import DataView  # , DatasetVersion
import json
import pandas as pd
from collections import defaultdict, Counter
import tqdm
import pickle
import matplotlib  # comment out for dgx
matplotlib.use('TkAgg')  # comment out for dgx
import matplotlib.pyplot as plt
import cv2
import numpy as np


site_id_to_name_json_path = './siteId_to_displayName.json'
with open(site_id_to_name_json_path, 'r') as file_site_id_to_name:
    site_id_to_name = json.load(file_site_id_to_name)


def save_dic(d, path):
    with open(path, 'wb') as f:
        print('saving dictionary', path)
        pickle.dump(d, f)
        print('done saving dictionary')


def parse_timestamp(created_at):
    timestamp = pd.to_datetime(created_at)
    # if pd.isnull(timestamp):
    #     return '', ''
    date = pd.to_datetime(timestamp.date(), format="%d/%m/%Y")
    time = pd.to_datetime(timestamp.time(), format="%H:%M:%S")
    return date, time


def get_field(frame, v, proc_if_str=True):
    output = frame.metadata[v] if v in frame.metadata else ''
    if proc_if_str and isinstance(output, str):
        output = output.strip().lower()
    return output


# def get_date_time(frame):
#     created_at = get_field(frame, 'created_at')
#     date, time = parse_timestamp(created_at)
#     return date, time


def get_timestamp(frame):
    created_at = get_field(frame, 'created_at', proc_if_str=False)
    timestamp = pd.to_datetime(created_at)
    return timestamp


def get_date(frame):
    timestamp = get_timestamp(frame)
    date = pd.to_datetime(timestamp.date(), format="%d/%m/%Y")
    return date


def get_time(frame):
    timestamp = get_timestamp(frame)
    time = pd.to_datetime(timestamp.time(), format="%H:%M:%S")
    return time


def get_frame_num(frame):
    return int(frame.metadata['frame'].split('_')[-1].split('.')[0])


def get_week(frame):
    timestamp = get_timestamp(frame)
    week = timestamp.isocalendar().week
    return week

def get_month(frame):
    timestamp = get_timestamp(frame)
    month = timestamp.month
    return month

def get_month_day(frame):
    timestamp = get_timestamp(frame)
    month = timestamp.month
    day = timestamp.day
    return '{}_{}'.format(month, day)


def get_hour(frame):
    timestamp = get_timestamp(frame)
    hour = timestamp.hour
    return hour


def get_site_name(frame, strip_lower=True):
    site_id = get_field(frame, 'siteId', proc_if_str=False)
    if site_id in site_id_to_name:
        site_name = site_id_to_name[site_id]
    else:
        site_name = 'unlisted__' + site_id
    if strip_lower:
        site_name = site_name.strip().lower()
    return site_name


def get_color(frame):
    tag_color = get_field(frame, 'color')  # frame.metadata['color']
    clip_color = get_field(frame, 'clip_color')
    out_color = clip_color if tag_color == '' else tag_color
    if 'grey' in out_color.lower():
        out_color = out_color.lower()
        out_color.replace('grey', 'gray')
    return out_color


def get_make(frame):
    make = get_field(frame, 'make')
    model = get_field(frame, 'model')
    if make.strip().lower() == 'toytoa':
        make = 'toyota'
    if make.strip().lower() == 'cchevy':
        make = 'chevy'
    if make.strip().lower() == 'chevy' or make.strip().lower() == 'chev' or make.strip().lower() == 'cheverolet':
        make = 'chevrolet'
    if make.strip().lower() == 'mitsu':
        make = 'mitsubishi'
    if make.strip().lower() == 'vw':
        make = 'volkswagen'
    if 'mercedes' in model.strip().lower():
        make = model
    if 'mercedes' in make.strip().lower() and 'benz' in make.strip().lower():
        make = 'mercedes'
    return make


def get_model(frame):
    model = get_field(frame, 'model')
    if 'mercedes' in model.strip().lower():
        return get_field(frame, 'make')
    return model


def get_make_model(frame):
    make = get_make(frame)  # get_field(frame, 'make')
    model = get_model(frame)  # get_field(frame, 'model')
    make_model = make + '_' + model
    if make_model == '_':
        make_model = ''
    return make_model


def get_labels(frame):
    labels = [label for ann in frame.annotations for label in ann.labels]
    if len(labels) == 0:
        # labels = ['None']
        return ''
    # return set(labels)
    return '__'.join(sorted(set(labels)))

def get_fg_percentage(frame):
    image_height, image_width = frame.height, frame.width
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for ann in frame.annotations:
        # if ann.labels[0] == 'Inspectability_problem':
        #     continue
        # polygon_np = None
        try:
            polygon_np = np.array([ann.polygon_xy], dtype=np.int32)
            # polygon_np = np.array([ann.polygons_xy], dtype=np.int32)
        except Exception:  # as e:
            try:
                polygon_np = np.array([ann.polygons_xy], dtype=np.int32)
            except Exception:  # as e:
                continue
        if polygon_np is not None:
            cv2.fillPoly(mask, polygon_np, color=255)
    fg_percentage = 100 * (np.count_nonzero(mask) / mask.size)
    return round(fg_percentage / 5) * 5


def get_attr(frame, attr):
    get_funcs = {'color': get_color, 'make': get_make, 'model': get_model, 'make_model': get_make_model, 'site': get_site_name,
                 'timestamp': get_timestamp, 'date': get_date, 'time': get_time, 'week': get_week, 'month': get_month, 'month_day': get_month_day, 'hour': get_hour,
                 'labels': get_labels, 'fg_percentage': get_fg_percentage, 'frame_num': get_frame_num, 'lp': lambda f: get_field(f, 'licensePlate')}
    if attr in get_funcs:
        return get_funcs[attr](frame)
    else:
        return get_field(frame, attr)


def get_fields(frame, attr_list):
    fields = {attr: get_attr(frame, attr) for attr in attr_list}
    return fields


def prepare_info_list(dataset, pop_empty=True):
    scans = defaultdict(list)
    for sample in dataset:
        attrs = list(sample.keys())
        for attr in attrs:
            if pop_empty and sample[attr] == '':
                sample.pop(attr)
                continue
            if attr != 'id' and isinstance(sample[attr], str):
                sample[attr] = sample[attr].strip().lower()

        scans[sample['scan_id']].append(sample)
    for scan_id, samples in scans.items():
        colors = [sample['color'] for sample in samples]
        if not all([c == colors[0] for c in colors]):
            # print('scan color missmatch fix', scan_id)
            color_counter = Counter(colors)
            most_common_color, count = color_counter.most_common(1)[0]
            for sample in samples:
                sample['color'] = most_common_color


def plot_bars(strings, field, source_string=None, log_scale=False, x_ticks_num=False, clearml_logger=None):
    # Step 1: Count the occurrences of each string
    counter = Counter(strings)

    total_samples = sum(counter.values())
    empty_count = 0  # counter.pop('', 0)

    # Step 2: Sort the data by string names
    sorted_items = sorted(counter.items())

    # Step 3: Prepare data for plotting
    labels, counts = zip(*sorted_items)
    show_labels = [i for i in range(len(labels))] if x_ticks_num else labels
    # if x_ticks_num:
    #     labels = [i for i in range(len(labels))]
    print('\n{} has {} options'.format(field, len(counts)))

    # Step 4: Plot the histogram
    if clearml_logger is not None:
        clearml_logger.current_logger().report_histogram(
            title=field,
            series=field,
            iteration=0,
            values=counts,
            xlabels=show_labels,
            # xaxis="title x",
            yaxis="Count",
        )
        return

    plt.figure(figsize=(10, 5))
    if source_string is None:
        plt.bar(show_labels, counts, color='skyblue')
    else:
        colors = ['skyblue'] * len(labels)
        source_counter = Counter(source_string)
        for ic, label in enumerate(labels):
            assert counts[ic] <= source_counter[label]
            if counts[ic] == source_counter[label]:
                colors[ic] = 'red'
        plt.bar(show_labels, counts, color=colors)

    # plt.xlabel('Strings')
    plt.ylabel('Count')
    title = field
    if log_scale:
        plt.yscale('log')  # Set y-axis to logarithmic scale
        title = '{} (log scale)'.format(title)
    if empty_count == 0:
        plt.title(title)
    else:
        plt.title(
            f'{title} (no info for: {empty_count} / {total_samples} = {round(100 * empty_count / total_samples, 2)}%)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()

    plt.show()


def dataset_to_stats(dataset, attributes):
    stats = defaultdict(list)
    for sample in dataset:
        for attr in attributes:
            stats[attr].append(sample[attr] if attr in sample else '')
    return stats


def plot_stats(dataset, attributes, source_dataset=None):
    if len(dataset) == 0:
        return

    stats = dataset_to_stats(dataset, attributes)
    source_stats = None if source_dataset is None else dataset_to_stats(source_dataset, attributes)

    for field in stats:
        strings = stats[field]
        source_string = None if source_stats is None else source_stats[field]
        plot_bars(strings, field, source_string=source_string)
        # plot_bars(strings, field, source_string=source_string, log_scale=False, x_ticks_num='model' in field)
        # plot_bars(strings, field, source_string=source_string, log_scale=True, x_ticks_num='model' in field)

